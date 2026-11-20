//! dmabufs from `/dev/dma_heap/system`, mapped into this process: the memory the hardware reads and writes. Each one
//! is allocated once and reused for every frame; dropping it unmaps and closes its fd.

use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{Context, Result};

pub const HEAP_PATH: &str = "/dev/dma_heap/system";

/// `struct dma_heap_allocation_data`
#[repr(C)]
struct HeapAllocation {
    len: u64,
    fd: u32,
    fd_flags: u32,
    heap_flags: u64,
}

/// `_IOWR('H', 0, struct dma_heap_allocation_data)`
const DMA_HEAP_IOCTL_ALLOC: libc::c_ulong = 0xc018_4800;
/// `_IOW('b', 0, struct dma_buf_sync)`
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x4008_6200;
const SYNC_READ: u64 = 1;
const SYNC_WRITE: u64 = 2;
const SYNC_START: u64 = 0;
const SYNC_END: u64 = 4;

/// The heap device, open only while allocating.
pub struct Heap {
    fd: OwnedFd,
}

impl Heap {
    pub fn open() -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC)
            .open(HEAP_PATH)
            .with_context(|| format!("opening {HEAP_PATH} (needs root)"))?;
        Ok(Self {
            fd: OwnedFd::from(file),
        })
    }

    pub fn alloc(&self, len: usize) -> Result<DmaBuf> {
        let mut request = HeapAllocation {
            len: len as u64,
            fd: 0,
            fd_flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
            heap_flags: 0,
        };
        // SAFETY: DMA_HEAP_IOCTL_ALLOC fills `fd` with a new dmabuf fd this process then owns.
        let ret =
            unsafe { libc::ioctl(self.fd.as_raw_fd(), DMA_HEAP_IOCTL_ALLOC as _, &mut request) };
        if ret < 0 {
            return Err(io::Error::last_os_error())
                .with_context(|| format!("allocating a {len}-byte dmabuf"));
        }
        // SAFETY: the kernel just created this fd for us.
        let fd = unsafe { OwnedFd::from_raw_fd(request.fd as RawFd) };
        // SAFETY: mapping the whole dmabuf shared; the mapping lives until Drop unmaps it.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error())
                .with_context(|| format!("mapping a {len}-byte dmabuf"));
        }
        Ok(DmaBuf {
            fd,
            ptr: ptr.cast(),
            len,
        })
    }
}

/// One mapped dmabuf. CPU access goes through [`DmaBuf::write`] / [`DmaBuf::read`], which bracket it with
/// `DMA_BUF_IOCTL_SYNC` so the CPU cache and the device agree.
pub struct DmaBuf {
    fd: OwnedFd,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the mapping is owned by this value and only reached through &self / &mut self.
unsafe impl Send for DmaBuf {}

impl DmaBuf {
    /// The fd the hardware registers take (the kernel translates it to an iova).
    pub fn fd(&self) -> u32 {
        self.fd.as_raw_fd() as u32
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    fn sync(&self, flags: u64) -> Result<()> {
        // SAFETY: DMA_BUF_IOCTL_SYNC takes a pointer to one u64 of flags.
        let ret = unsafe { libc::ioctl(self.fd.as_raw_fd(), DMA_BUF_IOCTL_SYNC as _, &flags) };
        if ret < 0 {
            return Err(io::Error::last_os_error()).context("DMA_BUF_IOCTL_SYNC");
        }
        Ok(())
    }

    /// Runs `fill` on the buffer's bytes for the CPU to write, then hands them back to the device.
    pub fn write<T>(&mut self, fill: impl FnOnce(&mut [u8]) -> T) -> Result<T> {
        self.sync(SYNC_START | SYNC_WRITE)?;
        // SAFETY: ptr..ptr+len is our live mapping and &mut self makes this the only view of it.
        let out = fill(unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) });
        self.sync(SYNC_END | SYNC_WRITE)?;
        Ok(out)
    }

    /// Runs `look` on the first `len` bytes as the device left them.
    pub fn read<T>(&self, len: usize, look: impl FnOnce(&[u8]) -> T) -> Result<T> {
        self.sync(SYNC_START | SYNC_READ)?;
        // SAFETY: ptr..ptr+len is our live mapping; the device is idle (its task has been polled to completion).
        let out = look(unsafe { std::slice::from_raw_parts(self.ptr, len.min(self.len)) });
        self.sync(SYNC_END | SYNC_READ)?;
        Ok(out)
    }
}

impl Drop for DmaBuf {
    fn drop(&mut self) {
        // SAFETY: unmapping the mapping created in Heap::alloc; the fd closes after this.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}
