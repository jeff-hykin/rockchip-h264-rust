//! MJPEG capture from a V4L2 device (a UVC camera) through raw ioctls: fixed memory-mapped buffers, set up once.
//! Struct layouts are the 64-bit Linux UAPI (`linux/videodev2.h`); each ioctl number encodes its struct's size, so a
//! layout mistake fails the call (ENOTTY) rather than corrupting memory.

use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};

const BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
const MEMORY_MMAP: u32 = 1;
const FIELD_NONE: u32 = 1;
const PIX_FMT_MJPEG: u32 = u32::from_le_bytes(*b"MJPG");

#[repr(C)]
#[derive(Clone, Copy)]
struct PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv_: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}

/// `struct v4l2_format`: the union is pointer-aligned, hence the padding after `type`
#[repr(C)]
struct Format {
    type_: u32,
    _pad: u32,
    pix: PixFormat,
    _rest: [u8; 200 - std::mem::size_of::<PixFormat>()],
}

#[repr(C)]
struct CaptureParm {
    capability: u32,
    capturemode: u32,
    numerator: u32,
    denominator: u32,
    extendedmode: u32,
    readbuffers: u32,
    reserved: [u32; 4],
}

/// `struct v4l2_streamparm`
#[repr(C)]
struct StreamParm {
    type_: u32,
    capture: CaptureParm,
    _rest: [u8; 200 - std::mem::size_of::<CaptureParm>()],
}

/// `struct v4l2_requestbuffers`
#[repr(C)]
struct RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    flags: u32,
}

/// `struct v4l2_buffer` (64-bit: timeval is 16 bytes, the `m` union 8)
#[repr(C)]
#[derive(Clone, Copy)]
struct Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    _pad: u32,
    timestamp: [i64; 2],
    timecode: [u32; 4],
    sequence: u32,
    memory: u32,
    offset: u64,
    length: u32,
    reserved2: u32,
    request_fd: u32,
    _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<Format>() == 208);
const _: () = assert!(std::mem::size_of::<StreamParm>() == 204);
const _: () = assert!(std::mem::size_of::<RequestBuffers>() == 20);
const _: () = assert!(std::mem::size_of::<Buffer>() == 88);

const fn iowr(nr: u32, size: usize) -> libc::c_ulong {
    (3 << 30 | (size as u32) << 16 | (b'V' as u32) << 8 | nr) as libc::c_ulong
}
const fn iow(nr: u32, size: usize) -> libc::c_ulong {
    (1 << 30 | (size as u32) << 16 | (b'V' as u32) << 8 | nr) as libc::c_ulong
}
const VIDIOC_S_FMT: libc::c_ulong = iowr(5, 208);
const VIDIOC_REQBUFS: libc::c_ulong = iowr(8, 20);
const VIDIOC_QUERYBUF: libc::c_ulong = iowr(9, 88);
const VIDIOC_QBUF: libc::c_ulong = iowr(15, 88);
const VIDIOC_DQBUF: libc::c_ulong = iowr(17, 88);
const VIDIOC_STREAMON: libc::c_ulong = iow(18, 4);
const VIDIOC_STREAMOFF: libc::c_ulong = iow(19, 4);
const VIDIOC_S_PARM: libc::c_ulong = iowr(22, 204);

fn xioctl<T>(fd: &OwnedFd, request: libc::c_ulong, arg: &mut T, what: &str) -> Result<()> {
    loop {
        // SAFETY: `arg` is the struct this ioctl number was built for (sizes asserted above).
        let ret = unsafe { libc::ioctl(fd.as_raw_fd(), request as _, arg as *mut T) };
        if ret >= 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error).context(what.to_string());
        }
    }
}

struct Mapping {
    ptr: *mut u8,
    len: usize,
}

/// An MJPEG stream off one V4L2 device. Dropping it stops streaming, unmaps the buffers and closes the device.
pub struct Capture {
    fd: OwnedFd,
    buffers: Vec<Mapping>,
    streaming: bool,
}

// SAFETY: the mappings are owned by this value and only reached through &mut self.
unsafe impl Send for Capture {}

impl Capture {
    pub fn open(
        device: &str,
        width: u32,
        height: u32,
        fps: u32,
        buffer_count: u32,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(device)
            .with_context(|| format!("opening {device}"))?;
        let mut capture = Self {
            fd: OwnedFd::from(file),
            buffers: Vec::new(),
            streaming: false,
        };
        // SAFETY: plain-old-data structs, all-zero is their documented initial state
        let mut format: Format = unsafe { std::mem::zeroed() };
        format.type_ = BUF_TYPE_VIDEO_CAPTURE;
        format.pix.width = width;
        format.pix.height = height;
        format.pix.pixelformat = PIX_FMT_MJPEG;
        format.pix.field = FIELD_NONE;
        xioctl(&capture.fd, VIDIOC_S_FMT, &mut format, "VIDIOC_S_FMT")?;
        ensure!(
            format.pix.pixelformat == PIX_FMT_MJPEG
                && (format.pix.width, format.pix.height) == (width, height),
            "{device} gave {}x{} {:?} instead of {width}x{height} MJPEG",
            format.pix.width,
            format.pix.height,
            format.pix.pixelformat.to_le_bytes().map(char::from)
        );
        // SAFETY: as above
        let mut parm: StreamParm = unsafe { std::mem::zeroed() };
        parm.type_ = BUF_TYPE_VIDEO_CAPTURE;
        parm.capture.numerator = 1;
        parm.capture.denominator = fps;
        xioctl(&capture.fd, VIDIOC_S_PARM, &mut parm, "VIDIOC_S_PARM")?;

        let mut request = RequestBuffers {
            count: buffer_count,
            type_: BUF_TYPE_VIDEO_CAPTURE,
            memory: MEMORY_MMAP,
            capabilities: 0,
            flags: 0,
        };
        xioctl(&capture.fd, VIDIOC_REQBUFS, &mut request, "VIDIOC_REQBUFS")?;
        ensure!(
            request.count >= 2,
            "{device} gave {} capture buffers",
            request.count
        );
        for index in 0..request.count {
            let mut buffer = Self::blank(index);
            xioctl(&capture.fd, VIDIOC_QUERYBUF, &mut buffer, "VIDIOC_QUERYBUF")?;
            // SAFETY: mapping the driver's buffer `index` at the offset it reported
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    buffer.length as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    capture.fd.as_raw_fd(),
                    buffer.offset as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error()).context("mapping a capture buffer");
            }
            capture.buffers.push(Mapping {
                ptr: ptr.cast(),
                len: buffer.length as usize,
            });
            xioctl(&capture.fd, VIDIOC_QBUF, &mut buffer, "VIDIOC_QBUF")?;
        }
        let mut kind = BUF_TYPE_VIDEO_CAPTURE as i32;
        xioctl(&capture.fd, VIDIOC_STREAMON, &mut kind, "VIDIOC_STREAMON")?;
        capture.streaming = true;
        Ok(capture)
    }

    fn blank(index: u32) -> Buffer {
        // SAFETY: plain-old-data
        let mut buffer: Buffer = unsafe { std::mem::zeroed() };
        buffer.index = index;
        buffer.type_ = BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = MEMORY_MMAP;
        buffer
    }

    /// Waits up to `timeout` for the next frame and hands its bytes to `take`, then gives the buffer back.
    pub fn next<T>(&mut self, timeout: Duration, take: impl FnOnce(&[u8]) -> T) -> Result<T> {
        let mut poll = libc::pollfd {
            fd: self.fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ms = timeout.as_millis().min(i32::MAX as u128) as i32;
        loop {
            // SAFETY: one pollfd
            let ret = unsafe { libc::poll(&mut poll, 1, ms) };
            if ret < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error).context("polling the camera");
            }
            if ret == 0 {
                bail!("no frame for {timeout:?}");
            }
            if poll.revents & libc::POLLERR != 0 {
                bail!("the camera reported an error (unplugged?)");
            }
            break;
        }
        let mut buffer = Self::blank(0);
        xioctl(&self.fd, VIDIOC_DQBUF, &mut buffer, "VIDIOC_DQBUF")?;
        let mapping = self
            .buffers
            .get(buffer.index as usize)
            .context("the driver returned an unknown buffer")?;
        // SAFETY: the dequeued buffer is ours until it is queued again below
        let bytes = unsafe {
            std::slice::from_raw_parts(mapping.ptr, (buffer.bytesused as usize).min(mapping.len))
        };
        let out = take(bytes);
        xioctl(&self.fd, VIDIOC_QBUF, &mut buffer, "VIDIOC_QBUF")?;
        Ok(out)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if self.streaming {
            let mut kind = BUF_TYPE_VIDEO_CAPTURE as i32;
            let _ = xioctl(&self.fd, VIDIOC_STREAMOFF, &mut kind, "VIDIOC_STREAMOFF");
        }
        for mapping in &self.buffers {
            // SAFETY: unmapping what open() mapped
            unsafe { libc::munmap(mapping.ptr.cast(), mapping.len) };
        }
    }
}
