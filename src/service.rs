//! The `/dev/mpp_service` ioctl ABI: one session per hardware client, a task is a batch of register writes, reads and
//! fd-offset fixups sent in one `MPP_IOC_CFG_V1` call, then a blocking poll until the hardware is done.
//!
//! Mirrors MPP `osal/driver/mpp_service.c` + `osal/inc/mpp_service.h` and the kernel's `mpp_common.c` (`mpp_msg_v1`).

use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{bail, Context, Result};

pub const SERVICE_PATH: &str = "/dev/mpp_service";

/// `_IOW('v', 1, unsigned int)`
const MPP_IOC_CFG_V1: libc::c_ulong = 0x4004_7601;

/// Command ids (`MppServiceCmdType`).
pub mod cmd {
    pub const QUERY_BASE: u32 = 0x000;
    pub const PROBE_HW_SUPPORT: u32 = 0x000;
    pub const QUERY_HW_ID: u32 = 0x001;
    pub const QUERY_CMD_SUPPORT: u32 = 0x002;
    pub const INIT_BASE: u32 = 0x100;
    pub const INIT_CLIENT_TYPE: u32 = 0x100;
    pub const SEND_BASE: u32 = 0x200;
    pub const SET_REG_WRITE: u32 = 0x200;
    pub const SET_REG_READ: u32 = 0x201;
    pub const SET_REG_ADDR_OFFSET: u32 = 0x202;
    pub const POLL_BASE: u32 = 0x300;
    pub const POLL_HW_FINISH: u32 = 0x300;
    pub const CONTROL_BASE: u32 = 0x400;
    pub const SEND_CODEC_INFO: u32 = 0x403;
}

const FLAG_MULTI_MSG: u32 = 0x01;
const FLAG_LAST_MSG: u32 = 0x02;
const FLAG_REG_OFFSET_ALONE: u32 = 0x10;

/// The kernel takes at most this many requests per task (`MPP_MAX_MSG_NUM`).
const MAX_REQUESTS: usize = 16;

/// Hardware client types (`MppClientType`): the index of the client's bit in `PROBE_HW_SUPPORT`.
pub mod client {
    pub const RKJPEGD: u32 = 13;
    pub const RKVENC: u32 = 16;
}

/// `mpp_msg_v1` / `MppReqV1`
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct Request {
    cmd: u32,
    flag: u32,
    size: u32,
    offset: u32,
    data_ptr: u64,
}

const _: () = assert!(std::mem::size_of::<Request>() == 24);

/// One register range to write: its byte offset in the device's register space and the words.
pub struct Write<'a> {
    pub offset: u32,
    pub words: &'a [u32],
}

/// One register range to read back once the task finishes.
pub struct Read<'a> {
    pub offset: u32,
    pub words: &'a mut [u32],
}

/// Adds `offset` bytes to the iova the kernel substitutes for the dmabuf fd in register `index`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddrOffset {
    pub index: u32,
    pub offset: u32,
}

fn open_service() -> Result<OwnedFd> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_CLOEXEC)
        .open(SERVICE_PATH)
        .with_context(|| format!("opening {SERVICE_PATH} (needs root)"))?;
    Ok(OwnedFd::from(file))
}

fn ioctl(fd: &OwnedFd, requests: &mut [Request]) -> io::Result<()> {
    // SAFETY: the kernel reads a chain of `mpp_msg_v1` ending at the one flagged LAST (or the first, if not MULTI);
    // every data pointer in it points at memory that outlives the call.
    let ret = unsafe { libc::ioctl(fd.as_raw_fd(), MPP_IOC_CFG_V1 as _, requests.as_mut_ptr()) };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// A single-message query or init command whose payload is one u32 (in and out).
fn single(fd: &OwnedFd, cmd: u32, value: &mut u32, size: u32) -> io::Result<()> {
    let mut request = [Request {
        cmd,
        flag: 0,
        size,
        offset: 0,
        data_ptr: value as *mut u32 as u64,
    }];
    ioctl(fd, &mut request)
}

/// `MPP_CMD_PROBE_HW_SUPPORT`: one bit per client type the driver has hardware for.
pub fn probe_hw_support() -> Result<u32> {
    let fd = open_service()?;
    let mut value = 0;
    single(&fd, cmd::PROBE_HW_SUPPORT, &mut value, 0).context("MPP_CMD_PROBE_HW_SUPPORT")?;
    Ok(value)
}

/// `MPP_CMD_QUERY_CMD_SUPPORT` for one command base: the first command id past the ones the driver supports.
pub fn query_cmd_support(base: u32) -> Result<u32> {
    let fd = open_service()?;
    let mut value = base;
    single(&fd, cmd::QUERY_CMD_SUPPORT, &mut value, 0)
        .with_context(|| format!("MPP_CMD_QUERY_CMD_SUPPORT {base:#x}"))?;
    Ok(value)
}

/// `MPP_CMD_QUERY_HW_ID` of one client (the kernel needs the client type set on the fd first).
pub fn query_hw_id(client: u32) -> Result<u32> {
    let session = Session::new(client)?;
    let mut value = 0;
    single(&session.fd, cmd::QUERY_HW_ID, &mut value, 4)
        .with_context(|| format!("MPP_CMD_QUERY_HW_ID client {client}"))?;
    Ok(value)
}

/// An open `/dev/mpp_service` fd bound to one hardware client. Dropping it closes the kernel session, which releases
/// every dmabuf the kernel imported for it.
pub struct Session {
    fd: OwnedFd,
}

impl Session {
    pub fn new(client: u32) -> Result<Self> {
        let fd = open_service()?;
        let mut value = client;
        single(&fd, cmd::INIT_CLIENT_TYPE, &mut value, 4)
            .with_context(|| format!("MPP_CMD_INIT_CLIENT_TYPE {client}"))?;
        Ok(Self { fd })
    }

    /// `MPP_CMD_SEND_CODEC_INFO`: the key/value pairs `/proc/mpp_service/*/sessions-info` shows.
    pub fn send_codec_info(&self, items: &[[u32; 4]]) -> Result<()> {
        let mut request = [Request {
            cmd: cmd::SEND_CODEC_INFO,
            flag: FLAG_LAST_MSG,
            size: std::mem::size_of_val(items) as u32,
            offset: 0,
            data_ptr: items.as_ptr() as u64,
        }];
        ioctl(&self.fd, &mut request).context("MPP_CMD_SEND_CODEC_INFO")
    }

    /// Runs one task: writes, reads and fd offsets in one request chain, then blocks until the hardware finishes and
    /// the kernel has filled `reads`.
    pub fn run(&self, writes: &[Write], reads: &mut [Read], offsets: &[AddrOffset]) -> Result<()> {
        let count = writes.len() + reads.len() + usize::from(!offsets.is_empty());
        if count == 0 || count > MAX_REQUESTS {
            bail!("{count} requests in one task (1..={MAX_REQUESTS})");
        }
        let mut requests = [Request::default(); MAX_REQUESTS];
        let mut at = 0;
        for write in writes {
            requests[at] = Request {
                cmd: cmd::SET_REG_WRITE,
                flag: FLAG_MULTI_MSG,
                size: (write.words.len() * 4) as u32,
                offset: write.offset,
                data_ptr: write.words.as_ptr() as u64,
            };
            at += 1;
        }
        for read in reads.iter_mut() {
            requests[at] = Request {
                cmd: cmd::SET_REG_READ,
                flag: FLAG_MULTI_MSG,
                size: (read.words.len() * 4) as u32,
                offset: read.offset,
                data_ptr: read.words.as_mut_ptr() as u64,
            };
            at += 1;
        }
        if !offsets.is_empty() {
            requests[at] = Request {
                cmd: cmd::SET_REG_ADDR_OFFSET,
                flag: FLAG_MULTI_MSG | FLAG_REG_OFFSET_ALONE,
                size: std::mem::size_of_val(offsets) as u32,
                offset: 0,
                data_ptr: offsets.as_ptr() as u64,
            };
            at += 1;
        }
        requests[at - 1].flag |= FLAG_LAST_MSG | FLAG_REG_OFFSET_ALONE;
        ioctl(&self.fd, &mut requests[..at]).context("sending the task to mpp_service")?;

        let mut poll = [Request {
            cmd: cmd::POLL_HW_FINISH,
            flag: FLAG_LAST_MSG | FLAG_REG_OFFSET_ALONE,
            ..Request::default()
        }];
        loop {
            match ioctl(&self.fd, &mut poll) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => {
                    return result.context("waiting for the hardware (MPP_CMD_POLL_HW_FINISH)")
                }
            }
        }
    }
}
