//! The few OpenCL calls the GPU remap needs, loaded at run time from the board's Mali driver (`libmali`), so nothing
//! links against OpenCL and a machine without it just reports the GPU path unavailable. Includes
//! `cl_arm_import_memory` so the GPU reads and writes the decoder's and encoder's dmabufs in place.

use std::ffi::{c_void, CString};
use std::ptr::null_mut;
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};

pub type Handle = *mut c_void;

const CL_SUCCESS: i32 = 0;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
pub const CL_MEM_READ_WRITE: u64 = 1 << 0;
pub const CL_MEM_WRITE_ONLY: u64 = 1 << 1;
pub const CL_MEM_READ_ONLY: u64 = 1 << 2;
const CL_MEM_COPY_HOST_PTR: u64 = 1 << 5;
const CL_IMPORT_TYPE_ARM: isize = 0x40B2;
const CL_IMPORT_TYPE_DMA_BUF_ARM: isize = 0x40B4;

/// The driver libraries tried, in order.
const LIBRARIES: [&str; 4] = [
    "libmali.so.1",
    "libMaliOpenCL.so.1",
    "libOpenCL.so.1",
    "/System/Library/Frameworks/OpenCL.framework/OpenCL",
];

macro_rules! api {
    ($($name:ident: fn($($arg:ty),*) -> $ret:ty;)*) => {
        #[allow(non_snake_case)]
        struct Api {
            $($name: unsafe extern "C" fn($($arg),*) -> $ret,)*
        }

        impl Api {
            unsafe fn load(lib: Handle) -> Result<Self> {
                Ok(Self {
                    $($name: {
                        let symbol = libc::dlsym(lib, concat!(stringify!($name), "\0").as_ptr().cast());
                        ensure!(!symbol.is_null(), concat!("the OpenCL driver has no ", stringify!($name)));
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> $ret>(symbol)
                    },)*
                })
            }
        }
    };
}

api! {
    clGetPlatformIDs: fn(u32, *mut Handle, *mut u32) -> i32;
    clGetDeviceIDs: fn(Handle, u64, u32, *mut Handle, *mut u32) -> i32;
    clCreateContext: fn(*const isize, u32, *const Handle, Handle, Handle, *mut i32) -> Handle;
    clCreateCommandQueue: fn(Handle, Handle, u64, *mut i32) -> Handle;
    clCreateProgramWithSource: fn(Handle, u32, *const *const std::ffi::c_char, *const usize, *mut i32) -> Handle;
    clBuildProgram: fn(Handle, u32, *const Handle, *const std::ffi::c_char, Handle, Handle) -> i32;
    clGetProgramBuildInfo: fn(Handle, Handle, u32, usize, *mut c_void, *mut usize) -> i32;
    clCreateKernel: fn(Handle, *const std::ffi::c_char, *mut i32) -> Handle;
    clCreateBuffer: fn(Handle, u64, usize, *mut c_void, *mut i32) -> Handle;
    clSetKernelArg: fn(Handle, u32, usize, *const c_void) -> i32;
    clEnqueueNDRangeKernel: fn(Handle, Handle, u32, *const usize, *const usize, *const usize, u32, *const Handle, *mut Handle) -> i32;
    clEnqueueWriteBuffer: fn(Handle, Handle, u32, usize, usize, *const c_void, u32, *const Handle, *mut Handle) -> i32;
    clEnqueueReadBuffer: fn(Handle, Handle, u32, usize, usize, *mut c_void, u32, *const Handle, *mut Handle) -> i32;
    clFinish: fn(Handle) -> i32;
    clReleaseMemObject: fn(Handle) -> i32;
    clReleaseKernel: fn(Handle) -> i32;
    clReleaseProgram: fn(Handle) -> i32;
    clReleaseCommandQueue: fn(Handle) -> i32;
    clReleaseContext: fn(Handle) -> i32;
}

fn check(code: i32, what: &str) -> Result<()> {
    if code != CL_SUCCESS {
        bail!("{what} failed (OpenCL error {code})");
    }
    Ok(())
}

type ImportMemoryArm =
    unsafe extern "C" fn(Handle, u64, *const isize, *mut c_void, usize, *mut i32) -> Handle;

struct Inner {
    api: Api,
    /// `clImportMemoryARM`, on drivers with `cl_arm_import_memory` (Mali)
    import_memory_arm: Option<ImportMemoryArm>,
    lib: Handle,
    device: Handle,
    context: Handle,
    queue: Handle,
}

// SAFETY: OpenCL objects may be used from any thread; each Gpu is driven by one thread at a time.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: the last reference: every buffer and kernel (each holding an Arc) is gone
        unsafe {
            (self.api.clReleaseCommandQueue)(self.queue);
            (self.api.clReleaseContext)(self.context);
            libc::dlclose(self.lib);
        }
    }
}

/// One GPU context and in-order queue.
#[derive(Clone)]
pub struct Gpu {
    inner: Arc<Inner>,
}

impl Gpu {
    /// Loads the driver and opens the first GPU.
    pub fn open() -> Result<Self> {
        let lib = LIBRARIES
            .iter()
            .find_map(|name| {
                let name = CString::new(*name).ok()?;
                // SAFETY: dlopen of a shared library by name
                let lib = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
                (!lib.is_null()).then_some(lib)
            })
            .with_context(|| format!("no OpenCL driver on this machine (tried {LIBRARIES:?})"))?;
        // SAFETY: the symbols are the OpenCL 1.2 entry points with the signatures declared above
        let api = unsafe { Api::load(lib)? };
        // SAFETY: an optional extension entry point; null where the driver lacks it
        let import_memory_arm = unsafe {
            let symbol = libc::dlsym(lib, c"clImportMemoryARM".as_ptr());
            (!symbol.is_null()).then(|| std::mem::transmute::<*mut c_void, ImportMemoryArm>(symbol))
        };
        let (mut platform, mut device) = (null_mut(), null_mut());
        let mut count = 0;
        // SAFETY: out-parameters sized for one entry
        unsafe {
            check(
                (api.clGetPlatformIDs)(1, &mut platform, &mut count),
                "clGetPlatformIDs",
            )?;
            ensure!(count > 0, "no OpenCL platform");
            check(
                (api.clGetDeviceIDs)(platform, CL_DEVICE_TYPE_GPU, 1, &mut device, &mut count),
                "clGetDeviceIDs",
            )?;
        }
        let mut error = 0;
        // SAFETY: one device, no properties or callback
        let context = unsafe {
            (api.clCreateContext)(
                std::ptr::null(),
                1,
                &device,
                null_mut(),
                null_mut(),
                &mut error,
            )
        };
        check(error, "clCreateContext")?;
        // SAFETY: a default in-order queue on that device
        let queue = unsafe { (api.clCreateCommandQueue)(context, device, 0, &mut error) };
        if let Err(e) = check(error, "clCreateCommandQueue") {
            // SAFETY: releasing what was just created
            unsafe { (api.clReleaseContext)(context) };
            return Err(e);
        }
        Ok(Self {
            inner: Arc::new(Inner {
                api,
                import_memory_arm,
                lib,
                device,
                context,
                queue,
            }),
        })
    }

    /// Builds `source` and returns its kernel `name`.
    pub fn kernel(&self, source: &str, name: &str) -> Result<Kernel> {
        let source = CString::new(source)?;
        let mut error = 0;
        let ptr = source.as_ptr();
        // SAFETY: one NUL-terminated source string
        let program = unsafe {
            (self.inner.api.clCreateProgramWithSource)(
                self.inner.context,
                1,
                &ptr,
                std::ptr::null(),
                &mut error,
            )
        };
        check(error, "clCreateProgramWithSource")?;
        // SAFETY: building for our one device
        let built = unsafe {
            (self.inner.api.clBuildProgram)(
                program,
                1,
                &self.inner.device,
                c"-cl-fast-relaxed-math".as_ptr(),
                null_mut(),
                null_mut(),
            )
        };
        if built != CL_SUCCESS {
            let mut log = vec![0u8; 16384];
            let mut len = 0;
            // SAFETY: a byte buffer for the log
            unsafe {
                (self.inner.api.clGetProgramBuildInfo)(
                    program,
                    self.inner.device,
                    CL_PROGRAM_BUILD_LOG,
                    log.len(),
                    log.as_mut_ptr().cast(),
                    &mut len,
                )
            };
            // SAFETY: releasing the failed program
            unsafe { (self.inner.api.clReleaseProgram)(program) };
            log.truncate(len);
            bail!(
                "OpenCL build failed ({built}): {}",
                String::from_utf8_lossy(&log)
            );
        }
        let name = CString::new(name)?;
        // SAFETY: a kernel of the built program
        let kernel = unsafe { (self.inner.api.clCreateKernel)(program, name.as_ptr(), &mut error) };
        if let Err(e) = check(error, "clCreateKernel") {
            // SAFETY: releasing the program
            unsafe { (self.inner.api.clReleaseProgram)(program) };
            return Err(e);
        }
        Ok(Kernel {
            inner: self.inner.clone(),
            program,
            kernel,
        })
    }

    /// A device buffer holding a copy of `data`.
    pub fn buffer_from<T: Copy>(&self, flags: u64, data: &[T]) -> Result<Buffer> {
        let mut error = 0;
        let size = std::mem::size_of_val(data).max(1);
        // SAFETY: COPY_HOST_PTR reads `size` bytes from `data` during the call
        let mem = unsafe {
            (self.inner.api.clCreateBuffer)(
                self.inner.context,
                flags | CL_MEM_COPY_HOST_PTR,
                size,
                data.as_ptr() as *mut c_void,
                &mut error,
            )
        };
        check(error, "clCreateBuffer")?;
        Ok(Buffer {
            inner: self.inner.clone(),
            mem,
            size,
        })
    }

    /// An empty device buffer of `size` bytes.
    pub fn buffer(&self, flags: u64, size: usize) -> Result<Buffer> {
        let mut error = 0;
        // SAFETY: a plain allocation
        let mem = unsafe {
            (self.inner.api.clCreateBuffer)(self.inner.context, flags, size, null_mut(), &mut error)
        };
        check(error, "clCreateBuffer")?;
        Ok(Buffer {
            inner: self.inner.clone(),
            mem,
            size,
        })
    }

    /// The GPU's view of a dmabuf (no copy): `cl_arm_import_memory` with `CL_IMPORT_TYPE_DMA_BUF_ARM`.
    pub fn import_dmabuf(&self, flags: u64, fd: i32, size: usize) -> Result<Buffer> {
        let properties = [CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0];
        let mut fd = fd;
        let mut error = 0;
        // SAFETY: the driver reads the fd through the pointer; it keeps its own reference to the dmabuf
        let import = self
            .inner
            .import_memory_arm
            .context("this OpenCL driver cannot import dmabufs (no cl_arm_import_memory)")?;
        let mem = unsafe {
            import(
                self.inner.context,
                flags,
                properties.as_ptr(),
                (&mut fd as *mut i32).cast(),
                size,
                &mut error,
            )
        };
        check(error, "clImportMemoryARM (dma_buf)")?;
        Ok(Buffer {
            inner: self.inner.clone(),
            mem,
            size,
        })
    }

    pub fn write(&self, buffer: &Buffer, data: &[u8]) -> Result<()> {
        ensure!(data.len() <= buffer.size, "write past the buffer");
        // SAFETY: a blocking write of `data`
        check(
            unsafe {
                (self.inner.api.clEnqueueWriteBuffer)(
                    self.inner.queue,
                    buffer.mem,
                    1,
                    0,
                    data.len(),
                    data.as_ptr().cast(),
                    0,
                    std::ptr::null(),
                    null_mut(),
                )
            },
            "clEnqueueWriteBuffer",
        )
    }

    pub fn read(&self, buffer: &Buffer, data: &mut [u8]) -> Result<()> {
        ensure!(data.len() <= buffer.size, "read past the buffer");
        // SAFETY: a blocking read into `data`
        check(
            unsafe {
                (self.inner.api.clEnqueueReadBuffer)(
                    self.inner.queue,
                    buffer.mem,
                    1,
                    0,
                    data.len(),
                    data.as_mut_ptr().cast(),
                    0,
                    std::ptr::null(),
                    null_mut(),
                )
            },
            "clEnqueueReadBuffer",
        )
    }

    /// Waits for everything queued.
    pub fn finish(&self) -> Result<()> {
        // SAFETY: our queue
        check(
            unsafe { (self.inner.api.clFinish)(self.inner.queue) },
            "clFinish",
        )
    }
}

pub struct Buffer {
    inner: Arc<Inner>,
    mem: Handle,
    size: usize,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: our memory object
        unsafe { (self.inner.api.clReleaseMemObject)(self.mem) };
    }
}

/// A kernel argument.
pub enum Arg<'a> {
    Buffer(&'a Buffer),
    U32(u32),
}

pub struct Kernel {
    inner: Arc<Inner>,
    program: Handle,
    kernel: Handle,
}

impl Kernel {
    /// Sets the arguments and queues a 2D launch of `global` work items (not waiting for it).
    pub fn launch(&self, args: &[Arg], global: [usize; 2]) -> Result<()> {
        for (i, arg) in args.iter().enumerate() {
            // SAFETY: the argument types match the kernel's (a buffer handle or a uint)
            let code = unsafe {
                match arg {
                    Arg::Buffer(b) => (self.inner.api.clSetKernelArg)(
                        self.kernel,
                        i as u32,
                        std::mem::size_of::<Handle>(),
                        (&b.mem as *const Handle).cast(),
                    ),
                    Arg::U32(v) => (self.inner.api.clSetKernelArg)(
                        self.kernel,
                        i as u32,
                        4,
                        (v as *const u32).cast(),
                    ),
                }
            };
            check(code, "clSetKernelArg")?;
        }
        // SAFETY: a 2D range, driver-chosen local size
        check(
            unsafe {
                (self.inner.api.clEnqueueNDRangeKernel)(
                    self.inner.queue,
                    self.kernel,
                    2,
                    std::ptr::null(),
                    global.as_ptr(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    null_mut(),
                )
            },
            "clEnqueueNDRangeKernel",
        )
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: our kernel and program
        unsafe {
            (self.inner.api.clReleaseKernel)(self.kernel);
            (self.inner.api.clReleaseProgram)(self.program);
        }
    }
}
