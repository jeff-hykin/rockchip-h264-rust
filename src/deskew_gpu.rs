//! The deskew remap on the RK3588's Mali-G610 through OpenCL: the same tap tables as the CPU path (so the output is
//! bit-identical), uploaded once per input size; each frame is two kernel launches reading the decoder's dmabuf and
//! writing the encoder's, both imported with `cl_arm_import_memory` (no copies).

use std::collections::HashMap;

use anyhow::{ensure, Result};

use crate::deskew::{chroma_rows, layout, Deskew};
use crate::dma::DmaBuf;
use crate::opencl::{
    Arg, Buffer, Gpu, Kernel, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY,
};

/// One work item per output sample; `step` bytes per sample (1 luma, 2 interleaved chroma). The arithmetic is the
/// CPU path's (`deskew::remap_plane`), 5-bit fixed-point weights included.
const KERNEL: &str = r#"
__kernel void remap(__global const uchar* src, uint src_offset, uint src_stride,
                    __global uchar* dst, uint dst_offset, uint dst_stride,
                    __global const uint* taps, uint out_width, uint fill, uint bilinear, uint step)
{
    uint u = get_global_id(0), v = get_global_id(1);
    uint tap = taps[v * out_width + u];
    uint o = dst_offset + v * dst_stride + u * step;
    if (tap == 0xffffffffu) {
        for (uint c = 0; c < step; c++) dst[o + c] = (uchar)fill;
        return;
    }
    uint i = src_offset + (tap >> 10);
    if (!bilinear) {
        for (uint c = 0; c < step; c++) dst[o + c] = src[i + c];
        return;
    }
    uint wx = (tap >> 5) & 31u, wy = tap & 31u;
    for (uint c = 0; c < step; c++) {
        uint a = src[i + c], b = src[i + step + c];
        uint d = src[i + src_stride + c], e = src[i + src_stride + step + c];
        uint top = a * (32u - wx) + b * wx;
        uint bottom = d * (32u - wx) + e * wx;
        dst[o + c] = (uchar)((top * (32u - wy) + bottom * wy + 512u) >> 10);
    }
}
"#;

pub struct GpuRemap {
    gpu: Gpu,
    kernel: Kernel,
    /// (luma, chroma) tap tables for the input the deskew was last prepared for
    taps: Option<(Buffer, Buffer)>,
    /// the [`Deskew::generation`] of the uploaded tables
    uploaded: u64,
    /// dmabufs imported so far, by fd: the pipeline reuses the same few buffers, so each is imported once
    imported: HashMap<i32, Buffer>,
}

impl GpuRemap {
    /// Opens the GPU and builds the kernel (fails clearly where there is no OpenCL driver).
    pub fn new() -> Result<Self> {
        let gpu = Gpu::open()?;
        let kernel = gpu.kernel(KERNEL, "remap")?;
        Ok(Self {
            gpu,
            kernel,
            taps: None,
            uploaded: 0,
            imported: HashMap::new(),
        })
    }

    fn upload(&mut self, deskew: &Deskew) -> Result<()> {
        ensure!(deskew.generation() != 0, "deskew tables not built");
        if self.uploaded == deskew.generation() {
            return Ok(());
        }
        let (luma, chroma) = deskew.tables().expect("built");
        self.taps = Some((
            self.gpu.buffer_from(CL_MEM_READ_ONLY, luma)?,
            self.gpu.buffer_from(CL_MEM_READ_ONLY, chroma)?,
        ));
        self.uploaded = deskew.generation();
        // a new size means new frame buffers: drop the old imports
        self.imported.clear();
        Ok(())
    }

    fn import(&mut self, buf: &DmaBuf, flags: u64) -> Result<()> {
        let fd = buf.fd() as i32;
        if !self.imported.contains_key(&fd) {
            let mem = self.gpu.import_dmabuf(flags, fd, buf.len())?;
            self.imported.insert(fd, mem);
        }
        Ok(())
    }

    /// Remaps `src` (the decoder's frame) into `dst` on the GPU and waits for it.
    pub fn apply(&mut self, deskew: &Deskew, src: &DmaBuf, dst: &DmaBuf) -> Result<()> {
        self.upload(deskew)?;
        self.import(src, CL_MEM_READ_WRITE)?;
        self.import(dst, CL_MEM_READ_WRITE)?;
        let (src_mem, dst_mem) = (
            &self.imported[&(src.fd() as i32)],
            &self.imported[&(dst.fd() as i32)],
        );
        self.launch(deskew, src_mem, dst_mem, src.len(), dst.len())
    }

    /// The same remap through copies (write the frame in, read the result out), for machines without dmabufs.
    pub fn apply_host(&mut self, deskew: &Deskew, src: &[u8], dst: &mut [u8]) -> Result<()> {
        self.upload(deskew)?;
        let src_mem = self.gpu.buffer(CL_MEM_READ_ONLY, src.len())?;
        let dst_mem = self.gpu.buffer(CL_MEM_WRITE_ONLY, dst.len())?;
        self.gpu.write(&src_mem, src)?;
        self.launch(deskew, &src_mem, &dst_mem, src.len(), dst.len())?;
        self.gpu.read(&dst_mem, dst)
    }

    fn launch(
        &self,
        deskew: &Deskew,
        src: &Buffer,
        dst: &Buffer,
        src_len: usize,
        dst_len: usize,
    ) -> Result<()> {
        let (width, height, sampling) = deskew.built_for().expect("uploaded");
        let out = deskew.output_camera(width, height);
        let (s_stride, s_rows) = layout(width, height);
        let (d_stride, d_rows) = layout(out.width, out.height);
        ensure!(
            src_len >= s_stride * (s_rows + chroma_rows(s_rows, sampling) + 2),
            "source buffer too small"
        );
        ensure!(
            dst_len >= d_stride * (d_rows + chroma_rows(d_rows, sampling)),
            "output buffer too small"
        );
        let (luma, chroma) = self.taps.as_ref().expect("uploaded");
        let bilinear = deskew.bilinear() as u32;
        let (ow, oh) = (out.width as usize, d_rows);
        for (taps, step, cols, rows, src_off, dst_off, fill) in [
            (luma, 1u32, ow, oh, 0usize, 0usize, 0u32),
            (
                chroma,
                2,
                ow / 2,
                chroma_rows(d_rows, sampling),
                s_stride * s_rows,
                d_stride * d_rows,
                128,
            ),
        ] {
            self.kernel.launch(
                &[
                    Arg::Buffer(src),
                    Arg::U32(src_off as u32),
                    Arg::U32(s_stride as u32),
                    Arg::Buffer(dst),
                    Arg::U32(dst_off as u32),
                    Arg::U32(d_stride as u32),
                    Arg::Buffer(taps),
                    Arg::U32(cols as u32),
                    Arg::U32(fill),
                    Arg::U32(bilinear),
                    Arg::U32(step),
                ],
                [cols, rows],
            )?;
        }
        self.gpu.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deskew::{Interpolation, Intrinsics, Model, Options, Planes, PlanesMut, Projection};
    use crate::jpeg::Sampling;

    /// Where any OpenCL device exists (the Mac's, say), the kernel must reproduce the CPU remap byte for byte.
    #[test]
    fn the_gpu_kernel_matches_the_cpu_where_a_gpu_exists() {
        let Ok(mut gpu) = GpuRemap::new() else {
            eprintln!("no OpenCL device here; skipped");
            return;
        };
        let (w, h) = (320u32, 180u32);
        for interpolation in [Interpolation::Bilinear, Interpolation::Nearest] {
            let camera = Intrinsics {
                width: w,
                height: h,
                fx: 122.0,
                fy: 122.0,
                cx: 160.0,
                cy: 90.0,
                model: Model::Equidistant,
                coefficients: vec![0.02, -0.01],
            };
            let options = Options {
                projection: Projection::Pinhole,
                output_size: None,
                balance: 0.5,
                interpolation,
            };
            let mut deskew = Deskew::new(camera, options).unwrap();
            deskew.prepare(w, h, Sampling::Yuv420);
            let (stride, rows) = layout(w, h);
            let src: Vec<u8> = (0..stride * (rows * 3 / 2 + 2))
                .map(|i| ((i * 37) ^ (i >> 7)) as u8)
                .collect();
            let mut cpu = vec![0u8; stride * rows * 3 / 2];
            deskew
                .apply(
                    &Planes {
                        width: w,
                        height: h,
                        stride,
                        rows,
                        sampling: Sampling::Yuv420,
                        bytes: &src,
                    },
                    &mut PlanesMut {
                        width: w,
                        height: h,
                        stride,
                        rows,
                        sampling: Sampling::Yuv420,
                        bytes: &mut cpu,
                    },
                )
                .unwrap();
            let mut on_gpu = vec![0u8; cpu.len()];
            gpu.apply_host(&deskew, &src, &mut on_gpu).unwrap();
            let differ = cpu.iter().zip(&on_gpu).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{interpolation:?}: {differ} bytes differ");
        }
    }
}
