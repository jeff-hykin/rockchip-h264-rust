//! Times the deskew remap on the CPU and on the GPU (dmabufs imported in place, and through copies) for one raw NV12
//! frame, checks the GPU output is identical to the CPU's, and writes it. Needs root (dmabufs) and the Mali driver.
//!
//! ```text
//! deskew_gpu_bench in.nv12 1280 720 out.nv12 --focal 486 --frames 300 [--projection pinhole] [--balance 0.5]
//! ```

use std::time::Instant;

use anyhow::{ensure, Context, Result};
use rockchip_h264::deskew::{layout, Deskew, Intrinsics, Options, Planes, PlanesMut};
use rockchip_h264::deskew_gpu::GpuRemap;
use rockchip_h264::dma::Heap;
use rockchip_h264::jpeg::Sampling;

fn thread_cpu_s() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: one timespec out-parameter
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// (median wall ms, median thread-cpu ms) of `frames` runs of `f`.
fn time(frames: u32, mut f: impl FnMut() -> Result<()>) -> Result<(f64, f64)> {
    let (mut wall, mut cpu) = (Vec::new(), Vec::new());
    for _ in 0..frames {
        let (t, c) = (Instant::now(), thread_cpu_s());
        f()?;
        wall.push(t.elapsed().as_secs_f64() * 1e3);
        cpu.push((thread_cpu_s() - c) * 1e3);
    }
    wall.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((wall[wall.len() / 2], cpu[cpu.len() / 2]))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (input, width, height, output) = (
        &args[1],
        args[2].parse::<u32>()?,
        args[3].parse::<u32>()?,
        &args[4],
    );
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let focal: f64 = flag("--focal").context("--focal")?.parse()?;
    let frames: u32 = flag("--frames")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(100);
    let options = Options {
        projection: flag("--projection").unwrap_or("pinhole".into()).parse()?,
        output_size: None,
        balance: flag("--balance")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(0.5),
        interpolation: flag("--interpolation")
            .unwrap_or("bilinear".into())
            .parse()?,
    };
    let camera = Intrinsics {
        width,
        height,
        fx: focal,
        fy: focal,
        cx: width as f64 / 2.0,
        cy: height as f64 / 2.0,
        model: flag("--model").unwrap_or("equidistant".into()).parse()?,
        coefficients: vec![],
    };
    let mut deskew = Deskew::new(camera, options)?;
    deskew.prepare(width, height, Sampling::Yuv420);
    let (stride, rows) = layout(width, height);
    let src_len = stride * (rows * 3 / 2 + 2);
    let dst_len = stride * rows * 3 / 2;

    // the frame in the decoder's layout (16-aligned stride and rows)
    let packed = std::fs::read(input)?;
    let (w, h) = (width as usize, height as usize);
    let mut frame = vec![0u8; src_len];
    for y in 0..h {
        frame[y * stride..y * stride + w].copy_from_slice(&packed[y * w..y * w + w]);
    }
    for y in 0..h / 2 {
        frame[stride * rows + y * stride..stride * rows + y * stride + w]
            .copy_from_slice(&packed[w * h + y * w..w * h + y * w + w]);
    }

    let mut cpu_out = vec![0u8; dst_len];
    let (cpu_wall, cpu_cpu) = time(frames, || {
        deskew.apply(
            &Planes {
                width,
                height,
                stride,
                rows,
                sampling: Sampling::Yuv420,
                bytes: &frame,
            },
            &mut PlanesMut {
                width,
                height,
                stride,
                rows,
                sampling: Sampling::Yuv420,
                bytes: &mut cpu_out,
            },
        )
    })?;
    println!("{width}x{height} cpu:          {cpu_wall:.2} ms/frame wall, {cpu_cpu:.2} ms cpu");

    let heap = Heap::open()?;
    let mut src = heap.alloc(src_len)?;
    let dst = heap.alloc(dst_len)?;
    src.write(|b| b[..frame.len()].copy_from_slice(&frame))?;
    let setup = Instant::now();
    let mut gpu = GpuRemap::new()?;
    gpu.apply(&deskew, &src, &dst)?;
    println!(
        "{width}x{height} gpu setup:    {:.1} ms (driver, kernel build, table upload, first frame)",
        setup.elapsed().as_secs_f64() * 1e3
    );
    let (gpu_wall, gpu_cpu) = time(frames, || gpu.apply(&deskew, &src, &dst))?;
    println!("{width}x{height} gpu dmabuf:   {gpu_wall:.2} ms/frame wall, {gpu_cpu:.2} ms cpu");
    let gpu_out = dst.read(dst_len, |b| b.to_vec())?;
    let mut host_out = vec![0u8; dst_len];
    let (host_wall, host_cpu) = time(frames.min(60), || {
        gpu.apply_host(&deskew, &frame, &mut host_out)
    })?;
    println!("{width}x{height} gpu copies:   {host_wall:.2} ms/frame wall, {host_cpu:.2} ms cpu");

    let (luma, total) = (stride * rows, dst_len);
    let differ = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    let (dy, dc) = (
        differ(&gpu_out[..luma], &cpu_out[..luma]),
        differ(&gpu_out[luma..total], &cpu_out[luma..total]),
    );
    println!(
        "gpu vs cpu: {dy} luma and {dc} chroma bytes differ; copies path differs in {}",
        differ(&host_out, &cpu_out)
    );
    ensure!(dy + dc == 0, "the GPU output differs from the CPU's");

    let mut packed_out = Vec::with_capacity(w * h * 3 / 2);
    for y in 0..h {
        packed_out.extend_from_slice(&gpu_out[y * stride..y * stride + w]);
    }
    for y in 0..h / 2 {
        packed_out.extend_from_slice(&gpu_out[luma + y * stride..luma + y * stride + w]);
    }
    std::fs::write(output, packed_out)?;
    Ok(())
}
