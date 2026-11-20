//! Deskews one raw NV12 frame (no hardware needed): writes the result, and times the remap at a given frame rate.
//!
//! ```text
//! ffmpeg -i frame.png -pix_fmt nv12 -f rawvideo in.nv12
//! deskew_bench in.nv12 1280 720 out.nv12 --model equidistant --focal 486 --coefficients 0,0,0,0 --frames 300 --fps 30
//! ffmpeg -f rawvideo -pix_fmt nv12 -s 1280x720 -i out.nv12 out.png
//! ```
//! The input is tightly packed (stride = width), as ffmpeg writes it; it is copied into the decoder's 16-aligned
//! layout first so the timing matches the real pipeline.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rockchip_h264::deskew::{layout, Deskew, Intrinsics, Options, Planes, PlanesMut};
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
    let coefficients: Vec<f64> = flag("--coefficients")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let options = Options {
        output_size: flag("--output-size").map(|s| {
            let (w, h) = s.split_once('x').expect("WxH");
            (w.parse().unwrap(), h.parse().unwrap())
        }),
        balance: flag("--balance")
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(0.0),
        interpolation: flag("--interpolation")
            .unwrap_or("bilinear".into())
            .parse()?,
    };
    let frames: u32 = flag("--frames")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1);
    let fps: f64 = flag("--fps")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(30.0);
    let camera = Intrinsics {
        width,
        height,
        fx: focal,
        fy: focal,
        cx: width as f64 / 2.0,
        cy: height as f64 / 2.0,
        model: flag("--model").unwrap_or("equidistant".into()).parse()?,
        coefficients,
    };

    let packed = std::fs::read(input)?;
    let (stride, rows) = layout(width, height);
    let mut src = vec![0u8; stride * (rows * 3 / 2 + 2)];
    let (w, h) = (width as usize, height as usize);
    for y in 0..h {
        src[y * stride..y * stride + w].copy_from_slice(&packed[y * w..y * w + w]);
    }
    for y in 0..h / 2 {
        let (s, d) = (w * h + y * w, stride * rows + y * stride);
        src[d..d + w].copy_from_slice(&packed[s..s + w]);
    }

    let built = Instant::now();
    let cpu0 = thread_cpu_s();
    let mut deskew = Deskew::new(camera, options)?;
    deskew.prepare(width, height, Sampling::Yuv420);
    let (ow, oh) = deskew.output_size(width, height);
    let (ostride, orows) = layout(ow, oh);
    let mut dst = vec![0u8; ostride * orows * 3 / 2];
    eprintln!(
        "tables built in {:.1} ms ({:.1} ms cpu)",
        built.elapsed().as_secs_f64() * 1e3,
        (thread_cpu_s() - cpu0) * 1e3
    );

    let period = Duration::from_secs_f64(1.0 / fps);
    let start = Instant::now();
    let mut cpu = Vec::with_capacity(frames as usize);
    for n in 0..frames {
        let c0 = thread_cpu_s();
        deskew.apply(
            &Planes {
                width,
                height,
                stride,
                rows,
                sampling: Sampling::Yuv420,
                bytes: &src,
            },
            &mut PlanesMut {
                width: ow,
                height: oh,
                stride: ostride,
                rows: orows,
                sampling: Sampling::Yuv420,
                bytes: &mut dst,
            },
        )?;
        cpu.push(thread_cpu_s() - c0);
        if let Some(wait) = (start + period * (n + 1)).checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
    }
    let wall = start.elapsed().as_secs_f64();
    cpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f64 = cpu.iter().sum();
    eprintln!(
        "{frames} frames {width}x{height} -> {ow}x{oh}: per frame cpu median {:.2} ms, p95 {:.2} ms, max {:.2} ms; {:.1}% of one core at {:.1} fps",
        cpu[cpu.len() / 2] * 1e3,
        cpu[cpu.len() * 95 / 100] * 1e3,
        cpu[cpu.len() - 1] * 1e3,
        total / wall * 100.0,
        frames as f64 / wall
    );
    let k = deskew.output_camera(width, height);
    eprintln!(
        "output camera fx {:.3} fy {:.3} cx {:.3} cy {:.3}",
        k.fx, k.fy, k.cx, k.cy
    );

    let (ow, oh) = (ow as usize, oh as usize);
    let mut packed_out = Vec::with_capacity(ow * oh * 3 / 2);
    for y in 0..oh {
        packed_out.extend_from_slice(&dst[y * ostride..y * ostride + ow]);
    }
    for y in 0..oh / 2 {
        let s = ostride * orows + y * ostride;
        packed_out.extend_from_slice(&dst[s..s + ow]);
    }
    std::fs::write(output, packed_out)?;
    Ok(())
}
