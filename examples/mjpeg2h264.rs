//! Test harness: MJPEG frames (a directory of JPEGs, looped, or a V4L2 camera) through one or more transcoders at
//! once, Annex-B H.264 out, per-stream rate / CPU and the process's fd count / RSS on stderr.
//!
//! ```text
//! mjpeg2h264 --check
//! mjpeg2h264 --jpeg-dir /tmp/jt --streams 2 --seconds 60 --out /tmp/out   # writes /tmp/out.0.h264, /tmp/out.1.h264
//! mjpeg2h264 --device /dev/video0 --seconds 10 --out - > cam.h264
//! ```

use std::io::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use rockchip_h264::{platform, v4l2, Config, Transcoder};

/// model, focal px at --width, coefficients, balance, output size, interpolation
type DeskewArgs = (String, f64, Vec<f64>, f64, Option<(u32, u32)>, String);

struct Args {
    check: bool,
    jpeg_dir: Option<String>,
    device: Option<String>,
    streams: usize,
    seconds: f64,
    out: Option<String>,
    pace: bool,
    stats_every: f64,
    config: Config,
    /// deskew every frame (`--deskew`)
    deskew: Option<DeskewArgs>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        check: false,
        jpeg_dir: None,
        device: None,
        streams: 1,
        seconds: f64::INFINITY,
        out: None,
        pace: true,
        stats_every: 5.0,
        config: Config::default(),
        deskew: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--check" => a.check = true,
            "--jpeg-dir" => a.jpeg_dir = Some(value()?),
            "--device" => a.device = Some(value()?),
            "--streams" => a.streams = value()?.parse()?,
            "--seconds" => a.seconds = value()?.parse()?,
            "--out" => a.out = Some(value()?),
            "--no-pace" => a.pace = false,
            "--stats-every" => a.stats_every = value()?.parse()?,
            "--fps" => a.config.fps = value()?.parse()?,
            "--bitrate" => a.config.bitrate_bps = value()?.parse()?,
            "--gop" => a.config.gop = value()?.parse()?,
            // e.g. --deskew equidistant:486:0,0,0,0:0.0:1280x720:bilinear (focal at the --width given)
            "--deskew" => {
                let v = value()?;
                let p: Vec<&str> = v.split(':').collect();
                anyhow::ensure!(
                    p.len() == 6,
                    "--deskew model:focal:k1,k2,...:balance:WxH|same:nearest|bilinear"
                );
                let size = (p[4] != "same").then(|| {
                    let (w, h) = p[4].split_once('x').expect("WxH");
                    (w.parse().unwrap(), h.parse().unwrap())
                });
                let coefficients = p[2]
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?;
                a.deskew = Some((
                    p[0].into(),
                    p[1].parse()?,
                    coefficients,
                    p[3].parse()?,
                    size,
                    p[5].into(),
                ));
            }
            "--width" => a.config.width = value()?.parse()?,
            "--height" => a.config.height = value()?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(a)
}

fn fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .map(|d| d.count())
        .unwrap_or(0)
}

fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

fn thread_cpu_s() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: one timespec out-parameter
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

fn process_cpu_s() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: one timespec out-parameter
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

#[derive(Default)]
struct Counters {
    frames: AtomicU64,
    idrs: AtomicU64,
    bytes: AtomicU64,
    errors: AtomicU64,
    /// microseconds of this stream's thread CPU
    cpu_us: AtomicU64,
    qp_sum_x100: AtomicU64,
    moving: AtomicU64,
}

fn run_stream(
    index: usize,
    args: &Args,
    jpegs: Arc<Vec<Vec<u8>>>,
    counters: Arc<Counters>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let mut transcoder = Transcoder::new(args.config)?;
    if let Some((model, focal, coefficients, balance, output_size, interpolation)) = &args.deskew {
        let (w, h) = (args.config.width, args.config.height);
        let camera = rockchip_h264::deskew::Intrinsics {
            width: w,
            height: h,
            fx: *focal,
            fy: *focal,
            cx: w as f64 / 2.0,
            cy: h as f64 / 2.0,
            model: model.parse()?,
            coefficients: coefficients.clone(),
        };
        let options = rockchip_h264::deskew::Options {
            output_size: *output_size,
            balance: *balance,
            interpolation: interpolation.parse()?,
        };
        transcoder.set_deskew(Some(rockchip_h264::deskew::Deskew::new(camera, options)?));
    }
    let mut sink: Box<dyn std::io::Write> = match args.out.as_deref() {
        None => Box::new(std::io::sink()),
        Some("-") => Box::new(std::io::stdout().lock()),
        Some(prefix) => Box::new(std::io::BufWriter::new(std::fs::File::create(format!(
            "{prefix}.{index}.h264"
        ))?)),
    };
    let mut capture = match &args.device {
        Some(device) => Some(v4l2::Capture::open(
            device,
            args.config.width,
            args.config.height,
            args.config.fps,
            4,
        )?),
        None => None,
    };
    let period = Duration::from_secs_f64(1.0 / args.config.fps as f64);
    let start = Instant::now();
    let mut au = Vec::with_capacity(1 << 20);
    let mut jpeg_buf = Vec::with_capacity(args.config.max_jpeg_bytes);
    let mut n: u64 = 0;
    while !stop.load(Ordering::Relaxed) {
        let cpu0 = thread_cpu_s();
        let result = match capture.as_mut() {
            Some(cap) => {
                cap.next(Duration::from_secs(2), |bytes| {
                    jpeg_buf.clear();
                    jpeg_buf.extend_from_slice(bytes);
                })?;
                transcoder.transcode(&jpeg_buf, &mut au)
            }
            None => transcoder.transcode(&jpegs[n as usize % jpegs.len()], &mut au),
        };
        match result {
            Ok(frame) => {
                sink.write_all(&au)?;
                counters.frames.fetch_add(1, Ordering::Relaxed);
                counters.idrs.fetch_add(frame.idr as u64, Ordering::Relaxed);
                counters.bytes.fetch_add(au.len() as u64, Ordering::Relaxed);
                counters
                    .qp_sum_x100
                    .fetch_add((frame.avg_qp * 100.0) as u64, Ordering::Relaxed);
                counters
                    .moving
                    .fetch_add(frame.moving_scene as u64, Ordering::Relaxed);
            }
            Err(error) => {
                counters.errors.fetch_add(1, Ordering::Relaxed);
                eprintln!("stream {index}: frame {n}: {error:#}");
            }
        }
        counters
            .cpu_us
            .fetch_add(((thread_cpu_s() - cpu0) * 1e6) as u64, Ordering::Relaxed);
        n += 1;
        if capture.is_none() && args.pace {
            let due = start + period * n as u32;
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
        }
    }
    sink.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.check {
        let probed = platform::probe()?;
        println!("{probed}");
        match platform::check(&probed) {
            Ok(tested) => println!("supported: {}", tested.name),
            Err(error) => {
                println!("{error:#}");
                std::process::exit(2);
            }
        }
        return Ok(());
    }
    let jpegs = match &args.jpeg_dir {
        Some(dir) => {
            let mut paths: Vec<_> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "jpg"))
                .collect();
            paths.sort();
            paths
                .iter()
                .map(std::fs::read)
                .collect::<std::io::Result<Vec<_>>>()?
        }
        None if args.device.is_some() => Vec::new(),
        None => bail!("--jpeg-dir or --device"),
    };
    if args.device.is_none() && jpegs.is_empty() {
        bail!("no .jpg files");
    }
    let jpegs = Arc::new(jpegs);
    let stop = Arc::new(AtomicBool::new(false));
    let args = Arc::new(args);
    let counters: Vec<Arc<Counters>> = (0..args.streams).map(|_| Arc::default()).collect();
    let workers: Vec<_> = (0..args.streams)
        .map(|i| {
            let (args, jpegs, counters, stop) = (
                args.clone(),
                jpegs.clone(),
                counters[i].clone(),
                stop.clone(),
            );
            std::thread::spawn(move || {
                let result = run_stream(i, &args, jpegs, counters, stop.clone());
                if let Err(error) = &result {
                    eprintln!("stream {i} stopped: {error:#}");
                    stop.store(true, Ordering::Relaxed);
                }
                result
            })
        })
        .collect();

    let start = Instant::now();
    let mut last = (
        Instant::now(),
        process_cpu_s(),
        vec![(0u64, 0u64, 0u64, 0u64, 0u64, 0u64); args.streams],
    );
    eprintln!("t_s,stream,fps,mbit_s,idr_per_s,avg_qp,cpu_pct,moving_frac,errors,fds,rss_kib,process_cpu_pct");
    while !stop.load(Ordering::Relaxed) && start.elapsed().as_secs_f64() < args.seconds {
        std::thread::sleep(Duration::from_millis(100));
        if last.0.elapsed().as_secs_f64() < args.stats_every {
            continue;
        }
        let dt = last.0.elapsed().as_secs_f64();
        let pcpu = process_cpu_s();
        let (fds, rss) = (fd_count(), rss_kib());
        for (i, c) in counters.iter().enumerate() {
            let now = (
                c.frames.load(Ordering::Relaxed),
                c.bytes.load(Ordering::Relaxed),
                c.idrs.load(Ordering::Relaxed),
                c.cpu_us.load(Ordering::Relaxed),
                c.qp_sum_x100.load(Ordering::Relaxed),
                c.moving.load(Ordering::Relaxed),
            );
            let p = last.2[i];
            let frames = (now.0 - p.0).max(1) as f64;
            eprintln!(
                "{:.1},{i},{:.2},{:.3},{:.2},{:.1},{:.1},{:.2},{},{fds},{rss},{:.1}",
                start.elapsed().as_secs_f64(),
                (now.0 - p.0) as f64 / dt,
                (now.1 - p.1) as f64 * 8e-6 / dt,
                (now.2 - p.2) as f64 / dt,
                (now.4 - p.4) as f64 / 100.0 / frames,
                (now.3 - p.3) as f64 * 1e-4 / dt,
                (now.5 - p.5) as f64 / frames,
                c.errors.load(Ordering::Relaxed),
                (pcpu - last.1) * 100.0 / dt,
            );
            last.2[i] = now;
        }
        last.0 = Instant::now();
        last.1 = pcpu;
        std::io::stderr().flush().ok();
    }
    stop.store(true, Ordering::Relaxed);
    for w in workers {
        w.join()
            .map_err(|_| anyhow::anyhow!("a stream panicked"))??;
    }
    Ok(())
}
