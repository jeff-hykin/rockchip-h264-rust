//! Deskewing (undistorting) a decoded frame before it is encoded: a remap table built once per input size from the
//! camera's intrinsics, then a table lookup per output sample (bilinear or nearest) with no per-frame allocation.
//!
//! The camera models are OpenCV's: `plumb_bob` (k1, k2, p1, p2, k3, as `cv::initUndistortRectifyMap`) and
//! `equidistant` (k1..k4, the Kannala-Brandt fisheye of `cv::fisheye`). The output camera matrix follows
//! `cv::fisheye::estimateNewCameraMatrixForUndistortRectify`: `balance` 0 keeps only pixels the camera saw (zoomed
//! in), 1 keeps the whole field of view, scaled to the output size.

use anyhow::{bail, ensure, Result};

use crate::jpeg::Sampling;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// k1, k2, p1, p2, k3 (missing terms are zero)
    PlumbBob,
    /// k1, k2, k3, k4: θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)
    Equidistant,
}

impl std::str::FromStr for Model {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "plumb_bob" => Ok(Self::PlumbBob),
            "equidistant" | "fisheye" => Ok(Self::Equidistant),
            other => bail!("unknown distortion model {other:?} (plumb_bob or equidistant)"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    Bilinear,
}

impl std::str::FromStr for Interpolation {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "nearest" => Ok(Self::Nearest),
            "bilinear" => Ok(Self::Bilinear),
            other => bail!("unknown interpolation {other:?} (nearest or bilinear)"),
        }
    }
}

/// A camera's intrinsics at one image size.
#[derive(Clone, Debug, PartialEq)]
pub struct Intrinsics {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub model: Model,
    pub coefficients: Vec<f64>,
}

impl Intrinsics {
    /// The same camera at another image size (the focal lengths and center scale with it).
    pub fn scaled_to(&self, width: u32, height: u32) -> Self {
        let (sx, sy) = (
            width as f64 / self.width as f64,
            height as f64 / self.height as f64,
        );
        Self {
            width,
            height,
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: self.cx * sx,
            cy: self.cy * sy,
            ..self.clone()
        }
    }

    fn k(&self, i: usize) -> f64 {
        self.coefficients.get(i).copied().unwrap_or(0.0)
    }

    /// Normalized undistorted (x, y) -> normalized distorted (x, y).
    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        match self.model {
            Model::PlumbBob => {
                let (k1, k2, p1, p2, k3) = (self.k(0), self.k(1), self.k(2), self.k(3), self.k(4));
                let r2 = x * x + y * y;
                let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
                (
                    x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                    y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
                )
            }
            Model::Equidistant => {
                let r = (x * x + y * y).sqrt();
                if r < 1e-12 {
                    return (x, y);
                }
                let theta = r.atan();
                let t2 = theta * theta;
                let theta_d = theta
                    * (1.0
                        + t2 * (self.k(0) + t2 * (self.k(1) + t2 * (self.k(2) + t2 * self.k(3)))));
                let scale = theta_d / r;
                (x * scale, y * scale)
            }
        }
    }

    /// Normalized distorted (x, y) -> normalized undistorted (x, y), solved iteratively as OpenCV does.
    fn undistort(&self, xd: f64, yd: f64) -> (f64, f64) {
        match self.model {
            Model::PlumbBob => {
                let (mut x, mut y) = (xd, yd);
                for _ in 0..20 {
                    let (dx, dy) = self.distort(x, y);
                    x += xd - dx;
                    y += yd - dy;
                }
                (x, y)
            }
            Model::Equidistant => {
                let theta_d = (xd * xd + yd * yd).sqrt();
                if theta_d < 1e-12 {
                    return (xd, yd);
                }
                let mut theta = theta_d;
                for _ in 0..20 {
                    let t2 = theta * theta;
                    let (k1, k2, k3, k4) = (self.k(0), self.k(1), self.k(2), self.k(3));
                    let f = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d;
                    let df =
                        1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)));
                    theta -= f / df;
                }
                // past 90 degrees there is no pinhole point; keep the ray at the edge of the plane
                let theta = theta.clamp(0.0, std::f64::consts::FRAC_PI_2 - 1e-6);
                let scale = theta.tan() / theta_d;
                (xd * scale, yd * scale)
            }
        }
    }
}

/// What to produce: the output size (the input's when `None`), how much of the field of view to keep, and how
/// samples are interpolated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    pub output_size: Option<(u32, u32)>,
    pub balance: f64,
    pub interpolation: Interpolation,
}

/// A pinhole camera matrix (no distortion) for the deskewed frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pinhole {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// `cv::fisheye::estimateNewCameraMatrixForUndistortRectify` (fov_scale 1, R identity), for either model.
pub fn new_camera(camera: &Intrinsics, options: &Options) -> Pinhole {
    let (w, h) = (camera.width as f64, camera.height as f64);
    let balance = options.balance.clamp(0.0, 1.0);
    let edges = [(w * 0.5, 0.0), (w, h * 0.5), (w * 0.5, h), (0.0, h * 0.5)];
    let points: Vec<(f64, f64)> = edges
        .iter()
        .map(|&(u, v)| camera.undistort((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy))
        .collect();
    let aspect = camera.fx / camera.fy;
    let cn = (
        points.iter().map(|p| p.0).sum::<f64>() / 4.0,
        aspect * points.iter().map(|p| p.1).sum::<f64>() / 4.0,
    );
    let xs = points.iter().map(|p| p.0);
    let ys: Vec<f64> = points.iter().map(|p| p.1 * aspect).collect();
    let (minx, maxx) = xs.fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(x), b.max(x)));
    let (miny, maxy) = ys
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &y| (a.min(y), b.max(y)));
    let fs = [
        w * 0.5 / (cn.0 - minx),
        w * 0.5 / (maxx - cn.0),
        h * 0.5 * aspect / (cn.1 - miny),
        h * 0.5 * aspect / (maxy - cn.1),
    ];
    let fmin = fs.iter().copied().fold(f64::MAX, f64::min);
    let fmax = fs.iter().copied().fold(f64::MIN, f64::max);
    let f = balance * fmin + (1.0 - balance) * fmax;
    let (mut fx, mut fy) = (f, f / aspect);
    let (mut cx, mut cy) = (-cn.0 * f + w * 0.5, (-cn.1 * f + h * aspect * 0.5) / aspect);
    let (ow, oh) = options.output_size.unwrap_or((camera.width, camera.height));
    let (sx, sy) = (ow as f64 / w, oh as f64 / h);
    fx *= sx;
    cx *= sx;
    fy *= sy;
    cy *= sy;
    Pinhole {
        width: ow,
        height: oh,
        fx,
        fy,
        cx,
        cy,
    }
}

/// One output sample's source, packed in a u32 as OpenCV's remap tables are: the top-left source sample's index
/// (high 22 bits) and the bilinear weights toward the next column and row (5 bits each, in 32nds).
/// [`OUTSIDE`]: no source sample (filled).
type Tap = u32;

const OUTSIDE: Tap = u32::MAX;
/// Bilinear weights are fixed point with this many fractional bits (OpenCV's `INTER_BITS`).
const WEIGHT_BITS: u32 = 5;
const INDEX_BITS: u32 = 32 - 2 * WEIGHT_BITS;

fn tap(index: usize, wx: u32, wy: u32) -> Tap {
    (index as u32) << (2 * WEIGHT_BITS) | wx << WEIGHT_BITS | wy
}

/// A plane's remap: one tap per output sample, and the source plane's row stride in bytes.
struct PlaneMap {
    taps: Vec<Tap>,
    out_width: usize,
    out_rows: usize,
    src_stride: usize,
    bilinear: bool,
}

/// One semi-planar YUV frame: luma (`stride` bytes per row, `rows` rows), then interleaved Cb/Cr with the same
/// stride, half the rows for 4:2:0 and all of them for 4:2:2.
pub struct Planes<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub rows: usize,
    pub sampling: Sampling,
    pub bytes: &'a [u8],
}

pub struct PlanesMut<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub rows: usize,
    pub sampling: Sampling,
    pub bytes: &'a mut [u8],
}

/// A remap from one input size and sampling to the output, built once.
pub struct Deskew {
    camera: Intrinsics,
    options: Options,
    /// (input width, height, sampling) the tables are for
    built_for: Option<(u32, u32, Sampling)>,
    output: Option<Pinhole>,
    luma: Option<PlaneMap>,
    chroma: Option<PlaneMap>,
}

/// `(stride, rows)` of a semi-planar frame of this size: the 16-aligned width and height the decoder writes.
pub fn layout(width: u32, height: u32) -> (usize, usize) {
    (
        width.next_multiple_of(16) as usize,
        height.next_multiple_of(16) as usize,
    )
}

fn chroma_rows(rows: usize, sampling: Sampling) -> usize {
    match sampling {
        Sampling::Yuv420 => rows / 2,
        Sampling::Yuv422 => rows,
    }
}

impl Deskew {
    /// `camera` may be given at any size: it is scaled to each input's size.
    pub fn new(camera: Intrinsics, options: Options) -> Result<Self> {
        ensure!(
            camera.fx > 0.0 && camera.fy > 0.0,
            "focal lengths must be positive"
        );
        ensure!(
            (0.0..=1.0).contains(&options.balance),
            "balance must be within 0..=1"
        );
        if let Some((w, h)) = options.output_size {
            ensure!(
                w >= 16 && h >= 16 && w % 2 == 0 && h % 2 == 0,
                "output size {w}x{h} must be even and at least 16x16"
            );
        }
        Ok(Self {
            camera,
            options,
            built_for: None,
            output: None,
            luma: None,
            chroma: None,
        })
    }

    /// The pinhole camera of the deskewed frames for this input size (what their CameraInfo should say).
    pub fn output_camera(&self, width: u32, height: u32) -> Pinhole {
        new_camera(&self.camera.scaled_to(width, height), &self.options)
    }

    /// The output size for an input of this size.
    pub fn output_size(&self, width: u32, height: u32) -> (u32, u32) {
        self.options.output_size.unwrap_or((width, height))
    }

    /// Builds the tables for this input, unless they already are.
    pub fn prepare(&mut self, width: u32, height: u32, sampling: Sampling) {
        if self.built_for == Some((width, height, sampling)) {
            return;
        }
        let camera = self.camera.scaled_to(width, height);
        let out = new_camera(&camera, &self.options);
        let (src_stride, src_rows) = layout(width, height);
        assert!(
            src_stride * src_rows < (1 << INDEX_BITS) - 1,
            "{width}x{height} is too large for the deskew tables"
        );
        let (_, out_rows) = layout(out.width, out.height);
        let bilinear = self.options.interpolation == Interpolation::Bilinear;
        // luma: output sample (u, v) -> source sample position
        self.luma = Some(build(
            out.width as usize,
            out_rows,
            src_stride,
            1,
            (width as f64, height as f64),
            bilinear,
            |u, v| source_position(&camera, &out, u, v),
        ));
        // chroma: one sample per 2 (and for 4:2:0, 2x2) luma samples, sited at their center
        let (sub_x, sub_y) = match sampling {
            Sampling::Yuv420 => (2.0, 2.0),
            Sampling::Yuv422 => (2.0, 1.0),
        };
        let chroma_out_rows = chroma_rows(out_rows, sampling);
        self.chroma = Some(build(
            out.width as usize / 2,
            chroma_out_rows,
            src_stride,
            2,
            (
                (width as f64 / sub_x).floor(),
                (height as f64 / sub_y).floor(),
            ),
            bilinear,
            |u, v| {
                let (lu, lv) = ((u + 0.5) * sub_x - 0.5, (v + 0.5) * sub_y - 0.5);
                let (x, y) = source_position(&camera, &out, lu, lv);
                ((x + 0.5) / sub_x - 0.5, (y + 0.5) / sub_y - 0.5)
            },
        ));
        self.output = Some(out);
        self.built_for = Some((width, height, sampling));
    }

    /// Remaps `src` into `dst` (built for `src`'s size with [`Deskew::prepare`]; both 8-bit semi-planar).
    pub fn apply(&self, src: &Planes, dst: &mut PlanesMut) -> Result<()> {
        ensure!(
            self.built_for == Some((src.width, src.height, src.sampling)),
            "deskew tables not built for this frame"
        );
        ensure!(
            src.bytes.len() >= src.stride * (src.rows + chroma_rows(src.rows, src.sampling) + 2),
            "source buffer too small: two rows of padding past the chroma plane are read (with weight 0)"
        );
        let out = self.output.expect("built");
        ensure!(
            (dst.width, dst.height) == (out.width, out.height),
            "output buffer is {}x{}, deskew makes {}x{}",
            dst.width,
            dst.height,
            out.width,
            out.height
        );
        let (luma, chroma) = (
            self.luma.as_ref().expect("built"),
            self.chroma.as_ref().expect("built"),
        );
        let src_luma = src.stride * src.rows;
        let dst_luma = dst.stride * dst.rows;
        // a tap at the last row or column reads one past it with weight 0: the next plane's (or the padding's) bytes
        let (dst_y, dst_c) = dst.bytes.split_at_mut(dst_luma);
        remap_plane::<1>(luma, src.bytes, dst_y, dst.stride, 0);
        remap_plane::<2>(chroma, &src.bytes[src_luma..], dst_c, dst.stride, 128);
        Ok(())
    }
}

/// Output pixel (u, v) of the deskewed frame -> position in the input frame, both in pixels.
fn source_position(camera: &Intrinsics, out: &Pinhole, u: f64, v: f64) -> (f64, f64) {
    let (x, y) = ((u - out.cx) / out.fx, (v - out.cy) / out.fy);
    let (xd, yd) = camera.distort(x, y);
    (camera.fx * xd + camera.cx, camera.fy * yd + camera.cy)
}

/// The taps of one plane: `position(u, v)` is the source position (in samples of this plane) of output sample
/// (u, v); `bounds` are the plane's valid width and height in samples.
fn build(
    out_width: usize,
    out_rows: usize,
    src_stride: usize,
    step: usize,
    bounds: (f64, f64),
    bilinear: bool,
    position: impl Fn(f64, f64) -> (f64, f64),
) -> PlaneMap {
    let one = (1u32 << WEIGHT_BITS) as f64;
    let mut taps = Vec::with_capacity(out_width * out_rows);
    // the weights round to 0..=32, and a weight of 32 is the next sample with weight 0
    for v in 0..out_rows {
        for u in 0..out_width {
            let (x, y) = position(u as f64, v as f64);
            let inside = x >= -0.5
                && y >= -0.5
                && x <= bounds.0 - 0.5
                && y <= bounds.1 - 0.5
                && x.is_finite()
                && y.is_finite();
            if !inside {
                taps.push(OUTSIDE);
                continue;
            }
            let (mut x, mut y) = (x.clamp(0.0, bounds.0 - 1.0), y.clamp(0.0, bounds.1 - 1.0));
            if !bilinear {
                (x, y) = (x.round(), y.round());
            }
            // in 32nds; the top-left neighbor stays one sample in from the last column / row so its partner exists
            let (fx, fy) = ((x * one).round(), (y * one).round());
            let x0 = (fx / one).floor().min(bounds.0 - 2.0).max(0.0);
            let y0 = (fy / one).floor().min(bounds.1 - 2.0).max(0.0);
            let (wx, wy) = ((fx - x0 * one) as u32, (fy - y0 * one) as u32);
            let (x0, y0, wx, wy) = match (wx > one as u32 - 1, wy > one as u32 - 1) {
                (false, false) => (x0, y0, wx, wy),
                (true, false) => (x0 + 1.0, y0, 0, wy),
                (false, true) => (x0, y0 + 1.0, wx, 0),
                (true, true) => (x0 + 1.0, y0 + 1.0, 0, 0),
            };
            taps.push(tap(y0 as usize * src_stride + x0 as usize * step, wx, wy));
        }
    }
    PlaneMap {
        taps,
        out_width,
        out_rows,
        src_stride,
        bilinear,
    }
}

/// Fills one destination plane from its taps; `fill` is the value outside the source (black luma, neutral chroma).
fn remap_plane<const STEP: usize>(
    map: &PlaneMap,
    src: &[u8],
    dst: &mut [u8],
    dst_stride: usize,
    fill: u8,
) {
    if !map.bilinear {
        return copy_plane::<STEP>(map, src, dst, dst_stride, fill);
    }
    let stride = map.src_stride;
    let one = 1u32 << WEIGHT_BITS;
    let mask = one - 1;
    // the furthest a tap reaches: its index plus one row and one sample
    let reach = stride + 2 * STEP;
    for v in 0..map.out_rows {
        let row = &map.taps[v * map.out_width..(v + 1) * map.out_width];
        let out = &mut dst[v * dst_stride..v * dst_stride + map.out_width * STEP];
        for (tap, px) in row.iter().zip(out.as_chunks_mut::<STEP>().0) {
            if *tap == OUTSIDE {
                px.fill(fill);
                continue;
            }
            let i = (*tap >> (2 * WEIGHT_BITS)) as usize;
            let (wx, wy) = ((*tap >> WEIGHT_BITS) & mask, *tap & mask);
            let s = &src[i..i + reach];
            for c in 0..STEP {
                let (a, b) = (s[c] as u32, s[STEP + c] as u32);
                let (d, e) = (s[stride + c] as u32, s[stride + STEP + c] as u32);
                let top = a * (one - wx) + b * wx;
                let bottom = d * (one - wx) + e * wx;
                px[c] = ((top * (one - wy) + bottom * wy + (1 << (2 * WEIGHT_BITS - 1)))
                    >> (2 * WEIGHT_BITS)) as u8;
            }
        }
    }
}

/// [`remap_plane`] for nearest-neighbor taps: one source sample per output sample, no arithmetic.
fn copy_plane<const STEP: usize>(
    map: &PlaneMap,
    src: &[u8],
    dst: &mut [u8],
    dst_stride: usize,
    fill: u8,
) {
    for v in 0..map.out_rows {
        let row = &map.taps[v * map.out_width..(v + 1) * map.out_width];
        let out = &mut dst[v * dst_stride..v * dst_stride + map.out_width * STEP];
        for (tap, px) in row.iter().zip(out.as_chunks_mut::<STEP>().0) {
            if *tap == OUTSIDE {
                px.fill(fill);
            } else {
                let i = (*tap >> (2 * WEIGHT_BITS)) as usize;
                px.copy_from_slice(&src[i..i + STEP]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fisheye() -> Intrinsics {
        Intrinsics {
            width: 1280,
            height: 720,
            fx: 486.0,
            fy: 486.0,
            cx: 640.0,
            cy: 360.0,
            model: Model::Equidistant,
            coefficients: vec![0.0; 4],
        }
    }

    #[test]
    fn distort_and_undistort_are_inverses() {
        for model in [Model::Equidistant, Model::PlumbBob] {
            let c = Intrinsics {
                model,
                coefficients: vec![-0.05, 0.01, 0.001, -0.002, 0.0],
                ..fisheye()
            };
            for (x, y) in [(0.1, -0.2), (0.6, 0.3), (-0.9, 0.5)] {
                let (xd, yd) = c.distort(x, y);
                let (xu, yu) = c.undistort(xd, yd);
                assert!(
                    (xu - x).abs() < 1e-9 && (yu - y).abs() < 1e-9,
                    "{model:?} {x},{y} -> {xu},{yu}"
                );
            }
        }
    }

    /// The values OpenCV's `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify` gives for the same camera.
    #[test]
    fn new_camera_matches_opencv() {
        let opts = Options {
            output_size: None,
            balance: 0.0,
            interpolation: Interpolation::Bilinear,
        };
        let k = new_camera(&fisheye(), &opts);
        assert!((k.fx - 393.679788894805).abs() < 1e-6, "{}", k.fx);
        assert!((k.cx - 640.0).abs() < 1e-6 && (k.cy - 360.0).abs() < 1e-6);
        let k = new_camera(
            &fisheye(),
            &Options {
                balance: 1.0,
                ..opts
            },
        );
        assert!((k.fx - 166.09655666).abs() < 1e-6, "{}", k.fx);
    }

    #[test]
    fn zero_distortion_at_balance_zero_is_the_identity() {
        let pinhole = Intrinsics {
            width: 64,
            height: 48,
            fx: 40.0,
            fy: 40.0,
            cx: 32.0,
            cy: 24.0,
            model: Model::PlumbBob,
            coefficients: vec![],
        };
        let mut d = Deskew::new(
            pinhole,
            Options {
                output_size: None,
                balance: 0.0,
                interpolation: Interpolation::Bilinear,
            },
        )
        .unwrap();
        let (w, h) = (64u32, 48u32);
        d.prepare(w, h, Sampling::Yuv420);
        let (stride, rows) = layout(w, h);
        let src: Vec<u8> = (0..stride * (rows * 3 / 2 + 2))
            .map(|i| (i * 7 % 251) as u8)
            .collect();
        let mut dst = vec![0u8; stride * rows * 3 / 2];
        d.apply(
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
                bytes: &mut dst,
            },
        )
        .unwrap();
        for y in 0..h as usize {
            assert_eq!(
                &dst[y * stride..y * stride + w as usize],
                &src[y * stride..y * stride + w as usize],
                "row {y}"
            );
        }
    }

    #[test]
    fn the_image_center_stays_put_and_corners_pull_inward() {
        let c = fisheye();
        let out = new_camera(
            &c,
            &Options {
                output_size: None,
                balance: 0.0,
                interpolation: Interpolation::Bilinear,
            },
        );
        let (x, y) = source_position(&c, &out, 640.0, 360.0);
        assert!((x - 640.0).abs() < 1e-9 && (y - 360.0).abs() < 1e-9);
        // a fisheye compresses the periphery: an output corner comes from closer to the center
        let (x, y) = source_position(&c, &out, 0.0, 0.0);
        assert!(x > 0.0 && y > 0.0, "{x},{y}");
    }

    #[test]
    fn output_size_scales_the_new_camera() {
        let half = Options {
            output_size: Some((640, 360)),
            balance: 0.0,
            interpolation: Interpolation::Nearest,
        };
        let full = Options {
            output_size: None,
            ..half
        };
        let (a, b) = (new_camera(&fisheye(), &half), new_camera(&fisheye(), &full));
        assert!((a.fx * 2.0 - b.fx).abs() < 1e-9 && (a.cx * 2.0 - b.cx).abs() < 1e-9);
        assert_eq!((a.width, a.height), (640, 360));
    }

    #[test]
    fn model_and_interpolation_names() {
        assert_eq!("equidistant".parse::<Model>().unwrap(), Model::Equidistant);
        assert_eq!("plumb_bob".parse::<Model>().unwrap(), Model::PlumbBob);
        assert!("kb4".parse::<Model>().is_err());
        assert_eq!(
            "bilinear".parse::<Interpolation>().unwrap(),
            Interpolation::Bilinear
        );
    }
}
