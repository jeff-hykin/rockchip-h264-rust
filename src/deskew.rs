//! Deskewing (undistorting) a decoded frame before it is encoded: a remap table built once per input size from the
//! camera's intrinsics, then a table lookup per output sample (bilinear or nearest) with no per-frame allocation.
//!
//! The lens models are the ones CameraInfo names, with OpenCV's math: `plumb_bob` (k1, k2, p1, p2, k3) and
//! `rational_polynomial` (k1, k2, p1, p2, k3, k4, k5, k6) as `cv::initUndistortRectifyMap`, and `equidistant` (k1..k4,
//! the Kannala-Brandt fisheye of `cv::fisheye`; also accepted as `fisheye` / `kannala_brandt`).
//!
//! The output is a pinhole image (straight lines stay straight), or for wide lenses a cylindrical or equirectangular
//! one (vertical lines stay straight; the whole field of view fits without the black wedges a pinhole needs). Its
//! scale follows `cv::fisheye::estimateNewCameraMatrixForUndistortRectify`, generalized to the output projection:
//! `balance` 0 keeps only what every edge of the frame saw, 1 the whole field of view, scaled to the output size.

use anyhow::{bail, ensure, Result};

use crate::jpeg::Sampling;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// k1, k2, p1, p2, k3 (missing terms are zero)
    PlumbBob,
    /// k1, k2, p1, p2, k3, k4, k5, k6: radial (1 + k1r² + k2r⁴ + k3r⁶) / (1 + k4r² + k5r⁴ + k6r⁶)
    RationalPolynomial,
    /// k1, k2, k3, k4: θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)
    Equidistant,
}

impl Model {
    /// How many coefficients the model takes (CameraInfo's D length).
    pub fn coefficient_count(self) -> usize {
        match self {
            Self::PlumbBob => 5,
            Self::RationalPolynomial => 8,
            Self::Equidistant => 4,
        }
    }

    /// The name CameraInfo uses.
    pub fn name(self) -> &'static str {
        match self {
            Self::PlumbBob => "plumb_bob",
            Self::RationalPolynomial => "rational_polynomial",
            Self::Equidistant => "equidistant",
        }
    }
}

impl std::str::FromStr for Model {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "plumb_bob" => Ok(Self::PlumbBob),
            "rational_polynomial" => Ok(Self::RationalPolynomial),
            "equidistant" | "fisheye" | "kannala_brandt" => Ok(Self::Equidistant),
            other => bail!(
                "unknown distortion model {other:?} (plumb_bob, rational_polynomial, or equidistant / fisheye / kannala_brandt)"
            ),
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

/// How the deskewed image maps view directions to pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// x = X/Z, y = Y/Z: straight lines stay straight; cannot hold 180 degrees
    Pinhole,
    /// x = azimuth, y = Y / horizontal distance: vertical lines stay straight, any horizontal field of view
    Cylindrical,
    /// x = azimuth, y = elevation: angles map linearly to pixels
    Equirectangular,
}

impl Projection {
    /// The name CameraInfo's `distortion_model` carries for an output in this projection.
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinhole => "pinhole",
            Self::Cylindrical => "cylindrical",
            Self::Equirectangular => "equirectangular",
        }
    }

    /// A view direction -> this projection's normalized image plane.
    fn to_plane(self, [x, y, z]: [f64; 3]) -> (f64, f64) {
        match self {
            Self::Pinhole => (x / z, y / z),
            Self::Cylindrical => (x.atan2(z), y / x.hypot(z)),
            Self::Equirectangular => (x.atan2(z), y.atan2(x.hypot(z))),
        }
    }

    /// This projection's normalized image plane -> a view direction.
    fn from_plane(self, a: f64, b: f64) -> [f64; 3] {
        match self {
            Self::Pinhole => [a, b, 1.0],
            Self::Cylindrical => [a.sin(), b, a.cos()],
            Self::Equirectangular => [a.sin() * b.cos(), b.sin(), a.cos() * b.cos()],
        }
    }
}

impl std::str::FromStr for Projection {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "pinhole" => Ok(Self::Pinhole),
            "cylindrical" => Ok(Self::Cylindrical),
            "equirectangular" => Ok(Self::Equirectangular),
            other => {
                bail!("unknown projection {other:?} (pinhole, cylindrical or equirectangular)")
            }
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
    /// Refuses coefficients the model doesn't take, and a camera that can't be right.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.fx > 0.0 && self.fy > 0.0,
            "focal lengths must be positive"
        );
        ensure!(
            self.width > 0 && self.height > 0,
            "the calibrated size must be positive"
        );
        let n = self.model.coefficient_count();
        ensure!(
            self.coefficients.len() <= n,
            "{} takes at most {n} distortion coefficients, got {}",
            self.model.name(),
            self.coefficients.len()
        );
        ensure!(
            self.coefficients.iter().all(|c| c.is_finite()),
            "distortion coefficients must be finite"
        );
        Ok(())
    }

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

    /// (k1, k2, p1, p2, k3, k4, k5, k6) of the plumb_bob / rational_polynomial models.
    fn brown(&self) -> [f64; 8] {
        std::array::from_fn(|i| self.k(i))
    }

    /// Radial factor and tangential offset of the Brown-Conrady models at normalized (x, y).
    fn brown_terms(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let [k1, k2, p1, p2, k3, k4, k5, k6] = self.brown();
        let r2 = x * x + y * y;
        let radial =
            (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)));
        let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        (radial, dx, dy)
    }

    fn theta_d(&self, theta: f64) -> f64 {
        let t2 = theta * theta;
        theta * (1.0 + t2 * (self.k(0) + t2 * (self.k(1) + t2 * (self.k(2) + t2 * self.k(3)))))
    }

    /// A view direction (camera frame, +z forward) -> the pixel the lens images it at, if it does.
    fn project(&self, [x, y, z]: [f64; 3]) -> Option<(f64, f64)> {
        let (xd, yd) = match self.model {
            Model::PlumbBob | Model::RationalPolynomial => {
                if z <= 1e-9 {
                    return None;
                }
                let (x, y) = (x / z, y / z);
                let (radial, dx, dy) = self.brown_terms(x, y);
                (x * radial + dx, y * radial + dy)
            }
            Model::Equidistant => {
                let r = x.hypot(y);
                if r < 1e-12 {
                    if z <= 0.0 {
                        return None;
                    }
                    (0.0, 0.0)
                } else {
                    let scale = self.theta_d(r.atan2(z)) / r;
                    (x * scale, y * scale)
                }
            }
        };
        Some((self.fx * xd + self.cx, self.fy * yd + self.cy))
    }

    /// A pixel -> the view direction it saw, inverting the lens iteratively as OpenCV does.
    fn unproject(&self, u: f64, v: f64) -> [f64; 3] {
        let (xd, yd) = ((u - self.cx) / self.fx, (v - self.cy) / self.fy);
        match self.model {
            Model::PlumbBob | Model::RationalPolynomial => {
                // Newton on the 2D distortion (a numeric Jacobian): converges where OpenCV's fixed-point loop stalls
                let distort = |x: f64, y: f64| {
                    let (radial, dx, dy) = self.brown_terms(x, y);
                    (x * radial + dx, y * radial + dy)
                };
                let (mut x, mut y) = (xd, yd);
                for _ in 0..30 {
                    let (fx, fy) = distort(x, y);
                    let (ex, ey) = (fx - xd, fy - yd);
                    if ex.abs() < 1e-14 && ey.abs() < 1e-14 {
                        break;
                    }
                    let h = 1e-7;
                    let (ax, ay) = distort(x + h, y);
                    let (bx, by) = distort(x, y + h);
                    let (j11, j21, j12, j22) =
                        ((ax - fx) / h, (ay - fy) / h, (bx - fx) / h, (by - fy) / h);
                    let det = j11 * j22 - j12 * j21;
                    if det.abs() < 1e-12 {
                        break;
                    }
                    x -= (j22 * ex - j12 * ey) / det;
                    y -= (-j21 * ex + j11 * ey) / det;
                }
                [x, y, 1.0]
            }
            Model::Equidistant => {
                let theta_d = xd.hypot(yd);
                if theta_d < 1e-12 {
                    return [0.0, 0.0, 1.0];
                }
                let mut theta = theta_d;
                for _ in 0..20 {
                    let t2 = theta * theta;
                    let (k1, k2, k3, k4) = (self.k(0), self.k(1), self.k(2), self.k(3));
                    let df =
                        1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)));
                    theta -= (self.theta_d(theta) - theta_d) / df;
                }
                let s = theta.sin() / theta_d;
                [xd * s, yd * s, theta.cos()]
            }
        }
    }
}

/// What to produce: the output projection and size (the input's when `None`), how much of the field of view to
/// keep, and how samples are interpolated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    pub projection: Projection,
    pub output_size: Option<(u32, u32)>,
    pub balance: f64,
    pub interpolation: Interpolation,
}

/// The deskewed frames' camera: no lens distortion, `projection` with these scales and center.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Output {
    pub projection: Projection,
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// `cv::fisheye::estimateNewCameraMatrixForUndistortRectify` (fov_scale 1, R identity), for any lens model and output
/// projection: the four edge midpoints' view directions, on the output's plane, set the scale and center.
pub fn new_camera(camera: &Intrinsics, options: &Options) -> Output {
    let (w, h) = (camera.width as f64, camera.height as f64);
    let balance = options.balance.clamp(0.0, 1.0);
    let edges = [(w * 0.5, 0.0), (w, h * 0.5), (w * 0.5, h), (0.0, h * 0.5)];
    let points: Vec<(f64, f64)> = edges
        .iter()
        .map(|&(u, v)| {
            let mut ray = camera.unproject(u, v);
            if options.projection == Projection::Pinhole {
                // past 90 degrees there is no pinhole point; keep the ray just inside the plane (as OpenCV does)
                let side = ray[0].hypot(ray[1]);
                let min_z = side * (1e-6f64).tan();
                ray[2] = ray[2].max(min_z);
            }
            options.projection.to_plane(ray)
        })
        .collect();
    let aspect = camera.fx / camera.fy;
    let cn = (
        points.iter().map(|p| p.0).sum::<f64>() / 4.0,
        aspect * points.iter().map(|p| p.1).sum::<f64>() / 4.0,
    );
    let (minx, maxx) = points
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.0), b.max(p.0)));
    let (miny, maxy) = points
        .iter()
        .map(|p| p.1 * aspect)
        .fold((f64::MAX, f64::MIN), |(a, b), y| (a.min(y), b.max(y)));
    let fs = [
        w * 0.5 / (cn.0 - minx),
        w * 0.5 / (maxx - cn.0),
        h * 0.5 * aspect / (cn.1 - miny),
        h * 0.5 * aspect / (maxy - cn.1),
    ];
    let fmin = fs.iter().copied().fold(f64::MAX, f64::min);
    let fmax = fs.iter().copied().fold(f64::MIN, f64::max);
    let f = balance * fmin + (1.0 - balance) * fmax;
    let (fx, fy) = (f, f / aspect);
    let (cx, cy) = (-cn.0 * f + w * 0.5, (-cn.1 * f + h * aspect * 0.5) / aspect);
    let (ow, oh) = options.output_size.unwrap_or((camera.width, camera.height));
    let (sx, sy) = (ow as f64 / w, oh as f64 / h);
    Output {
        projection: options.projection,
        width: ow,
        height: oh,
        fx: fx * sx,
        fy: fy * sy,
        cx: cx * sx,
        cy: cy * sy,
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
    /// a process-unique id of the current tables, so a copy of them elsewhere (the GPU's) knows when it is stale
    generation: u64,
    output: Option<Output>,
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

pub(crate) fn chroma_rows(rows: usize, sampling: Sampling) -> usize {
    match sampling {
        Sampling::Yuv420 => rows / 2,
        Sampling::Yuv422 => rows,
    }
}

impl Deskew {
    /// `camera` may be given at any size: it is scaled to each input's size.
    pub fn new(camera: Intrinsics, options: Options) -> Result<Self> {
        camera.validate()?;
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
            generation: 0,
            output: None,
            luma: None,
            chroma: None,
        })
    }

    /// The pinhole camera of the deskewed frames for this input size (what their CameraInfo should say).
    pub fn output_camera(&self, width: u32, height: u32) -> Output {
        new_camera(&self.camera.scaled_to(width, height), &self.options)
    }

    /// The input the tables are built for.
    pub fn built_for(&self) -> Option<(u32, u32, Sampling)> {
        self.built_for
    }

    /// The (luma, chroma) tap tables, once built (for the GPU path).
    pub(crate) fn tables(&self) -> Option<(&[u32], &[u32])> {
        Some((
            self.luma.as_ref()?.taps.as_slice(),
            self.chroma.as_ref()?.taps.as_slice(),
        ))
    }

    pub(crate) fn bilinear(&self) -> bool {
        self.options.interpolation == Interpolation::Bilinear
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
            |u, v| source_position(&camera, &out, u, v).unwrap_or((f64::NAN, f64::NAN)),
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
                let (x, y) = source_position(&camera, &out, lu, lv).unwrap_or((f64::NAN, f64::NAN));
                ((x + 0.5) / sub_x - 0.5, (y + 0.5) / sub_y - 0.5)
            },
        ));
        self.output = Some(out);
        self.built_for = Some((width, height, sampling));
        static GENERATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        self.generation = GENERATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Changes with every table build (0: none yet).
    pub fn generation(&self) -> u64 {
        self.generation
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

/// Output pixel (u, v) of the deskewed frame -> position in the input frame, both in pixels (`None`: a direction the
/// lens doesn't see).
fn source_position(camera: &Intrinsics, out: &Output, u: f64, v: f64) -> Option<(f64, f64)> {
    let ray = out
        .projection
        .from_plane((u - out.cx) / out.fx, (v - out.cy) / out.fy);
    camera.project(ray)
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

    fn opts(projection: Projection, balance: f64) -> Options {
        Options {
            projection,
            output_size: None,
            balance,
            interpolation: Interpolation::Bilinear,
        }
    }

    /// `cv2.initUndistortRectifyMap` / `cv2.fisheye.initUndistortRectifyMap` (CV_32FC1) for one camera
    /// (fx 600, fy 590, cx 652.5, cy 351.25) into one pinhole (fx 420, fy 410, cx 640, cy 360), at six output pixels.
    #[test]
    fn remap_positions_match_opencv_for_every_model() {
        type Case = (Model, &'static [f64], [(f64, f64, f64, f64); 6]);
        let cases: [Case; 3] = [
            (
                Model::PlumbBob,
                &[-0.28, 0.07, 0.0012, -0.0009, -0.008],
                [
                    (0.0, 0.0, 131.93, 59.423),
                    (100.0, 50.0, 160.713, 69.132),
                    (640.0, 360.0, 652.5, 351.25),
                    (1000.0, 600.0, 1047.635, 617.754),
                    (1279.0, 719.0, 1168.568, 646.43),
                    (320.0, 700.0, 311.457, 716.424),
                ],
            ),
            (
                Model::RationalPolynomial,
                &[0.9, -0.3, 0.0011, -0.0007, 0.02, 1.2, -0.1, 0.05],
                [
                    (0.0, 0.0, 388.048, 204.151),
                    (100.0, 50.0, 291.632, 144.553),
                    (640.0, 360.0, 652.5, 351.25),
                    (1000.0, 600.0, 1030.22, 605.909),
                    (1279.0, 719.0, 915.514, 502.829),
                    (320.0, 700.0, 335.176, 691.126),
                ],
            ),
            (
                Model::Equidistant,
                &[0.03, -0.012, 0.004, -0.001],
                [
                    (0.0, 0.0, 92.361, 33.866),
                    (100.0, 50.0, 135.188, 52.102),
                    (640.0, 360.0, 652.5, 351.25),
                    (1000.0, 600.0, 1056.965, 622.866),
                    (1279.0, 719.0, 1212.371, 668.095),
                    (320.0, 700.0, 303.968, 724.275),
                ],
            ),
        ];
        let out = Output {
            projection: Projection::Pinhole,
            width: 1280,
            height: 720,
            fx: 420.0,
            fy: 410.0,
            cx: 640.0,
            cy: 360.0,
        };
        for (model, d, points) in cases {
            let camera = Intrinsics {
                width: 1280,
                height: 720,
                fx: 600.0,
                fy: 590.0,
                cx: 652.5,
                cy: 351.25,
                model,
                coefficients: d.to_vec(),
            };
            for (u, v, x, y) in points {
                let (gx, gy) = source_position(&camera, &out, u, v).unwrap();
                assert!(
                    (gx - x).abs() < 2e-3 && (gy - y).abs() < 2e-3,
                    "{model:?} ({u},{v}): {gx},{gy} vs OpenCV {x},{y}"
                );
            }
        }
    }

    /// `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify` for fx = fy = 486, k = 0.02, -0.01, 0.003, 0.
    #[test]
    fn new_camera_matches_opencv() {
        let c = Intrinsics {
            coefficients: vec![0.02, -0.01, 0.003, 0.0],
            ..fisheye()
        };
        for (balance, f) in [(0.0, 398.56284), (0.5, 290.976456), (1.0, 183.390073)] {
            let k = new_camera(&c, &opts(Projection::Pinhole, balance));
            assert!(
                (k.fx - f).abs() < 1e-5 && (k.fy - f).abs() < 1e-5,
                "balance {balance}: {}",
                k.fx
            );
            assert!((k.cx - 640.0).abs() < 1e-6 && (k.cy - 360.0).abs() < 1e-6);
        }
    }

    #[test]
    fn unproject_inverts_project_for_every_model() {
        for (model, d) in [
            (Model::PlumbBob, vec![-0.2, 0.05, 0.001, -0.002, 0.0]),
            (
                Model::RationalPolynomial,
                vec![0.9, -0.3, 0.0011, -0.0007, 0.02, 1.2, -0.1, 0.05],
            ),
            (Model::Equidistant, vec![0.03, -0.012, 0.004, -0.001]),
        ] {
            let c = Intrinsics {
                model,
                coefficients: d,
                ..fisheye()
            };
            for (u, v) in [(640.0, 360.0), (300.0, 200.0), (1100.0, 600.0)] {
                let (pu, pv) = c.project(c.unproject(u, v)).unwrap();
                assert!(
                    (pu - u).abs() < 1e-6 && (pv - v).abs() < 1e-6,
                    "{model:?} {u},{v} -> {pu},{pv}"
                );
            }
        }
    }

    #[test]
    fn a_fisheye_sees_past_ninety_degrees_and_a_pinhole_model_does_not() {
        let side = [1.0, 0.0, -0.1];
        assert!(fisheye().project(side).is_some());
        assert!(Intrinsics {
            model: Model::PlumbBob,
            ..fisheye()
        }
        .project(side)
        .is_none());
    }

    #[test]
    fn unknown_models_and_too_many_coefficients_are_config_errors() {
        assert_eq!(
            "kannala_brandt".parse::<Model>().unwrap(),
            Model::Equidistant
        );
        assert_eq!("fisheye".parse::<Model>().unwrap(), Model::Equidistant);
        assert_eq!(
            "rational_polynomial".parse::<Model>().unwrap(),
            Model::RationalPolynomial
        );
        assert!(
            format!("{:#}", "double_sphere".parse::<Model>().unwrap_err())
                .contains("unknown distortion model")
        );
        let c = Intrinsics {
            coefficients: vec![0.0; 5],
            ..fisheye()
        };
        assert!(format!(
            "{:#}",
            Deskew::new(c, opts(Projection::Pinhole, 0.0))
                .err()
                .unwrap()
        )
        .contains("at most 4"));
        assert!("mercator".parse::<Projection>().is_err());
    }

    #[test]
    fn cylindrical_keeps_azimuth_linear_and_fits_the_whole_width() {
        // a zero-coefficient fisheye's edge midpoints are 640/486 rad (sideways) and 360/486 rad (up) off axis
        let out = new_camera(&fisheye(), &opts(Projection::Cylindrical, 1.0));
        // balance 1 keeps the whole field of view: the smaller of the two scales (here the vertical one) wins
        let vertical = 360.0 / (360.0f64 / 486.0).tan();
        assert!((out.fx - vertical).abs() < 1e-6, "{}", out.fx);
        // balance 0 fills the frame: the horizontal extent sets the scale
        let fill = new_camera(&fisheye(), &opts(Projection::Cylindrical, 0.0));
        assert!((fill.fx - 486.0).abs() < 1e-6, "{}", fill.fx);
        // along the center row the source position moves linearly in angle
        let (x, _) = source_position(&fisheye(), &fill, fill.cx + fill.fx * 0.5, fill.cy).unwrap();
        assert!((x - (640.0 + 486.0 * 0.5)).abs() < 1e-6, "{x}");
    }

    #[test]
    fn equirectangular_maps_angles_linearly_both_ways() {
        let out = new_camera(&fisheye(), &opts(Projection::Equirectangular, 0.5));
        let (x, y) = source_position(&fisheye(), &out, out.cx, out.cy + out.fy * 0.3).unwrap();
        assert!(
            (x - 640.0).abs() < 1e-6 && (y - (360.0 + 486.0 * 0.3)).abs() < 1e-6,
            "{x},{y}"
        );
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
        let mut d = Deskew::new(pinhole, opts(Projection::Pinhole, 0.0)).unwrap();
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
    fn output_size_scales_the_new_camera() {
        let half = Options {
            output_size: Some((640, 360)),
            ..opts(Projection::Pinhole, 0.0)
        };
        let (a, b) = (
            new_camera(&fisheye(), &half),
            new_camera(&fisheye(), &opts(Projection::Pinhole, 0.0)),
        );
        assert!((a.fx * 2.0 - b.fx).abs() < 1e-9 && (a.cx * 2.0 - b.cx).abs() < 1e-9);
        assert_eq!((a.width, a.height), (640, 360));
    }
}
