//! MJPEG -> H.264 on the Rockchip RK3588 in pure Rust: the hardware JPEG decoder and the VEPU580 H.264 encoder,
//! driven straight through the kernel's `/dev/mpp_service` ioctls (no MPP, no GStreamer). Every buffer is allocated
//! once up front, so a stream runs indefinitely without growing its fd count or memory.
//!
//! Before touching the hardware, [`Transcoder::new`] checks the platform against the exact kernel / driver / SoC /
//! hardware-id combinations this crate was verified on ([`platform::TESTED`]) and refuses anything else.
//!
//! ```no_run
//! let mut t = rockchip_h264::Transcoder::new(rockchip_h264::Config::default()).unwrap();
//! let mut access_unit = Vec::new();
//! let jpeg = std::fs::read("frame.jpg").unwrap();
//! t.transcode(&jpeg, &mut access_unit).unwrap();
//! ```

pub mod dma;
pub mod h264;
pub mod jpeg;
pub mod jpegd;
pub mod platform;
pub mod rc;
pub mod service;
pub mod v4l2;
pub mod vepu580;
#[cfg(test)]
mod vepu580_fixtures;
pub mod vepu580_regs;
pub mod vepu580_setup;
pub mod vepu580_tables;
pub mod vepu580_tune;

use anyhow::Result;

pub use vepu580::Frame;

/// The stream to produce: what `mpph264enc rc-mode=cbr bps=... gop=...` was told. `width` and `height` are what
/// the stream is expected to be; each JPEG's own size is what gets encoded (see [`Transcoder::transcode`]).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_bps: u32,
    /// keyframe interval in frames
    pub gop: u32,
    /// largest JPEG accepted
    pub max_jpeg_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps: 30,
            bitrate_bps: 4_000_000,
            gop: 30,
            max_jpeg_bytes: 4 << 20,
        }
    }
}

/// One MJPEG -> H.264 stream: a JPEG decoder session, an encoder session and the frame buffer between them.
pub struct Transcoder {
    config: Config,
    heap: dma::Heap,
    decoder: jpegd::JpegDecoder,
    /// the encoder and the decoder's output buffer, both sized for the current frame size (None between sizes)
    sized: Option<(vepu580::H264Encoder, dma::DmaBuf)>,
    size: (u32, u32),
    resizes: u32,
    platform: &'static platform::Tested,
}

impl Transcoder {
    /// Checks the platform (refusing an untested one) and the configured size, then opens both hardware sessions and
    /// allocates every buffer.
    pub fn new(config: Config) -> Result<Self> {
        let (_, tested) = platform::probe_and_check()?;
        let heap = dma::Heap::open()?;
        let sized = Self::sized(&heap, &config, config.width, config.height)?;
        Ok(Self {
            config,
            decoder: jpegd::JpegDecoder::new(&heap, config.max_jpeg_bytes)?,
            heap,
            sized: Some(sized),
            size: (config.width, config.height),
            resizes: 0,
            platform: tested,
        })
    }

    fn sized(
        heap: &dma::Heap,
        config: &Config,
        width: u32,
        height: u32,
    ) -> Result<(vepu580::H264Encoder, dma::DmaBuf)> {
        let encoder = vepu580::H264Encoder::new(
            heap,
            vepu580::EncoderConfig {
                width,
                height,
                fps: config.fps,
                bitrate_bps: config.bitrate_bps,
                gop: config.gop,
            },
        )?;
        Ok((encoder, heap.alloc(jpegd::output_size(width, height))?))
    }

    /// Which tested platform this is.
    pub fn platform(&self) -> &'static platform::Tested {
        self.platform
    }

    /// The size being encoded now.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// How many times the stream changed size.
    pub fn resizes(&self) -> u32 {
        self.resizes
    }

    /// Decodes one JPEG and encodes it as one Annex-B access unit into `out` (replacing its contents).
    ///
    /// The size encoded is the JPEG's own (its SOF header), not the configured one: if it changes, the encoder and
    /// its buffers are rebuilt for the new size (the old ones released first) and the stream restarts with new
    /// parameter sets and an IDR. A JPEG that fails to parse or decode is an error and leaves the stream intact; an
    /// encoder failure makes the next frame an IDR.
    pub fn transcode(&mut self, jpeg: &[u8], out: &mut Vec<u8>) -> Result<Frame> {
        let header = jpeg::parse(&jpeg[..jpeg::trimmed_len(jpeg)])?;
        if (header.width, header.height) != self.size || self.sized.is_none() {
            vepu580::check_size(header.width, header.height)?;
            // release the old size's session and buffers before taking the new ones
            self.sized = None;
            self.sized = Some(Self::sized(
                &self.heap,
                &self.config,
                header.width,
                header.height,
            )?);
            if (header.width, header.height) != self.size {
                self.resizes += 1;
            }
            self.size = (header.width, header.height);
        }
        let (encoder, frame) = self.sized.as_mut().expect("set above");
        let sampling = self.decoder.decode(jpeg, &header, frame)?;
        encoder.encode(frame, sampling, out)
    }

    /// Makes the next frame an IDR.
    pub fn force_idr(&mut self) {
        if let Some((encoder, _)) = self.sized.as_mut() {
            encoder.force_idr();
        }
    }
}
