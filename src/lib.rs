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
pub mod vepu580_template;
pub mod vepu580_tune;

use anyhow::Result;

pub use vepu580::Frame;

/// The stream to produce: what `mpph264enc rc-mode=cbr bps=... gop=...` was told.
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
            max_jpeg_bytes: 2 << 20,
        }
    }
}

/// One MJPEG -> H.264 stream: a JPEG decoder session, an encoder session and the frame buffer between them.
pub struct Transcoder {
    decoder: jpegd::JpegDecoder,
    encoder: vepu580::H264Encoder,
    frame: dma::DmaBuf,
    platform: &'static platform::Tested,
}

impl Transcoder {
    /// Checks the platform (refusing an untested one), then opens both hardware sessions and allocates every buffer.
    pub fn new(config: Config) -> Result<Self> {
        let (_, tested) = platform::probe_and_check()?;
        let heap = dma::Heap::open()?;
        Ok(Self {
            decoder: jpegd::JpegDecoder::new(
                &heap,
                config.width,
                config.height,
                config.max_jpeg_bytes,
            )?,
            encoder: vepu580::H264Encoder::new(
                &heap,
                vepu580::EncoderConfig {
                    width: config.width,
                    height: config.height,
                    fps: config.fps,
                    bitrate_bps: config.bitrate_bps,
                    gop: config.gop,
                },
            )?,
            // semi-planar YUV, room for 4:2:2
            frame: heap.alloc((config.width * config.height.next_multiple_of(16) * 2) as usize)?,
            platform: tested,
        })
    }

    /// Which tested platform this is.
    pub fn platform(&self) -> &'static platform::Tested {
        self.platform
    }

    /// Decodes one JPEG and encodes it as one Annex-B access unit into `out` (replacing its contents). A JPEG that
    /// fails to decode is an error and leaves the stream intact; an encoder failure makes the next frame an IDR.
    pub fn transcode(&mut self, jpeg: &[u8], out: &mut Vec<u8>) -> Result<Frame> {
        let sampling = self.decoder.decode(jpeg, &self.frame)?;
        self.encoder.encode(&self.frame, sampling, out)
    }

    /// Makes the next frame an IDR.
    pub fn force_idr(&mut self) {
        self.encoder.force_idr();
    }
}
