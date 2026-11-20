//! The RK3588 VEPU580 hardware H.264 encoder (RKVENC): one task per frame, IDR then P frames off one reference,
//! the hardware writing each slice (header included, with its start code) into a bitstream dmabuf.
//!
//! The register image starts from the vendor MPP build's own (`vepu580_template.rs`), and every per-frame field is
//! set here by name, following MPP `mpp/hal/rkenc/h264e/hal_h264e_vepu580.c` (`setup_vepu580_rc_base`,
//! `setup_vepu580_io_buf`, `setup_vepu580_recn_refr`, `setup_vepu580_dual_core`, `setup_vepu580_codec`,
//! `hal_h264e_vepu580_start` / `_wait` / `_ret_task`). Field positions come from `hal_h264e_vepu580_reg.h` and
//! `mpp/hal/rkenc/common/vepu580_common.h`.

use anyhow::{bail, ensure, Context, Result};

use crate::dma::{DmaBuf, Heap};
use crate::h264;
use crate::jpeg::Sampling;
use crate::rc::{Plan, RateControl, QP_MAX, QP_MIN};
use crate::service::{client, AddrOffset, Read, Session, Write};
use crate::vepu580_template as tpl;
use crate::vepu580_tune::Tune;

/// One register field: byte offset of its 32-bit register, lowest bit, width.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub addr: u32,
    pub lsb: u32,
    pub width: u32,
}

pub const fn f(addr: u32, lsb: u32, width: u32) -> Field {
    Field { addr, lsb, width }
}

const fn mask(width: u32) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1 << width) - 1
    }
}

const OSD_OFFSET: u32 = 0x3000;
const OSD_WORDS: usize = 288;
const INT_STA: u32 = 0x2c;
const ST_OFFSET: u32 = 0x4000;
const ST_WORDS: usize = 180;

/// The register classes MPP writes, in the order it writes them (`hal_h264e_vepu580_start`).
#[derive(Clone)]
pub struct Regs {
    pub ctl: [u32; tpl::CTL.len()],
    pub base: [u32; tpl::BASE.len()],
    pub rc: [u32; tpl::RC.len()],
    pub s3: [u32; tpl::S3.len()],
    pub rdo: [u32; tpl::RDO.len()],
    pub scl: [u32; tpl::SCL.len()],
    pub osd: [u32; OSD_WORDS],
}

impl Regs {
    pub fn template(i_slice: bool) -> Box<Self> {
        let mut regs = Box::new(Self {
            ctl: tpl::CTL,
            base: tpl::BASE,
            rc: tpl::RC,
            s3: tpl::S3,
            rdo: tpl::RDO,
            scl: tpl::SCL,
            osd: [0; OSD_WORDS],
        });
        if !i_slice {
            for (addr, value) in tpl::P_OVERRIDES {
                *regs.word_mut(addr) = value;
            }
        }
        regs
    }

    fn classes(&self) -> [(u32, &[u32]); 7] {
        [
            (tpl::CTL_OFFSET, &self.ctl),
            (tpl::BASE_OFFSET, &self.base),
            (tpl::RC_OFFSET, &self.rc),
            (tpl::S3_OFFSET, &self.s3),
            (tpl::RDO_OFFSET, &self.rdo),
            (tpl::SCL_OFFSET, &self.scl),
            (OSD_OFFSET, &self.osd),
        ]
    }

    fn word_mut(&mut self, addr: u32) -> &mut u32 {
        let class: &mut [u32] = match addr {
            a if a < tpl::BASE_OFFSET => &mut self.ctl,
            a if a < tpl::RC_OFFSET => &mut self.base,
            a if a < tpl::S3_OFFSET => &mut self.rc,
            a if a < tpl::RDO_OFFSET => &mut self.s3,
            a if a < tpl::SCL_OFFSET => &mut self.rdo,
            a if a < OSD_OFFSET => &mut self.scl,
            _ => &mut self.osd,
        };
        let base = match addr {
            a if a < tpl::BASE_OFFSET => tpl::CTL_OFFSET,
            a if a < tpl::RC_OFFSET => tpl::BASE_OFFSET,
            a if a < tpl::S3_OFFSET => tpl::RC_OFFSET,
            a if a < tpl::RDO_OFFSET => tpl::S3_OFFSET,
            a if a < tpl::SCL_OFFSET => tpl::RDO_OFFSET,
            a if a < OSD_OFFSET => tpl::SCL_OFFSET,
            _ => OSD_OFFSET,
        };
        &mut class[((addr - base) / 4) as usize]
    }

    pub fn word(&self, addr: u32) -> u32 {
        for (base, words) in self.classes() {
            if addr >= base && ((addr - base) / 4) < words.len() as u32 {
                return words[((addr - base) / 4) as usize];
            }
        }
        panic!("register {addr:#x} is outside every class");
    }

    pub fn set(&mut self, field: Field, value: u32) {
        let m = mask(field.width);
        let w = self.word_mut(field.addr);
        *w = (*w & !(m << field.lsb)) | ((value & m) << field.lsb);
    }

    pub fn get(&self, field: Field) -> u32 {
        (self.word(field.addr) >> field.lsb) & mask(field.width)
    }
}

/// The status the kernel copies back after the task: `int_sta` and the status class.
pub struct Status {
    pub int_sta: [u32; 1],
    pub st: [u32; ST_WORDS],
}

impl Status {
    pub fn word(&self, addr: u32) -> u32 {
        self.st[((addr - ST_OFFSET) / 4) as usize]
    }

    pub fn get(&self, field: Field) -> u32 {
        (self.word(field.addr) >> field.lsb) & mask(field.width)
    }
}

// reg_base buffer addresses (register index = byte offset / 4)
const ADR_SRC0: u32 = 0x280;
const ADR_SRC1: u32 = 0x284;
const ADR_SRC2: u32 = 0x288;
const RFPW_H: u32 = 0x28c;
const RFPW_B: u32 = 0x290;
const RFPR_H: u32 = 0x294;
const RFPR_B: u32 = 0x298;
const DSPW: u32 = 0x2a4;
const DSPR: u32 = 0x2a8;
const BSBT: u32 = 0x2b0;
const BSBB: u32 = 0x2b4;
const BSBR: u32 = 0x2b8;
const BSBS: u32 = 0x2bc;

const PIC_QP: Field = f(0x300, 8, 6);
const CUR_FRM_REF: Field = f(0x300, 2, 1);
const DCHS_TXID: Field = f(0x304, 0, 2);
const DCHS_RXID: Field = f(0x304, 2, 2);
const DCHS_TXE: Field = f(0x304, 4, 1);
const DCHS_RXE: Field = f(0x304, 5, 1);
const DCHS_OFST: Field = f(0x304, 16, 11);
const PIC_WD8_M1: Field = f(0x310, 0, 11);
const PIC_HD8_M1: Field = f(0x310, 16, 11);
const SRC_CFMT: Field = f(0x318, 2, 4);
const SRC_STRD0: Field = f(0x334, 0, 17);
const SRC_STRD1: Field = f(0x338, 0, 16);
const RC_CTU_NUM: Field = f(0x350, 12, 20);
const RC_QP_RANGE: Field = f(0x354, 16, 4);
const RC_MAX_QP: Field = f(0x354, 20, 6);
const RC_MIN_QP: Field = f(0x354, 26, 6);
const CTU_EBIT: Field = f(0x358, 0, 20);
const NAL_REF_IDC: Field = f(0x3b0, 0, 2);
const NAL_UNIT_TYPE: Field = f(0x3b0, 2, 5);
const MAX_FNUM: Field = f(0x3b4, 0, 4);
const MPOC_LM4: Field = f(0x3b4, 5, 4);
const PIC_INIT_QP: Field = f(0x3b8, 7, 6);
const CB_OFST: Field = f(0x3b8, 13, 5);
const CR_OFST: Field = f(0x3b8, 18, 5);
const SLI_TYPE: Field = f(0x3bc, 0, 2);
const FRM_NUM: Field = f(0x3bc, 16, 16);
const IDR_PID: Field = f(0x3c0, 0, 16);
const POC_LSB: Field = f(0x3c0, 16, 16);
const RC_DTHD: u32 = 0x1008;
const ROI_QP_AREAS: [Field; 16] = [
    f(0x1030, 0, 6),
    f(0x1030, 6, 6),
    f(0x1030, 12, 6),
    f(0x1030, 18, 6),
    f(0x1030, 24, 6),
    f(0x1034, 0, 6),
    f(0x1034, 6, 6),
    f(0x1034, 12, 6),
    f(0x1034, 18, 6),
    f(0x1034, 24, 6),
    f(0x1038, 0, 6),
    f(0x1038, 6, 6),
    f(0x1038, 12, 6),
    f(0x1038, 18, 6),
    f(0x1038, 24, 6),
    f(0x103c, 0, 6),
];

const ST_BS_LGTH_L32: u32 = 0x4000;
const ST_QP_SUM: u32 = 0x400c;
/// int_sta: bit-stream overflow, bus write full, bus write error, bus read error, watchdog
const INT_STA_ERRORS: u32 = 0x1f << 4;

/// `VEPU5xx_FMT_*`
fn src_cfmt(sampling: Sampling) -> u32 {
    match sampling {
        Sampling::Yuv420 => 6,
        Sampling::Yuv422 => 4,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_bps: u32,
    /// keyframe interval in frames
    pub gop: u32,
}

/// The only resolution the register template was captured and verified for.
pub const SUPPORTED_SIZES: &[(u32, u32)] = &[(1280, 720)];

/// One reconstructed frame (the encoder's reference): its FBC pixel buffer and downscaled "thumb".
struct Recon {
    pixel: DmaBuf,
    thumb: DmaBuf,
}

/// What one frame produced.
pub struct Frame {
    pub idr: bool,
    pub qp: u32,
    pub avg_qp: f64,
    pub target_bits: u32,
    pub bits: u32,
    pub moving_scene: bool,
}

pub struct H264Encoder {
    config: EncoderConfig,
    session: Session,
    recon: [Recon; 2],
    /// slot the next frame writes its reconstruction into
    curr_slot: usize,
    out: DmaBuf,
    fbc_header_size: u32,
    frames_since_idr: u32,
    idr_count: u32,
    /// dual-core handshake id the next frame transmits under, and the id each recon slot was written under
    dchs_curr: u32,
    dchs_slot_txid: [u32; 2],
    rc: RateControl,
    tune: Tune,
    force_idr: bool,
    sps_pps: Vec<u8>,
    aud: Vec<u8>,
}

impl H264Encoder {
    pub fn new(heap: &Heap, config: EncoderConfig) -> Result<Self> {
        if !SUPPORTED_SIZES.contains(&(config.width, config.height)) {
            bail!("no Rust version for {}x{} (the encoder's register image is verified for {SUPPORTED_SIZES:?} only)", config.width, config.height);
        }
        ensure!(
            config.fps > 0 && config.gop > 0 && config.bitrate_bps > 0,
            "fps, gop and bitrate must be positive"
        );
        // setup_hal_bufs: FBC reconstruction needs 64-aligned width and 16 extra lines
        let aligned_w = config.width.next_multiple_of(64);
        let aligned_h = config.height.next_multiple_of(16) + 16;
        let fbc_header_size = (aligned_w * aligned_h / 64).next_multiple_of(8192);
        let pixel_size = fbc_header_size + aligned_w * aligned_h * 3 / 2;
        let thumb_size = (aligned_w / 64 * aligned_h / 64 * 256).next_multiple_of(8192);
        let recon = || -> Result<Recon> {
            Ok(Recon {
                pixel: heap.alloc(pixel_size as usize)?,
                thumb: heap.alloc(thumb_size as usize)?,
            })
        };
        let session = Session::new(client::RKVENC).context("opening the H.264 encoder")?;
        // what /proc/mpp_service/*/sessions-info shows: width, height, format, fps in/out, rc mode, bitrate, gop
        let info = |key: u32, flag: u32, value: u32| [key, flag, value, 0];
        let _ = session.send_codec_info(&[
            info(1, 1, config.width),
            info(2, 1, config.height),
            info(3, 2, u32::from_le_bytes(*b"h264")),
            info(4, 1, config.fps),
            info(5, 1, config.fps),
            info(6, 2, u32::from_le_bytes(*b"cbr\0")),
            info(7, 1, config.bitrate_bps),
            info(8, 1, config.gop),
            info(10, 2, u32::from_le_bytes(*b"high")),
        ]);
        let params = h264::StreamParams {
            width: config.width,
            height: config.height,
            fps: config.fps,
        };
        Ok(Self {
            config,
            session,
            recon: [recon()?, recon()?],
            curr_slot: 0,
            // the vendor build's packet buffer: one luma plane's worth
            out: heap.alloc((config.width * config.height) as usize)?,
            fbc_header_size,
            frames_since_idr: 0,
            idr_count: 0,
            dchs_curr: 0,
            dchs_slot_txid: [0; 2],
            rc: RateControl::new(config.bitrate_bps, config.fps, config.gop),
            tune: Tune::default(),
            force_idr: true,
            sps_pps: [h264::sps(params), h264::pps()].concat(),
            aud: h264::aud(),
        })
    }

    /// Makes the next frame an IDR (it also follows every `gop` frames, and any failed frame).
    pub fn force_idr(&mut self) {
        self.force_idr = true;
    }

    /// The register image and fd offsets for one frame.
    fn registers(
        &mut self,
        src: &DmaBuf,
        sampling: Sampling,
        idr: bool,
        plan: Plan,
    ) -> (Box<Regs>, [AddrOffset; 6]) {
        let c = self.config;
        let mut regs = Regs::template(idr);
        let (mb_w, mb_h) = (c.width.div_ceil(16), c.height.div_ceil(16));

        // prep: source format and strides; resolution
        regs.set(SRC_CFMT, src_cfmt(sampling));
        regs.set(SRC_STRD0, c.width);
        regs.set(SRC_STRD1, c.width);
        regs.set(PIC_WD8_M1, c.width.div_ceil(8) - 1);
        regs.set(PIC_HD8_M1, c.height.div_ceil(8) - 1);

        // codec: the parameter sets' fields the hardware needs for the slice header
        regs.set(MAX_FNUM, h264::LOG2_MAX_FRAME_NUM_MINUS4);
        regs.set(MPOC_LM4, h264::LOG2_MAX_POC_LSB_MINUS4);
        regs.set(PIC_INIT_QP, h264::PIC_INIT_QP);
        regs.set(CB_OFST, h264::CHROMA_QP_INDEX_OFFSET as u32);
        regs.set(CR_OFST, h264::CHROMA_QP_INDEX_OFFSET as u32);
        let frame_num = self.frames_since_idr;
        regs.set(NAL_REF_IDC, if idr { 3 } else { 2 });
        regs.set(NAL_UNIT_TYPE, if idr { 5 } else { 1 });
        regs.set(SLI_TYPE, if idr { 2 } else { 0 });
        regs.set(FRM_NUM, frame_num);
        regs.set(IDR_PID, if idr { self.idr_count } else { 0xffff });
        regs.set(POC_LSB, 2 * frame_num);

        // dual core handshake: the reference's writer id, so whichever core takes this frame waits for it
        let refr_slot = 1 - self.curr_slot;
        let (rxe, rxid) = if idr {
            self.dchs_curr = 0;
            (0, 0)
        } else {
            (1, self.dchs_slot_txid[refr_slot])
        };
        self.dchs_slot_txid[self.curr_slot] = self.dchs_curr;
        regs.set(DCHS_TXID, self.dchs_curr);
        regs.set(DCHS_RXID, rxid);
        regs.set(DCHS_TXE, regs.get(CUR_FRM_REF));
        regs.set(DCHS_RXE, rxe);
        regs.set(DCHS_OFST, 9);
        self.dchs_curr = (self.dchs_curr + 1) % 4;

        // rate control
        set_rate_control(&mut regs, mb_w, mb_h, plan);

        // buffers: source, reconstruction (written) and reference (read), bitstream
        let curr = &self.recon[self.curr_slot];
        let refr = if idr { curr } else { &self.recon[refr_slot] };
        for addr in [ADR_SRC0, ADR_SRC1, ADR_SRC2] {
            regs.set(f(addr, 0, 32), src.fd());
        }
        regs.set(f(RFPW_H, 0, 32), curr.pixel.fd());
        regs.set(f(RFPW_B, 0, 32), curr.pixel.fd());
        regs.set(f(DSPW, 0, 32), curr.thumb.fd());
        regs.set(f(RFPR_H, 0, 32), refr.pixel.fd());
        regs.set(f(RFPR_B, 0, 32), refr.pixel.fd());
        regs.set(f(DSPR, 0, 32), refr.thumb.fd());
        for addr in [BSBT, BSBB, BSBR, BSBS] {
            regs.set(f(addr, 0, 32), self.out.fd());
        }
        let chroma = c.width * c.height;
        let offsets = [
            AddrOffset {
                index: ADR_SRC1 / 4,
                offset: chroma,
            },
            AddrOffset {
                index: ADR_SRC2 / 4,
                offset: chroma,
            },
            AddrOffset {
                index: BSBT / 4,
                offset: self.out.len() as u32,
            },
            AddrOffset {
                index: BSBS / 4,
                offset: 0,
            },
            AddrOffset {
                index: RFPW_B / 4,
                offset: self.fbc_header_size,
            },
            AddrOffset {
                index: RFPR_B / 4,
                offset: self.fbc_header_size,
            },
        ];

        self.tune.patch(&mut regs, idr);
        (regs, offsets)
    }

    /// Encodes one YUV frame in `src` (the JPEG decoder's output) into one Annex-B access unit: an access unit
    /// delimiter, then on an IDR the SPS and PPS, then the slice.
    pub fn encode(&mut self, src: &DmaBuf, sampling: Sampling, out: &mut Vec<u8>) -> Result<Frame> {
        let idr = self.force_idr || self.frames_since_idr >= self.config.gop;
        if idr {
            self.frames_since_idr = 0;
        }
        let plan = self.rc.plan(idr);
        let (regs, offsets) = self.registers(src, sampling, idr, plan);
        let mut status = Status {
            int_sta: [0],
            st: [0; ST_WORDS],
        };
        let classes = regs.classes();
        let writes = classes.map(|(offset, words)| Write { offset, words });
        let result = self.session.run(
            &writes,
            &mut [
                Read {
                    offset: INT_STA,
                    words: &mut status.int_sta,
                },
                Read {
                    offset: ST_OFFSET,
                    words: &mut status.st,
                },
            ],
            &offsets,
        );
        // anything that goes wrong leaves no trustworthy reference: start over at an IDR
        self.force_idr = true;
        result?;
        let int_sta = status.int_sta[0];
        ensure!(
            int_sta & 1 != 0 && int_sta & INT_STA_ERRORS == 0,
            "H.264 encode failed: interrupt status {int_sta:#06x}"
        );
        let len = status.word(ST_BS_LGTH_L32) as usize;
        ensure!(
            len > 4 && len <= self.out.len(),
            "H.264 encoder reported a {len}-byte slice"
        );
        out.clear();
        out.extend_from_slice(&self.aud);
        if idr {
            out.extend_from_slice(&self.sps_pps);
        }
        self.out.read(len, |slice| out.extend_from_slice(slice))?;
        ensure!(
            out[out.len() - len..].starts_with(&[0, 0, 0, 1]),
            "the encoder's slice has no start code"
        );

        let mbs = self.config.width.div_ceil(16) * self.config.height.div_ceil(16);
        let avg_qp = status.word(ST_QP_SUM) as f64 / mbs as f64;
        let bits = (len * 8) as u32;
        self.rc.update(idr, plan, bits, avg_qp);
        self.tune.update(&status);
        if idr {
            self.idr_count = (self.idr_count + 1) & 0xffff;
        }
        self.frames_since_idr += 1;
        self.curr_slot = 1 - self.curr_slot;
        self.force_idr = false;
        Ok(Frame {
            idr,
            qp: plan.qp,
            avg_qp,
            target_bits: plan.target_bits,
            bits,
            moving_scene: self.tune.moving_scene(),
        })
    }
}

/// `setup_vepu580_rc_base`: the frame's starting QP and the row-level rate control's bit target and thresholds.
fn set_rate_control(regs: &mut Regs, mb_w: u32, mb_h: u32, plan: Plan) {
    let mut mb_target_bits_mul_16 = ((plan.target_bits as u64) << 4) / (mb_w * mb_h) as u64;
    if mb_target_bits_mul_16 >= 0x100000 {
        mb_target_bits_mul_16 = 0x50000;
    }
    let mb_target_bits = ((mb_target_bits_mul_16 * mb_w as u64) >> 4) as i32;
    let negative_bits_thd = -(5 * mb_target_bits / 16);
    let positive_bits_thd = 5 * mb_target_bits / 16;
    regs.set(PIC_QP, plan.qp);
    regs.set(RC_CTU_NUM, mb_w);
    regs.set(RC_QP_RANGE, 2);
    regs.set(RC_MAX_QP, QP_MAX);
    regs.set(RC_MIN_QP, QP_MIN);
    regs.set(CTU_EBIT, mb_target_bits_mul_16 as u32);
    let thresholds = [
        4 * negative_bits_thd,
        negative_bits_thd,
        positive_bits_thd,
        4 * positive_bits_thd,
    ];
    for (k, t) in thresholds.into_iter().enumerate() {
        regs.set(f(RC_DTHD + 4 * k as u32, 0, 32), t as u32);
    }
    for (k, area) in ROI_QP_AREAS.iter().enumerate() {
        regs.set(*area, if k % 2 == 0 { QP_MIN } else { QP_MAX });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frame 0 (IDR) of the vendor capture: QP 26, 1025550 target bits (ctu_ebit 4558), dchs 0.
    #[test]
    fn rate_control_fields_match_the_vendor_idr() {
        let mut regs = Regs::template(true);
        set_rate_control(
            &mut regs,
            80,
            45,
            Plan {
                qp: 26,
                target_bits: 4558 * 3600 / 16 + 1,
            },
        );
        assert_eq!(regs.get(CTU_EBIT), 4558);
        assert_eq!(regs.word(RC_DTHD) as i32, -28484);
        assert_eq!(regs.word(RC_DTHD + 4) as i32, -7121);
        assert_eq!(regs.word(RC_DTHD + 8) as i32, 7121);
        assert_eq!(regs.word(RC_DTHD + 12) as i32, 28484);
        assert_eq!(regs.get(PIC_QP), 26);
        // the template already holds the vendor's values for everything else rate control sets
        assert_eq!(regs.base, Regs::template(true).base);
        assert_eq!(regs.rc, Regs::template(true).rc);
    }

    #[test]
    fn the_template_carries_the_vendor_codec_fields() {
        let regs = Regs::template(true);
        assert_eq!(regs.get(MAX_FNUM), h264::LOG2_MAX_FRAME_NUM_MINUS4);
        assert_eq!(regs.get(MPOC_LM4), h264::LOG2_MAX_POC_LSB_MINUS4);
        assert_eq!(regs.get(PIC_INIT_QP), h264::PIC_INIT_QP);
        assert_eq!(
            regs.get(CB_OFST),
            (h264::CHROMA_QP_INDEX_OFFSET as u32) & 31
        );
        assert_eq!(regs.get(SRC_CFMT), src_cfmt(Sampling::Yuv420));
        assert_eq!(regs.get(SRC_STRD0), 1280);
        assert_eq!((regs.get(PIC_WD8_M1), regs.get(PIC_HD8_M1)), (159, 89));
        assert_eq!(regs.get(RC_MIN_QP), QP_MIN);
        assert_eq!(regs.get(RC_MAX_QP), QP_MAX);
        for (k, area) in ROI_QP_AREAS.iter().enumerate() {
            assert_eq!(regs.get(*area), if k % 2 == 0 { QP_MIN } else { QP_MAX });
        }
    }

    #[test]
    fn the_static_scene_tune_reproduces_the_template() {
        for idr in [true, false] {
            let mut regs = Regs::template(idr);
            Tune::default().patch(&mut regs, idr);
            let want = Regs::template(idr);
            assert_eq!(regs.s3, want.s3, "idr {idr}");
            assert_eq!(regs.rdo, want.rdo, "idr {idr}");
            assert_eq!(regs.rc, want.rc, "idr {idr}");
            assert_eq!(regs.base, want.base, "idr {idr}");
        }
    }
}
