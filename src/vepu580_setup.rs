//! The VEPU580 H.264 register image for one frame, built the way MPP builds it: every `setup_vepu580_*` function of
//! `mpp/hal/rkenc/h264e/hal_h264e_vepu580.c` that runs for this stream, in `hal_h264e_vepu580_gen_regs` order, from
//! a zeroed register set. The scene tuner's patch (`vepu580_tune.rs`) is applied on top by the caller.
//!
//! This stream's configuration, as MPP has it for `mpph264enc rc-mode=cbr`: High profile, CABAC, 8x8 transform,
//! one reference frame, no slice split, no ROI / OSD / intra refresh / two-pass, NV12 or NV16 input.

use crate::h264;
use crate::jpeg::Sampling;
use crate::rc::{Plan, QP_MAX, QP_MIN};
use crate::service::AddrOffset;
use crate::vepu580::Regs;
use crate::vepu580_regs::{
    reg_base as b, reg_ctl as c, reg_rc_klut as k, reg_rdo as r, reg_s3 as s,
};
use crate::vepu580_tune::KLUT_WEIGHT;

/// Everything about one frame the registers depend on.
#[derive(Clone, Copy, Debug)]
pub struct FrameSetup {
    pub width: u32,
    pub height: u32,
    pub sampling: Sampling,
    pub idr: bool,
    pub frame_num: u32,
    pub idr_pic_id: u32,
    pub poc_lsb: u32,
    pub plan: Plan,
    pub dchs_txid: u32,
    pub dchs_rxid: u32,
    pub dchs_rxe: bool,
    pub level_idc: u32,
    pub src_fd: u32,
    /// reconstruction written by this frame, and the reference read: (FBC pixel buffer fd, thumb buffer fd)
    pub recn: (u32, u32),
    pub refr: (u32, u32),
    pub fbc_header_size: u32,
    pub out_fd: u32,
    pub out_size: u32,
}

/// `h264_aq_tthd_default` / `h264_I_aq_step_default` / `h264_P_aq_step_default` (hal_h264e_vepu580.c)
const AQ_THRD: [i32; 16] = [0, 0, 0, 0, 3, 3, 5, 5, 8, 8, 8, 15, 15, 20, 25, 35];
const AQ_STEP_I: [i32; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8];
const AQ_STEP_P: [i32; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 8];
/// `reg_rc_klut.aq_tthd[16]` and `aq_step[16]`: byte arrays (not in the generated field list)
const AQ_TTHD_ADDR: u32 = 0x1044;
const AQ_STEP_ADDR: u32 = 0x1054;
/// `reg_scl` starts with the flat scaling list table
const SCL_ADDR: u32 = 0x2200;
/// `h264_mode_bias` and MPP's default `hw->mode_bias[1] = 8`
const MODE_BIAS: [u32; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 64, 128];
const HW_MODE_BIAS_1: usize = 8;
/// `hw->qp_delta_row_i` / `hw->qp_delta_row` defaults (hal_h264e_vepu580_init)
const QP_DELTA_ROW_I: u32 = 2;
const QP_DELTA_ROW: u32 = 2;

const fn align(v: u32, a: u32) -> u32 {
    v.div_ceil(a) * a
}

/// The register image and the fd offsets the kernel adds to the address registers.
pub fn generate(p: &FrameSetup) -> (Box<Regs>, [AddrOffset; 6]) {
    let mut regs = Regs::zeroed();
    let rg = &mut *regs;
    setup_normal(rg);
    setup_prep(rg, p);
    setup_codec(rg, p);
    setup_dual_core(rg, p);
    setup_rdo_pred(rg, p);
    setup_rdo_cfg(rg);
    setup_rdo_bias_cfg(rg);
    for (i, w) in crate::vepu580_tables::FLAT_SCALING_LIST.iter().enumerate() {
        rg.set_word(SCL_ADDR + 4 * i as u32, *w);
    }
    setup_rc_base(rg, p);
    let offsets = setup_io_buf_and_recn(rg, p);
    setup_me(rg, p);
    setup_l2(rg, p.idr);
    (regs, offsets)
}

/// `setup_vepu580_normal`
fn setup_normal(rg: &mut Regs) {
    rg.set(c::COMMON_ENC_STRT_VEPU_CMD, 1);
    rg.set(c::COMMON_FUNC_EN_CKE, 1);
    rg.set(c::COMMON_FUNC_EN_RESETN_HW_EN, 1);
    rg.set(c::COMMON_FUNC_EN_ENC_DONE_TMVP_EN, 1);
    for field in [
        c::COMMON_INT_EN_ENC_DONE_EN,
        c::COMMON_INT_EN_LKT_NODE_DONE_EN,
        c::COMMON_INT_EN_SCLR_DONE_EN,
        c::COMMON_INT_EN_BSF_OFLW_EN,
        c::COMMON_INT_EN_BRSP_OTSD_EN,
        c::COMMON_INT_EN_WBUS_ERR_EN,
        c::COMMON_INT_EN_RBUS_ERR_EN,
        c::COMMON_INT_EN_WDG_EN,
    ] {
        rg.set(field, 1);
    }
    rg.set(c::COMMON_ENC_WDG_VS_LOAD_THD, 0x1fffff);
    rg.set(c::COMMON_DTRNS_MAP_BSW_BUS_EDIN, 7);
    rg.set(c::COMMON_DTRNS_MAP_AFBC_BSIZE, 1);
    rg.set(c::COMMON_DTRNS_CFG_DSPR_OTSD, 1);
}

/// `setup_vepu580_prep` for NV12 / NV16 (`vepu5xx_yuv_cfg`: no swaps, little endian, no CSC)
fn setup_prep(rg: &mut Regs, p: &FrameSetup) {
    rg.set(b::COMMON_ENC_RSL_PIC_WD8_M1, align(p.width, 16) / 8 - 1);
    rg.set(b::COMMON_SRC_FILL_PIC_WFILL, align(p.width, 16) - p.width);
    rg.set(b::COMMON_ENC_RSL_PIC_HD8_M1, align(p.height, 16) / 8 - 1);
    rg.set(b::COMMON_SRC_FILL_PIC_HFILL, align(p.height, 16) - p.height);
    rg.set(b::COMMON_SRC_FMT_SRC_CFMT, src_cfmt(p.sampling));
    rg.set(b::COMMON_SRC_FMT_OUT_FMT, 1);
    // current MPP sets src_range = 1 for any YUV input; the vendor build on AOS (2023 MPP) left it 0, and that is
    // the stream that was verified, so this follows the vendor
    rg.set(b::COMMON_SRC_FMT_SRC_RANGE, 0);
    // the decoder's planes: luma stride the 16-aligned width, semi-planar chroma the same stride
    let y_stride = luma_stride(p.width);
    rg.set(b::COMMON_SRC_STRD0_SRC_STRD0, y_stride);
    rg.set(b::COMMON_SRC_STRD1_SRC_STRD1, y_stride);
    rg.set(b::SLI_CFG_SLI_CRS_EN, 1);
}

/// `VEPU5xx_FMT_*`
pub fn src_cfmt(sampling: Sampling) -> u32 {
    match sampling {
        Sampling::Yuv420 => 6,
        Sampling::Yuv422 => 4,
    }
}

/// Bytes per luma row of the JPEG decoder's output (and the encoder's input).
pub fn luma_stride(width: u32) -> u32 {
    align(width, 16)
}

/// Rows of luma the decoder writes, so the chroma plane starts `luma_stride * luma_rows` in.
pub fn luma_rows(height: u32) -> u32 {
    align(height, 16)
}

/// `setup_vepu580_codec` for a one-reference stream: no reordering, no memory-management operations
fn setup_codec(rg: &mut Regs, p: &FrameSetup) {
    let nal_ref_idc = if p.idr { 3 } else { 2 };
    rg.set(b::COMMON_ENC_PIC_ENC_STND, 0);
    rg.set(b::COMMON_ENC_PIC_CUR_FRM_REF, 1);
    rg.set(b::COMMON_ENC_PIC_BS_SCP, 1);
    rg.set(b::SYNT_NAL_NAL_REF_IDC, nal_ref_idc);
    rg.set(b::SYNT_NAL_NAL_UNIT_TYPE, if p.idr { 5 } else { 1 });
    rg.set(b::SYNT_SPS_MAX_FNUM, h264::LOG2_MAX_FRAME_NUM_MINUS4);
    rg.set(b::SYNT_SPS_DRCT_8X8, 1);
    rg.set(b::SYNT_SPS_MPOC_LM4, h264::LOG2_MAX_POC_LSB_MINUS4);
    rg.set(b::SYNT_PPS_ETPY_MODE, 1);
    rg.set(b::SYNT_PPS_TRNS_8X8, 1);
    rg.set(b::SYNT_PPS_PIC_INIT_QP, h264::PIC_INIT_QP);
    rg.set(b::SYNT_PPS_CB_OFST, h264::CHROMA_QP_INDEX_OFFSET as u32);
    rg.set(b::SYNT_PPS_CR_OFST, h264::CHROMA_QP_INDEX_OFFSET as u32);
    rg.set(b::SYNT_PPS_DBF_CP_FLG, 1);
    rg.set(b::SYNT_SLI0_SLI_TYPE, if p.idr { 2 } else { 0 });
    rg.set(b::SYNT_SLI0_FRM_NUM, p.frame_num);
    rg.set(
        b::SYNT_SLI1_IDR_PID,
        if p.idr { p.idr_pic_id } else { 0xffff },
    );
    rg.set(b::SYNT_SLI1_POC_LSB, p.poc_lsb);
}

/// `setup_vepu580_dual_core` (MPP's two-task mode, which the vendor build ran)
fn setup_dual_core(rg: &mut Regs, p: &FrameSetup) {
    rg.set(b::COMMON_DUAL_CORE_DCHS_TXID, p.dchs_txid);
    rg.set(b::COMMON_DUAL_CORE_DCHS_RXID, p.dchs_rxid);
    rg.set(
        b::COMMON_DUAL_CORE_DCHS_TXE,
        rg.get(b::COMMON_ENC_PIC_CUR_FRM_REF),
    );
    rg.set(b::COMMON_DUAL_CORE_DCHS_RXE, p.dchs_rxe as u32);
    rg.set(b::COMMON_DUAL_CORE_DCHS_OFST, 9);
}

/// `setup_vepu580_rdo_pred` (High profile, CABAC, no scaling matrix)
fn setup_rdo_pred(rg: &mut Regs, p: &FrameSetup) {
    let (ofst, from) = if p.idr { (0, 0) } else { (3, 4) };
    rg.set(k::KLUT_OFST_CHRM_KLUT_OFST, ofst);
    for (i, w) in KLUT_WEIGHT[from..from + 24].iter().enumerate() {
        rg.set_word(k::KLUT_WGT0_CHRM_KLUT_WGT0.addr + 4 * i as u32, *w);
    }
    rg.set(b::IPRD_CSTS_VTHD_Y, 9);
    rg.set(b::IPRD_CSTS_VTHD_C, 63);
    rg.set(b::RDO_CFG_RECT_SIZE, 0);
    rg.set(b::RDO_CFG_INTER_4X4, 1);
    rg.set(b::RDO_CFG_VLC_LMT, 0);
    rg.set(b::RDO_CFG_CHRM_SPCL, 1);
    rg.set(b::RDO_CFG_RDO_MASK, 0);
    rg.set(b::RDO_CFG_CCWA_E, 1);
    rg.set(b::RDO_CFG_SCL_LST_SEL, 0);
    rg.set(b::RDO_CFG_ATR_E, 1);
    rg.set(b::RDO_CFG_ATF_INTRA_E, 1);
}

/// `setup_vepu580_rdo_cfg`
fn setup_rdo_cfg(rg: &mut Regs) {
    let values: &[(crate::vepu580::Field, u32)] = &[
        (r::RDO_SQI_CFG_ATF_PSKIP_EN, 1),
        (r::RDO_INTRA_CIME_THD0_ATF_RDO_INTRA_CIME_THD0, 28),
        (r::RDO_INTRA_CIME_THD0_ATF_RDO_INTRA_CIME_THD1, 44),
        (r::RDO_INTRA_CIME_THD1_ATF_RDO_INTRA_CIME_THD2, 72),
        (r::RDO_INTRA_VAR_THD0_ATF_RDO_INTRA_VAR_THD00, 25),
        (r::RDO_INTRA_VAR_THD0_ATF_RDO_INTRA_VAR_THD01, 64),
        (r::RDO_INTRA_VAR_THD1_ATF_RDO_INTRA_VAR_THD10, 25),
        (r::RDO_INTRA_VAR_THD1_ATF_RDO_INTRA_VAR_THD11, 64),
        (r::RDO_INTRA_VAR_THD2_ATF_RDO_INTRA_VAR_THD20, 70),
        (r::RDO_INTRA_VAR_THD2_ATF_RDO_INTRA_VAR_THD21, 100),
        (r::RDO_INTRA_VAR_THD3_ATF_RDO_INTRA_VAR_THD30, 70),
        (r::RDO_INTRA_VAR_THD3_ATF_RDO_INTRA_VAR_THD31, 100),
        (r::RDO_INTRA_ATF_WGT0_ATF_RDO_INTRA_WGT00, 24),
        (r::RDO_INTRA_ATF_WGT0_ATF_RDO_INTRA_WGT01, 22),
        (r::RDO_INTRA_ATF_WGT0_ATF_RDO_INTRA_WGT02, 21),
        (r::RDO_INTRA_ATF_WGT1_ATF_RDO_INTRA_WGT10, 22),
        (r::RDO_INTRA_ATF_WGT1_ATF_RDO_INTRA_WGT11, 21),
        (r::RDO_INTRA_ATF_WGT1_ATF_RDO_INTRA_WGT12, 20),
        (r::RDO_INTRA_ATF_WGT2_ATF_RDO_INTRA_WGT20, 20),
        (r::RDO_INTRA_ATF_WGT2_ATF_RDO_INTRA_WGT21, 19),
        (r::RDO_INTRA_ATF_WGT2_ATF_RDO_INTRA_WGT22, 18),
        (r::RDO_INTRA_ATF_WGT3_ATF_RDO_INTRA_WGT30, 16),
        (r::RDO_INTRA_ATF_WGT3_ATF_RDO_INTRA_WGT31, 16),
        (r::RDO_INTRA_ATF_WGT3_ATF_RDO_INTRA_WGT32, 16),
        (r::RDO_SKIP_CIME_THD0_ATF_RDO_SKIP_CIME_THD0, 10),
        (r::RDO_SKIP_CIME_THD0_ATF_RDO_SKIP_CIME_THD1, 8),
        (r::RDO_SKIP_CIME_THD1_ATF_RDO_SKIP_CIME_THD2, 15),
        (r::RDO_SKIP_CIME_THD1_ATF_RDO_SKIP_CIME_THD3, 25),
        (r::RDO_SKIP_VAR_THD0_ATF_RDO_SKIP_VAR_THD10, 25),
        (r::RDO_SKIP_VAR_THD0_ATF_RDO_SKIP_VAR_THD11, 40),
        (r::RDO_SKIP_VAR_THD1_ATF_RDO_SKIP_VAR_THD20, 25),
        (r::RDO_SKIP_VAR_THD1_ATF_RDO_SKIP_VAR_THD21, 40),
        (r::RDO_SKIP_VAR_THD2_ATF_RDO_SKIP_VAR_THD30, 70),
        (r::RDO_SKIP_VAR_THD2_ATF_RDO_SKIP_VAR_THD31, 100),
        (r::RDO_SKIP_VAR_THD3_ATF_RDO_SKIP_VAR_THD40, 70),
        (r::RDO_SKIP_VAR_THD3_ATF_RDO_SKIP_VAR_THD41, 100),
        (r::RDO_SKIP_ATF_WGT0_ATF_RDO_SKIP_ATF_WGT00, 20),
        (r::RDO_SKIP_ATF_WGT0_ATF_RDO_SKIP_ATF_WGT10, 16),
        (r::RDO_SKIP_ATF_WGT0_ATF_RDO_SKIP_ATF_WGT11, 16),
        (r::RDO_SKIP_ATF_WGT0_ATF_RDO_SKIP_ATF_WGT12, 16),
        (r::RDO_SKIP_ATF_WGT1_ATF_RDO_SKIP_ATF_WGT20, 16),
        (r::RDO_SKIP_ATF_WGT1_ATF_RDO_SKIP_ATF_WGT21, 16),
        (r::RDO_SKIP_ATF_WGT1_ATF_RDO_SKIP_ATF_WGT22, 16),
        (r::RDO_SKIP_ATF_WGT2_ATF_RDO_SKIP_ATF_WGT30, 16),
        (r::RDO_SKIP_ATF_WGT2_ATF_RDO_SKIP_ATF_WGT31, 16),
        (r::RDO_SKIP_ATF_WGT2_ATF_RDO_SKIP_ATF_WGT32, 16),
        (r::RDO_SKIP_ATF_WGT3_ATF_RDO_SKIP_ATF_WGT40, 16),
        (r::RDO_SKIP_ATF_WGT3_ATF_RDO_SKIP_ATF_WGT41, 16),
        (r::RDO_SKIP_ATF_WGT3_ATF_RDO_SKIP_ATF_WGT42, 16),
    ];
    for (field, v) in values {
        rg.set(*field, *v);
    }
}

/// `setup_vepu580_rdo_bias_cfg` with MPP's default mode bias (skip bias off)
fn setup_rdo_bias_cfg(rg: &mut Regs) {
    let bias = MODE_BIAS[HW_MODE_BIAS_1];
    let floors = [
        (r::RDO_INTRA_ATF_WGT0_ATF_RDO_INTRA_WGT00, 24),
        (r::RDO_INTRA_ATF_WGT0_ATF_RDO_INTRA_WGT01, 22),
        (r::RDO_INTRA_ATF_WGT0_ATF_RDO_INTRA_WGT02, 21),
        (r::RDO_INTRA_ATF_WGT1_ATF_RDO_INTRA_WGT10, 22),
        (r::RDO_INTRA_ATF_WGT1_ATF_RDO_INTRA_WGT11, 21),
        (r::RDO_INTRA_ATF_WGT1_ATF_RDO_INTRA_WGT12, 20),
        (r::RDO_INTRA_ATF_WGT2_ATF_RDO_INTRA_WGT20, 20),
        (r::RDO_INTRA_ATF_WGT2_ATF_RDO_INTRA_WGT21, 19),
        (r::RDO_INTRA_ATF_WGT2_ATF_RDO_INTRA_WGT22, 18),
        (r::RDO_INTRA_ATF_WGT3_ATF_RDO_INTRA_WGT30, 0),
        (r::RDO_INTRA_ATF_WGT3_ATF_RDO_INTRA_WGT31, 0),
        (r::RDO_INTRA_ATF_WGT3_ATF_RDO_INTRA_WGT32, 0),
    ];
    for (field, floor) in floors {
        rg.set(field, bias.max(floor));
    }
}

/// `setup_vepu580_rc_base` (CBR): starting QP, row-level rate control target and thresholds, ROI QP limits
pub fn setup_rc_base(rg: &mut Regs, p: &FrameSetup) {
    let mb_w = align(p.width, 16) / 16;
    let mb_h = align(p.height, 16) / 16;
    let (qp_min, qp_max) = (QP_MIN, QP_MAX);
    let areas = [
        (k::ROI_QTHD0_QPMIN_AREA0, qp_min),
        (k::ROI_QTHD0_QPMAX_AREA0, qp_max),
        (k::ROI_QTHD0_QPMIN_AREA1, qp_min),
        (k::ROI_QTHD0_QPMAX_AREA1, qp_max),
        (k::ROI_QTHD0_QPMIN_AREA2, qp_min),
        (k::ROI_QTHD1_QPMAX_AREA2, qp_max),
        (k::ROI_QTHD1_QPMIN_AREA3, qp_min),
        (k::ROI_QTHD1_QPMAX_AREA3, qp_max),
        (k::ROI_QTHD1_QPMIN_AREA4, qp_min),
        (k::ROI_QTHD1_QPMAX_AREA4, qp_max),
        (k::ROI_QTHD2_QPMIN_AREA5, qp_min),
        (k::ROI_QTHD2_QPMAX_AREA5, qp_max),
        (k::ROI_QTHD2_QPMIN_AREA6, qp_min),
        (k::ROI_QTHD2_QPMAX_AREA6, qp_max),
        (k::ROI_QTHD2_QPMIN_AREA7, qp_min),
        (k::ROI_QTHD3_QPMAX_AREA7, qp_max),
        (k::ROI_QTHD3_QPMAP_MODE, 1),
    ];
    for (field, v) in areas {
        rg.set(field, v);
    }
    let mut mb_target_bits_mul_16 = ((p.plan.target_bits as u64) << 4) / (mb_w * mb_h) as u64;
    if mb_target_bits_mul_16 >= 0x100000 {
        mb_target_bits_mul_16 = 0x50000;
    }
    let mb_target_bits = ((mb_target_bits_mul_16 * mb_w as u64) >> 4) as i32;
    let negative_bits_thd = -(5 * mb_target_bits / 16);
    let positive_bits_thd = 5 * mb_target_bits / 16;
    rg.set(b::COMMON_ENC_PIC_PIC_QP, p.plan.qp);
    rg.set(b::COMMON_RC_CFG_RC_EN, 1);
    rg.set(b::COMMON_RC_CFG_AQ_EN, 1);
    rg.set(b::COMMON_RC_CFG_AQ_MODE, 0);
    rg.set(b::COMMON_RC_CFG_RC_CTU_NUM, mb_w);
    rg.set(
        b::COMMON_RC_QP_RC_QP_RANGE,
        if p.idr { QP_DELTA_ROW_I } else { QP_DELTA_ROW },
    );
    rg.set(b::COMMON_RC_QP_RC_MAX_QP, qp_max);
    rg.set(b::COMMON_RC_QP_RC_MIN_QP, qp_min);
    rg.set(b::COMMON_RC_TGT_CTU_EBIT, mb_target_bits_mul_16 as u32);
    let adj: [(crate::vepu580::Field, i32); 9] = [
        (k::RC_ADJ0_QP_ADJ0, -2),
        (k::RC_ADJ0_QP_ADJ1, -1),
        (k::RC_ADJ0_QP_ADJ2, 0),
        (k::RC_ADJ0_QP_ADJ3, 1),
        (k::RC_ADJ0_QP_ADJ4, 2),
        (k::RC_ADJ1_QP_ADJ5, 0),
        (k::RC_ADJ1_QP_ADJ6, 0),
        (k::RC_ADJ1_QP_ADJ7, 0),
        (k::RC_ADJ1_QP_ADJ8, 1),
    ];
    for (field, v) in adj {
        rg.set(field, v as u32);
    }
    let dthd = [
        4 * negative_bits_thd,
        negative_bits_thd,
        positive_bits_thd,
        4 * positive_bits_thd,
    ];
    for i in 0..9 {
        let v = dthd.get(i).copied().unwrap_or(0x7fff_ffff);
        rg.set_word(k::RC_DTHD_0_8.addr + 4 * i as u32, v as u32);
    }
    for (field, v) in [
        (k::MD_SAD_THD_MD_SAD_THD0, 4),
        (k::MD_SAD_THD_MD_SAD_THD1, 9),
        (k::MD_SAD_THD_MD_SAD_THD2, 15),
        (k::MADI_THD_MADI_THD0, 4),
        (k::MADI_THD_MADI_THD1, 9),
        (k::MADI_THD_MADI_THD2, 15),
    ] {
        rg.set(field, v);
    }
}

/// `setup_vepu580_io_buf` + `setup_vepu580_recn_refr`: the buffer fds, and the offsets into them (register index,
/// bytes) as MPP passes them through `mpp_dev_multi_offset_update`
fn setup_io_buf_and_recn(rg: &mut Regs, p: &FrameSetup) -> [AddrOffset; 6] {
    for field in [b::COMMON_ADR_SRC0, b::COMMON_ADR_SRC1, b::COMMON_ADR_SRC2] {
        rg.set(field, p.src_fd);
    }
    for field in [
        b::COMMON_BSBB_ADDR,
        b::COMMON_BSBR_ADDR,
        b::COMMON_ADR_BSBS,
        b::COMMON_BSBT_ADDR,
    ] {
        rg.set(field, p.out_fd);
    }
    rg.set(b::COMMON_RFPW_H_ADDR, p.recn.0);
    rg.set(b::COMMON_RFPW_B_ADDR, p.recn.0);
    rg.set(b::COMMON_DSPW_ADDR, p.recn.1);
    rg.set(b::COMMON_RFPR_H_ADDR, p.refr.0);
    rg.set(b::COMMON_RFPR_B_ADDR, p.refr.0);
    rg.set(b::COMMON_DSPR_ADDR, p.refr.1);
    // semi-planar: both chroma addresses at the end of the (16-row aligned) luma plane
    let chroma = luma_stride(p.width) * luma_rows(p.height);
    let index = |field: crate::vepu580::Field| field.addr / 4;
    [
        AddrOffset {
            index: index(b::COMMON_ADR_SRC1),
            offset: chroma,
        },
        AddrOffset {
            index: index(b::COMMON_ADR_SRC2),
            offset: chroma,
        },
        AddrOffset {
            index: index(b::COMMON_BSBT_ADDR),
            offset: p.out_size,
        },
        // our parameter sets go out separately, so the slice starts at the buffer's first byte
        AddrOffset {
            index: index(b::COMMON_ADR_BSBS),
            offset: 0,
        },
        AddrOffset {
            index: index(b::COMMON_RFPW_B_ADDR),
            offset: p.fbc_header_size,
        },
        AddrOffset {
            index: index(b::COMMON_RFPR_B_ADDR),
            offset: p.fbc_header_size,
        },
    ]
}

/// `setup_vepu580_me` and `calc_cime_parameter` (no global motion vector)
fn setup_me(rg: &mut Regs, p: &FrameSetup) {
    let mbs_w = align(p.width, 16) / 16;
    let mbs_h = align(p.height, 16) / 16;
    let (cime_w, cime_h) = (176i32, 112i32);
    let mut cime_blk_w_max = 44i32;
    let mut cime_blk_h_max = match p.level_idc {
        10 | 9 => 12,
        _ => 28,
    };
    if cime_w < cime_blk_w_max * 4 {
        cime_blk_w_max = cime_w / 4;
    }
    if cime_h < cime_blk_h_max * 4 {
        cime_blk_h_max = cime_h / 4;
    }
    if cime_blk_w_max / 4 * 2 > (mbs_w as i32 * 2 + 1) / 2 {
        cime_blk_w_max = (mbs_w as i32 * 2 + 1) / 2 / 2 * 4;
    }
    let h64 = align(mbs_h * 16, 64) as i32;
    if cime_blk_h_max / 4 > h64 / 128 * 4 {
        cime_blk_h_max = h64 / 128 * 16;
    }
    rg.set(b::COMMON_ME_RNGE_CME_SRCH_H, (cime_blk_w_max / 4) as u32);
    rg.set(b::COMMON_ME_RNGE_CME_SRCH_V, (cime_blk_h_max / 4) as u32);
    rg.set(b::COMMON_ME_RNGE_RME_SRCH_H, 7);
    rg.set(b::COMMON_ME_RNGE_RME_SRCH_V, 5);
    rg.set(b::COMMON_ME_RNGE_DLT_FRM_NUM, 0);
    let mdst = if p.idr { 0 } else { 5 };
    rg.set(b::ME_CFG_PMV_MDST_H, mdst);
    rg.set(b::ME_CFG_PMV_MDST_V, mdst);
    let mv_limit = if p.level_idc > 20 {
        2
    } else if p.level_idc >= 11 {
        1
    } else {
        0
    };
    rg.set(b::ME_CFG_MV_LIMIT, mv_limit);
    rg.set(b::ME_CFG_PMV_NUM, 2);

    let (linebuf_w, rama_h, rama_max) = cime_parameters(
        rg.get(b::COMMON_ME_RNGE_CME_SRCH_H) as i32 * 4,
        rg.get(b::COMMON_ME_RNGE_CME_SRCH_V) as i32 * 4,
        (align(mbs_w * 16, 64) / 64) as i32,
    );
    rg.set(b::ME_CACH_CME_LINEBUF_W, linebuf_w);
    rg.set(b::ME_CACH_CME_RAMA_H, rama_h);
    rg.set(b::ME_CACH_CME_RAMA_MAX, rama_max);
}

/// `calc_cime_parameter` with gmv (0, 0): (cme_linebuf_w, cme_rama_h, cme_rama_max)
fn cime_parameters(srch_w: i32, srch_h: i32, pic_wd64: i32) -> (u32, u32, u32) {
    let (x_gmv, y_gmv) = (0i32, 0i32);
    let frm_sta = if x_gmv - srch_w < 0 {
        (x_gmv - srch_w - 15) / 16
    } else {
        (x_gmv - srch_w) / 16
    }
    .clamp(0, pic_wd64 - 1);
    let frm_end = if x_gmv + srch_w < 0 {
        pic_wd64 - 1 + (x_gmv + srch_w) / 16
    } else {
        pic_wd64 - 1 + (x_gmv + srch_w + 15) / 16
    }
    .clamp(0, pic_wd64 - 1);
    let pic_w = (frm_end - frm_sta + 1) * 64;
    let linebuf_w = (if pic_w != 0 { pic_w } else { 64 }) / 64;

    let cime_cacha_max = 2464i32;
    let ctu_4_h = 1i32;
    let mut cime_cacha_h = ctu_4_h;
    // C's % truncates toward zero, as Rust's does
    let mut cur_srch_16_w = if x_gmv % 16 - srch_w % 16 < 0 {
        (16 + (x_gmv % 16 - srch_w % 16) % 16 + srch_w * 2 + 15) / 16 + 1
    } else {
        ((x_gmv % 16 - srch_w % 16) % 16 + srch_w * 2 + 15) / 16 + 1
    };
    let cur_srch_4_h = if y_gmv % 4 - srch_h % 4 < 0 {
        (4 + (y_gmv % 4 - srch_h % 4) % 4 + srch_h * 2 + 3) / 4 + ctu_4_h
    } else {
        ((y_gmv % 4 - srch_h % 4) % 4 + srch_h * 2 + 3) / 4 + ctu_4_h
    };
    let cur_srch_max = cur_srch_4_h;
    if linebuf_w < cur_srch_16_w {
        cur_srch_16_w = linebuf_w;
    }
    let mut ramb_h = cur_srch_4_h;
    while cime_cacha_h < cur_srch_max
        && cime_cacha_max > (cime_cacha_h - ctu_4_h) * linebuf_w * 4 + ramb_h * 4 * cur_srch_16_w
    {
        cime_cacha_h += ctu_4_h;
        ramb_h = if ramb_h > 2 * ctu_4_h {
            ramb_h - ctu_4_h
        } else {
            ctu_4_h
        };
    }
    if cur_srch_4_h == ctu_4_h {
        cime_cacha_h += ctu_4_h;
        ramb_h = 0;
    }
    if cime_cacha_max < (cime_cacha_h - ctu_4_h) * linebuf_w * 4 + ramb_h * 4 * cur_srch_16_w {
        cime_cacha_h -= ctu_4_h;
    }
    let ram_col_h = (cime_cacha_h - ctu_4_h) / ctu_4_h;
    let rama_max = ram_col_h * linebuf_w + cur_srch_16_w;
    (linebuf_w as u32, cime_cacha_h as u32, rama_max as u32)
}

/// `setup_vepu580_l2` with MPP's default hardware config
fn setup_l2(rg: &mut Regs, idr: bool) {
    let lambda = &crate::vepu580_tables::LAMBDA_DEFAULT_58[6..58];
    for (i, w) in lambda.iter().enumerate() {
        rg.set_word(s::COMMON_RDO_WGTA_QP_GRPA_0_51.addr + 4 * i as u32, *w);
    }
    for i in 0..s::COMMON_IPRD_WGT_QP_HEVC_0_51.len {
        rg.set_word(s::COMMON_IPRD_WGT_QP_HEVC_0_51.addr + 4 * i, 0);
    }
    rg.set(k::MADI_CFG_MADI_MODE, 0);
    rg.set(k::MADI_CFG_MADI_THD, 25);
    let values: &[(crate::vepu580::Field, u32)] = &[
        (s::LVL32_INTRA_CST_THD0_LVL4_INTRA_CST_THD0, 1),
        (s::LVL32_INTRA_CST_THD0_LVL4_INTRA_CST_THD1, 4),
        (s::LVL32_INTRA_CST_THD1_LVL4_INTRA_CST_THD2, 9),
        (s::LVL32_INTRA_CST_THD1_LVL4_INTRA_CST_THD3, 36),
        (s::LVL16_INTRA_CST_THD0_LVL8_INTRA_CHRM_CST_THD0, 1),
        (s::LVL16_INTRA_CST_THD0_LVL8_INTRA_CHRM_CST_THD1, 4),
        (s::LVL16_INTRA_CST_THD1_LVL8_INTRA_CHRM_CST_THD2, 9),
        (s::LVL16_INTRA_CST_THD1_LVL8_INTRA_CHRM_CST_THD3, 36),
        (s::LVL8_INTRA_CST_THD0_LVL8_INTRA_CST_THD0, 1),
        (s::LVL8_INTRA_CST_THD0_LVL8_INTRA_CST_THD1, 4),
        (s::LVL8_INTRA_CST_THD1_LVL8_INTRA_CST_THD2, 9),
        (s::LVL8_INTRA_CST_THD1_LVL8_INTRA_CST_THD3, 36),
        (s::LVL16_INTRA_UL_CST_THD_LVL16_INTRA_UL_CST_THLD, 0),
        (s::LVL32_INTRA_CST_WGT0_LVL8_INTRA_CST_WGT0, 48),
        (s::LVL32_INTRA_CST_WGT0_LVL8_INTRA_CST_WGT1, 60),
        (s::LVL32_INTRA_CST_WGT0_LVL8_INTRA_CST_WGT2, 40),
        (s::LVL32_INTRA_CST_WGT0_LVL8_INTRA_CST_WGT3, 48),
        (s::LVL32_INTRA_CST_WGT1_LVL4_INTRA_CST_WGT0, 48),
        (s::LVL32_INTRA_CST_WGT1_LVL4_INTRA_CST_WGT1, 60),
        (s::LVL32_INTRA_CST_WGT1_LVL4_INTRA_CST_WGT2, 40),
        (s::LVL32_INTRA_CST_WGT1_LVL4_INTRA_CST_WGT3, 48),
        (s::LVL16_INTRA_CST_WGT0_LVL16_INTRA_CST_WGT0, 48),
        (s::LVL16_INTRA_CST_WGT0_LVL16_INTRA_CST_WGT1, 60),
        (s::LVL16_INTRA_CST_WGT0_LVL16_INTRA_CST_WGT2, 40),
        (s::LVL16_INTRA_CST_WGT0_LVL16_INTRA_CST_WGT3, 48),
        (s::LVL16_INTRA_CST_WGT1_LVL8_INTRA_CHRM_CST_WGT0, 36),
        (s::LVL16_INTRA_CST_WGT1_LVL8_INTRA_CHRM_CST_WGT1, 42),
        (s::LVL16_INTRA_CST_WGT1_LVL8_INTRA_CHRM_CST_WGT2, 28),
        (s::LVL16_INTRA_CST_WGT1_LVL8_INTRA_CHRM_CST_WGT3, 32),
        (s::RDO_QUANT_QUANT_F_BIAS_P, 171),
    ];
    for (field, v) in values {
        rg.set(*field, *v);
    }
    let (bias_i, thd0, thd1, thd2) = if idr {
        (683, 1, 4, 36)
    } else {
        (583, 4, 16, 81)
    };
    rg.set(s::RDO_QUANT_QUANT_F_BIAS_I, bias_i);
    rg.set(s::ATR_THD0_ATR_THD0, thd0);
    rg.set(s::ATR_THD0_ATR_THD1, thd1);
    rg.set(s::ATR_THD1_ATR_THD2, thd2);
    rg.set(s::ATR_THD1_ATR_THDQP, 45);
    let atr: [u32; 9] = if idr {
        [16, 16, 16, 22, 21, 20, 20, 18, 16]
    } else {
        [25, 20, 16, 25, 20, 18, 25, 20, 16]
    };
    let atr_fields = [
        s::LVL16_ATR_WGT_LVL16_ATR_WGT0,
        s::LVL16_ATR_WGT_LVL16_ATR_WGT1,
        s::LVL16_ATR_WGT_LVL16_ATR_WGT2,
        s::LVL8_ATR_WGT_LVL8_ATR_WGT0,
        s::LVL8_ATR_WGT_LVL8_ATR_WGT1,
        s::LVL8_ATR_WGT_LVL8_ATR_WGT2,
        s::LVL4_ATR_WGT_LVL4_ATR_WGT0,
        s::LVL4_ATR_WGT_LVL4_ATR_WGT1,
        s::LVL4_ATR_WGT_LVL4_ATR_WGT2,
    ];
    for (field, v) in atr_fields.into_iter().zip(atr) {
        rg.set(field, v);
    }
    let me: &[(crate::vepu580::Field, u32)] = &[
        (s::COMMON_CIME_SQI_CFG_CIME_SAD_MOD_SEL, 0),
        (s::COMMON_CIME_SQI_CFG_CIME_SAD_USE_BIG_BLOCK, 1),
        (s::COMMON_CIME_SQI_CFG_CIME_PMV_SET_ZERO, 1),
        (s::COMMON_CIME_SQI_CFG_CIME_PMV_NUM, 3),
        (s::COMMON_CIME_SQI_THD_CIME_MVD_TH0, 32),
        (s::COMMON_CIME_SQI_THD_CIME_MVD_TH1, 80),
        (s::COMMON_CIME_SQI_THD_CIME_MVD_TH2, 128),
        (s::COMMON_CIME_SQI_MULTI0_CIME_MULTI0, 4),
        (s::COMMON_CIME_SQI_MULTI0_CIME_MULTI1, 8),
        (s::COMMON_CIME_SQI_MULTI1_CIME_MULTI2, 24),
        (s::COMMON_CIME_SQI_MULTI1_CIME_MULTI3, 24),
        (s::COMMON_RIME_SQI_THD_CIME_SAD_TH0, 50),
        (s::COMMON_RIME_SQI_THD_RIME_MVD_TH0, 3),
        (s::COMMON_RIME_SQI_THD_RIME_MVD_TH1, 8),
        (s::COMMON_RIME_SQI_MULTI_RIME_MULTI0, 4),
        (s::COMMON_RIME_SQI_MULTI_RIME_MULTI1, 32),
        (s::COMMON_RIME_SQI_MULTI_RIME_MULTI2, 128),
        (s::COMMON_FME_SQI_THD0_CIME_SAD_PU16_TH, 2),
        (s::COMMON_FME_SQI_THD1_MOVE_LAMBDA, 1),
    ];
    for (field, v) in me {
        rg.set(*field, *v);
    }
    let steps = if idr { &AQ_STEP_I } else { &AQ_STEP_P };
    for i in 0..16u32 {
        rg.set_byte(AQ_TTHD_ADDR + i, AQ_THRD[i as usize] as u8);
        rg.set_byte(AQ_STEP_ADDR + i, (steps[i as usize] & 0x3f) as u8);
    }
}
