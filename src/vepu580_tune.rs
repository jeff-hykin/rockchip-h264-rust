//! The VEPU580 H.264 scene tuner: classifies motion from each frame's statistics and switches the motion-search and
//! RDO weights between a static-scene and a moving-scene set.
//!
//! A line-for-line port of MPP `mpp/hal/rkenc/h264e/hal_h264e_vepu580_tune.c` (`vepu580_h264e_tune_reg_patch`,
//! `vepu580_h264e_tune_stat_update`) and its tables in `mpp/hal/rkenc/common/vepu580_tune.c`, with the scene mode
//! MPP uses for a non-IPC stream (`ap_motion_flag = 1`), which is what the vendor build ran with on AOS.

use crate::vepu580::{f, Field, Regs, Status};

const MD_WIN_LEN: usize = 5;
const MD_SHOW_LEN: usize = 4;
const FRAME_MOTION_ANALYSIS_NUM: usize = 5;

const MOTION_LEVEL_STILL: i32 = 0;
const MOTION_LEVEL_MOTION: i32 = 1;
const MOTION_LEVEL_BIG_MOTION: i32 = 2;
const MOTION_LEVEL_UNKNOW_SCENE: i32 = 3;

const MB_AVG_MADP_THD: [i32; 6] = [192, 128, 64, 192, 128, 64];
#[rustfmt::skip]
const CTU_MADP_CNT_THD: [[i32; 8]; 6] = [
    [50, 100, 130, 50, 100, 550, 500, 550],
    [100, 150, 200, 80, 120, 500, 450, 550],
    [150, 200, 250, 100, 150, 450, 400, 450],
    [50, 100, 130, 50, 100, 550, 500, 550],
    [100, 150, 200, 80, 120, 500, 450, 550],
    [150, 200, 250, 100, 150, 450, 400, 450],
];
const MADP_NUM_MAP: [[i32; 4]; 5] = [
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [1, 1, 1, 1],
];

#[rustfmt::skip]
const ATR_WGT: [[u32; 9]; 4] = [
    [22, 19, 16, 22, 19, 18, 22, 19, 16],
    [19, 19, 19, 19, 19, 19, 19, 19, 19],
    [22, 19, 16, 22, 19, 18, 22, 19, 16],
    [20, 20, 20, 20, 20, 20, 20, 20, 20],
];
const SKIP_ATF_WGT: [[u32; 13]; 4] = [[16; 13]; 4];
#[rustfmt::skip]
const INTRA_ATF_WGT: [[u32; 12]; 4] = [
    [24, 22, 21, 22, 21, 20, 20, 19, 18, 16, 16, 16],
    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    [22, 21, 20, 21, 20, 19, 20, 19, 18, 16, 16, 16],
    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
];
const CIME_MULTI: [[u32; 4]; 4] = [[4, 8, 24, 24], [4, 7, 20, 20], [4, 7, 20, 20], [4, 4, 4, 4]];
const RIME_MULTI: [[u32; 3]; 4] = [[4, 32, 128], [4, 16, 64], [4, 16, 64], [4, 4, 4]];

/// `h264e_klut_weight` (hal_h264e_vepu580.c): a P frame takes 24 words from index 0 (moving) or 4 (static).
#[rustfmt::skip]
pub const KLUT_WEIGHT: [u32; 30] = [
    0x0a000010, 0x00064000, 0x14000020, 0x000c8000, 0x28000040, 0x00194000, 0x50800080, 0x0032c000,
    0xa1000100, 0x00658000, 0x42800200, 0x00cb0001, 0x85000400, 0x01964002, 0x0a000800, 0x032c8005,
    0x14001000, 0x0659400a, 0x28802000, 0x0cb2c014, 0x51004000, 0x1965c028, 0xa2808000, 0x32cbc050,
    0x4500ffff, 0x659780a1, 0x8a81fffe, 0xcc000142, 0xff83ffff, 0x000001ff,
];
const KLUT_WGT0: u32 = 0x1084;

// reg_base
const VTHD_Y: Field = f(0x3a4, 0, 12);
const VTHD_C: Field = f(0x3a4, 16, 12);
const ATF_INTRA_E: Field = f(0x3a0, 19, 1);
// reg_rc_klut
const MD_SAD_THD: [Field; 3] = [f(0x1070, 0, 8), f(0x1070, 8, 8), f(0x1070, 16, 8)];
const MADI_THD: [Field; 3] = [f(0x1074, 0, 8), f(0x1074, 8, 8), f(0x1074, 16, 8)];
const CHRM_KLUT_OFST: Field = f(0x1080, 0, 3);
// reg_s3
const LVL16_INTRA_UL_CST_THLD: Field = f(0x1718, 0, 12);
const QUANT_F_BIAS_P: Field = f(0x1730, 10, 10);
const ATR_THDQP: Field = f(0x1744, 16, 6);
#[rustfmt::skip]
const ATR_WGT_FIELDS: [Field; 9] = [
    f(0x1750, 0, 8), f(0x1750, 8, 8), f(0x1750, 16, 8),
    f(0x1754, 0, 8), f(0x1754, 8, 8), f(0x1754, 16, 8),
    f(0x1758, 0, 8), f(0x1758, 8, 8), f(0x1758, 16, 8),
];
const CIME_PMV_SET_ZERO: Field = f(0x1760, 2, 1);
const CIME_MULTI_FIELDS: [Field; 4] = [
    f(0x1768, 0, 10),
    f(0x1768, 16, 10),
    f(0x176c, 0, 10),
    f(0x176c, 16, 10),
];
const CIME_SAD_TH0: Field = f(0x1770, 0, 12);
const RIME_MULTI_FIELDS: [Field; 3] = [f(0x1774, 0, 10), f(0x1774, 10, 10), f(0x1774, 20, 10)];
const CIME_SAD_PU16_TH: Field = f(0x1778, 0, 12);
const MOVE_LAMBDA: Field = f(0x177c, 16, 4);
// reg_rdo
const ATF_PSKIP_EN: Field = f(0x2000, 0, 1);
const INTRA_CIME_THD: [Field; 3] = [f(0x20cc, 0, 12), f(0x20cc, 16, 12), f(0x20d0, 0, 12)];
#[rustfmt::skip]
const INTRA_ATF_WGT_FIELDS: [Field; 12] = [
    f(0x20e4, 0, 8), f(0x20e4, 8, 8), f(0x20e4, 16, 8),
    f(0x20e8, 0, 8), f(0x20e8, 8, 8), f(0x20e8, 16, 8),
    f(0x20ec, 0, 8), f(0x20ec, 8, 8), f(0x20ec, 16, 8),
    f(0x20f0, 0, 8), f(0x20f0, 8, 8), f(0x20f0, 16, 8),
];
const SKIP_CIME_THD: [Field; 4] = [
    f(0x211c, 0, 12),
    f(0x211c, 16, 12),
    f(0x2120, 0, 12),
    f(0x2120, 16, 12),
];
#[rustfmt::skip]
const SKIP_ATF_WGT_FIELDS: [Field; 13] = [
    f(0x2134, 0, 8), f(0x2134, 8, 8), f(0x2134, 16, 8), f(0x2134, 24, 8),
    f(0x2138, 0, 8), f(0x2138, 8, 8), f(0x2138, 16, 8),
    f(0x213c, 0, 8), f(0x213c, 8, 8), f(0x213c, 16, 8),
    f(0x2140, 0, 8), f(0x2140, 8, 8), f(0x2140, 16, 8),
];
// reg_st
const ST_MADP: u32 = 0x4120;
const ST_NUM_CTU: Field = f(0x4124, 0, 21);
const ST_MADI: u32 = 0x4128;
const ST_NUM_B16: Field = f(0x412c, 0, 23);
const ST_MD_SAD_B16NUM: u32 = 0x4134;
const ST_MADI_B16NUM: u32 = 0x4144;

pub struct Tune {
    curr_scene_motion_flag: i32,
    ap_motion_flag: i32,
    md_madp: [i32; MD_WIN_LEN],
    md_flag_matrix: [i32; MD_SHOW_LEN],
    scene_motion_flag_matrix: [i32; FRAME_MOTION_ANALYSIS_NUM],
    pre_madp: [i32; 2],
    pre_madi: [i32; 2],
}

impl Default for Tune {
    fn default() -> Self {
        Self {
            curr_scene_motion_flag: 0,
            ap_motion_flag: 1,
            md_madp: [0; MD_WIN_LEN],
            md_flag_matrix: [0; MD_SHOW_LEN],
            scene_motion_flag_matrix: [0; FRAME_MOTION_ANALYSIS_NUM],
            pre_madp: [-1; 2],
            pre_madi: [-1; 2],
        }
    }
}

impl Tune {
    pub fn moving_scene(&self) -> bool {
        self.curr_scene_motion_flag != 0
    }

    /// `vepu580_h264e_tune_reg_patch`
    pub fn patch(&self, regs: &mut Regs, i_slice: bool) {
        let smf = (self.ap_motion_flag * 2 + self.curr_scene_motion_flag) as usize;
        if !i_slice {
            let from = if self.curr_scene_motion_flag != 0 {
                0
            } else {
                4
            };
            for (k, &w) in KLUT_WEIGHT[from..from + 24].iter().enumerate() {
                regs.set(f(KLUT_WGT0 + 4 * k as u32, 0, 32), w);
            }
        }
        for (field, v) in MD_SAD_THD.iter().zip([4, 9, 15]) {
            regs.set(*field, v);
        }
        for (field, v) in MADI_THD.iter().zip([4, 9, 15]) {
            regs.set(*field, v);
        }
        if self.curr_scene_motion_flag != 0 {
            regs.set(LVL16_INTRA_UL_CST_THLD, 2501);
            regs.set(QUANT_F_BIAS_P, 341);
            regs.set(VTHD_Y, 0);
            regs.set(VTHD_C, 0);
            regs.set(CHRM_KLUT_OFST, 0);
            regs.set(ATF_INTRA_E, 0);
            regs.set(ATF_PSKIP_EN, 0);
            regs.set(ATR_THDQP, 51);
            regs.set(CIME_PMV_SET_ZERO, 0);
            regs.set(CIME_SAD_TH0, 0);
            regs.set(CIME_SAD_PU16_TH, 0);
            regs.set(MOVE_LAMBDA, 8);
        }
        for (field, v) in INTRA_CIME_THD.iter().zip([28, 44, 72]) {
            regs.set(*field, v);
        }
        for (field, v) in INTRA_ATF_WGT_FIELDS.iter().zip(INTRA_ATF_WGT[smf]) {
            regs.set(*field, v);
        }
        for (field, v) in SKIP_CIME_THD.iter().zip([10, 8, 15, 25]) {
            regs.set(*field, v);
        }
        for (field, v) in SKIP_ATF_WGT_FIELDS.iter().zip(SKIP_ATF_WGT[smf]) {
            regs.set(*field, v);
        }
        if !i_slice {
            for (field, v) in ATR_WGT_FIELDS.iter().zip(ATR_WGT[smf]) {
                regs.set(*field, v);
            }
        }
        for (field, v) in CIME_MULTI_FIELDS.iter().zip(CIME_MULTI[smf]) {
            regs.set(*field, v);
        }
        for (field, v) in RIME_MULTI_FIELDS.iter().zip(RIME_MULTI[smf]) {
            regs.set(*field, v);
        }
    }

    /// `vepu580_h264e_tune_stat_update`, from the status registers of the frame just encoded.
    #[allow(clippy::if_same_then_else, clippy::explicit_counter_loop)]
    pub fn update(&mut self, st: &Status) {
        let num = |base: u32, i: u32| st.word(base + 4 * i) as i32;
        let md_sad = [0, 1, 2, 3].map(|i| num(ST_MD_SAD_B16NUM, i));
        let madp_cnt: [i32; 5] = MADP_NUM_MAP.map(|m| {
            (0..4)
                .map(|k| md_sad[k].wrapping_mul(m[k]))
                .fold(0i32, i32::wrapping_add)
        });
        let num_b16 = st.get(ST_NUM_B16) as i32;
        let num_ctu = st.get(ST_NUM_CTU) as i32;
        self.pre_madi[0] = if num_b16 == 0 {
            0
        } else {
            (st.word(ST_MADI) as i32).wrapping_div(num_b16)
        };
        self.pre_madp[0] = if num_ctu == 0 {
            0
        } else {
            (st.word(ST_MADP) as i32).wrapping_div(num_ctu)
        };
        let mb_num = (0..4)
            .map(|i| num(ST_MADI_B16NUM, i))
            .fold(0i32, i32::wrapping_add);
        let mb_num = if mb_num != 0 { mb_num } else { 1 };
        let mvbit = if self.ap_motion_flag != 0 { 15 } else { 10 };
        let thd = |base: usize, k: usize| mb_num.wrapping_mul(CTU_MADP_CNT_THD[base][k]) >> mvbit;

        let mut madp = MOTION_LEVEL_STILL;
        if madp_cnt[4] != 0 {
            let mut base = self.ap_motion_flag as usize * 3;
            for _ in 0..3 {
                if self.pre_madp[0] >= MB_AVG_MADP_THD[base] {
                    if madp_cnt[0] > thd(base, 0)
                        || madp_cnt[1] > thd(base, 1)
                        || madp_cnt[2] > thd(base, 2)
                    {
                        madp = MOTION_LEVEL_BIG_MOTION;
                    } else if (madp_cnt[0] > thd(base, 3) || madp_cnt[1] > thd(base, 4))
                        && madp_cnt[3] < thd(base, 5)
                    {
                        madp = MOTION_LEVEL_BIG_MOTION;
                    } else if madp_cnt[3] < thd(base, 6) {
                        madp = MOTION_LEVEL_BIG_MOTION;
                    } else if madp_cnt[3] < thd(base, 7) {
                        madp = MOTION_LEVEL_MOTION;
                    }
                    break;
                }
                base += 1;
            }
        } else {
            madp = MOTION_LEVEL_UNKNOW_SCENE;
        }

        let mut score_threshold = ((MD_WIN_LEN as i32 - 2) * 6 + 2 * 8 + 2 * 11 + 2 * 13) / 2;
        let mut score = if madp != MOTION_LEVEL_UNKNOW_SCENE {
            madp * 13 + self.md_madp[0] * 11 + self.md_madp[1] * 8
        } else {
            score_threshold -= 13;
            self.md_madp[0] * 11 + self.md_madp[1] * 8
        };
        for j in 2..MD_WIN_LEN {
            score += self.md_madp[j] * 6;
        }
        let md_flag = (score >= score_threshold) as i32;

        let m = &self.md_flag_matrix;
        self.curr_scene_motion_flag = 0;
        if m[0] != 0 && m[1] != 0 && m[2] != 0 {
            self.curr_scene_motion_flag = 1;
        } else if (m[0] != 0 && m[1] != 0) || (m[1] != 0 && m[2] != 0 && m[3] != 0) {
            self.curr_scene_motion_flag = md_flag;
        }

        if madp != MOTION_LEVEL_UNKNOW_SCENE {
            self.md_madp.copy_within(0..MD_WIN_LEN - 1, 1);
            self.md_madp[0] = madp;
        }
        self.md_flag_matrix.copy_within(0..MD_SHOW_LEN - 1, 1);
        self.md_flag_matrix[0] = md_flag;
        self.scene_motion_flag_matrix
            .copy_within(0..FRAME_MOTION_ANALYSIS_NUM - 1, 1);
        self.scene_motion_flag_matrix[0] = self.curr_scene_motion_flag;
        self.pre_madi[1] = self.pre_madi[0];
        self.pre_madp[1] = self.pre_madp[0];
    }
}
