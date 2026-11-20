//! The H.264 parameter sets and access unit delimiter written in software (the hardware writes the slices, header
//! included). Field values follow what the vendor MPP build wrote on AOS (MPP `mpp/codec/enc/h264/h264e_sps.c`,
//! `h264e_pps.c`), so the stream matches the gst pipeline's byte for byte up to the slice data.

/// Exp-Golomb / fixed-width bit writer for one RBSP.
#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    cur: u32,
    used: u32,
}

impl Bits {
    fn put(&mut self, width: u32, value: u32) {
        for i in (0..width).rev() {
            self.cur = self.cur << 1 | (value >> i) & 1;
            self.used += 1;
            if self.used == 8 {
                self.bytes.push(self.cur as u8);
                self.cur = 0;
                self.used = 0;
            }
        }
    }

    fn flag(&mut self, on: bool) {
        self.put(1, on as u32);
    }

    fn ue(&mut self, value: u32) {
        let v = value + 1;
        let len = 32 - v.leading_zeros();
        self.put(len - 1, 0);
        self.put(len, v);
    }

    fn se(&mut self, value: i32) {
        self.ue(if value > 0 {
            2 * value as u32 - 1
        } else {
            (-2 * value) as u32
        });
    }

    /// rbsp_trailing_bits
    fn finish(mut self) -> Vec<u8> {
        self.put(1, 1);
        while self.used != 0 {
            self.put(1, 0);
        }
        self.bytes
    }
}

/// A NAL unit with a 4-byte start code, emulation prevention applied to its RBSP.
fn nal(ref_idc: u8, kind: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut out = vec![0, 0, 0, 1, ref_idc << 5 | kind];
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct StreamParams {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

/// log2_max_frame_num_minus4 and log2_max_pic_order_cnt_lsb_minus4 (MPP's 16-bit frame_num / POC lsb)
pub const LOG2_MAX_FRAME_NUM_MINUS4: u32 = 12;
pub const LOG2_MAX_POC_LSB_MINUS4: u32 = 12;
pub const CHROMA_QP_INDEX_OFFSET: i32 = -6;
pub const PIC_INIT_QP: u32 = 26;

/// High profile, level 4.0, one reference frame, CABAC, 8x8 transform, VUI with timing (as MPP writes it).
pub fn sps(p: StreamParams) -> Vec<u8> {
    let mbs_w = p.width.div_ceil(16);
    let mbs_h = p.height.div_ceil(16);
    let mut b = Bits::default();
    b.put(8, 100); // profile_idc: High
    b.put(8, 0x10); // constraint_set3_flag (as MPP sets it)
    b.put(8, 40); // level_idc
    b.ue(0); // seq_parameter_set_id
    b.ue(1); // chroma_format_idc: 4:2:0
    b.ue(0); // bit_depth_luma_minus8
    b.ue(0); // bit_depth_chroma_minus8
    b.flag(false); // qpprime_y_zero_transform_bypass_flag
    b.flag(false); // seq_scaling_matrix_present_flag
    b.ue(LOG2_MAX_FRAME_NUM_MINUS4);
    b.ue(0); // pic_order_cnt_type
    b.ue(LOG2_MAX_POC_LSB_MINUS4);
    b.ue(1); // max_num_ref_frames
    b.flag(true); // gaps_in_frame_num_value_allowed_flag
    b.ue(mbs_w - 1);
    b.ue(mbs_h - 1);
    b.flag(true); // frame_mbs_only_flag
    b.flag(true); // direct_8x8_inference_flag
    let crop_right = (mbs_w * 16 - p.width) / 2;
    let crop_bottom = (mbs_h * 16 - p.height) / 2;
    b.flag(crop_right != 0 || crop_bottom != 0);
    if crop_right != 0 || crop_bottom != 0 {
        b.ue(0);
        b.ue(crop_right);
        b.ue(0);
        b.ue(crop_bottom);
    }
    b.flag(true); // vui_parameters_present_flag
    b.flag(false); // aspect_ratio_info_present_flag
    b.flag(false); // overscan_info_present_flag
    b.flag(false); // video_signal_type_present_flag
    b.flag(false); // chroma_loc_info_present_flag
    b.flag(true); // timing_info_present_flag
    b.put(32, 1); // num_units_in_tick
    b.put(32, 2 * p.fps); // time_scale
    b.flag(true); // fixed_frame_rate_flag
    b.flag(false); // nal_hrd_parameters_present_flag
    b.flag(false); // vcl_hrd_parameters_present_flag
    b.flag(false); // pic_struct_present_flag
    b.flag(true); // bitstream_restriction_flag
    b.flag(true); // motion_vectors_over_pic_boundaries_flag
    b.ue(0); // max_bytes_per_pic_denom
    b.ue(0); // max_bits_per_mb_denom
    b.ue(16); // log2_max_mv_length_horizontal
    b.ue(16); // log2_max_mv_length_vertical
    b.ue(0); // max_num_reorder_frames
    b.ue(1); // max_dec_frame_buffering
    nal(3, 7, &b.finish())
}

pub fn pps() -> Vec<u8> {
    let mut b = Bits::default();
    b.ue(0); // pic_parameter_set_id
    b.ue(0); // seq_parameter_set_id
    b.flag(true); // entropy_coding_mode_flag: CABAC
    b.flag(false); // bottom_field_pic_order_in_frame_present_flag
    b.ue(0); // num_slice_groups_minus1
    b.ue(0); // num_ref_idx_l0_default_active_minus1
    b.ue(0); // num_ref_idx_l1_default_active_minus1
    b.flag(false); // weighted_pred_flag
    b.put(2, 0); // weighted_bipred_idc
    b.se(PIC_INIT_QP as i32 - 26); // pic_init_qp_minus26
    b.se(0); // pic_init_qs_minus26
    b.se(CHROMA_QP_INDEX_OFFSET);
    b.flag(true); // deblocking_filter_control_present_flag
    b.flag(false); // constrained_intra_pred_flag
    b.flag(false); // redundant_pic_cnt_present_flag
    b.flag(true); // transform_8x8_mode_flag
    b.flag(false); // pic_scaling_matrix_present_flag
    b.se(CHROMA_QP_INDEX_OFFSET); // second_chroma_qp_index_offset
    nal(3, 8, &b.finish())
}

/// An access unit delimiter (primary_pic_type 7: any slice type), as `h264parse` puts in front of each unit.
pub fn aud() -> Vec<u8> {
    let mut b = Bits::default();
    b.put(3, 7);
    nal(0, 9, &b.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// The first 46 bytes of the gst pipeline's 1280x720@30 output on AOS (`mppjpegdec ! mpph264enc ! h264parse`).
    #[test]
    fn headers_match_the_vendor_stream() {
        let p = StreamParams {
            width: 1280,
            height: 720,
            fps: 30,
        };
        assert_eq!(aud(), hex("0000000109f0"));
        assert_eq!(
            sps(p),
            hex("0000000167641028ac1b1aa05005ba10000003001000000303c8f08846a0")
        );
        assert_eq!(pps(), hex("0000000168ee31b21b"));
    }

    #[test]
    fn emulation_prevention() {
        assert_eq!(
            nal(0, 1, &[0, 0, 1, 0, 0, 0]),
            vec![0, 0, 0, 1, 1, 0, 0, 3, 1, 0, 0, 3, 0]
        );
    }
}
