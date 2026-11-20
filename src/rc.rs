//! Frame-level constant-bitrate control: picks each frame's starting QP and bit target; the VEPU580's own row-level
//! rate control (programmed from the target, see `vepu580::set_rate_control`) steers QP within the frame.
//!
//! This is a small CBR controller of our own, not a port of MPP's `rc_model_v2.c`: it splits a GOP's bits between
//! the IDR and the P frames by their measured size ratio, pays back any overshoot or undershoot over about a second,
//! and moves QP by the bits error in the 6-QP-per-doubling model H.264 quantization follows.

/// QP limits; the register template's ROI QP limits are the same (10..=51, see the tests in `vepu580`).
pub const QP_MIN: u32 = 10;
pub const QP_MAX: u32 = 51;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub qp: u32,
    pub target_bits: u32,
}

pub struct RateControl {
    nominal_bits: f64,
    gop: f64,
    fps: f64,
    /// QP the next P frame starts from
    qp_p: f64,
    /// IDR QP relative to the P QP
    i_offset: f64,
    /// IDR bits / P bits, measured
    i_ratio: f64,
    /// bits spent beyond the nominal rate so far (negative: under)
    debt_bits: f64,
    /// recent P frame size, for the IDR ratio
    p_bits_avg: f64,
}

impl RateControl {
    pub fn new(bitrate_bps: u32, fps: u32, gop: u32) -> Self {
        Self {
            nominal_bits: bitrate_bps as f64 / fps as f64,
            gop: gop.max(1) as f64,
            fps: fps as f64,
            qp_p: crate::h264::PIC_INIT_QP as f64,
            i_offset: -2.0,
            i_ratio: 6.0,
            debt_bits: 0.0,
            p_bits_avg: 0.0,
        }
    }

    fn p_target(&self) -> f64 {
        let base = self.nominal_bits * self.gop / (self.i_ratio + self.gop - 1.0);
        (base - self.debt_bits / self.fps).clamp(0.5 * base, 1.5 * base)
    }

    pub fn plan(&self, idr: bool) -> Plan {
        let (qp, target) = if idr {
            (self.qp_p + self.i_offset, self.p_target() * self.i_ratio)
        } else {
            (self.qp_p, self.p_target())
        };
        Plan {
            qp: (qp.round() as i64).clamp(QP_MIN as i64, QP_MAX as i64) as u32,
            target_bits: target.max(1.0) as u32,
        }
    }

    /// Feeds back what the frame cost: its size and the average QP the hardware ended up using.
    pub fn update(&mut self, idr: bool, plan: Plan, bits: u32, avg_qp: f64) {
        let bits_f = (bits as f64).max(1.0);
        let error_qp = 6.0 * (bits_f / plan.target_bits as f64).log2();
        // a whole second of bitrate either way at most: a stretch of static scene can't bank unlimited credit
        self.debt_bits = (self.debt_bits + bits_f - self.nominal_bits)
            .clamp(-self.nominal_bits * self.fps, self.nominal_bits * self.fps);
        if idr {
            if self.p_bits_avg > 0.0 {
                self.i_ratio =
                    (0.5 * self.i_ratio + 0.5 * bits_f / self.p_bits_avg).clamp(1.5, 20.0);
            }
            self.i_offset = (avg_qp - self.qp_p + 0.7 * error_qp).clamp(-10.0, 6.0);
        } else {
            let step = (avg_qp - self.qp_p + 0.7 * error_qp).clamp(-3.0, 3.0);
            self.qp_p = (self.qp_p + step).clamp(QP_MIN as f64, QP_MAX as f64);
            self.p_bits_avg = if self.p_bits_avg == 0.0 {
                bits_f
            } else {
                0.9 * self.p_bits_avg + 0.1 * bits_f
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame whose size halves every 6 QP steps, `complexity` bits at QP 26.
    fn model(qp: u32, complexity: f64) -> u32 {
        (complexity * 2f64.powf((26.0 - qp as f64) / 6.0)) as u32
    }

    #[test]
    fn settles_at_the_bitrate() {
        let mut rc = RateControl::new(4_000_000, 30, 30);
        let mut total = 0u64;
        let frames = 30 * 60;
        for i in 0..frames {
            let idr = i % 30 == 0;
            let plan = rc.plan(idr);
            let bits = model(plan.qp, if idr { 900_000.0 } else { 300_000.0 });
            if i >= 30 * 10 {
                total += bits as u64;
            }
            rc.update(idr, plan, bits, plan.qp as f64);
        }
        let bps = total as f64 / ((frames - 300) as f64 / 30.0);
        assert!((bps - 4e6).abs() < 0.05 * 4e6, "{bps}");
    }

    #[test]
    fn a_trivial_scene_sits_at_the_minimum_qp() {
        let mut rc = RateControl::new(4_000_000, 30, 30);
        for i in 0..300 {
            let idr = i % 30 == 0;
            let plan = rc.plan(idr);
            rc.update(idr, plan, model(plan.qp, 2_000.0), plan.qp as f64);
        }
        assert_eq!(rc.plan(false).qp, QP_MIN);
    }
}
