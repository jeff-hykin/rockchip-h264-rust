//! The startup gate: everything MPP itself branches on (`osal/mpp_platform.c`, `osal/driver/mpp_service.c`,
//! `osal/mpp_soc.c`) read off this machine and compared against the exact combinations this crate was tested on.
//! Anything else is refused, naming what changed, because the register programming here is only known-good there.

use std::fmt;

use anyhow::{bail, Result};

use crate::service::{self, client, cmd};

/// What MPP reads about the platform before it picks a hardware path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Platform {
    /// `uname -r` (MPP `check_kernel_version` reads `/proc/version`)
    pub kernel_release: String,
    /// `/dev/mpp_service` present: MPP's `IOCTL_MPP_SERVICE_V1` (otherwise the legacy `vcodec_service` ABI)
    pub ioctl_mpp_service_v1: bool,
    /// `/proc/mpp_service/version`: the driver's own version string
    pub driver_version: String,
    /// `/proc/device-tree/compatible`, the strings MPP matches its SoC table against
    pub soc_compatible: Vec<String>,
    /// `MPP_CMD_QUERY_CMD_SUPPORT` for the query, init, send, poll and control bases
    pub cmd_support: [u32; 5],
    /// `MPP_CMD_PROBE_HW_SUPPORT`: the client-type bitmask
    pub hw_support: u32,
    /// `MPP_CMD_QUERY_HW_ID` of the H.264 encoder (RKVENC, VEPU580)
    pub rkvenc_hw_id: u32,
    /// `MPP_CMD_QUERY_HW_ID` of the JPEG decoder (RKJPEGD)
    pub rkjpegd_hw_id: u32,
}

/// One platform combination this crate has been run and verified on.
#[derive(Debug)]
pub struct Tested {
    pub name: &'static str,
    pub kernel_release: &'static str,
    pub driver_version: &'static str,
    pub soc_compatible: &'static str,
    pub cmd_support: [u32; 5],
    pub hw_support: u32,
    pub rkvenc_hw_id: u32,
    pub rkjpegd_hw_id: u32,
}

/// Every combination the register programming has been verified on; see the README's version table.
pub const TESTED: &[Tested] = &[Tested {
    name: "Deep Robotics M20 AOS (RK3588, Rockchip BSP 5.10.198)",
    kernel_release: "5.10.198",
    driver_version: "a700c86c0 author: cyc 2025-07-29 Initcommit for rk3588_sdk",
    soc_compatible: "rockchip,rk3588",
    cmd_support: [0x003, 0x103, 0x205, 0x302, 0x404],
    hw_support: 0x100d_221a,
    rkvenc_hw_id: 0x5060_3312,
    rkjpegd_hw_id: 0xdb1f_0005,
}];

impl fmt::Display for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "kernel release      {}", self.kernel_release)?;
        writeln!(
            f,
            "ioctl interface     {}",
            if self.ioctl_mpp_service_v1 {
                "mpp_service v1"
            } else {
                "vcodec_service (legacy)"
            }
        )?;
        writeln!(f, "driver version      {}", self.driver_version)?;
        writeln!(f, "soc compatible      {}", self.soc_compatible.join(" "))?;
        let cmds: Vec<String> = self
            .cmd_support
            .iter()
            .map(|c| format!("{c:#05x}"))
            .collect();
        writeln!(f, "cmd support         {}", cmds.join(" "))?;
        writeln!(f, "hw support          {:#010x}", self.hw_support)?;
        writeln!(f, "rkvenc hw id        {:#010x}", self.rkvenc_hw_id)?;
        write!(f, "rkjpegd hw id       {:#010x}", self.rkjpegd_hw_id)
    }
}

fn read_trimmed(path: &str) -> String {
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Reads the platform the way MPP does at startup (needs root for the mpp_service ioctls).
pub fn probe() -> Result<Platform> {
    let ioctl_mpp_service_v1 = std::path::Path::new(service::SERVICE_PATH).exists();
    let mut platform = Platform {
        kernel_release: read_trimmed("/proc/sys/kernel/osrelease"),
        ioctl_mpp_service_v1,
        driver_version: read_trimmed("/proc/mpp_service/version"),
        soc_compatible: std::fs::read("/proc/device-tree/compatible")
            .unwrap_or_default()
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect(),
        cmd_support: [0; 5],
        hw_support: 0,
        rkvenc_hw_id: 0,
        rkjpegd_hw_id: 0,
    };
    if !ioctl_mpp_service_v1 {
        return Ok(platform);
    }
    let bases = [
        cmd::QUERY_BASE,
        cmd::INIT_BASE,
        cmd::SEND_BASE,
        cmd::POLL_BASE,
        cmd::CONTROL_BASE,
    ];
    for (slot, base) in platform.cmd_support.iter_mut().zip(bases) {
        *slot = service::query_cmd_support(base)?;
    }
    platform.hw_support = service::probe_hw_support()?;
    if platform.hw_support & (1 << client::RKVENC) != 0 {
        platform.rkvenc_hw_id = service::query_hw_id(client::RKVENC)?;
    }
    if platform.hw_support & (1 << client::RKJPEGD) != 0 {
        platform.rkjpegd_hw_id = service::query_hw_id(client::RKJPEGD)?;
    }
    Ok(platform)
}

/// The differences between `platform` and one tested combination, each as "<what>: found X, tested Y".
fn differences(platform: &Platform, tested: &Tested) -> Vec<String> {
    let mut diffs = Vec::new();
    let mut check = |what: &str, found: String, want: String| {
        if found != want {
            diffs.push(format!("{what}: found {found:?}, tested {want:?}"));
        }
    };
    check(
        "kernel release",
        platform.kernel_release.clone(),
        tested.kernel_release.into(),
    );
    check(
        "ioctl interface",
        if platform.ioctl_mpp_service_v1 {
            "mpp_service v1"
        } else {
            "vcodec_service"
        }
        .into(),
        "mpp_service v1".into(),
    );
    check(
        "mpp_service driver version",
        platform.driver_version.clone(),
        tested.driver_version.into(),
    );
    check(
        "SoC",
        if platform
            .soc_compatible
            .iter()
            .any(|c| c == tested.soc_compatible)
        {
            tested.soc_compatible.into()
        } else {
            platform.soc_compatible.join(" ")
        },
        tested.soc_compatible.into(),
    );
    check(
        "mpp_service command support",
        format!("{:x?}", platform.cmd_support),
        format!("{:x?}", tested.cmd_support),
    );
    check(
        "mpp_service hardware bitmask",
        format!("{:#010x}", platform.hw_support),
        format!("{:#010x}", tested.hw_support),
    );
    check(
        "H.264 encoder (RKVENC) hardware id",
        format!("{:#010x}", platform.rkvenc_hw_id),
        format!("{:#010x}", tested.rkvenc_hw_id),
    );
    check(
        "JPEG decoder (RKJPEGD) hardware id",
        format!("{:#010x}", platform.rkjpegd_hw_id),
        format!("{:#010x}", tested.rkjpegd_hw_id),
    );
    diffs
}

/// The tested combination `platform` matches exactly, or an error naming everything that differs from the closest.
pub fn check(platform: &Platform) -> Result<&'static Tested> {
    let mut closest: Option<Vec<String>> = None;
    for tested in TESTED {
        let diffs = differences(platform, tested);
        if diffs.is_empty() {
            return Ok(tested);
        }
        if closest.as_ref().is_none_or(|c| diffs.len() < c.len()) {
            closest = Some(diffs);
        }
    }
    let diffs = closest.unwrap_or_default();
    bail!(
        "no Rust version for this platform ({}); refusing to program the hardware",
        diffs.join("; ")
    )
}

/// Probes this machine and refuses anything not in [`TESTED`].
pub fn probe_and_check() -> Result<(Platform, &'static Tested)> {
    let platform = probe()?;
    let tested = check(&platform)?;
    Ok((platform, tested))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aos() -> Platform {
        let t = &TESTED[0];
        Platform {
            kernel_release: t.kernel_release.into(),
            ioctl_mpp_service_v1: true,
            driver_version: t.driver_version.into(),
            soc_compatible: vec![
                "rockchip,rk3588-firefly-itx-3588j".into(),
                "rockchip,rk3588".into(),
            ],
            cmd_support: t.cmd_support,
            hw_support: t.hw_support,
            rkvenc_hw_id: t.rkvenc_hw_id,
            rkjpegd_hw_id: t.rkjpegd_hw_id,
        }
    }

    fn refusal(platform: &Platform) -> String {
        format!("{:#}", check(platform).unwrap_err())
    }

    #[test]
    fn the_tested_platform_passes() {
        assert_eq!(check(&aos()).unwrap().name, TESTED[0].name);
    }

    #[test]
    fn another_kernel_is_refused_by_name() {
        let mut p = aos();
        p.kernel_release = "5.10.209".into();
        let msg = refusal(&p);
        assert!(
            msg.starts_with("no Rust version for this platform"),
            "{msg}"
        );
        assert!(msg.contains("kernel release: found \"5.10.209\""), "{msg}");
    }

    #[test]
    fn the_legacy_ioctl_interface_is_refused() {
        let mut p = aos();
        p.ioctl_mpp_service_v1 = false;
        assert!(refusal(&p).contains("ioctl interface: found \"vcodec_service\""));
    }

    #[test]
    fn a_new_driver_build_is_refused() {
        let mut p = aos();
        p.driver_version = "b1234567 author: someone 2027-01-01".into();
        assert!(refusal(&p).contains("mpp_service driver version"));
    }

    #[test]
    fn another_soc_is_refused() {
        let mut p = aos();
        p.soc_compatible = vec!["rockchip,rk3568".into()];
        assert!(refusal(&p).contains("SoC: found \"rockchip,rk3568\""));
    }

    #[test]
    fn a_changed_command_set_is_refused() {
        let mut p = aos();
        p.cmd_support[4] = 0x405;
        assert!(refusal(&p).contains("mpp_service command support"));
    }

    #[test]
    fn changed_hardware_bits_or_ids_are_refused() {
        let mut p = aos();
        p.hw_support &= !(1 << client::RKJPEGD);
        assert!(refusal(&p).contains("hardware bitmask"));
        let mut p = aos();
        p.rkvenc_hw_id = 0x5060_3313;
        assert!(refusal(&p).contains("RKVENC) hardware id"));
        let mut p = aos();
        p.rkjpegd_hw_id = 0;
        assert!(refusal(&p).contains("RKJPEGD) hardware id"));
    }

    #[test]
    fn every_difference_is_named_at_once() {
        let mut p = aos();
        p.kernel_release = "6.1.75".into();
        p.rkvenc_hw_id = 1;
        let msg = refusal(&p);
        assert!(
            msg.contains("kernel release") && msg.contains("RKVENC"),
            "{msg}"
        );
    }
}
