# rockchip-h264

MJPEG in, H.264 out, on the Rockchip RK3588's hardware: the JPEG decoder (RKJPEGD) and the VEPU580 H.264 encoder
(RKVENC), driven from Rust straight through the kernel's `/dev/mpp_service` ioctls. No MPP library, no GStreamer,
no C. Every dmabuf and hardware session is created once per stream, so a stream's fd count and memory stay flat
however long it runs.

Built for the Deep Robotics M20's AOS board, replacing a `gst-launch-1.0 ... mppjpegdec ! mpph264enc ...` child
whose vendor plugin leaked a dmabuf fd per frame.

```rust
let mut t = rockchip_h264::Transcoder::new(rockchip_h264::Config::default())?; // 1280x720@30, 4 Mbps CBR, GOP 30
let mut access_unit = Vec::new();
let frame = t.transcode(&jpeg_bytes, &mut access_unit)?; // [AUD][SPS PPS on IDR][slice], Annex-B
```

`rockchip_h264::v4l2::Capture` reads MJPEG off a UVC camera. `examples/mjpeg2h264.rs` is the test harness: JPEG
files or a camera through N streams at once, H.264 to files, stats on stderr (`--check` prints the platform and the
gate verdict).

Needs root (the `mpp_service` and `dma_heap` device nodes are root-only on AOS).

## The startup gate

`Transcoder::new` first reads everything MPP itself branches on and refuses anything that isn't an exact match for
a tested combination, with an error naming what changed (`no Rust version for this platform (kernel release: found
"5.10.209", tested "5.10.198"; ...)`):

| what | where it comes from | tested value |
|---|---|---|
| kernel release | `/proc/sys/kernel/osrelease` (MPP: `check_kernel_version`) | `5.10.198` |
| ioctl interface | `/dev/mpp_service` present (MPP: `IOCTL_MPP_SERVICE_V1`) | mpp_service v1 |
| driver version | `/proc/mpp_service/version` | `a700c86c0 author: cyc 2025-07-29 Initcommit for rk3588_sdk` |
| SoC | `/proc/device-tree/compatible` (MPP: `mpp_get_soc_info`) | contains `rockchip,rk3588` |
| command support | `MPP_CMD_QUERY_CMD_SUPPORT` per base (MPP: `check_mpp_service_cap`) | `0x003 0x103 0x205 0x302 0x404` |
| hardware bitmask | `MPP_CMD_PROBE_HW_SUPPORT` | `0x100d221a` |
| encoder hardware id | `MPP_CMD_QUERY_HW_ID`, client RKVENC (16) | `0x50603312` |
| decoder hardware id | `MPP_CMD_QUERY_HW_ID`, client RKJPEGD (13) | `0xdb1f0005` |

Tested combinations (`platform::TESTED`):

| name | board | notes |
|---|---|---|
| Deep Robotics M20 AOS | RK3588 (Firefly ITX-3588J), Ubuntu 20.04, Rockchip BSP 5.10.198 | the only one so far |

The encoder also refuses any resolution but 1280x720 (`vepu580::SUPPORTED_SIZES`): its register image was captured
and verified at that size only.

## What it mirrors

The register programming follows these files of [rockchip-linux/mpp](https://github.com/rockchip-linux/mpp) at
`9b56348cc61393c84662e6d6dd78feae67773a36` (2026-11-03), git blob hashes for a watcher to compare:

| file | blob | used for |
|---|---|---|
| `mpp/hal/rkenc/h264e/hal_h264e_vepu580.c` | `1ed5cde375f26510b261ab413f1e098aed206b0e` | per-frame encoder registers, rate-control registers, buffers, dual-core ids, task submit/poll, status |
| `mpp/hal/rkenc/h264e/hal_h264e_vepu580_reg.h` | `baabf4e38e5989cbe2aac54203aba60e08cefe8b` | encoder register field positions |
| `mpp/hal/rkenc/common/vepu580_common.h` | `28f4e374e93748ca34de202e6339a8f8b1537450` | encoder register field positions (common part) |
| `mpp/hal/rkenc/common/vepu5xx_common.h` | `4fddd4674806e50379b649f682449c4cb379a850` | source format codes |
| `mpp/hal/rkenc/h264e/hal_h264e_vepu580_tune.c` | `91ac271730c91fcbd2d58b264277186dd95cf72d` | scene tuner (ported line for line in `vepu580_tune.rs`) |
| `mpp/hal/rkenc/common/vepu580_tune.c` | `d948904e703ecfb552e4a534f877d072a337b4a8` | tuner tables |
| `mpp/hal/vpu/jpegd/hal_jpegd_rkv.c` | `9f5eab621cd0a3d39a57e775a366c52ea2f859e2` | JPEG decoder registers |
| `mpp/hal/vpu/jpegd/hal_jpegd_rkv_reg.h` | `f903dedf867af83b40221a8283c375ca59468fc2` | JPEG decoder register field positions |
| `mpp/hal/vpu/jpegd/hal_jpegd_vpu7xx_com.c` | `2d25dd9fb37b8e64cf5469faeff3634e62fd6698` | Huffman / quantization table layout |
| `mpp/hal/vpu/jpegd/hal_jpegd_vpu7xx_com.h` | `eae2b125219ffb06d46a72d577ff7382430f5b67` | table offsets, mode codes |
| `mpp/hal/vpu/jpegd/hal_jpegd_common.h` | `4878244014a2b7c9e95eeae28909092cda404bd6` | zig-zag order |
| `mpp/codec/dec/jpeg/jpegd_parser.c` | `88662dd0fb6a1dc748455221ed23c66b7288ae22` | JPEG header parsing, default Huffman tables |
| `mpp/codec/enc/h264/h264e_sps.c` | `2e9454eae19e538db0e1bb41939451e863de8530` | SPS fields |
| `mpp/codec/enc/h264/h264e_pps.c` | `6a7b9e007f783dd9e2845bb21c6742de3e02c36a` | PPS fields |
| `osal/driver/mpp_service.c` | `0346164a112041435154613dd7c23d70429addd4` | ioctl request chains, capability probe |
| `osal/inc/mpp_service.h` | `213c07ffd91d5b119a3d8e7c42fc420db461766b` | ioctl ABI, command ids, flags |
| `osal/mpp_platform.c` | `a7c2c5f65125fd6fb443126d795a3b3542ccba81` | what the startup gate checks |
| `osal/inc/mpp_platform.h` | `e89dee5b2d60fcb4a0bc703dadcc255a91679897` | ioctl / kernel version enums |

and of the kernel driver, [rockchip-linux/kernel](https://github.com/rockchip-linux/kernel) `develop-5.10` at
`95a3ad83f7f90e0e672b9c99b0858b4964318667` (2026-08-28):

| file | blob | used for |
|---|---|---|
| `drivers/video/rockchip/mpp/mpp_common.c` | `cb30abb4394d8074b29e488a9e05898508f79f86` | `mpp_msg_v1`, request collection, poll |
| `drivers/video/rockchip/mpp/mpp_common.h` | `e26381e372713f1177c38c4833789038716e7c68` | message limits, device types |
| `drivers/video/rockchip/mpp/mpp_service.c` | `009b36d273783806e5029434831d8e8851bb42a6` | procfs `version` / `supports-cmd` |
| `drivers/video/rockchip/mpp/mpp_rkvenc2.c` | `035dae0ffd556102d2e0bc84ce4427c27d2a658c` | encoder register classes, fd translation table, dual-core handshake |
| `drivers/video/rockchip/mpp/mpp_jpgdec.c` | `d846a73514e99d2b848fc4423b4167008f4dba81` | decoder fd translation table |

AOS runs the vendor's own import of that driver (the `version` string above), not this exact commit; the gate pins
the vendor build.

### The register template

Most of the VEPU580's ~1450 register words are static for a given resolution and mode (scaling lists, RDO and
motion-search tables). `src/vepu580_template.rs` is the image the vendor MPP build (the one in DeepRobotics' gst
plugin on AOS) programs for this exact stream, captured with `tools/mppshim.c` (an `LD_PRELOAD` logger of
`MPP_IOC_CFG_V1` requests, a dev tool only) and turned into Rust by `tools/vepu580_template.py`:

```sh
gcc -O2 -shared -fPIC -o mppshim.so tools/mppshim.c -ldl -lpthread           # on AOS
sudo env LD_PRELOAD=$PWD/mppshim.so SHIM_LOG=ref.log gst-launch-1.0 -q multifilesrc location=f%03d.jpg \
  caps=image/jpeg,width=1280,height=720,framerate=30/1 ! jpegparse ! mppjpegdec \
  ! mpph264enc rc-mode=cbr bps=4000000 bps-max=5000000 gop=30 ! h264parse config-interval=-1 \
  ! video/x-h264,stream-format=byte-stream,alignment=au ! filesink location=ref.h264
tools/vepu580_template.py ref.log > src/vepu580_template.rs
```

Every field that changes from frame to frame (buffers, QP, rate-control targets and thresholds, slice syntax,
dual-core ids, the tuner's fields) is set by name in `vepu580.rs` / `vepu580_tune.rs`, following MPP. The capture
showed which ones those are: across 90 frames of the vendor's own output nothing else changed.

## Differences from MPP

- Frame-level rate control is this crate's own small CBR controller (`rc.rs`), not a port of MPP's
  `rc_model_v2.c`; the hardware's row-level rate control is programmed exactly as MPP does it.
- The encoder runs one task at a time per stream (MPP keeps two in flight). The dual-core handshake ids are still
  set the way MPP sets them for its two-task mode, which is what the vendor build ran.
- The tuner is current MPP's. The vendor build (2023 source) switches one field (`quant_f_bias_P`, 171 -> 341) a
  few frames earlier on moving content; everything else it programs matches.
- JPEG: baseline 8-bit YCbCr only, 4:2:0 (2x2,1x1,1x1) or 4:2:2 (2x1,1x1,1x1), Cb and Cr sharing Huffman tables;
  anything else is refused (as MPP refuses it).
- The SPS has no `video_signal_type`, as the vendor stream: a full-range camera JPEG is labeled limited range, same
  as before.

## License

Apache-2.0. The register programming is derived from Rockchip's MPP (Apache-2.0) and the Rockchip kernel driver.
