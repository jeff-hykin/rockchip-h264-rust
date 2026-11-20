# rockchip-h264

Pure-Rust MJPEG → H.264 on the Rockchip RK3588's hardware (the RKJPEGD JPEG decoder and the VEPU580 encoder), driven
straight through `/dev/mpp_service`. No MPP library, no GStreamer, no C.

```rust
let mut t = rockchip_h264::Transcoder::new(rockchip_h264::Config::default())?; // 1280x720@30, 4 Mbps CBR, GOP 30
let mut access_unit = Vec::new();
t.transcode(&jpeg_bytes, &mut access_unit)?; // Annex-B, SPS/PPS before each IDR
```

`rockchip_h264::v4l2::Capture` reads MJPEG off a UVC camera; `examples/mjpeg2h264.rs` is a test harness. Needs root.

`nix build` (crate2nix) builds the library and the harness, and `nix flake check` runs the tests; for the board, `cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.31`.

`Transcoder::new` refuses to run on anything but a tested platform (kernel release, `mpp_service` driver version,
SoC, command support, hardware ids), naming what differs. Tested: Deep Robotics M20 AOS board, RK3588, Rockchip BSP
kernel 5.10.198.

Any size from 16x16 to 4096x4096 (even width and height) is encoded; the size comes from each JPEG's own header, and a
change mid-stream rebuilds the encoder (new SPS/PPS, an IDR, buffers resized). Every register of every frame is
generated the way MPP generates it (`src/vepu580_setup.rs`, field positions generated from MPP's header by
`tools/vepu580_fields.py`), and tested word-for-word against what the vendor MPP build on AOS programmed at each of
the M20 cameras' MJPEG sizes (1920x1080, 1280x960, 1280x720, 1024x768, 848x480, 800x600, 640x480, 640x360), captured
with `tools/mppshim.c` (`tools/captures/`, `tools/vepu580_fixtures.py`). One field follows the vendor build rather
than current MPP: `src_range` (0, where current MPP writes 1).

## Tracks

The register programming and ioctl ABI follow these files:

- [rockchip-linux/mpp @ 9b56348](https://github.com/rockchip-linux/mpp/tree/9b56348cc61393c84662e6d6dd78feae67773a36):
  [hal_h264e_vepu580.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/h264e/hal_h264e_vepu580.c),
  [hal_h264e_vepu580_reg.h](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/h264e/hal_h264e_vepu580_reg.h),
  [hal_h264e_vepu580_tune.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/h264e/hal_h264e_vepu580_tune.c),
  [vepu580_common.h](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/common/vepu580_common.h),
  [vepu5xx_common.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/common/vepu5xx_common.c),
  [vepu5xx_common.h](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/common/vepu5xx_common.h),
  [vepu580_tune.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/common/vepu580_tune.c),
  [hal_jpegd_rkv.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/vpu/jpegd/hal_jpegd_rkv.c),
  [hal_jpegd_rkv_reg.h](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/vpu/jpegd/hal_jpegd_rkv_reg.h),
  [hal_jpegd_vpu7xx_com.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/vpu/jpegd/hal_jpegd_vpu7xx_com.c),
  [jpegd_parser.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/codec/dec/jpeg/jpegd_parser.c),
  [h264e_sps.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/codec/enc/h264/h264e_sps.c),
  [h264e_pps.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/codec/enc/h264/h264e_pps.c),
  [mpp_service.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/osal/driver/mpp_service.c),
  [mpp_service.h](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/osal/inc/mpp_service.h),
  [mpp_platform.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/osal/mpp_platform.c)
- [rockchip-linux/kernel develop-5.10 @ 95a3ad8](https://github.com/rockchip-linux/kernel/tree/95a3ad83f7f90e0e672b9c99b0858b4964318667/drivers/video/rockchip/mpp):
  [mpp_common.c](https://github.com/rockchip-linux/kernel/blob/95a3ad83f7f90e0e672b9c99b0858b4964318667/drivers/video/rockchip/mpp/mpp_common.c),
  [mpp_service.c](https://github.com/rockchip-linux/kernel/blob/95a3ad83f7f90e0e672b9c99b0858b4964318667/drivers/video/rockchip/mpp/mpp_service.c),
  [mpp_rkvenc2.c](https://github.com/rockchip-linux/kernel/blob/95a3ad83f7f90e0e672b9c99b0858b4964318667/drivers/video/rockchip/mpp/mpp_rkvenc2.c),
  [mpp_jpgdec.c](https://github.com/rockchip-linux/kernel/blob/95a3ad83f7f90e0e672b9c99b0858b4964318667/drivers/video/rockchip/mpp/mpp_jpgdec.c)

## License

Apache-2.0, as MPP.
