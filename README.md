# rockchip-h264

Pure-Rust MJPEG → H.264 on the Rockchip RK3588's hardware (the RKJPEGD JPEG decoder and the VEPU580 encoder), driven
straight through `/dev/mpp_service`. No MPP library, no GStreamer, no C.

```rust
let mut t = rockchip_h264::Transcoder::new(rockchip_h264::Config::default())?; // 1280x720@30, 4 Mbps CBR, GOP 30
let mut access_unit = Vec::new();
t.transcode(&jpeg_bytes, &mut access_unit)?; // Annex-B, SPS/PPS before each IDR
```

`rockchip_h264::v4l2::Capture` reads MJPEG off a UVC camera; `examples/mjpeg2h264.rs` is a test harness. Needs root.

`Transcoder::new` refuses to run on anything but a tested platform (kernel release, `mpp_service` driver version,
SoC, command support, hardware ids), naming what differs. Tested: Deep Robotics M20 AOS board, RK3588, Rockchip BSP
kernel 5.10.198. Only 1280x720 is supported.

## Tracks

The register programming and ioctl ABI follow these files:

- [rockchip-linux/mpp @ 9b56348](https://github.com/rockchip-linux/mpp/tree/9b56348cc61393c84662e6d6dd78feae67773a36):
  [hal_h264e_vepu580.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/h264e/hal_h264e_vepu580.c),
  [hal_h264e_vepu580_reg.h](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/h264e/hal_h264e_vepu580_reg.h),
  [hal_h264e_vepu580_tune.c](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/h264e/hal_h264e_vepu580_tune.c),
  [vepu580_common.h](https://github.com/rockchip-linux/mpp/blob/9b56348cc61393c84662e6d6dd78feae67773a36/mpp/hal/rkenc/common/vepu580_common.h),
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
