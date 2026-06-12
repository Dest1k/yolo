#!/usr/bin/env python3
"""
Heads-up: the RKNN sidecar is NOT for the Raspberry Pi 5.

RKNN models run on Rockchip NPUs (RK3588 / RK3576 / RK3568 …) — the Pi 5 has no
NPU, so a .rknn won't run there. For the Pi 5 use the NCNN sidecars
(yolo-fastestv2 / nanodet) or MediaPipe; each has its own get_model.py for a
COCO smoke-test.

If you ARE on a Rockchip board: prebuilt COCO .rknn models live in the official
model zoo — grab one for your exact chip (they're chip-specific):

    git clone --depth 1 https://github.com/airockchip/rknn_model_zoo.git
    # e.g. rknn_model_zoo/examples/yolov5/model/  has a download/convert script
    #      per target (RK3588 etc.). Build the .rknn with rknn-toolkit2 for your SoC.

Then point the sidecar at it:
    RKNN_MODEL=yolov5s.rknn YOLO_SOURCE=0 python3 yolo_rknn_sidecar.py
"""
print(__doc__)
