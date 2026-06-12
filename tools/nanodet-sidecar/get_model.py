#!/usr/bin/env python3
"""
Get a COCO NanoDet-Plus model for a first smoke-test of the sidecar.

Unlike YOLO-FastestV2 / MediaPipe, NanoDet does NOT publish a ready-to-download
NCNN (or stable ONNX) COCO model upstream — its pretrained weights are PyTorch
checkpoints. So the honest quickest paths are:

  1) Convert an official COCO checkpoint you already have:
         python get_model.py --ckpt nanodet-plus-m_416.ckpt \
                             --cfg nanodet/config/nanodet-plus-m_416.yml
     (grab the checkpoint from RangiLyu/nanodet's Model Zoo, then this runs
      export_ncnn.py → a verified .param/.bin.)

  2) Just train a quick model — the trainer auto-exports a verified ncnn model.
     Set EPOCHS low in train_nanodet.py's CONFIG for a fast smoke model:
         python train_nanodet.py

  3) Test the decode on the desktop headless runner with a NanoDet ONNX instead
     of ncnn (no conversion tools needed):
         YOLO_MODEL=nanodet.onnx YOLO_DECODE=nanodet YOLO_INPUT=416 <headless>

This script does option (1) when you pass --ckpt/--cfg, otherwise it prints the
above so you're never stuck guessing.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Convert a NanoDet COCO checkpoint to ncnn for a smoke-test.")
    ap.add_argument("--ckpt", help="official NanoDet-Plus .ckpt (from the repo's Model Zoo)")
    ap.add_argument("--cfg", help="the matching config .yml (e.g. nanodet/config/nanodet-plus-m_416.yml)")
    ap.add_argument("--repo", default="nanodet", help="cloned RangiLyu/nanodet dir")
    ap.add_argument("--input", type=int, default=416)
    ap.add_argument("--reg-max", type=int, default=7)
    a = ap.parse_args()

    if not (a.ckpt and a.cfg):
        print(__doc__)
        print("No --ckpt/--cfg given — see the options above. Nothing downloaded "
              "(NanoDet ships no prebuilt ncnn/ONNX COCO model).")
        return

    if not os.path.isdir(a.repo):
        sys.exit(f"ERROR: nanodet repo not found at '{a.repo}'. Clone it first:\n"
                 f"  git clone --depth 1 https://github.com/RangiLyu/nanodet.git {a.repo}")
    sys.path.insert(0, HERE)
    import export_ncnn
    res = export_ncnn.run_export(a.repo, a.cfg, os.path.join(HERE, "nanodet"),
                                 a.input, a.reg_max, classes=None, ckpt=a.ckpt)
    if not res:
        sys.exit(1)
    param, binf = res
    print("\nDone. Smoke-test the sidecar:")
    print(f"  ND_PARAM={param} ND_BIN={binf} ND_INPUT={a.input} \\")
    print(f"    YOLO_SOURCE=rpicam python3 nanodet_ncnn_sidecar.py --inspect")


if __name__ == "__main__":
    main()
