#!/usr/bin/env python3
"""
Fetch a ready, COCO-pretrained YOLO-FastestV2 NCNN model for a first smoke-test of
the sidecar — before you train your own. One command, no GPU, no conversion:

    python get_model.py
    # then it prints the exact line to run the sidecar.

Downloads the upstream repo's optimised ncnn model (dog-qiuqiu/Yolo-FastestV2,
COCO 80 classes, 352 input) + a coco.names. The sidecar auto-detects the input/
output blob names, so it just runs.
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "https://raw.githubusercontent.com/dog-qiuqiu/Yolo-FastestV2/master/sample/ncnn/model"
FILES = {
    "yolo-fastestv2-opt.param": f"{RAW}/yolo-fastestv2-opt.param",
    "yolo-fastestv2-opt.bin":   f"{RAW}/yolo-fastestv2-opt.bin",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}


def fetch(name, url):
    dst = os.path.join(HERE, name)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  have {name} ({os.path.getsize(dst)} B)"); return dst
    print(f"  downloading {name} …")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
            f.write(r.read())
    except Exception as e:
        sys.exit(f"ERROR: failed to download {url}\n  {e}")
    print(f"    → {dst} ({os.path.getsize(dst)} B)")
    return dst


def main():
    print("Fetching a COCO-pretrained YOLO-FastestV2 ncnn model…")
    param = fetch("yolo-fastestv2-opt.param", FILES["yolo-fastestv2-opt.param"])
    binf  = fetch("yolo-fastestv2-opt.bin",   FILES["yolo-fastestv2-opt.bin"])
    names = fetch("coco.names",               FILES["coco.names"])
    print("\nDone. Smoke-test the sidecar (this model's input is 352):")
    print(f"  YF_PARAM={param} YF_BIN={binf} YF_INPUT=352 \\")
    print(f"    YOLO_LABELS={names} YOLO_SOURCE=rpicam \\")
    print(f"    python3 yolofastest_ncnn_sidecar.py")
    print("  (USB cam: YOLO_SOURCE=0 · sanity-check first with the same line + --inspect)")


if __name__ == "__main__":
    main()
