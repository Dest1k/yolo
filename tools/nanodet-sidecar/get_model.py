#!/usr/bin/env python3
"""
Fetch a COCO-pretrained NanoDet-Plus and convert it to NCNN for a first smoke-test
of the sidecar — fully automatic:

    python get_model.py
    # → clones nanodet, downloads the official COCO checkpoint, exports a verified
    #   .param/.bin, and prints the line to run the sidecar.

NanoDet ships no prebuilt ncnn model, but it does publish the COCO checkpoint as a
stable GitHub release asset (no Google Drive), so this needs no manual downloads.
Requires torch + a converter (onnx2ncnn or `pip install pnnx`) on this machine — the
same env you'd train in. Override with --ckpt/--cfg/--repo if you already have them.
"""
import argparse
import os
import sys
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/RangiLyu/nanodet.git"
# Official COCO pretrain checkpoint (trained with config/nanodet-plus-m_416.yml).
CKPT_URL = ("https://github.com/RangiLyu/nanodet/releases/download/"
            "v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt")


def fetch(url, dst, timeout=300):
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst)} B)"); return dst
    print(f"  downloading {os.path.basename(dst)} …")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if not os.path.getsize(dst):
        raise RuntimeError(f"downloaded 0 bytes from {url}")
    print(f"    → {dst} ({os.path.getsize(dst)} B)")
    return dst


def main():
    ap = argparse.ArgumentParser(description="Auto-fetch + convert a COCO NanoDet-Plus to ncnn.")
    ap.add_argument("--repo", default=os.path.join(HERE, "nanodet"), help="nanodet repo dir")
    ap.add_argument("--ckpt", default=None, help="checkpoint .ckpt (default: auto-download COCO)")
    ap.add_argument("--cfg", default=None, help="config .yml (default: repo nanodet-plus-m_416.yml)")
    ap.add_argument("--input", type=int, default=416)
    ap.add_argument("--reg-max", type=int, default=7)
    a = ap.parse_args()

    if not os.path.isdir(a.repo):
        print("Cloning RangiLyu/nanodet…")
        if subprocess.call(["git", "clone", "--depth", "1", REPO_URL, a.repo]):
            sys.exit("ERROR: git clone failed")
    cfg = a.cfg or os.path.join(a.repo, "config", "nanodet-plus-m_416.yml")
    if not os.path.isfile(cfg):
        sys.exit(f"ERROR: config not found: {cfg}")

    ckpt = a.ckpt
    if not ckpt:
        try:
            ckpt = fetch(CKPT_URL, os.path.join(HERE, "nanodet-plus-m_416_checkpoint.ckpt"))
        except Exception as e:
            sys.exit(f"ERROR: could not download the COCO checkpoint\n  {e}\n"
                     f"  Grab it by hand from {CKPT_URL} and pass --ckpt.")

    sys.path.insert(0, HERE)
    import export_ncnn
    res = export_ncnn.run_export(a.repo, cfg, os.path.join(HERE, "nanodet"),
                                 a.input, a.reg_max, classes=80, ckpt=ckpt)
    if not res:
        sys.exit(1)
    param, binf = res
    print("\nDone. Smoke-test the sidecar (COCO 80 classes):")
    print(f"  ND_PARAM={param} ND_BIN={binf} ND_INPUT={a.input} \\")
    print(f"    YOLO_SOURCE=rpicam python3 nanodet_ncnn_sidecar.py --inspect")
    print("  (USB cam: YOLO_SOURCE=0)")


if __name__ == "__main__":
    main()
