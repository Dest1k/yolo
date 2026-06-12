#!/usr/bin/env python3
"""
Fetch a COCO-pretrained PicoDet and convert it to NCNN for a first smoke-test of the
sidecar — before you train your own. No GPU and no PaddlePaddle *runtime* needed:
PaddleDetection ships a downloadable inference model, and paddle2onnx converts it on
CPU.

    pip install paddle2onnx
    pip install pnnx        # or have onnx2ncnn on PATH
    python get_model.py
    # then it prints the line to run the sidecar.

Pulls picodet_s_320_coco_lcnet (COCO 80, 320 input) from PaddleDetection's bcebos
host → ONNX (opset 11) → ncnn. PicoDet's exact output layout depends on the export,
so verify with `--inspect` and tune the sidecar's PICODET_* env if boxes look off.
"""
import os
import sys
import shutil
import tarfile
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TAR_URL = "https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_320_coco_lcnet.tar"
NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
STEM = "picodet_s_320"


def fetch(url, dst, timeout=180):
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  have {os.path.basename(dst)}"); return dst
    print(f"  downloading {os.path.basename(dst)} …")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    return dst


def main():
    print("Fetching PicoDet (COCO) inference model…")
    tar = os.path.join(HERE, "picodet_s_320_coco_lcnet.tar")
    try:
        fetch(TAR_URL, tar)
        fetch(NAMES_URL, os.path.join(HERE, "coco.names"))
    except Exception as e:
        sys.exit(f"ERROR: download failed\n  {e}")
    mdl_dir = os.path.join(HERE, "picodet_s_320_coco_lcnet")
    with tarfile.open(tar) as t:
        t.extractall(HERE)
    # locate the paddle inference files
    pdmodel = next((os.path.join(r, f) for r, _, fs in os.walk(mdl_dir) for f in fs
                    if f.endswith(".pdmodel")), None)
    if not pdmodel:
        sys.exit("ERROR: no .pdmodel found in the downloaded archive")
    src_dir = os.path.dirname(pdmodel)

    onnx = os.path.join(HERE, STEM + ".onnx")
    if not shutil.which("paddle2onnx") and not _module("paddle2onnx"):
        sys.exit("Need paddle2onnx for the Paddle→ONNX step:  pip install paddle2onnx")
    print("  paddle2onnx → ONNX (opset 11)…")
    rc = subprocess.call([sys.executable, "-m", "paddle2onnx",
                          "--model_dir", src_dir,
                          "--model_filename", os.path.basename(pdmodel),
                          "--params_filename", os.path.basename(pdmodel).replace(".pdmodel", ".pdiparams"),
                          "--opset_version", "11", "--save_file", onnx])
    if rc != 0 or not os.path.isfile(onnx):
        sys.exit("ERROR: paddle2onnx failed")

    param = os.path.join(HERE, STEM + ".param"); binf = os.path.join(HERE, STEM + ".bin")
    if shutil.which("onnx2ncnn"):
        subprocess.call(["onnx2ncnn", onnx, param, binf])
    elif shutil.which("pnnx"):
        subprocess.call(["pnnx", onnx, "inputshape=[1,3,320,320]"])
        p, b = onnx[:-5] + ".ncnn.param", onnx[:-5] + ".ncnn.bin"
        if os.path.isfile(p): shutil.copyfile(p, param); shutil.copyfile(b, binf)
    else:
        sys.exit("Need an ONNX→ncnn converter:  pip install pnnx  (or onnx2ncnn on PATH)\n"
                 f"  (ONNX is ready at {onnx})")

    if not os.path.isfile(param):
        sys.exit("ERROR: ncnn conversion produced no .param")
    print("\nDone. Smoke-test (verify the layout first — PicoDet exports vary):")
    print(f"  PICODET_PARAM={param} PICODET_BIN={binf} \\")
    print(f"    YOLO_LABELS={os.path.join(HERE, 'coco.names')} YOLO_SOURCE=rpicam \\")
    print(f"    python3 picodet_ncnn_sidecar.py --inspect")


def _module(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":
    main()
