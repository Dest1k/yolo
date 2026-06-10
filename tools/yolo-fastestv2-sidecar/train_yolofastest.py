#!/usr/bin/env python3
"""
Train YOLO-FastestV2 on your dataset and export it for the NCNN sidecar.

Orchestrates the upstream repo (dog-qiuqiu/Yolo-FastestV2): clones it, computes
anchors for your data (genanchors), trains, and points you at the ONNX→NCNN export.

Good news vs PicoDet: this is plain **PyTorch** and light. It can use a GPU — and
unlike Paddle/TF, a recent PyTorch (cu128) supports Blackwell, so your RTX 5080
can actually train this. CPU works too (it's a tiny model), just slower.

Env:
  YF_DATA      path to the .data file (from make_yolofastest_data.py)   [required]
  YF_DIR       where to clone Yolo-FastestV2          (default ./Yolo-FastestV2)
  YF_GENANCHORS 1 = recompute anchors for your data and patch .data     (default 1)

Install PyTorch first (pick the build for your machine):
  # NVIDIA GPU incl. RTX 50xx/Blackwell — needs CUDA 12.8 build:
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
  # or CPU only:
  pip install torch torchvision
  pip install opencv-python numpy tqdm
"""

import os
import re
import sys
import subprocess

def env(k, d=None):
    v = os.environ.get(k)
    return v if v not in (None, "") else d

YF_DATA = env("YF_DATA")
YF_DIR = env("YF_DIR", "Yolo-FastestV2")
GENANCHORS = env("YF_GENANCHORS", "1") == "1"


def sh(cmd, cwd=None):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd)


def main():
    if not YF_DATA or not os.path.isfile(YF_DATA):
        sys.exit("ERROR: set YF_DATA to your .data file (run make_yolofastest_data.py first)")
    data_abs = os.path.abspath(YF_DATA)

    if not os.path.isdir(YF_DIR):
        print("[1/4] Cloning Yolo-FastestV2…")
        if sh(["git", "clone", "--depth", "1", "https://github.com/dog-qiuqiu/Yolo-FastestV2.git", YF_DIR]):
            sys.exit("ERROR: git clone failed")
    else:
        print(f"[1/4] Using existing repo at {YF_DIR}")

    # 2) Anchors for your data (kmeans) → patch the .data
    if GENANCHORS and os.path.isfile(os.path.join(YF_DIR, "genanchors.py")):
        print("[2/4] Computing anchors (genanchors)…")
        train_txt = _read_kv(data_abs, "train")
        if sh([sys.executable, "genanchors.py", "--traintxt", train_txt], cwd=YF_DIR) == 0:
            anc = os.path.join(YF_DIR, "anchors6.txt")
            if os.path.isfile(anc):
                line = open(anc).readline().strip()
                _patch_kv(data_abs, "anchors", line.replace(" ", ", ") if line else None)
                _write_sidecar_env(data_abs, line)
                print(f"  patched anchors → {data_abs}")
    else:
        print("[2/4] Skipping genanchors (using anchors already in .data)")

    # 3) Train
    print("[3/4] Training (PyTorch — GPU if available)…")
    if sh([sys.executable, "train.py", "--data", data_abs], cwd=YF_DIR):
        sys.exit("ERROR: training failed (see output above)")

    # 4) Export pointers
    print("[4/4] Export to NCNN:")
    print("  # 1) export the trained .pth to ONNX (repo provides test/export helpers):")
    print(f"  #    python test.py --data {data_abs} --weights <best>.pth  (or the repo's onnx export)")
    print("  # 2) simplify + convert:")
    print("  #    python -m onnxsim model.onnx model-sim.onnx")
    print("  #    onnx2ncnn model-sim.onnx yolofastestv2.param yolofastestv2.bin")
    print("\nThen on the Pi:")
    print("  YF_PARAM=yolofastestv2.param YF_BIN=yolofastestv2.bin YOLO_LABELS=custom.names \\")
    print("    python3 yolofastest_ncnn_sidecar.py --inspect   # get output blob names")
    print("  # set YF_OUTPUTS from --inspect, match YF_INPUT/anchors, then run.")


def _read_kv(data_path, key):
    for line in open(data_path):
        m = re.match(rf"\s*{key}\s*=\s*(.+)", line)
        if m:
            return m.group(1).strip()
    sys.exit(f"ERROR: '{key}=' not found in {data_path}")


def _write_sidecar_env(data_path, anchor_line):
    """Emit a sidecar_env.sh so the trained model is zero-config on the Pi:
    the anchors aren't recoverable from the model, so we hand them over here."""
    try:
        nums = [x for x in re.split(r"[,\s]+", anchor_line.strip()) if x]
        if len(nums) != 12:
            return
        inp = _read_kv(data_path, "width")
        a16 = ",".join(nums[:6]); a32 = ",".join(nums[6:])
        out = os.path.join(os.path.dirname(os.path.abspath(data_path)) or ".", "sidecar_env.sh")
        with open(out, "w") as f:
            f.write(f"# source this before running the sidecar with your trained model\n"
                    f"export YF_INPUT={inp}\n"
                    f"export YF_ANCHORS_16={a16}\n"
                    f"export YF_ANCHORS_32={a32}\n")
        print(f"  wrote sidecar env (anchors/input) → {out}")
    except Exception:
        pass


def _patch_kv(data_path, key, value):
    if not value:
        return
    lines = open(data_path).read().splitlines()
    out = []
    for line in lines:
        out.append(re.sub(rf"^(\s*{key}\s*=).*", rf"\g<1>{value}", line) if re.match(rf"\s*{key}\s*=", line) else line)
    open(data_path, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
