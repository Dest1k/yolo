#!/usr/bin/env python3
r"""
Train YOLO-FastestV2 on your dataset and export it for the NCNN sidecar.

>>> Everything is configured in the CONFIG block below — edit it and run:
        python train_yolofastest.py
    (No environment variables. Windows-friendly.)

What it does: builds the .data/.names/file-lists from your YOLO dataset, clones the
upstream repo (dog-qiuqiu/Yolo-FastestV2), computes anchors for your data, patches
the repo's trainer for high hardware utilisation, trains, and prints the
ONNX->NCNN export steps.

It's plain PyTorch and light. A recent CUDA-12.8 PyTorch trains it on an RTX 50xx
(Blackwell) GPU; CPU works too. Install (Windows, in your venv):
    # GPU (RTX 50xx/Blackwell):
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    # or CPU only:
    pip install torch torchvision
    pip install opencv-python numpy tqdm onnx onnxsim

⚠️ Export ONNX with opset 11 (the repo's pytorch2onnx.py) — onnx2ncnn needs it.
   A new torch may force opset 18; pass dynamo=False, or export in a stable CPU
   torch venv. (See the README.)
"""

import os
import re
import sys
import glob
import subprocess

# ========================= CONFIG — EDIT ME ==================================
DATASET    = r"C:\Users\dest\Desktop\test\merged_dataset"   # YOLO dataset root (images/{train,val}, labels/{train,val})
CLASSES    = ["Birds", "Drones", "Dron2"]                   # in YOLO id order

INPUT      = 352            # train/infer square size (YOLO-FastestV2 default 352)
EPOCHS     = 300            # training epochs (repo default 300)
LR         = 0.001          # learning rate

# ── Hardware utilisation (tuned for Ultra 9 275HX + RTX 5080 16GB + 64GB) ──────
DEVICE     = "gpu"          # "gpu" (RTX 5080, needs cu128 torch) or "cpu"
BATCH      = 96             # tiny model → big batch is fine on 16GB. Underused GPU? raise to 128/192. OOM? lower to 64/48.
WORKERS    = 16             # dataloader workers — THE main lever for a tiny model (24-thread CPU; 16 leaves headroom)
CUDNN_BENCHMARK = True      # fixed input size → let cuDNN pick the fastest kernels

# ── Plumbing ──────────────────────────────────────────────────────────────────
OUT        = "yf_data"                 # where the .data/lists are written
REPO_DIR   = "Yolo-FastestV2"          # where the upstream repo is cloned
GENANCHORS = True                      # recompute anchors for your data (recommended)
ANCHORS    = "12.64,19.39, 37.88,51.48, 55.71,138.31, 126.91,78.23, 131.57,214.55, 279.92,258.87"
# =============================================================================


def sh(cmd, cwd=None, extra_env=None):
    print(f"  $ {' '.join(cmd)}")
    e = dict(os.environ)
    if extra_env:
        e.update(extra_env)
    return subprocess.call(cmd, cwd=cwd, env=e)


def list_images(split):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files += glob.glob(os.path.join(DATASET, "images", split, ext))
    return sorted(os.path.abspath(f) for f in files)


def build_data():
    if not os.path.isdir(DATASET):
        sys.exit(f"ERROR: DATASET not found: {DATASET}")
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "val"):
        imgs = list_images(split)
        if not imgs:
            print(f"  WARNING: no images in {DATASET}/images/{split}")
        with open(os.path.join(OUT, f"{split}.txt"), "w") as f:
            f.write("\n".join(imgs) + ("\n" if imgs else ""))
        print(f"  {split}: {len(imgs)} images")
    names_path = os.path.abspath(os.path.join(OUT, "custom.names"))
    with open(names_path, "w") as f:
        f.write("\n".join(CLASSES) + "\n")
    data_path = os.path.abspath(os.path.join(OUT, "custom.data"))
    with open(data_path, "w") as f:
        f.write(f"""[name]
model_name=custom

[train-configure]
epochs={EPOCHS}
steps={int(EPOCHS * 0.6)},{int(EPOCHS * 0.85)}
batch_size={BATCH}
subdivisions=1
learning_rate={LR}

[model-configure]
pre_weights=None
classes={len(CLASSES)}
width={INPUT}
height={INPUT}
anchor_num=3
anchors={ANCHORS}

[data-configure]
train={os.path.abspath(os.path.join(OUT, 'train.txt'))}
val={os.path.abspath(os.path.join(OUT, 'val.txt'))}
names={names_path}
""")
    print(f"  data -> {data_path}  (classes={CLASSES}, batch={BATCH}, epochs={EPOCHS})")
    return data_path


def patch_train(repo):
    """Bump dataloader workers + enable cuDNN autotune in the repo's trainer for
    higher utilisation. Best-effort and version-tolerant."""
    p = os.path.join(repo, "train.py")
    if not os.path.isfile(p):
        print("  (train.py not found to patch — using repo defaults)")
        return
    s = open(p, encoding="utf-8").read(); orig = s
    s, n = re.subn(r"num_workers\s*=\s*\d+", f"num_workers={WORKERS}", s)
    if CUDNN_BENCHMARK and "cudnn.benchmark" not in s:
        s = re.sub(r"(^import torch\b.*$)", r"\1\ntorch.backends.cudnn.benchmark = True",
                   s, count=1, flags=re.M)
    if s != orig:
        open(p, "w", encoding="utf-8").write(s)
        print(f"  patched train.py: num_workers={WORKERS} (x{n}), cudnn.benchmark={CUDNN_BENCHMARK}")


def patch_anchors(data_path, repo):
    if not (GENANCHORS and os.path.isfile(os.path.join(repo, "genanchors.py"))):
        return
    print("Computing anchors for your data (genanchors)…")
    train_txt = os.path.abspath(os.path.join(OUT, "train.txt"))
    if sh([sys.executable, "genanchors.py", "--traintxt", train_txt], cwd=repo) != 0:
        return
    anc = os.path.join(repo, "anchors6.txt")
    if not os.path.isfile(anc):
        return
    line = open(anc).readline().strip()
    if not line:
        return
    nums = re.split(r"[,\s]+", line)
    val = ", ".join(nums)
    s = open(data_path, encoding="utf-8").read()
    s = re.sub(r"(?m)^(anchors\s*=).*", rf"\g<1>{val}", s)
    open(data_path, "w", encoding="utf-8").write(s)
    print(f"  patched anchors into {data_path}")


def main():
    print("[1/4] Building .data from CONFIG…")
    data_path = build_data()

    print("[2/4] Getting the repo (clone if missing)…")
    if not os.path.isdir(REPO_DIR):
        if sh(["git", "clone", "--depth", "1", "https://github.com/dog-qiuqiu/Yolo-FastestV2.git", REPO_DIR]):
            sys.exit("ERROR: git clone failed")
    patch_anchors(data_path, REPO_DIR)
    patch_train(REPO_DIR)

    print(f"[3/4] Training on {DEVICE.upper()} (batch={BATCH}, workers={WORKERS})…")
    extra_env = {"CUDA_VISIBLE_DEVICES": ""} if DEVICE == "cpu" else None
    if sh([sys.executable, "train.py", "--data", data_path], cwd=REPO_DIR, extra_env=extra_env):
        sys.exit("ERROR: training failed (see output above)")

    print("[4/4] Export to NCNN (run these after training):")
    print(f"  python pytorch2onnx.py --data {data_path} --weights <best>.pth   # opset 11! (dynamo=False)")
    print("  python -m onnxsim model.onnx model-sim.onnx")
    print("  onnx2ncnn model-sim.onnx yolofastestv2.param yolofastestv2.bin")
    print("\nThen on the Pi:")
    print("  YF_PARAM=yolofastestv2.param YF_BIN=yolofastestv2.bin YOLO_LABELS=yf_data/custom.names \\")
    print("    python3 yolofastest_ncnn_sidecar.py --inspect")


if __name__ == "__main__":
    main()
