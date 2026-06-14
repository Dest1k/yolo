#!/usr/bin/env python3
r"""
Train YOLO-FastestV2 on your dataset AND auto-export an optimised NCNN model — one
command, no manual onnx2ncnn dance.

>>> Everything is configured in the CONFIG block below — edit it and run:
        python train_yolofastest.py
    (No environment variables. Windows-friendly.)

What it does: builds the .data/.names/file-lists from your YOLO dataset, clones the
upstream repo (dog-qiuqiu/Yolo-FastestV2), computes anchors for your data, patches
the repo's trainer for high hardware utilisation, trains, then runs export_ncnn.py
to produce a verified, fp16-optimised .param/.bin ready for the Pi sidecar and the
phone app. (Set EXPORT=False to stop after training.)

It's plain PyTorch and light. A recent CUDA-12.8 PyTorch trains it on an RTX 50xx
(Blackwell) GPU; CPU works too. Install (Windows, in your venv):
    # GPU (RTX 50xx/Blackwell):
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    # or CPU only:  pip install torch torchvision
    pip install opencv-python numpy tqdm onnx onnxsim ncnn pnnx

The export forces opset 11 + dynamo=False regardless of your torch version (so the
cu128 nightly's opset-18 default can't break onnx2ncnn), prefers onnx2ncnn+
ncnnoptimize if on PATH else pnnx, and loads the result back to verify the head
blobs actually extract. See export_ncnn.py / the README for details.
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

# ── Hardware utilisation (tuned for Ultra 9 285K + RTX 5090 32GB + 128GB) ──────
DEVICE     = "gpu"          # "gpu" (RTX 5090, needs cu128 nightly torch) or "cpu"
BATCH      = 192            # tiny model → big batch is fine on 32GB. Underused GPU? raise to 256/384. OOM? lower.
WORKERS    = 20             # dataloader workers — THE main lever for a tiny model (285K = 24 cores; 20 leaves headroom)
CUDNN_BENCHMARK = True      # fixed input size → let cuDNN pick the fastest kernels

# ── Plumbing ──────────────────────────────────────────────────────────────────
OUT        = "yf_data"                 # where the .data/lists are written
REPO_DIR   = "Yolo-FastestV2"          # where the upstream repo is cloned
EXPORT     = True                      # after training, auto-export an optimised, verified ncnn model
OUT_STEM   = "yolofastestv2"           # output stem for the exported .param/.bin
GENANCHORS = False                     # recompute anchors for your data. Off first (default
                                       # anchors work fine); turn True once training runs to optimise.
ANCHORS    = "12.64,19.39, 37.88,51.48, 55.71,138.31, 126.91,78.23, 131.57,214.55, 279.92,258.87"
# =============================================================================

# ── Env overrides — so the command center (or any launcher) can drive training
#    without editing this file. Same TRAIN_* keys for every trainer. ───────────
def _envc(name, cur, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else cur
DATASET = _envc("TRAIN_DATASET", DATASET)
if os.environ.get("TRAIN_CLASSES"):
    CLASSES = [c.strip() for c in os.environ["TRAIN_CLASSES"].split(",") if c.strip()]
INPUT   = _envc("TRAIN_INPUT", INPUT, int)
EPOCHS  = _envc("TRAIN_EPOCHS", EPOCHS, int)
BATCH   = _envc("TRAIN_BATCH", BATCH, int)
WORKERS = _envc("TRAIN_WORKERS", WORKERS, int)
DEVICE  = _envc("TRAIN_DEVICE", DEVICE)
if os.environ.get("TRAIN_EXPORT"):
    EXPORT = os.environ["TRAIN_EXPORT"].lower() not in ("0", "false", "no", "off")


def sh(cmd, cwd=None, extra_env=None):
    print(f"  $ {' '.join(cmd)}")
    e = dict(os.environ)
    if extra_env:
        e.update(extra_env)
    return subprocess.call(cmd, cwd=cwd, env=e)


IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def _has_imgs(d):
    if not os.path.isdir(d):
        return False
    for e in IMG_EXT:
        if glob.glob(os.path.join(d, e)) or glob.glob(os.path.join(d, "**", e), recursive=True):
            return True
    return False


def _split_dir(split):
    """Find the images dir for a split across common YOLO layouts."""
    for c in (os.path.join(DATASET, "images", split),   # images/train
              os.path.join(DATASET, split, "images"),   # train/images   <-- your layout
              os.path.join(DATASET, split)):            # train/
        if _has_imgs(c):
            return c
    return None


def list_images(split):
    d = _split_dir(split)
    if not d:
        return []
    files = []
    for e in IMG_EXT:
        files += glob.glob(os.path.join(d, e))
        files += glob.glob(os.path.join(d, "**", e), recursive=True)
    return sorted(set(os.path.abspath(f) for f in files))


def build_data():
    if not os.path.isdir(DATASET):
        sys.exit(f"ERROR: DATASET not found: {DATASET}")
    os.makedirs(OUT, exist_ok=True)
    counts = {}
    for split in ("train", "val"):
        imgs = list_images(split)
        counts[split] = len(imgs)
        d = _split_dir(split)
        print(f"  {split}: {len(imgs)} images" + (f"  (from {d})" if d else ""))
        with open(os.path.join(OUT, f"{split}.txt"), "w") as f:
            f.write("\n".join(imgs) + ("\n" if imgs else ""))
        if imgs and "images" not in imgs[0].replace("\\", "/"):
            print("  WARNING: image paths have no 'images' folder — the repo derives label\n"
                  "           paths by replacing 'images'->'labels'; labels may not resolve.")
    if counts["train"] == 0:
        top = ", ".join(sorted(os.listdir(DATASET))[:20])
        sys.exit(
            f"ERROR: no training images found under {DATASET}\n"
            f"  looked in: images/train, train/images, train\n"
            f"  {DATASET} contains: {top}\n"
            f"  Expected a YOLO layout with labels parallel to images (images->labels).")
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


def patch_datasets(repo):
    """Fix the repo's label-path derivation in utils/datasets.py. Different
    forks/versions are broken in different ways (an unbound `line`, a hardcoded
    `.jpg`, or labels looked up next to the image instead of in a parallel labels/
    folder). Replace it with one robust line that works for any extension and the
    standard images/ + labels/ layout, derived from self.data_list[index]."""
    p = os.path.join(repo, "utils", "datasets.py")
    if not os.path.isfile(p):
        print("  (utils/datasets.py not found to patch)")
        return
    s = open(p, encoding="utf-8").read(); orig = s
    robust = ('label_path = os.path.splitext(self.data_list[index])[0]'
              '.replace("images", "labels") + ".txt"')
    s, n = re.subn(r"label_path\s*=\s*.*", robust, s, count=1)
    if "import os" not in s:
        s = "import os\n" + s
    if s != orig and n:
        open(p, "w", encoding="utf-8").write(s)
        print(f"  patched utils/datasets.py: robust label path (images->labels, any ext)")
    elif not n:
        print("  WARNING: couldn't find label_path line in utils/datasets.py to patch")


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
    patch_datasets(REPO_DIR)

    print(f"[3/4] Training on {DEVICE.upper()} (batch={BATCH}, workers={WORKERS})…")
    extra_env = {"CUDA_VISIBLE_DEVICES": ""} if DEVICE == "cpu" else None
    if sh([sys.executable, "train.py", "--data", data_path], cwd=REPO_DIR, extra_env=extra_env):
        sys.exit("ERROR: training failed (see output above)")

    if EXPORT:
        print("[4/4] Exporting an optimised, verified NCNN model…")
        # Same-folder import so it works wherever the script is run from.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import export_ncnn
            res = export_ncnn.run_export(REPO_DIR, data_path, OUT_STEM, INPUT)
        except Exception as e:
            res = None
            print(f"  export step errored: {e}")
        if res:
            param, binf = res
            names = os.path.abspath(os.path.join(OUT, "custom.names"))
            print("\n✅ Done — model trained AND exported. Run it:")
            print(f"  Pi sidecar:  YF_PARAM={param} YF_BIN={binf} YF_INPUT={INPUT} \\")
            print(f"               YOLO_LABELS={names} python3 yolofastest_ncnn_sidecar.py --inspect")
            print(f"  Phone:       load the .param/.bin, version=FastestV2, input={INPUT}, "
                  f"classes from custom.names")
            return
        print("  Auto-export didn't complete — fall back to the manual steps:")
    else:
        print("[4/4] Export to NCNN (EXPORT=False — run these manually):")
    print(f"  python export_ncnn.py --repo {REPO_DIR} --data {data_path} --out {OUT_STEM} --input {INPUT}")
    print("  (or the long way: pytorch2onnx.py [opset 11] → onnxsim → onnx2ncnn → ncnnoptimize)")


if __name__ == "__main__":
    main()
