#!/usr/bin/env python3
r"""
Train NanoDet-Plus on your dataset AND auto-export an optimised NCNN model — one
command, same paradigm as train_yolofastest.py.

>>> Edit the CONFIG block below and run:   python train_nanodet.py
    (No environment variables needed. Windows-friendly.)
    Prefer clicking? Run the GUI:           python train_nanodet_gui.py
    (every field below — including fine-tuning — set in a window, live log under it.)

Fine-tuning (дообучение): point WEIGHTS at a finished .ckpt to continue a model on
new/expanded data, or RESUME at an interrupted run. Both also drivable from the GUI.

What it does: converts your YOLO dataset to COCO JSON (no image copying), clones
RangiLyu/nanodet, writes a custom config off the stock nanodet-plus-m, trains, then
runs export_ncnn.py → a verified, fp16 .param/.bin for the NanoDet sidecar.

Why NanoDet-Plus over YOLO-FastestV2: an FPN over 3-4 strides (8/16/32/64) means
much better SMALL-object detection, while still CPU-real-time on a Pi 5. It's plain
PyTorch + PyTorch-Lightning, so a cu128 nightly trains it on a Blackwell GPU (unlike
PaddleDetection/PicoDet, which won't run on Blackwell at all).

Install (Windows, in your venv):
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install opencv-python numpy onnx onnxsim ncnn pnnx pytorch-lightning pycocotools omegaconf

⚠️ This is an orchestration wrapper around the upstream repo. NanoDet's config schema
   is stable but not frozen — the trainer reports every field it patches; if it warns
   that a key wasn't found, open the generated config and set it by hand.
"""

import os
import re
import sys
import glob
import json
import shutil
import subprocess

# Windows consoles default to a legacy code page (e.g. cp1251) that can't encode the
# arrows/emoji we print -> UnicodeEncodeError. Replace un-encodable chars instead of
# crashing training. (Also covers export_ncnn.py, imported in-process below.)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

# ========================= CONFIG — EDIT ME ==================================
DATASET    = r"C:\Users\dest\Desktop\test\merged_dataset"   # YOLO root (images/{train,val} + labels/{train,val})
CLASSES    = ["Birds", "Drones", "Dron2"]                   # in YOLO id order

INPUT      = 416            # square input (NanoDet-Plus-m default 416; 320 = faster, less small-object reach)
EPOCHS     = 200            # NanoDet converges faster than from-scratch YOLO; 100-300 typical
REG_MAX    = 7             # DFL bins per side − 1 (nanodet-plus default 7) — keep unless you change the head
LR         = None           # learning rate; None = keep the stock config's. Fine-tuning wants a lower one (e.g. 0.0005)

# ── Fine-tuning / resuming (дообучение) ───────────────────────────────────────
#   WEIGHTS — load these weights as a STARTING point and train fresh on your data
#             (transfer-learning / continue a finished model on new/expanded data).
#   RESUME  — continue an INTERRUPTED run, keeping optimizer state + epoch counter.
#   Give a .ckpt path to either (leave "" to train from scratch). WEIGHTS is the
#   usual "дообучение" lever; RESUME is for picking a crashed run back up.
WEIGHTS    = ""             # e.g. r"C:\...\nanodet\workspace\custom\model_best\model_best.ckpt"
RESUME     = ""             # e.g. r"C:\...\nanodet\workspace\custom\model_last.ckpt"

# ── Hardware (Ultra 9 285K + RTX 5090 32GB + 128GB) ───────────────────────────
DEVICE     = "gpu"          # "gpu" (RTX 5090, cu128 nightly torch) or "cpu"
GPU_IDS    = [0]
BATCH      = 96             # nanodet-plus-m is small; 96-160 fits 32GB. OOM? lower.
WORKERS    = 20             # dataloader workers per GPU (285K = 24 cores)

# ── Plumbing ──────────────────────────────────────────────────────────────────
OUT        = "nd_data"                 # where the COCO .json files are written
REPO_DIR   = "nanodet"                 # where RangiLyu/nanodet is cloned
BASE_CFG   = "config/nanodet-plus-m_416.yml"   # stock config to start from
EXPORT     = True                      # after training, auto-export a verified ncnn model
OUT_STEM   = "nanodet"                 # output stem for the exported .param/.bin
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
REG_MAX = _envc("TRAIN_REG_MAX", REG_MAX, int)
BATCH   = _envc("TRAIN_BATCH", BATCH, int)
WORKERS = _envc("TRAIN_WORKERS", WORKERS, int)
DEVICE  = _envc("TRAIN_DEVICE", DEVICE)
WEIGHTS = _envc("TRAIN_WEIGHTS", WEIGHTS)
RESUME  = _envc("TRAIN_RESUME", RESUME)
if os.environ.get("TRAIN_LR"):
    LR = float(os.environ["TRAIN_LR"])
if os.environ.get("TRAIN_GPU_IDS"):
    GPU_IDS = [int(x) for x in re.split(r"[,\s]+", os.environ["TRAIN_GPU_IDS"].strip()) if x]
if os.environ.get("TRAIN_EXPORT"):
    EXPORT = os.environ["TRAIN_EXPORT"].lower() not in ("0", "false", "no", "off")

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def sh(cmd, cwd=None, extra_env=None):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    e = dict(os.environ)
    if extra_env:
        e.update(extra_env)
    return subprocess.call([str(c) for c in cmd], cwd=cwd, env=e)


def setup_repo_env():
    """Install the cloned nanodet's dependencies — WITHOUT clobbering your torch — so the
    next steps don't die on a missing sub-module or the wrong pytorch-lightning.

    `tools/train.py` does `import nanodet`; that's handled by PYTHONPATH=<repo> (set on the
    training call), so we don't pip-install the package itself (avoids a setup.py build).
    Here we only install the repo's requirements EXCEPT torch/vision/audio, which pulls the
    right pytorch-lightning/omegaconf/tabulate/… versions while keeping your cu128 nightly
    torch. Skip the whole thing with TRAIN_PIP=0."""
    if os.environ.get("TRAIN_PIP", "1").lower() in ("0", "false", "no", "off"):
        print("  (TRAIN_PIP=0 — skipping auto dep install; relying on PYTHONPATH + your env)")
        return
    req = os.path.join(os.path.abspath(REPO_DIR), "requirements.txt")
    if not os.path.isfile(req):
        return
    pkgs = []
    for line in open(req, encoding="utf-8"):
        spec = line.strip()
        if not spec or spec.startswith(("#", "-")):
            continue
        name = re.split(r"[<>=!~;\[ ]", spec)[0].strip().lower()
        if name in ("torch", "torchvision", "torchaudio"):
            continue                                  # keep the user's cu128 nightly build
        pkgs.append(spec)
    if pkgs:
        print(f"[2b/4] Installing nanodet deps (minus torch): {', '.join(pkgs)}")
        sh([sys.executable, "-m", "pip", "install", *pkgs])


def _has_imgs(d):
    return os.path.isdir(d) and any(
        glob.glob(os.path.join(d, e)) or glob.glob(os.path.join(d, "**", e), recursive=True)
        for e in IMG_EXT)


def _split_dir(split):
    """Find the images dir for a split across common YOLO layouts."""
    for c in (os.path.join(DATASET, "images", split),
              os.path.join(DATASET, split, "images"),
              os.path.join(DATASET, split)):
        if _has_imgs(c):
            return c
    return None


def _list_images(d):
    files = []
    for e in IMG_EXT:
        files += glob.glob(os.path.join(d, e))
    return sorted(set(os.path.abspath(f) for f in files))


def _img_size(path):
    """(w, h) without fully decoding when possible (PIL), else OpenCV."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        import cv2
        im = cv2.imread(path)
        if im is None:
            return None
        return im.shape[1], im.shape[0]


def _label_path(img_path):
    return os.path.splitext(img_path)[0].replace("images", "labels") + ".txt"


def build_coco(split):
    """YOLO → COCO JSON (no image copy). Returns (img_dir, json_path, n_imgs)."""
    d = _split_dir(split)
    if not d:
        return None, None, 0
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
    img_id, ann_id, n_obj = 1, 1, 0
    for img in _list_images(d):
        wh = _img_size(img)
        if not wh:
            continue
        w, h = wh
        coco["images"].append({"id": img_id, "file_name": os.path.basename(img), "width": w, "height": h})
        lp = _label_path(img)
        if os.path.isfile(lp):
            for line in open(lp):
                p = line.split()
                if len(p) < 5:
                    continue
                cls = int(float(p[0])); xc, yc, bw, bh = (float(x) for x in p[1:5])
                x = (xc - bw / 2) * w; y = (yc - bh / 2) * h; bw *= w; bh *= h
                if bw <= 1 or bh <= 1:
                    continue
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cls,
                    "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
                ann_id += 1; n_obj += 1
        img_id += 1
    os.makedirs(OUT, exist_ok=True)
    jp = os.path.abspath(os.path.join(OUT, f"instances_{split}.json"))
    json.dump(coco, open(jp, "w"))
    print(f"  {split}: {img_id - 1} images, {n_obj} objects  (img_path={d})")
    return os.path.abspath(d), jp, img_id - 1


def write_config(train_img, train_json, val_img, val_json):
    """Copy the stock nanodet-plus config and patch only the dataset/class/schedule
    fields. Reports each patch; warns on any key it couldn't find (schema drift)."""
    src = os.path.join(REPO_DIR, BASE_CFG)
    if not os.path.isfile(src):
        sys.exit(f"ERROR: base config not found: {src}\n  (check BASE_CFG / the clone)")
    s = open(src, encoding="utf-8").read()

    def sub(pattern, repl, n, what, flags=0):
        nonlocal s
        s, c = re.subn(pattern, repl, s, count=n, flags=flags)
        print(f"  patch {what}: {c} site(s)" + ("  [!] NOT FOUND" if c == 0 else ""))

    names = ", ".join(f"'{c}'" for c in CLASSES)
    sub(r"(?m)^save_dir:.*$", "save_dir: workspace/custom", 1, "save_dir")
    sub(r"num_classes:\s*\d+", f"num_classes: {len(CLASSES)}", 0, "num_classes")
    sub(r"(?m)^class_names:.*$", f"class_names: &class_names [{names}]", 1, "class_names")
    # img_path / ann_path appear train-then-val; replace the first two of each in order.
    img_it = iter([train_img, val_img]); ann_it = iter([train_json, val_json])
    sub(r"(?m)^(\s*img_path:\s*).*$", lambda m: m.group(1) + next(img_it), 2, "img_path (train,val)")
    sub(r"(?m)^(\s*ann_path:\s*).*$", lambda m: m.group(1) + next(ann_it), 2, "ann_path (train,val)")
    sub(r"input_size:\s*\[\s*\d+\s*,\s*\d+\s*\]", f"input_size: [{INPUT}, {INPUT}]", 0, "input_size")
    sub(r"gpu_ids:\s*\[.*?\]", f"gpu_ids: {GPU_IDS if DEVICE == 'gpu' else []}", 0, "gpu_ids")
    sub(r"workers_per_gpu:\s*\d+", f"workers_per_gpu: {WORKERS}", 0, "workers_per_gpu")
    sub(r"batchsize_per_gpu:\s*\d+", f"batchsize_per_gpu: {BATCH}", 0, "batchsize_per_gpu")
    sub(r"total_epochs:\s*\d+", f"total_epochs: {EPOCHS}", 0, "total_epochs")
    if LR is not None:                       # first lr: under schedule.optimizer
        sub(r"(?m)^(\s*lr:\s*).*$", lambda m: m.group(1) + repr(float(LR)), 1, "optimizer lr")

    # Fine-tuning / resume: nanodet reads schedule.load_model (transfer weights, fresh
    # schedule) and schedule.resume (continue a run). The stock config ships them as
    # commented placeholders; drop any existing copy and re-insert under `schedule:`.
    def set_schedule_key(key, value):
        nonlocal s
        s = re.sub(rf"(?m)^[ \t]*#?[ \t]*{key}:.*$\n?", "", s)          # strip old/commented
        ins = f'\\g<1>\n  {key}: "{value}"'                             # quote: YAML-safe paths
        s, c = re.subn(r"(?m)^(schedule:)[ \t]*$", ins, s, count=1)     # insert right under schedule:
        print(f"  patch schedule.{key}: {value}" + ("  [!] 'schedule:' not found" if c == 0 else ""))
    if WEIGHTS:
        set_schedule_key("load_model", os.path.abspath(WEIGHTS).replace("\\", "/"))
    if RESUME:
        set_schedule_key("resume", os.path.abspath(RESUME).replace("\\", "/"))

    os.makedirs(OUT, exist_ok=True)
    out_cfg = os.path.abspath(os.path.join(OUT, "custom.yml"))
    open(out_cfg, "w", encoding="utf-8").write(s)
    print(f"  config -> {out_cfg}")
    return out_cfg


def main():
    if sys.version_info >= (3, 13):
        print(f"  [!] You're on Python {sys.version_info.major}.{sys.version_info.minor}. The training "
              "stack (torch, pytorch-lightning, pycocotools) often has NO wheels yet for such a new\n"
              "      Python, so installs silently land in the wrong place / fail to build. If anything\n"
              "      below errors on imports, use Python 3.10 or 3.11 for the training box (see README).")
    print("[1/4] Converting dataset to COCO...")
    if not os.path.isdir(DATASET):
        sys.exit(f"ERROR: DATASET not found: {DATASET}")
    tr_img, tr_json, n_tr = build_coco("train")
    va_img, va_json, n_va = build_coco("val")
    if n_tr == 0:
        sys.exit(f"ERROR: no training images under {DATASET} (looked in images/train, train/images, train)")
    if n_va == 0:                       # NanoDet needs a val set; fall back to train
        print("  WARNING: no val split found — using train as val (metrics will be optimistic)")
        va_img, va_json = tr_img, tr_json

    print("[2/4] Getting the repo (clone if missing)...")
    if not os.path.isdir(REPO_DIR):
        if sh(["git", "clone", "--depth", "1", "https://github.com/RangiLyu/nanodet.git", REPO_DIR]):
            sys.exit("ERROR: git clone failed")
    setup_repo_env()                         # make `import nanodet` work + deps (no torch touch)
    cfg = write_config(tr_img, tr_json, va_img, va_json)

    mode = (f"fine-tune from {os.path.basename(WEIGHTS)}" if WEIGHTS
            else f"resume {os.path.basename(RESUME)}" if RESUME else "from scratch")
    print(f"[3/4] Training on {DEVICE.upper()} ({mode}; batch={BATCH}, workers={WORKERS}, epochs={EPOCHS})...")
    # PYTHONPATH=<repo> guarantees `import nanodet` resolves to the cloned repo even if
    # the pip step was skipped/offline or a different 'nanodet' is installed elsewhere.
    train_env = {"PYTHONPATH": os.path.abspath(REPO_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if sh([sys.executable, "tools/train.py", cfg], cwd=REPO_DIR, extra_env=train_env):
        sys.exit("ERROR: training failed (see output above). If it's a pytorch-lightning API\n"
                 "  error, the auto-installed nanodet deps set the right PL version — re-run; or\n"
                 "  pip install -r nanodet/requirements.txt (keep your torch). Check config warnings.")

    if EXPORT:
        print("[4/4] Exporting an optimised, verified NCNN model...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import export_ncnn
            res = export_ncnn.run_export(REPO_DIR, cfg, OUT_STEM, INPUT, REG_MAX, classes=len(CLASSES))
        except Exception as e:
            res = None; print(f"  export step errored: {e}")
        if res:
            param, binf = res
            names = os.path.abspath(os.path.join(OUT, "classes.txt"))
            open(names, "w").write("\n".join(CLASSES) + "\n")
            print("\n[OK] Done - trained AND exported. Run it:")
            print(f"  Pi sidecar:  ND_PARAM={param} ND_BIN={binf} ND_INPUT={INPUT} \\")
            print(f"               YOLO_LABELS={names} python3 nanodet_ncnn_sidecar.py --inspect")
            print(f"  Phone:       NanoDet needs an Android decoder (GFL/DFL) - Pi-only for now.")
            return
        print("  Auto-export didn't complete - run it manually:")
    else:
        print("[4/4] Export (EXPORT=False) - run manually:")
    print(f"  python export_ncnn.py --repo {REPO_DIR} --cfg {cfg} --out {OUT_STEM} "
          f"--input {INPUT} --reg-max {REG_MAX} --classes {len(CLASSES)}")


if __name__ == "__main__":
    main()
