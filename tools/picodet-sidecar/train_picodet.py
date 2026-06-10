#!/usr/bin/env python3
"""
Train PicoDet on your dataset with PaddleDetection and export it for the NCNN
sidecar. This is an orchestration wrapper: it clones PaddleDetection, writes a
custom config that points at your COCO dataset, runs training, then exports an
inference model (→ ONNX → NCNN).

⚠️ REALITY CHECK — read before you start:
  • PicoDet training uses **PaddleDetection (PaddlePaddle)** and is HEAVY: the
    stock PicoDet configs train for ~300 epochs. On a CPU that's days-to-weeks —
    not practical. You want a supported NVIDIA GPU.
  • Your RTX 5080 is Blackwell (sm_120) — too new for PaddlePaddle's CUDA too, so
    it won't train locally either. Use **Google Colab / a cloud GPU** (or any
    older supported NVIDIA GPU). This wrapper runs the same way there.
  • Dataset must be COCO format — convert your YOLO/VOC set with `dataset_to_coco.py`.

Env knobs:
  PD_DATASET     path to the COCO dataset (output of dataset_to_coco.py)  [required]
  PD_CLASSES     comma-separated class names, in order               [required]
  PD_DIR         where to clone PaddleDetection            (default ./PaddleDetection)
  PD_BASE        base config (PicoDet variant)   (default configs/picodet/picodet_l_640_coco_lcnet.yml)
  PD_EPOCHS      epochs                                                (default 80)
  PD_BATCH       train batch size                                     (default 24 — lower on small GPUs)
  PD_DEVICE      gpu | cpu                                            (default gpu)
  PD_EVAL        1 = --eval during training (slower, gives mAP)       (default 1)

After training, the wrapper exports an inference model and tries paddle2onnx +
onnx2ncnn if they're installed; otherwise it prints the exact commands to run.
The resulting .param/.bin go to the sidecar via PICODET_PARAM / PICODET_BIN.

Install (on the GPU host):
  pip install paddlepaddle-gpu      # or paddlepaddle (CPU) — match your CUDA
  pip install paddle2onnx
  (onnx2ncnn comes from the ncnn tools build)
"""

import os
import sys
import subprocess

def env(k, d=None):
    v = os.environ.get(k)
    return v if v not in (None, "") else d

PD_DATASET = env("PD_DATASET")
PD_CLASSES = env("PD_CLASSES")
PD_DIR     = env("PD_DIR", "PaddleDetection")
PD_BASE    = env("PD_BASE", "configs/picodet/picodet_l_640_coco_lcnet.yml")
PD_EPOCHS  = int(env("PD_EPOCHS", "80"))
PD_BATCH   = int(env("PD_BATCH", "24"))
PD_DEVICE  = env("PD_DEVICE", "gpu")
PD_EVAL    = env("PD_EVAL", "1") == "1"


def sh(cmd, cwd=None):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd)


def main():
    if not PD_DATASET or not PD_CLASSES:
        sys.exit("ERROR: set PD_DATASET (COCO dataset dir) and PD_CLASSES (comma names)")
    classes = [c.strip() for c in PD_CLASSES.split(",") if c.strip()]
    dataset_abs = os.path.abspath(PD_DATASET)
    if not os.path.isdir(os.path.join(dataset_abs, "annotations")):
        sys.exit(f"ERROR: {dataset_abs}/annotations not found — run dataset_to_coco.py first")

    # 1) PaddleDetection
    if not os.path.isdir(PD_DIR):
        print("[1/4] Cloning PaddleDetection…")
        if sh(["git", "clone", "--depth", "1", "https://github.com/PaddlePaddle/PaddleDetection.git", PD_DIR]):
            sys.exit("ERROR: git clone failed")
        sh([sys.executable, "-m", "pip", "install", "-r", os.path.join(PD_DIR, "requirements.txt")])
    else:
        print(f"[1/4] Using existing PaddleDetection at {PD_DIR}")

    # 2) Custom config that points at your dataset.
    cfg_dir = os.path.join(PD_DIR, "configs", "custom")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "picodet_custom.yml")
    cfg = f"""_BASE_: [
  '../../{PD_BASE}',
]
weights: output/picodet_custom/best_model
epoch: {PD_EPOCHS}
num_classes: {len(classes)}

TrainDataset:
  !COCODataSet
    image_dir: train
    anno_path: annotations/instances_train.json
    dataset_dir: {dataset_abs}
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: val
    anno_path: annotations/instances_val.json
    dataset_dir: {dataset_abs}

TestDataset:
  !ImageFolder
    anno_path: annotations/instances_val.json
    dataset_dir: {dataset_abs}

TrainReader:
  batch_size: {PD_BATCH}
worker_num: 4
"""
    with open(cfg_path, "w") as f:
        f.write(cfg)
    print(f"[2/4] Wrote config → {cfg_path}  (classes={classes}, epochs={PD_EPOCHS}, batch={PD_BATCH})")

    # 3) Train
    print("[3/4] Training (this is the long part — GPU strongly recommended)…")
    train_cmd = [sys.executable, "tools/train.py", "-c", "configs/custom/picodet_custom.yml",
                 "-o", f"use_gpu={'true' if PD_DEVICE == 'gpu' else 'false'}"]
    if PD_EVAL:
        train_cmd.insert(3, "--eval")
    if sh(train_cmd, cwd=PD_DIR):
        sys.exit("ERROR: training failed (see PaddleDetection output above)")

    # 4) Export inference model → ONNX → NCNN
    print("[4/4] Exporting inference model…")
    sh([sys.executable, "tools/export_model.py", "-c", "configs/custom/picodet_custom.yml",
        "-o", "weights=output/picodet_custom/best_model", "--output_dir=inference_model"], cwd=PD_DIR)
    inf = os.path.join(PD_DIR, "inference_model", "picodet_custom")
    onnx_out = os.path.join(PD_DIR, "picodet_custom.onnx")

    print("\nNext, convert to ONNX then NCNN:")
    print(f"  paddle2onnx --model_dir {inf} --model_filename model.pdmodel "
          f"--params_filename model.pdiparams --opset_version 11 --save_file {onnx_out}")
    print(f"  python -m onnxsim {onnx_out} {onnx_out}        # optional but recommended")
    print(f"  onnx2ncnn {onnx_out} picodet.param picodet.bin  # from the ncnn tools build")
    print("\nThen on the Pi:")
    print("  PICODET_PARAM=picodet.param PICODET_BIN=picodet.bin YOLO_LABELS=labels.txt \\")
    print("    python3 picodet_ncnn_sidecar.py --inspect      # get the output blob names")
    print("  # set PICODET_CLS_BLOBS / PICODET_REG_BLOBS from --inspect, then run for real.")

    # Best-effort automatic paddle2onnx if available.
    if subprocess.call(["bash", "-lc", "command -v paddle2onnx"], stdout=subprocess.DEVNULL) == 0:
        print("\npaddle2onnx found — exporting ONNX automatically…")
        sh(["paddle2onnx", "--model_dir", inf, "--model_filename", "model.pdmodel",
            "--params_filename", "model.pdiparams", "--opset_version", "11", "--save_file", onnx_out])
        print(f"  ONNX → {onnx_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
