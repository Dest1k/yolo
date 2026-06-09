#!/usr/bin/env python3
"""
Train a custom EfficientDet-Lite object detector with MediaPipe Model Maker and
export a `.tflite` you can drop straight into the MediaPipe sidecar
(`YOLO_MODEL=...`).

IMPORTANT — where to run this:
  MediaPipe Model Maker is NOT supported on native Windows (no tensorflow-text
  wheels, protobuf/TF conflicts — exactly the "runtime_version" / fake
  tensorflow_text errors you hit). Run it on one of:
    • WSL2 / Linux / macOS with a clean Python 3.9–3.11 venv (this script), or
    • Google Colab (free GPU).
  No monkey-patching of tensorflow_text / tensorflow_addons is needed there.

Performance tuning (all via env vars, with sensible defaults):
  MM_BATCH        batch size                         (default 16; 32–64 on a GPU)
  MM_EPOCHS       epochs                             (default 50)
  MM_MODEL        lite0 | lite2                      (default lite0; lite2 = slower/accurate)
  MM_LR           learning rate                      (default: Model Maker's own)
  MM_THREADS      CPU op threads                     (default: all logical cores)
  MM_FORCE_CPU    1 = ignore the GPU, use CPU only   (default 0)
  MM_MIXED        1 = mixed_float16 (GPU speedup)    (default 0)
  MM_XLA          1 = XLA JIT (may speed up / may break)  (default 0)
  MM_CACHE        dataset cache dir                  (default ./cache)

GPU note (RTX 50-series / Blackwell, sm_120): the TensorFlow that Model Maker
pins is older than Blackwell, so the GPU may error ("no kernel image is available
for execution on the device") or simply not be picked up. If that happens, set
MM_FORCE_CPU=1 — a 24-thread Ultra 9 trains EfficientDet-Lite0 on a custom dataset
quickly anyway. The script prints which device it actually uses.

Install (WSL2 / Linux, Python 3.9–3.11):
    python -m pip install --upgrade pip
    python -m pip install mediapipe-model-maker

Dataset layout (Pascal VOC — what LabelImg / Roboflow VOC export produces):
    mediapipe_dataset/
      train/  images/*.jpg   Annotations/*.xml   (class names read from <name> tags)
      val/    images/*.jpg   Annotations/*.xml

Run:
    python train_object_detector.py
Output:
    exported_model/model.tflite   ← copy to the Pi, use as YOLO_MODEL
"""

import os

# ── Performance knobs — set BEFORE importing TensorFlow so they take effect ────
def _env(k, d=None): return os.environ.get(k, d)

_FORCE_CPU = _env("MM_FORCE_CPU", "0") == "1"
_THREADS = _env("MM_THREADS") or str(os.cpu_count() or 8)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")            # quieter TF logs
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _THREADS)    # use all the Ultra 9 cores
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
if _FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                   # hide the GPU entirely

import tensorflow as tf

# GPU: enable memory growth (don't grab all VRAM up front) and report what we got.
_gpus = [] if _FORCE_CPU else tf.config.list_physical_devices("GPU")
for g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
if _env("MM_MIXED", "0") == "1" and _gpus:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
if _env("MM_XLA", "0") == "1":
    tf.config.optimizer.set_jit(True)

from mediapipe_model_maker import object_detector

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR  = "mediapipe_dataset/train"   # must contain images/ and Annotations/
VAL_DIR    = "mediapipe_dataset/val"
EXPORT_DIR = "exported_model"
EPOCHS     = int(_env("MM_EPOCHS", "50"))
BATCH_SIZE = int(_env("MM_BATCH", "16"))
CACHE      = _env("MM_CACHE", "cache")
MODEL_NAME = _env("MM_MODEL", "lite0")   # resolved against the installed version below
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_model(name):
    """
    Pick a SupportedModels member that actually exists in the installed Model
    Maker. Enum members vary by version — recent releases dropped EfficientDet and
    expose only MobileNet detectors — so map friendly names and fall back sanely.
    Any of these trains a .tflite the sidecar runs identically.
    """
    sm = object_detector.SupportedModels
    available = [m.name for m in sm]
    wanted = {
        "lite0": "EFFICIENTDET_LITE0", "lite2": "EFFICIENTDET_LITE2", "lite4": "EFFICIENTDET_LITE4",
        "mobilenet": "MOBILENET_V2", "mobilenet_i320": "MOBILENET_V2_I320",
        "mobilenet_multi": "MOBILENET_MULTI_AVG",
    }.get(name.lower(), name.upper())
    # Try the requested model first, then graceful fallbacks across versions.
    for cand in (wanted, "EFFICIENTDET_LITE0", "MOBILENET_V2", "MOBILENET_MULTI_AVG", *available):
        if hasattr(sm, cand):
            if cand != wanted:
                print(f"  note: '{name}' → not in this Model Maker; using {cand}")
            return getattr(sm, cand)
    raise SystemExit(f"No usable model. SupportedModels in this version: {available}")


def _normalise_voc(split_dir):
    """MediaPipe expects subfolders named exactly `images` and `Annotations`."""
    imgs = os.path.join(split_dir, "images")
    ann = os.path.join(split_dir, "Annotations")
    ann_lower = os.path.join(split_dir, "annotations")
    if not os.path.isdir(ann) and os.path.isdir(ann_lower):
        os.rename(ann_lower, ann)            # annotations/ → Annotations/ (Linux is case-sensitive)
        print(f"  renamed {ann_lower} → {ann}")
    if not os.path.isdir(imgs):
        raise SystemExit(f"ERROR: missing {imgs}  (put your images there)")
    if not os.path.isdir(ann):
        raise SystemExit(f"ERROR: missing {ann}  (put your Pascal VOC .xml there)")


def main():
    model = _resolve_model(MODEL_NAME)
    dev = f"GPU ×{len(_gpus)} ({_gpus[0].name})" if _gpus else "CPU"
    print(f"Device: {dev}   threads={_THREADS}   batch={BATCH_SIZE}   epochs={EPOCHS}   "
          f"model={model.name}   mixed={_env('MM_MIXED', '0')}   xla={_env('MM_XLA', '0')}")
    print(f"  Model Maker supports: {[m.name for m in object_detector.SupportedModels]}")
    if not _gpus and not _FORCE_CPU:
        print("  (no usable GPU seen — training on CPU; on Blackwell this is often the fast, safe path)")

    for d in (TRAIN_DIR, VAL_DIR):
        _normalise_voc(d)

    print("Loading dataset (Pascal VOC)…")
    train_data = object_detector.Dataset.from_pascal_voc_folder(TRAIN_DIR, cache_dir=os.path.join(CACHE, "train"))
    val_data   = object_detector.Dataset.from_pascal_voc_folder(VAL_DIR,   cache_dir=os.path.join(CACHE, "val"))
    print(f"  train: {train_data.size} images, classes: {train_data.label_names}")
    print(f"  val:   {val_data.size} images")

    hp_kwargs = dict(export_dir=EXPORT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE)
    if _env("MM_LR"):
        hp_kwargs["learning_rate"] = float(_env("MM_LR"))
    options = object_detector.ObjectDetectorOptions(
        supported_model=model, hparams=object_detector.HParams(**hp_kwargs))

    print("Training…")
    model = object_detector.ObjectDetector.create(
        train_data=train_data, validation_data=val_data, options=options)

    print("Evaluating…")
    loss, coco_metrics = model.evaluate(val_data, batch_size=BATCH_SIZE)
    print(f"  loss={loss}  metrics={coco_metrics}")

    # Exports an int8-quantised model.tflite (with label metadata) into EXPORT_DIR.
    model.export_model()
    print(f"Done → {os.path.join(EXPORT_DIR, 'model.tflite')}")
    print("Copy it to the Pi and run the sidecar with YOLO_MODEL=path/to/model.tflite")


if __name__ == "__main__":
    main()
