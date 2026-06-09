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
  MM_MAX_IMAGES   cap the training set (faster runs) (default: all)
  MM_QUIET        1 = silence TF/Keras/TFA chatter   (default 1; set 0 to debug)
  MM_QUANT        float | int8 — int8 = QAT, smaller/faster on the Pi  (default float)
  MM_QAT_EPOCHS   int8 QAT fine-tune epochs          (default 10)
  MM_QAT_BATCH    int8 QAT batch size                (default 4)

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
# Quiet by default (MM_QUIET=0 to see everything). Level 3 = errors only; this kills
# the cuDNN/cuFFT/cuBLAS "already registered" + GPU dlopen spam at import.
_QUIET = _env("MM_QUIET", "1") == "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3" if _QUIET else "1")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _THREADS)    # use all the Ultra 9 cores
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
if _FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                   # hide the GPU entirely

import logging
import warnings
import tensorflow as tf

# Silence the endless deprecation / "Gradients do not exist" / TFA / protobuf
# chatter so the screen shows just the training progress.
if _QUIET:
    warnings.filterwarnings("ignore")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)
    try:
        from absl import logging as _absl_logging
        _absl_logging.set_verbosity(_absl_logging.ERROR)
    except Exception:
        pass

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

    NOTE on input resolution: unlike YOLO (free imgsz), each model variant has a
    FIXED input size baked in. You pick it by choosing the variant (or by passing
    256/320/384 directly). The .tflite stores the size in metadata and the sidecar
    resizes frames to it automatically — nothing to set there.
    """
    sm = object_detector.SupportedModels
    available = [m.name for m in sm]
    wanted = {
        "lite0": "EFFICIENTDET_LITE0", "lite2": "EFFICIENTDET_LITE2", "lite4": "EFFICIENTDET_LITE4",
        "mobilenet": "MOBILENET_V2", "mobilenet_i320": "MOBILENET_V2_I320",
        "mobilenet_multi": "MOBILENET_MULTI_AVG",
        # pick by input resolution directly (intuitive if you used 320 before)
        "256": "MOBILENET_V2", "320": "MOBILENET_V2_I320", "384": "MOBILENET_MULTI_AVG_I384",
    }.get(name.lower(), name.upper())
    # Try the requested model first, then graceful fallbacks across versions.
    for cand in (wanted, "EFFICIENTDET_LITE0", "MOBILENET_V2", "MOBILENET_MULTI_AVG", *available):
        if hasattr(sm, cand):
            if cand != wanted:
                print(f"  note: '{name}' → not in this Model Maker; using {cand}")
            return getattr(sm, cand)
    raise SystemExit(f"No usable model. SupportedModels in this version: {available}")


# Input resolution baked into each variant (for display; the .tflite metadata is
# the source of truth and the sidecar reads it automatically).
_INPUT_RES = {
    "MOBILENET_V2": 256, "MOBILENET_V2_I320": 320,
    "MOBILENET_MULTI_AVG": 256, "MOBILENET_MULTI_AVG_I384": 384,
    "EFFICIENTDET_LITE0": 320, "EFFICIENTDET_LITE2": 448, "EFFICIENTDET_LITE4": 640,
}


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


def _make_hparams(**kw):
    """Build HParams keeping only fields this Model Maker version actually has."""
    import dataclasses
    try:
        names = {f.name for f in dataclasses.fields(object_detector.HParams)}
        kw = {k: v for k, v in kw.items() if k in names}
    except Exception:
        kw = {k: kw[k] for k in ("export_dir", "epochs", "batch_size") if k in kw}
    return object_detector.HParams(**kw)


def _quantize_int8(model, train_data, val_data):
    """
    Make the exported .tflite int8 via Quantization-Aware Training — the supported
    int8 path for the MediaPipe object detector. It's a short extra fine-tune on the
    quantised graph; afterwards export_model() emits an int8 model (~2–4× smaller,
    notably faster on the Pi's CPU). Returns True if QAT ran, False to fall back to
    float (older Model Maker without QAT, or any failure — never blocks the export).
    """
    import dataclasses
    try:
        qkw = dict(
            learning_rate=float(_env("MM_QAT_LR", "0.3")),
            batch_size=int(_env("MM_QAT_BATCH", "4")),
            epochs=int(_env("MM_QAT_EPOCHS", "10")),
            decay_steps=int(_env("MM_QAT_DECAY_STEPS", "6")),
            decay_rate=float(_env("MM_QAT_DECAY_RATE", "0.96")),
        )
        qat_cls = object_detector.QATHParams
        names = {f.name for f in dataclasses.fields(qat_cls)}
        qat = qat_cls(**{k: v for k, v in qkw.items() if k in names})
        print(f"Quantization-aware training (int8): epochs={qkw['epochs']} batch={qkw['batch_size']}…")
        model.quantization_aware_training(train_data, val_data, qat_hparams=qat)
        qloss, qmetrics = model.evaluate(val_data, batch_size=qkw["batch_size"])
        print(f"  int8 (QAT) eval: loss={qloss}  metrics={qmetrics}")
        return True
    except Exception as e:
        print(f"  WARNING: int8 QAT unavailable/failed → exporting float32 instead "
              f"({type(e).__name__}: {e})")
        return False


def main():
    import math
    spec = _resolve_model(MODEL_NAME)
    res = _INPUT_RES.get(spec.name)
    res_str = f"{res}×{res}" if res else "model default"
    dev = f"GPU ×{len(_gpus)} ({_gpus[0].name})" if _gpus else "CPU"
    print(f"Device: {dev}   threads={_THREADS}   batch={BATCH_SIZE}   epochs={EPOCHS}   "
          f"model={spec.name} (input {res_str})   mixed={_env('MM_MIXED', '0')}   xla={_env('MM_XLA', '0')}")
    print(f"  Model Maker supports: {[m.name for m in object_detector.SupportedModels]}")
    if not _gpus and not _FORCE_CPU:
        print("  (no usable GPU — training on CPU. Blackwell isn't supported by Model Maker's"
              " TensorFlow; for GPU use Google Colab. See the README.)")

    for d in (TRAIN_DIR, VAL_DIR):
        _normalise_voc(d)

    max_images = int(_env("MM_MAX_IMAGES", "0")) or None   # cap the training set for faster runs
    print("Loading dataset (Pascal VOC)…")
    train_data = object_detector.Dataset.from_pascal_voc_folder(
        TRAIN_DIR, cache_dir=os.path.join(CACHE, "train"), max_num_images=max_images)
    val_data = object_detector.Dataset.from_pascal_voc_folder(
        VAL_DIR, cache_dir=os.path.join(CACHE, "val"))
    print(f"  train: {train_data.size} images, classes: {train_data.label_names}")
    print(f"  val:   {val_data.size} images")

    steps = math.ceil(train_data.size / BATCH_SIZE)
    print(f"  ≈ {steps} steps/epoch × {EPOCHS} epochs = {steps * EPOCHS} steps total")
    if train_data.size > 5000 and max_images is None and not _gpus:
        print("  TIP: big dataset on CPU is slow. Try MM_EPOCHS=10 and/or "
              "MM_MAX_IMAGES=4000 for a quick first model.")

    hp_kwargs = dict(export_dir=EXPORT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     steps_per_epoch=steps)   # steps_per_epoch makes Keras show a real ETA
    if _env("MM_LR"):
        hp_kwargs["learning_rate"] = float(_env("MM_LR"))
    options = object_detector.ObjectDetectorOptions(
        supported_model=spec, hparams=_make_hparams(**hp_kwargs))

    print("Training… (per-epoch line shows step/total · time/step · losses · ETA)")
    model = object_detector.ObjectDetector.create(
        train_data=train_data, validation_data=val_data, options=options)

    print("Evaluating (float)…")
    loss, coco_metrics = model.evaluate(val_data, batch_size=BATCH_SIZE)
    print(f"  loss={loss}  metrics={coco_metrics}")

    # Optional int8 quantization (faster on the Pi). MM_QUANT=int8 runs QAT first.
    quant = _env("MM_QUANT", "float").lower()
    is_int8 = quant in ("int8", "qat") and _quantize_int8(model, train_data, val_data)

    # export_model() emits int8 after QAT, else float32. Writes into EXPORT_DIR.
    model.export_model("model.tflite")
    out = os.path.join(EXPORT_DIR, "model.tflite")
    print(f"Done ({'int8' if is_int8 else 'float32'}) → {out}")
    print("Copy it to the Pi and run the sidecar with YOLO_MODEL=path/to/model.tflite")


if __name__ == "__main__":
    main()
