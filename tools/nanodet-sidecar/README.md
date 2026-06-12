# NanoDet-Plus (NCNN) sidecar — Raspberry Pi 5

**NanoDet-Plus** (RangiLyu) is an anchor-free detector with a GFL head and an FPN
over 3–4 strides (8/16/32/64). It's heavier than YOLO-FastestV2 but reads **small
objects much better** thanks to the stride-8 level, while still running
**CPU-real-time on a Pi 5**. Same panel/stream/manual-capture/gimbal scaffolding as
the other sidecars — only the model and decode differ.

**When to use which** (Pi 5, trained on your own GPU):
- **YOLO-FastestV2** — absolute max FPS (e.g. ~78 FPS @320), small objects are its weak spot.
- **NanoDet-Plus** — a step up in small-object accuracy at still-comfortable FPS. This sidecar.
- (PicoDet/MediaPipe — *not* recommended for this setup: PaddlePaddle won't train on
  Blackwell GPUs, MediaPipe needs WSL2+CPU and only outputs `.tflite`.)

Both YOLO-FastestV2 and NanoDet-Plus train on your **RTX 5090 (Blackwell)** with a
cu128 nightly PyTorch and export to NCNN with **one command**.

## Pieces in this folder

| File | What it does |
|---|---|
| `nanodet_ncnn_sidecar.py` | inference + stream + panel (run this on the Pi) |
| `train_nanodet.py` | one command: YOLO→COCO, clone repo, train, **and auto-export an optimised ncnn model** |
| `export_ncnn.py` | `.ckpt` → ONNX(opset 11) → onnxsim → ncnn → ncnnoptimize(fp16) → **verify** |

## 1. Install on the Pi

```bash
pip3 install ncnn numpy opencv-python
```

## 2. Run

```bash
ND_PARAM=nanodet.param ND_BIN=nanodet.bin ND_INPUT=416 \
  YOLO_LABELS=classes.txt YOLO_SOURCE=rpicam \
  python3 nanodet_ncnn_sidecar.py
```
Open `http://<pi-ip>:8080`. Drag a box to lock a target; **H** toggles the gimbal
panel; **Space** toggles auto-follow — same as the other sidecars.

First time / if boxes look wrong, run `--inspect`: it sweeps input sizes and prints
the output shape and point count so you can confirm `ND_INPUT` / `ND_STRIDES` /
`ND_REG_MAX` match your export.
```bash
ND_PARAM=… ND_BIN=… python3 nanodet_ncnn_sidecar.py --inspect
```

### Environment variables
| Var | Meaning | Default |
|---|---|---|
| `ND_PARAM` / `ND_BIN` | ncnn model files | **required** |
| `ND_INPUT` | square input size (match training) | `416` |
| `ND_STRIDES` | FPN strides | `8,16,32,64` |
| `ND_REG_MAX` | DFL bins per side − 1 | `7` |
| `ND_OUTPUT` | head output blob name (from `--inspect`) | last output |
| `ND_INPUT_BLOB` | model input blob name | first input |
| `ND_MEAN` / `ND_STD` | BGR normalisation | nanodet ImageNet |
| `ND_THREADS` | inference threads | all cores |
| `YOLO_CV_THREADS` | OpenCV threads (1 = don't fight inference for cores) | `1` |
| `YOLO_TRACK_HOLD` | seconds a box lingers after it stops being detected (lower = tighter/less ghosting) | `0.3` |
| `YOLO_SOURCE` / `YOLO_LABELS` / `YOLO_CONF` / `YOLO_NMS` / `YOLO_FILTER` / `YOLO_PORT` / `YOLO_JPEG_Q` / `YOLO_CAM_*` / `YOLO_TRACK` / `YOLO_GIMBAL` | as the other sidecars | |

The decode mirrors RangiLyu/nanodet's `demo_ncnn`: per grid point, argmax class
score (sigmoid auto-applied if the export left logits), then each of the 4 box sides
is a softmax-integral over `reg_max+1` bins → distance from the cell centre. The
decode math is unit-tested against an independent reference (≤1e-14).

## 3. Train on your own data (one command)

YOLO-format dataset in, verified ncnn model out — same paradigm as
`train_yolofastest.py`.

```bash
# on your GPU box (Windows + RTX 5090):
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install opencv-python numpy onnx onnxsim ncnn pnnx pytorch-lightning pycocotools omegaconf

# edit the CONFIG block in train_nanodet.py (DATASET, CLASSES, INPUT, BATCH…), then:
python train_nanodet.py
```
It converts your YOLO labels to COCO JSON (no image copying), clones
`RangiLyu/nanodet`, writes a custom config off the stock `nanodet-plus-m_416.yml`
(patching classes, data paths, input size, batch/workers/epochs — it reports every
field it patches), trains, then runs `export_ncnn.py` to produce a **verified,
fp16** `.param`/`.bin`. The final line prints the exact command to run it.

**Hardware defaults** (CONFIG block) are tuned for **Ultra 9 285K + RTX 5090 32GB +
128GB**: `BATCH=96`, `WORKERS=20`, `GPU_IDS=[0]`, `INPUT=416`. NanoDet-Plus is plain
PyTorch + Lightning, so the cu128 nightly torch trains it natively on Blackwell.

> **Why the export is stable:** it forces **opset 11 + `dynamo=False`** (so a new
> torch can't emit opset-18 that `onnx2ncnn` mis-converts), prefers
> `onnx2ncnn`+`ncnnoptimize`(fp16) else **`pnnx`**, and loads the model back in
> ncnn-python to confirm the head blob extracts and its channel count fits
> `nc + 4*(reg_max+1)` before declaring success.

> **Orchestration caveat:** `train_nanodet.py` wraps the upstream repo. NanoDet's
> config schema is stable but not frozen — the trainer prints every field it patches;
> if it warns a key wasn't found, open `nd_data/custom.yml` and set it by hand.

## 4. Where it runs

- **Pi 5 sidecar** — yes (this folder).
- **Desktop headless runner** — supported (see `desktop/`); pick the NanoDet model type.
- **Android phone app** — NanoDet uses a GFL/DFL head; the app decodes it once the
  NanoDet path is selected (input must match training).

## Status / caveats

The stream/panel/capture/gimbal scaffolding is the same proven code as the other
sidecars (including the recent panel/JSON fixes). The **decode math is unit-tested**
against an independent reference, but it **hasn't been run against your exact
exported model here** — `--inspect`, `ND_INPUT`, `ND_STRIDES`, `ND_REG_MAX`,
`ND_MEAN`/`ND_STD` and `ND_OUTPUT` are all overridable so you can converge quickly on
the real model.
