# YOLO-FastestV2 (NCNN) sidecar — Raspberry Pi 5

Runs **YOLO-FastestV2** (dog-qiuqiu/Yolo-FastestV2 — a ~0.25M-param detector, one
of the smallest/fastest around) on **NCNN**, with the same annotated MJPEG stream +
web panel as the other sidecars: manual drag-to-lock capture, IoU tracking, SIYI
gimbal. Open `http://<board-ip>:8080`. On a bare Pi 5 CPU this is typically the
**highest-FPS** option here.

> Why NCNN + a custom decoder: YOLO-FastestV2 is anchor-based (ShuffleNetV2
> backbone, 2 scales, decoupled head — `reg(4*na) + obj(na) + cls(nc)` per cell with
> class scores shared across anchors). NCNN gives the raw head, so the sidecar
> decodes it (anchors → box, obj×class score, NMS). The box/score formulas, anchors
> and blob names depend on the export, so they're env-tunable — use `--inspect`.

## Pieces in this folder

| File | What it does |
|---|---|
| `yolofastest_ncnn_sidecar.py` | inference + stream + panel (run this on the Pi) |
| `make_yolofastest_data.py` | build train/val lists + `.data`/`.names` from a YOLO dataset |
| `train_yolofastest.py` | one command: build `.data`, clone repo, compute anchors, train, **and auto-export an optimised ncnn model** |
| `export_ncnn.py` | `.pth` → ONNX(opset 11) → onnxsim → ncnn → ncnnoptimize(fp16) → **verify**; run by the trainer or standalone |

## 1. Install on the Pi

```bash
sudo apt update && sudo apt install -y python3-pip python3-opencv
pip3 install ncnn numpy
```

## 2. Get a ready model (for testing)

The upstream repo ships a ready NCNN model (COCO, 80 classes):

```bash
git clone --depth 1 https://github.com/dog-qiuqiu/Yolo-FastestV2.git
ls Yolo-FastestV2/model/*.param Yolo-FastestV2/model/*.bin
```
Use those `.param`/`.bin` to test the pipeline before training your own.

## 3. First run / tuning

**It auto-configures.** On startup the sidecar probes the model and auto-detects the
**strides and head output blobs** from the output grid sizes, so the usual case is
just:

```bash
YF_PARAM=model/yolo-fastestv2.param YF_BIN=model/yolo-fastestv2.bin \
  YF_INPUT=352 YOLO_SOURCE=rpicam YOLO_CONF=0.3 \
  python3 yolofastest_ncnn_sidecar.py
```
It prints `auto: strides=… outputs=…`. Open `http://<board-ip>:8080`. (For a model
you trained, `source sidecar_env.sh` first — the trainer writes your anchors/input
there, since anchors can't be read from the model.)

Only if auto-detect fails on a non-standard export do you set `YF_OUTPUTS` by hand
(`--inspect` lists names; the env var overrides auto-detect).

**`--autotune`** removes the last manual bit (the box/score formula): point the
camera at your objects and add `--autotune` once — it samples ~30 frames, tries
`v5`/`plain` × `sqrt`/`mul`, and keeps the combo whose boxes look sanest:
```bash
YF_PARAM=… YF_BIN=… YOLO_SOURCE=rpicam python3 yolofastest_ncnn_sidecar.py --autotune
```
It prints `autotune: box=… score=…`; bake the winner into `YF_BOX_DECODE`/`YF_SCORE`
(or the systemd unit) so you don't re-run it each boot.

**If boxes are wrong / missing**, in order:
- **`NCNN extract failed … code -100` + "det" racing to ~300 fps** → extraction
  itself fails (so `detect()` returns nothing instantly). ncnn `-100` is an
  alloc/forward failure (a layer produced an empty blob). Top causes:
  - **ONNX opset too new.** `onnx2ncnn` expects **opset 11**. Exporting with opset
    13/17/18 mis-converts shape ops (Reshape/Slice/Resize/…) → the model loads but
    `-100`s on forward. **`export_ncnn.py` handles this for you** (forces opset 11 +
    `dynamo=False` + `onnxsim`, then verifies), so a normal `python train_yolofastest.py`
    is safe even on the cu128 nightly torch. Only relevant if you export by hand:
    re-export with **opset 11**, run **`onnxsim`**, or use **`pnnx`** for new opsets.
  - **`YF_INPUT` ≠ export size** (esp. after `ncnnoptimize`, which bakes fixed
    shapes). `--inspect` now sweeps input sizes and reports which one works.
- Nothing detected (but extraction OK) → wrong `YF_OUTPUTS` (re-check `--inspect`).
- Boxes the wrong size/position → switch the box formula: `YF_BOX_DECODE=plain`
  (default `v5`). Also make sure `YF_INPUT` matches the model (default 352) and the
  anchors are right (`YF_ANCHORS_16` / `YF_ANCHORS_32`, defaults = repo COCO anchors).
- Too many/too few boxes → tune `YOLO_CONF` / `YOLO_NMS`, or try `YF_SCORE=mul`
  (default `sqrt`, i.e. score = √(obj·cls)).

### Environment variables
| Var | Meaning | Default |
|---|---|---|
| `YF_PARAM` / `YF_BIN` | ncnn model files | **required** |
| `YF_INPUT` | square input size | `352` |
| `YF_STRIDES` | detection strides | `16,32` |
| `YF_ANCHORS_PER` | anchors per cell | `3` |
| `YF_ANCHORS_16` / `YF_ANCHORS_32` | per-stride anchors `w,h,...` (px @ input) | repo COCO |
| `YF_BOX_DECODE` | `v5` or `plain` box formula | `v5` |
| `YF_SCORE` | `sqrt` or `mul` (final score) | `sqrt` |
| `YF_OUTPUTS` | head output blob names per stride (from `--inspect`) | common names |
| `YF_INPUT_BLOB` | input blob name | first input |
| `YF_THREADS` | inference threads | all cores |
| `YOLO_CV_THREADS` | OpenCV threads (1 = don't fight inference for cores) | `1` |
| `YOLO_TRACK_HOLD` | seconds a box lingers after it stops being detected (lower = tighter/less ghosting) | `0.3` |
| `YOLO_SOURCE` | `0`/`1` USB, `rpicam` (CSI), rtsp/http URL, GStreamer | `0` |
| `YOLO_LABELS` | labels.txt (one per line) | COCO 80 |
| `YOLO_CONF` / `YOLO_NMS` | score / NMS-IoU thresholds | `0.3` / `0.45` |
| `YOLO_FILTER` / `YOLO_PORT` / `YOLO_JPEG_Q` / `YOLO_CAM_*` / `YOLO_TRACK` / `YOLO_GIMBAL` | as other sidecars | |

Manual capture (drag a box), gimbal control and the panel work exactly like the
other sidecars. The manual lock uses OpenCV's **CSRT** tracker (robust to camera
pan/rotation/scale) when your OpenCV has it, else KCF, else a template fallback —
override with `MANUAL_TRACKER=csrt|kcf|ncc` (CSRT = most stable, KCF = faster). The
startup log prints which one is active.

## 4. Train on your own data

YOLO-FastestV2 trains directly on **YOLO-format** labels (no box conversion!) and
is plain **PyTorch** — light, and it can use your GPU (incl. RTX 50xx/Blackwell with
a CUDA-12.8 PyTorch build; see `train_yolofastest.py` for the pip line).

Both training scripts are configured by a **`CONFIG` block at the top of the file**
(no environment variables — Windows-friendly). Dataset layout (standard Ultralytics):
```
<dataset>/images/{train,val}/*.jpg
<dataset>/labels/{train,val}/*.txt   (YOLO: class xc yc w h, normalised)
```

```bash
# install PyTorch (GPU build for Blackwell, or CPU) + tools, all in your venv
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install opencv-python numpy tqdm onnx onnxsim ncnn pnnx

# edit the CONFIG block in train_yolofastest.py (DATASET, CLASSES, DEVICE, BATCH,
# WORKERS, EPOCHS…) once, then run it — ONE command does the whole pipeline:
#   build .data → clone repo → compute anchors → patch+train → export+verify ncnn
python train_yolofastest.py
```
That's it. The final line prints the exact commands to run the model on the Pi and
the phone. `make_yolofastest_data.py` does only the data-prep step (same CONFIG
block) if you want it separately.

### What "auto-export" does (the part that used to be manual)

After training, the trainer calls `export_ncnn.py`, which is deliberately
**stable** — it neutralises the two things that usually break a YOLO-FastestV2
export:

1. **opset.** A new PyTorch (the cu128 nightly for Blackwell) defaults to the
   TorchDynamo exporter and emits **opset 18**, which `onnx2ncnn` mis-converts
   (→ ncnn `-100` on forward). `export_ncnn.py` monkeypatches `torch.onnx.export`
   to force **opset 11 + `dynamo=False`** and then runs the repo's own
   `pytorch2onnx.py` — so it's correct on *any* torch version, no second venv.
2. **the converter.** Prefers `onnx2ncnn` + `ncnnoptimize` (fp16) if they're on
   PATH; otherwise uses **`pnnx`** (`pip install pnnx` — one wheel that converts
   *and* optimises). `ncnnoptimize` is only kept if the optimised model still
   passes verification (it can fuse away the head's output blobs on some exports).

Then it **loads the model back in ncnn-python and confirms the head blobs extract**
(and that the channel count matches your class count) — so a green "VERIFY OK" means
the file genuinely works before you ever copy it to the Pi. fp16 storage + the
runtime's automatic fp16/int8 inference = "all optimisations" for a no-calibration
export (int8 with a calibration set is a separate, optional step).

Run it standalone on an existing checkpoint:
```bash
python export_ncnn.py --repo Yolo-FastestV2 --data yf_data/custom.data --out yolofastestv2 --input 352
```

**Max-utilisation knobs** (in `train_yolofastest.py`'s CONFIG), tuned for an
**Ultra 9 285K + RTX 5090 32GB + 128GB**: `BATCH=192` (the model is tiny — raise to
256/384 if the GPU is underused, lower on OOM), `WORKERS=20` (dataloader workers —
the main lever for such a small net; the trainer patches them into the repo's
loader; 285K = 24 cores), `CUDNN_BENCHMARK=True`, `DEVICE=gpu`. A Blackwell card
needs the cu128 nightly torch above; otherwise set `DEVICE=cpu`.

> **Input size & anchors across resolutions:** YOLO-FastestV2 anchors are absolute
> pixels tied to the training input — they do **not** auto-scale. Train and infer at
> the **same** `INPUT` (e.g. 480), and set `YF_INPUT`/the phone's input to match.
> With `GENANCHORS=False` (default) the repo anchors are used as-is, so nothing to
> change; if you regenerate them, paste the new `anchors=` into `YF_ANCHORS_16/32`
> (sidecar) and the phone's "FastestV2 anchors" field.

## 5. Autostart (systemd)

`/etc/systemd/system/yolo-fastestv2.service`:
```ini
[Unit]
Description=YOLO-FastestV2 sidecar (NCNN MJPEG broadcast)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/yolo/tools/yolo-fastestv2-sidecar
Environment=YF_PARAM=/home/pi/models/yolofastestv2.param
Environment=YF_BIN=/home/pi/models/yolofastestv2.bin
Environment=YF_OUTPUTS=out16,out32
Environment=YOLO_SOURCE=rpicam
ExecStart=/usr/bin/python3 /home/pi/yolo/tools/yolo-fastestv2-sidecar/yolofastest_ncnn_sidecar.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now yolo-fastestv2
journalctl -u yolo-fastestv2 -f
```

## Status / caveats

Scaffolding (stream/panel/capture/gimbal/manual-capture) is the same proven code as
the other sidecars. The **anchor decode hasn't been validated against your exact
exported model here** — that's why `YF_OUTPUTS`, `YF_BOX_DECODE`, `YF_SCORE`,
`YF_INPUT` and the anchors are all overridable. Use `--inspect` + the step-3
checklist on the real model, then lock the working config into the systemd unit.
