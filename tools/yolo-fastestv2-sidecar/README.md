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
| `train_yolofastest.py` | clone the repo, compute anchors, train, point to ONNX/NCNN export |

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
  alloc/forward failure — almost always **`YF_INPUT` doesn't match the size the
  model was exported at** (especially after `ncnnoptimize`, which can bake fixed
  shapes). Run `--inspect` (it now does a probe forward and prints OK/FAIL + grid
  per output) and set `YF_INPUT` to the export size (try 256/320/352/416). The
  sidecar also auto-ignores a bad `YF_OUTPUTS` and re-detects working outputs.
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
| `YF_THREADS` | inference threads | `4` |
| `YOLO_SOURCE` | `0`/`1` USB, `rpicam` (CSI), rtsp/http URL, GStreamer | `0` |
| `YOLO_LABELS` | labels.txt (one per line) | COCO 80 |
| `YOLO_CONF` / `YOLO_NMS` | score / NMS-IoU thresholds | `0.3` / `0.45` |
| `YOLO_FILTER` / `YOLO_PORT` / `YOLO_JPEG_Q` / `YOLO_CAM_*` / `YOLO_TRACK` / `YOLO_GIMBAL` | as other sidecars | |

Manual capture (drag a box), gimbal control and the panel work exactly like the
other sidecars.

## 4. Train on your own data

YOLO-FastestV2 trains directly on **YOLO-format** labels (no box conversion!) and
is plain **PyTorch** — light, and it can use your GPU (incl. RTX 50xx/Blackwell with
a CUDA-12.8 PyTorch build; see `train_yolofastest.py` for the pip line).

Dataset layout (standard Ultralytics):
```
<dataset>/images/{train,val}/*.jpg
<dataset>/labels/{train,val}/*.txt   (YOLO: class xc yc w h, normalised)
```

```bash
# a) build the file lists + .data/.names
YF_DATASET=/path/to/yolo_dataset YF_CLASSES="Birds,Drones,Dron2" \
  python make_yolofastest_data.py            # → yf_data/custom.data

# b) install PyTorch (GPU build for Blackwell, or CPU), then train + get anchors
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install opencv-python numpy tqdm
YF_DATA=yf_data/custom.data python train_yolofastest.py
```
It clones the repo, runs `genanchors` for your data, trains, and prints the
ONNX→`onnx2ncnn` export commands. Copy the resulting `.param`/`.bin` + `custom.names`
to the Pi and run as in step 3 (set `YF_OUTPUTS` from `--inspect`).

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
