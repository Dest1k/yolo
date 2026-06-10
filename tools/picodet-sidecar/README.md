# PicoDet (NCNN) sidecar — Raspberry Pi 5

Runs a **PicoDet** detector (PaddleDetection) on the **NCNN** runtime — one of the
fastest paths on an ARM CPU — and serves the same annotated MJPEG stream + web
panel as the other sidecars: **manual drag-to-lock capture**, IoU tracking, and
optional SIYI gimbal control. Open `http://<board-ip>:8080`.

> Why NCNN + a custom decoder: PicoDet's output is **not** YOLO-shaped (it's a GFL
> head — per-stride class scores + a discrete box-distance distribution), and NCNN
> can't run its baked-in NMS. So the sidecar decodes the head itself (softmax over
> the distance bins → distance2bbox → NMS), like the canonical ncnn picodet example.
> The decode is **parametrised** because blob names/strides depend on your export —
> see "First run / tuning" below.

This is a standalone Python process (NCNN has no JVM binding), same idea as the
RKNN and MediaPipe sidecars.

## Pieces in this folder

| File | What it does |
|---|---|
| `picodet_ncnn_sidecar.py` | inference + stream + panel (run this on the Pi) |
| `dataset_to_coco.py` | convert a YOLO/VOC dataset → COCO (for training) |
| `train_picodet.py` | train PicoDet on your data via PaddleDetection, export to ONNX/NCNN |

## 1. Install on the Pi

```bash
sudo apt update && sudo apt install -y python3-pip python3-opencv
pip3 install ncnn numpy
```

## 2. Get a ready model (for testing the pipeline)

Easiest is the **ncnn model zoo / PaddleDetection deploy** PicoDet, already in
`.param`/`.bin`. A known-good test model is the PicoDet from the ncnn examples
(COCO, 80 classes). Put `picodet.param` + `picodet.bin` somewhere on the Pi.

> If you only have an ONNX (e.g. from PaddleDetection export), convert it:
> `onnx2ncnn picodet.onnx picodet.param picodet.bin` (from the ncnn tools build;
> run it through `python -m onnxsim` first). PicoDet must be exported **without**
> NMS (`export_model … -o exclude_nms=True`) so NCNN gets the raw head outputs.

## 3. First run / tuning (important)

The output **blob names** and a couple of decode constants depend on how the model
was exported, so the very first time you must point the decoder at the right
outputs:

```bash
# 1) list the model's input/output blob names
PICODET_PARAM=picodet.param PICODET_BIN=picodet.bin \
  python3 picodet_ncnn_sidecar.py --inspect
```
You'll get something like the class-score outputs and the box-distribution outputs
for each stride. Set them (ordered by stride 8,16,32,64) and run:

```bash
PICODET_PARAM=picodet.param PICODET_BIN=picodet.bin \
  PICODET_INPUT=416 \
  PICODET_CLS_BLOBS=cls_8,cls_16,cls_32,cls_64 \
  PICODET_REG_BLOBS=dis_8,dis_16,dis_32,dis_64 \
  YOLO_SOURCE=rpicam YOLO_CONF=0.4 \
  python3 picodet_ncnn_sidecar.py
```
Then open `http://<board-ip>:8080`.

**If boxes are wrong / missing**, in order:
- No detections at all → wrong `PICODET_CLS_BLOBS`/`PICODET_REG_BLOBS` (re-check `--inspect`).
- Boxes shifted by ~half a cell → set `PICODET_CELL_OFFSET=0` (default 0.5).
- Boxes the right shape but wrong scale → wrong `PICODET_INPUT` (must match how the
  model was exported: PicoDet-S 320/416, PicoDet-L 640).
- Fewer/more FPN levels → set `PICODET_STRIDES` (default `8,16,32,64`).

### Environment variables
| Var | Meaning | Default |
|---|---|---|
| `PICODET_PARAM` / `PICODET_BIN` | ncnn model files | **required** |
| `PICODET_INPUT` | square input size (S 320/416, L 640) | `416` |
| `PICODET_STRIDES` | FPN strides | `8,16,32,64` |
| `PICODET_REG_MAX` | box-distribution bins − 1 | `7` |
| `PICODET_CELL_OFFSET` | grid-cell centre offset (try `0` if shifted) | `0.5` |
| `PICODET_CLS_BLOBS` / `PICODET_REG_BLOBS` | output blob names per stride (from `--inspect`) | common defaults |
| `PICODET_INPUT_BLOB` | input blob name | first input |
| `PICODET_THREADS` | inference threads | `4` |
| `YOLO_SOURCE` | `0`/`1` USB, `rpicam` (Pi CSI), rtsp/http URL, GStreamer | `0` |
| `YOLO_LABELS` | labels.txt (one per line) | COCO 80 |
| `YOLO_CONF` / `YOLO_NMS` | score / NMS-IoU thresholds | `0.4` / `0.5` |
| `YOLO_FILTER` | keep only these classes (names or indices) | all |
| `YOLO_PORT` / `YOLO_JPEG_Q` / `YOLO_CAM_W/H/FPS` / `YOLO_TRACK` | as other sidecars | |
| `YOLO_GIMBAL` / `YOLO_GIMBAL_HOST` / `YOLO_GIMBAL_PORT` / `YOLO_TRACK_SPEED` | SIYI gimbal | off |

Manual capture (drag a box on the video), gimbal control, and the panel work
exactly like the MediaPipe sidecar.

## 4. Train PicoDet on your own data

> ⚠️ **This is heavy.** PicoDet trains with **PaddleDetection** and the stock
> configs run ~300 epochs — on CPU that's days. You need a **supported NVIDIA GPU**.
> Your RTX 5080 (Blackwell) is too new for PaddlePaddle's CUDA too, so train on
> **Google Colab / a cloud GPU** or any older supported NVIDIA card.

**a) Convert your dataset to COCO** (edit `CLASSES` and `INPUT_FORMAT` at the top):
```bash
python dataset_to_coco.py
# → picodet_dataset/{train,val}/*.jpg + annotations/instances_{train,val}.json
```

**b) Train + export** (on the GPU host):
```bash
pip install paddlepaddle-gpu paddle2onnx        # match your CUDA; CPU: paddlepaddle
PD_DATASET=picodet_dataset PD_CLASSES="Birds,Drones,Dron2" \
  PD_BASE=configs/picodet/picodet_l_640_coco_lcnet.yml \
  PD_EPOCHS=80 PD_BATCH=24 \
  python train_picodet.py
```
It clones PaddleDetection, writes a config pointing at your dataset, trains, and
exports an inference model — then prints the exact `paddle2onnx` + `onnx2ncnn`
commands to produce `picodet.param` / `picodet.bin` for the sidecar. Remember to
export **without NMS** for NCNN.

**c) Copy `picodet.param`/`picodet.bin` (and a `labels.txt`) to the Pi** and run as
in step 3.

## 5. Autostart (systemd)

`/etc/systemd/system/yolo-picodet.service`:
```ini
[Unit]
Description=YOLO PicoDet sidecar (NCNN MJPEG broadcast)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/yolo/tools/picodet-sidecar
Environment=PICODET_PARAM=/home/pi/models/picodet.param
Environment=PICODET_BIN=/home/pi/models/picodet.bin
Environment=PICODET_CLS_BLOBS=cls_8,cls_16,cls_32,cls_64
Environment=PICODET_REG_BLOBS=dis_8,dis_16,dis_32,dis_64
Environment=YOLO_SOURCE=rpicam
ExecStart=/usr/bin/python3 /home/pi/yolo/tools/picodet-sidecar/picodet_ncnn_sidecar.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now yolo-picodet
journalctl -u yolo-picodet -f
```

## Status / caveats

The NCNN decode is written to the canonical PicoDet/GFL reference but **hasn't been
validated against your specific exported model here** — that's why blob names and
`PICODET_CELL_OFFSET`/`PICODET_INPUT`/`PICODET_STRIDES` are all overridable. Use
`--inspect` and the tuning checklist in step 3 on the real model. Once a config
works, lock it into the systemd unit.
