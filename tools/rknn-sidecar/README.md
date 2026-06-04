# YOLO RKNN sidecar (Orange Pi 5 / RK3588 NPU)

NPU-accelerated detector for Rockchip boards. Runs a `.rknn` YOLO model on the
RK3588 NPU (~6 TOPS — far faster than CPU) and broadcasts an annotated MJPEG
stream on the LAN, exactly like the JVM headless runner. Open
`http://<board-ip>:8080` in a browser or VLC.

This is a standalone Python process (the RKNN runtime has no Java binding, so the
NPU is driven from Python). Capture/inference are decoupled and an IoU tracker
stabilises boxes — same behaviour as the rest of the project.

---

## 1. Convert your model to `.rknn` (on an x86-64 Linux host)

The NPU needs an `.rknn` file, converted from ONNX with **rknn-toolkit2** (runs
on a PC, not on the board).

```bash
pip install rknn-toolkit2          # x86-64 Linux + Python 3.8–3.11

python3 - <<'PY'
from rknn.api import RKNN
rknn = RKNN()
# RK3588; mean/std bake normalisation into the model so we feed raw uint8 frames
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
rknn.load_onnx(model='yolov8n.onnx')          # the same ONNX the desktop app uses
rknn.build(do_quantization=True, dataset='dataset.txt')   # int8; dataset.txt = list of sample images
rknn.export_rknn('yolov8n.rknn')
PY
```

- `dataset.txt` — a few dozen representative `.jpg` paths (one per line) for int8
  calibration. For a quick test you can set `do_quantization=False` (fp16, larger/slower).
- Keep the ONNX output layout standard (e.g. `[1,84,8400]` for YOLOv8/v11, or
  `[1,N,6]` for YOLOv10) — the sidecar decodes both, auto-detecting pixel vs
  normalised coords, just like the JVM `OnnxDetector`.
- Copy the resulting `yolov8n.rknn` to the board.

## 2. Install the runtime on the board

```bash
sudo apt update && sudo apt install -y python3-pip python3-opencv
pip3 install rknn-toolkit-lite2 numpy
```
(The `librknnrt.so` runtime ships with Rockchip Orange Pi OS images. If missing,
install it from Rockchip's `rknn-toolkit2` repo under `rknpu2/runtime/Linux`.)

## 3. Run

```bash
YOLO_MODEL=/home/orangepi/models/yolov8n.rknn YOLO_SOURCE=0 \
  python3 yolo_rknn_sidecar.py
```

Then open `http://<board-ip>:8080`. FPS counter (stream | detect) is bottom-left.

### Environment variables
| Var | Meaning | Default |
|---|---|---|
| `YOLO_MODEL` | path to `.rknn` model | **required** |
| `YOLO_SOURCE` | camera index `0`/`1`, a GStreamer/V4L2 pipeline string, or http MJPEG URL | `0` |
| `YOLO_INPUT` | model input size (square) | `640` |
| `YOLO_CLASSES` | number of classes | `80` |
| `YOLO_CONF` | confidence threshold | `0.25` |
| `YOLO_NMS` | NMS IoU threshold | `0.45` |
| `YOLO_PORT` | MJPEG server port | `8080` |
| `YOLO_CAM_W` / `YOLO_CAM_H` / `YOLO_CAM_FPS` | capture geometry (index sources) | `640` / `480` / `30` |
| `YOLO_TRACK` | `on` / `off` IoU tracking | `on` |

> CSI camera on Orange Pi 5: pass a GStreamer pipeline as `YOLO_SOURCE`, e.g.
> `YOLO_SOURCE="v4l2src device=/dev/video0 ! videoconvert ! appsink"` (exact
> pipeline depends on your camera/driver), or use a USB webcam (`YOLO_SOURCE=0`).

## 4. Autostart (systemd)

`/etc/systemd/system/yolo-rknn.service`:
```ini
[Unit]
Description=YOLO RKNN sidecar (NPU MJPEG broadcast)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=orangepi
WorkingDirectory=/home/orangepi/yolo/tools/rknn-sidecar
Environment=YOLO_MODEL=/home/orangepi/models/yolov8n.rknn
Environment=YOLO_SOURCE=0
ExecStart=/usr/bin/python3 /home/orangepi/yolo/tools/rknn-sidecar/yolo_rknn_sidecar.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now yolo-rknn
journalctl -u yolo-rknn -f      # logs + stream URL
```
