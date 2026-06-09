# MediaPipe Object Detection sidecar (Raspberry Pi 5 / CPU boards)

A drop-in detector that runs **MediaPipe Tasks ObjectDetector** (a `.tflite` model,
e.g. EfficientDet-Lite) and broadcasts an annotated MJPEG stream on the LAN — open
`http://<board-ip>:8080` in a browser or VLC, exactly like the JVM headless runner
(`:desktop:runHeadless`). Capture and inference are decoupled, an IoU tracker
stabilises boxes, and the same web panel is served — including **manual target
capture** (drag a box to lock & follow any object) and optional SIYI gimbal control.

This is a standalone Python process: MediaPipe Tasks has Python/Android/iOS/Web
bindings but no maintained desktop-Java one, so on a Pi it's driven from Python —
same idea as the RKNN sidecar next door.

## Hardware on a Raspberry Pi 5

MediaPipe runs the TFLite graph on the **CPU with the XNNPACK delegate**, which is
well optimised for the Pi 5's quad-core Cortex-A76. This is the realistic
accelerated path on a Pi — there is **no usable MediaPipe GPU/NPU delegate** for the
Pi's VideoCore, so don't expect a GPU option here. Pick the model for your speed
budget:

| Model | Input | Speed (Pi 5, CPU) | Accuracy |
|---|---|---|---|
| EfficientDet-Lite0 | 320 | fastest (real-time-ish) | lower |
| EfficientDet-Lite2 | 448 | ~2× slower | higher |

There's nothing to convert — point `YOLO_MODEL` at the `.tflite` and run.

## 1. Install

```bash
sudo apt update && sudo apt install -y python3-pip python3-opencv
pip3 install mediapipe          # pulls TFLite runtime + XNNPACK (aarch64 wheel)
```
> If `pip3 install mediapipe` can't find a wheel, use a 64-bit Raspberry Pi OS
> (Bookworm) with Python 3.9–3.12. `mediapipe` ships prebuilt aarch64 wheels.

## 2. Get a model

Download a ready MediaPipe ObjectDetector model (COCO classes):

```bash
mkdir -p ~/models
# EfficientDet-Lite0 (fast)
wget -O ~/models/efficientdet_lite0.tflite \
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
# or EfficientDet-Lite2 (more accurate)
wget -O ~/models/efficientdet_lite2.tflite \
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/int8/1/efficientdet_lite2.tflite
```

You can also train your own with **MediaPipe Model Maker** (`pip install
mediapipe-model-maker`, on a PC) and export a `.tflite` — it carries its class
names, so the panel labels them automatically. `YOLO_LABELS` overrides them.

## 3. Run

```bash
# USB webcam
YOLO_MODEL=~/models/efficientdet_lite0.tflite YOLO_SOURCE=0 \
  python3 yolo_mediapipe_sidecar.py

# Raspberry Pi CSI camera (ribbon) — via rpicam-vid/libcamera-vid
YOLO_MODEL=~/models/efficientdet_lite0.tflite YOLO_SOURCE=rpicam \
  python3 yolo_mediapipe_sidecar.py
```

Then open `http://<board-ip>:8080`. FPS counter (stream | detect) is bottom-left.

**Manual target capture:** drag a rectangle on the video to lock any object — a
cyan `LOCK` box sticks to it (independent of the model, survives brief occlusions
and re-acquires when it reappears). **C / Esc** clears the lock. With a gimbal,
the manual lock takes priority for follow. **H** toggles the gimbal controls
(hidden when there's no gimbal).

### Environment variables
| Var | Meaning | Default |
|---|---|---|
| `YOLO_MODEL` | path to a MediaPipe `.tflite` ObjectDetector model | **required** |
| `YOLO_SOURCE` | `0`/`1` USB cam, `rpicam`/`libcamera` (Pi CSI), `rtsp://…` / http MJPEG URL, or a GStreamer pipeline | `0` |
| `YOLO_LABELS` | path to labels.txt (one per line) to override the model's names | model names |
| `YOLO_FILTER` | keep only these classes (names or indices, comma-separated) | all |
| `YOLO_CONF` | score threshold `0..1` (raise to `0.5`+ on noisy/wide-angle scenes) | `0.4` |
| `YOLO_MAX_DETS` | max detections per frame | `25` |
| `YOLO_MAX_AREA` | drop boxes bigger than this fraction of the frame (full-frame false positives) | `0.9` |
| `YOLO_PORT` | MJPEG / control panel port | `8080` |
| `YOLO_JPEG_Q` | MJPEG quality `1..100` | `75` |
| `YOLO_CAM_W` / `YOLO_CAM_H` / `YOLO_CAM_FPS` | capture geometry | `1280` / `720` / `30` |
| `YOLO_TRACK` | `on` / `off` IoU tracking (box persistence) | `on` |
| `YOLO_GIMBAL` | `on` / `off` SIYI gimbal control (auto-on for SIYI source) | `off` |
| `YOLO_GIMBAL_HOST` / `YOLO_GIMBAL_PORT` | SIYI camera UDP address | `192.168.144.25` / `37260` |
| `YOLO_TRACK_SPEED` | max gimbal follow speed | `40` |
| `YOLO_TRACK_INVERT_YAW` / `YOLO_TRACK_INVERT_PITCH` | flip an axis if it chases away | `off` |

### Black screen (no image), but the stream FPS looks fine

The frames are flowing but empty — almost always the **wrong source**. The default
`YOLO_SOURCE=0` opens a USB/V4L2 device; on a Pi a **CSI camera is not at index 0**
(`cv2.VideoCapture(0)` reads black). Use `YOLO_SOURCE=rpicam` for the ribbon camera
(`rpicam-hello --list-cameras` should list it). For a real USB cam at index 0 the
sidecar now forces the V4L2 backend + MJPG, which fixes most black/low-FPS cases.

### Noisy detections / a box "sticks" over the whole frame

EfficientDet-Lite0 is a small COCO model and struggles on dark, cluttered or
wide-angle/fisheye scenes — it can emit a low-confidence full-frame false positive
(a classic is a "keyboard" over the whole image). Two guards handle it:

- **`YOLO_MAX_AREA`** (default `0.9`) drops any box bigger than that fraction of the
  frame — those are almost always junk. Lower it (e.g. `0.6`) if huge boxes persist.
- **`YOLO_CONF`** — raise it (`0.5`–`0.6`) to cut weak detections.

If a junk box still appears to "freeze", it's the IoU tracker holding it for ~0.8 s
to smooth gaps; run with **`YOLO_TRACK=off`** to confirm, then rely on the area/conf
guards. A genuinely frozen *frame* (everything stops) instead means the camera
stalled — check the capture source. For better accuracy on hard scenes, use
EfficientDet-Lite2, or run your own YOLO model via the JVM `:desktop:runHeadless`.

> CSI on Pi 5: `YOLO_SOURCE=rpicam` spawns `rpicam-vid` (needs `rpicam-apps`).
> For a USB cam use `YOLO_SOURCE=0`. For an exotic camera you can pass a full
> GStreamer pipeline string as `YOLO_SOURCE` (OpenCV must be built with GStreamer).

### SIYI gimbal control

Same as the JVM headless, served on the **same port** at `/` (when `YOLO_GIMBAL=on`
or the source is the SIYI camera): video with controls overlaid — movement,
zoom/focus, modes, photo/record, and **target follow** (Space toggles auto-follow).
HTTP endpoints (`/rotate`, `/angle`, `/zoom`, `/mode`, `/track`, `/pick`, `/lock`,
`/unlock`, `/status`) match the JVM app.

## 4. Autostart (systemd)

`/etc/systemd/system/yolo-mediapipe.service`:
```ini
[Unit]
Description=YOLO MediaPipe sidecar (CPU MJPEG broadcast)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/yolo/tools/mediapipe-sidecar
Environment=YOLO_MODEL=/home/pi/models/efficientdet_lite0.tflite
Environment=YOLO_SOURCE=rpicam
ExecStart=/usr/bin/python3 /home/pi/yolo/tools/mediapipe-sidecar/yolo_mediapipe_sidecar.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now yolo-mediapipe
journalctl -u yolo-mediapipe -f      # logs + stream URL
```

## When to use this vs. the JVM runner

- **JVM `:desktop:runHeadless`** — your own ONNX/PT YOLO models, hardware H.264
  recording, the full SIYI feature set. The main path.
- **MediaPipe sidecar (this)** — you want MediaPipe's ObjectDetector / Model Maker
  models, or a pure-Python deployment on a Pi. Same panel, same manual capture.
- **RKNN sidecar** — Rockchip boards (Orange Pi 5 / RK3588) with an NPU.
