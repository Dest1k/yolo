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

You can also **train your own** model — see "Train a custom model" below.

## Train a custom model (your own classes)

Use **MediaPipe Model Maker** with the ready script
[`train_object_detector.py`](train_object_detector.py). It reads Pascal VOC XML
(LabelImg / Roboflow VOC export) directly and exports a `.tflite` with class-name
metadata — so the sidecar labels your classes automatically (`YOLO_LABELS`
overrides if needed).

> ⚠️ **Do NOT train on native Windows.** MediaPipe Model Maker isn't supported
> there — you'll hit `tensorflow_text` / `tensorflow_addons` / protobuf
> `runtime_version` errors no matter how you patch it. Use Google Colab (easiest)
> or WSL2 / Linux / macOS with Python 3.9–3.11. The training itself is unchanged;
> only the host differs.

Dataset layout (Pascal VOC):
```
mediapipe_dataset/
  train/  images/*.jpg   Annotations/*.xml
  val/    images/*.jpg   Annotations/*.xml
```

Already have a **YOLO dataset** (the one you trained YOLO with)? Convert it with
[`yolo_to_voc.py`](yolo_to_voc.py) — set `CLASSES` to your `data.yaml` `names:` (same
order!) and run it from the dataset root; it writes `mediapipe_dataset/` ready for
the trainer (skipping degenerate boxes and out-of-range class ids).

### WSL2 setup on your Windows PC (recommended)

MediaPipe Model Maker needs **Python 3.9–3.11** — Ubuntu 24.04 ships 3.12, so we
install 3.11 explicitly.

One-time, in **PowerShell (admin)**:
```powershell
wsl --install            # installs WSL2 + Ubuntu; reboot when asked
```
Then open **Ubuntu** and set up a clean venv:
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv python3.11-dev libgl1 libglib2.0-0
python3.11 -m venv ~/mm && source ~/mm/bin/activate
pip install --upgrade pip
pip install mediapipe-model-maker          # pulls a matching TF/protobuf — no Windows hacks
```

Get the script next to your dataset (your Windows files are under `/mnt/c`), then
train. Grab just the one file with `curl` (replace with your dataset path):
```bash
cd /mnt/c/Users/dest/Desktop/test/merged_dataset
curl -L -o train_object_detector.py \
  https://raw.githubusercontent.com/dest1k/yolo/main/tools/mediapipe-sidecar/train_object_detector.py
python train_object_detector.py
```
Result: `exported_model/model.tflite` → copy to the Pi, run the sidecar with
`YOLO_MODEL=/path/to/model.tflite`.

The script prints the models your installed Model Maker actually supports and
auto-picks one — recent versions expose **MobileNetV2** (EfficientDet was dropped
from Model Maker), which the sidecar runs identically.

**Input resolution** isn't a free number like YOLO's `imgsz` — it's *fixed per
model variant*. Pick it via `MM_MODEL` (you can pass the resolution directly):

| `MM_MODEL` | model | input |
|---|---|---|
| `mobilenet` / `256` | `MOBILENET_V2` | 256×256 |
| `320` / `mobilenet_i320` | `MOBILENET_V2_I320` | 320×320 |
| `384` | `MOBILENET_MULTI_AVG_I384` | 384×384 |

The `.tflite` stores its input size in metadata, and the sidecar resizes frames to
it automatically — there's no `YOLO_INPUT` to set (unlike the JVM ONNX path). So to
train "like 320 before", use `MM_MODEL=320`.

If training later dies on `cannot import name 'runtime_version' from
'google.protobuf'` (a too-new `tensorflow-metadata` vs the pinned protobuf), pin it
down once and re-run:
```bash
pip install "tensorflow-metadata<1.16" "tensorflow-datasets==4.9.3"
```

#### GPU vs CPU (RTX 5080 / Ultra 9 275HX)

The script auto-detects the GPU and prints which device it uses. **Reality check:**
the RTX 5080 is **Blackwell (sm_120)**, newer than the TensorFlow that Model Maker
pins, and that TF doesn't ship CUDA kernels for it — so in WSL you'll see
`Skipping registering GPU devices` and it trains on the **CPU**. That's expected,
not a misconfig. A 24-thread Ultra 9 275HX handles it; the script uses all cores.

- **Want a GPU?** Use **Google Colab** (free, supported GPU) — see below. There's no
  practical way to use a Blackwell GPU with Model Maker's pinned TF locally today.
- The output is **quiet by default** (`MM_QUIET=1`): the cuDNN/TFA/"Gradients do not
  exist"/deprecation spam is silenced, leaving Keras's per-epoch progress line
  (`step/total · time/step · losses · ETA`). Set `MM_QUIET=0` to see everything.
- **Big dataset?** 46k images × 50 epochs on CPU is ~a day. For a quick first model
  use `MM_EPOCHS=10` and/or `MM_MAX_IMAGES=4000`.

Performance / run knobs (env vars):

| Var | Meaning | Default |
|---|---|---|
| `MM_EPOCHS` | training epochs (lower = faster) | `50` |
| `MM_MAX_IMAGES` | cap the training set for quick runs | all |
| `MM_BATCH` | batch size (raise it — you have the RAM/VRAM) | `16` |
| `MM_MODEL` | model name; auto-resolves to what's installed (`lite0`, `mobilenet`…) | `lite0` |
| `MM_LR` | learning rate (raise with big batches) | Model Maker default |
| `MM_THREADS` | CPU op threads | all logical cores |
| `MM_QUANT` | `float` or `int8` — int8 = smaller/faster on the Pi (via QAT) | `float` |
| `MM_QAT_EPOCHS` | int8 QAT fine-tune epochs | `10` |
| `MM_QUIET` | `1` = quiet output (just progress); `0` = full TF logs | `1` |
| `MM_FORCE_CPU` | `1` = ignore the GPU | `0` |
| `MM_MIXED` | `1` = mixed_float16 (only helps a working GPU) | `0` |
| `MM_XLA` | `1` = XLA JIT (may speed up / may break) | `0` |
| `MM_CACHE` | dataset cache dir | `cache` |

Examples:
```bash
# quick first model on CPU: 10 epochs, 4k images, clean output
MM_EPOCHS=10 MM_MAX_IMAGES=4000 python train_object_detector.py
# int8 model (faster on the Pi) at 320 input
MM_MODEL=320 MM_QUANT=int8 python train_object_detector.py
# full run, see every TF log (debugging)
MM_QUIET=0 python train_object_detector.py
```

#### int8 (the FPS lever on the Pi)

`MM_QUANT=int8` makes the exported `.tflite` **int8** instead of float32 —
typically **~2–4× smaller and noticeably faster** on the Pi 5 CPU (often the
difference between ~13 and ~20–30 FPS). For the MediaPipe object detector int8 is
done via **Quantization-Aware Training (QAT)**: the script first trains the float
model, then runs a short QAT fine-tune (`MM_QAT_EPOCHS`, default 10), and
`export_model()` then emits the int8 model. If your Model Maker version doesn't
support QAT it falls back to float32 automatically (you'll see a WARNING). The Pi
sidecar runs int8 and float models the same way — no config change.

### Alternative — Google Colab (zero setup, free + supported GPU)

```python
!pip install -q mediapipe-model-maker
# upload + unzip your dataset and train_object_detector.py, then:
!python train_object_detector.py
# download exported_model/model.tflite
```


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
