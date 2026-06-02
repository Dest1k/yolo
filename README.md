# YOLO Detector

Real-time object detection on **Android** and **Desktop** (Linux / Windows / macOS).  
Load any YOLOv5/v8/v10 model in NCNN or ONNX/PT format, point it at a camera or MJPEG stream, and get live bounding boxes with per-class counters at full camera framerate.

---

## Highlights

### Live MJPEG stream server
Toggle an HTTP server straight from the camera screen.  
Any browser, VLC, or another instance of this app can open `http://<device-IP>:8080` and see the annotated video in real time.  
Frame compositing (drawing boxes) happens on every captured frame — the server runs at the full camera capture rate (~30 fps) regardless of how fast inference is.

### MJPEG stream input
Instead of the phone camera you can feed the app any HTTP MJPEG stream (e.g. an IP camera, another phone, or a desktop webcam server).  
The same YOLO inference pipeline runs on the incoming frames — useful for processing a fixed camera without carrying a phone.

### Smart recording
Two video-recording modes:
- **Always** — records continuously while the button is active.
- **Smart** — starts automatically the moment any object is detected; stops 3 seconds after the last detection disappears. No manual intervention needed.

### GPU acceleration on Desktop
The desktop app tries GPU execution providers in order at startup:
1. **CUDA** (NVIDIA on Linux / Windows)
2. **DirectML** (AMD or NVIDIA on Windows)
3. **CPU fallback** — always works, no special drivers needed

You can pin a specific mode (AUTO / CUDA / DirectML / CPU) from the sidebar. The active provider is shown after the model loads.

### PyTorch `.pt` model support (Desktop)
Alongside ONNX, the desktop app loads TorchScript-exported `.pt` files via DJL.  
Both YOLOv5 anchor-free output `[1, N, 4+C]` and YOLOv10 NMS-free output `[1, N, 6]` are supported.

---

## Feature matrix

| Feature | Android | Desktop |
|---------|:-------:|:-------:|
| Live camera / webcam | ✓ | ✓ |
| MJPEG stream input | ✓ | ✓ |
| MJPEG stream output server | ✓ | ✓ |
| NCNN inference (CPU + Vulkan GPU) | ✓ | — |
| ONNX Runtime inference | ✓ | ✓ |
| PyTorch TorchScript (`.pt`) | — | ✓ |
| CUDA / DirectML GPU (desktop) | — | ✓ |
| Screenshot with bounding boxes | ✓ | ✓ |
| Video recording — Always mode | ✓ | — |
| Video recording — Smart mode | ✓ | — |
| Per-class object counter | ✓ | ✓ |
| Model library (auto-download) | ✓ | — |
| Camera resolution switcher | ✓ | — |

---

## Supported model formats

### NCNN — Android only
- `.param` + `.bin` file pair
- YOLOv5 (anchor-based), YOLOv8 (anchor-free), YOLOv10 (NMS-free)
- CPU or GPU (Vulkan) via NCNN native library

### ONNX — Android + Desktop
- Single `.onnx` file
- NMS-free output shape `[1, N, 6]` (YOLOv10 style)
- Android: NNAPI acceleration available
- Desktop: CUDA EP (NVIDIA) + DirectML EP (AMD/NVIDIA Windows); auto CPU fallback

### PyTorch TorchScript — Desktop only
- Single `.pt` file exported with `model.export(format="torchscript")`
- YOLOv5 output `[1, N, 4+C]` and YOLOv10 output `[1, N, 6]`
- GPU via CUDA 12.1 (NVIDIA); CPU fallback

---

## Android

### Requirements
- Android 8.0+ (API 26)
- Camera permission (for camera mode)
- Network permission (for stream input/output)

### Build
```bash
git clone https://github.com/dest1k/yolo.git
cd yolo
./gradlew :app:assembleDebug
# APK → app/build/outputs/apk/debug/app-debug.apk
```

### Usage

1. **Load a model**
   - Tap **Библиотека моделей** to download a preset NCNN model, or
   - Tap **Обзор** next to `.param` / `.bin` for NCNN files, or
   - Tap **Обзор** next to ONNX for a `.onnx` file

2. **Configure** via **Настройки модели**:
   - YOLO version (v5 / v8 / v10), input size, class count
   - Confidence & NMS thresholds, CPU threads, GPU toggle
   - Output layer names (or auto-detect with **Определить выходы**)

3. **Stream input** (optional) — enter an HTTP MJPEG URL in the stream field to use a network camera instead of the built-in camera

4. Tap **Запустить камеру** (or **Запустить стрим**)

### In-camera controls

| Button | Function |
|--------|----------|
| **MJPEG** | Start / stop the built-in MJPEG server. URL shown below the button. |
| **720p** | Cycle camera resolution: 480p → 720p → 1080p |
| 📷 FAB | Save current annotated frame to gallery |
| 🎥 FAB | Start / stop **Always** recording |
| ⚙ FAB | Open settings mid-session |
| 🔍 FAB (bottom-left) | Toggle **Smart record** mode |
| ↔ FAB | Flip front / back camera |

---

## Desktop

### Requirements
- JDK 17+
- Linux x86-64, Windows x86-64, or macOS

### Build & run
```bash
# Run from source
./gradlew :desktop:run

# Package installer for current OS
./gradlew :desktop:packageDeb      # Linux  → desktop/build/compose/binaries/main/deb/
./gradlew :desktop:packageMsi      # Windows → desktop/build/compose/binaries/main/msi/
./gradlew :desktop:packageDmg      # macOS   → desktop/build/compose/binaries/main/dmg/
```

### Usage

1. Pick **model type** — ONNX or PT
2. **Browse** to your model file
3. Set **video source** — webcam index (0 = default) or `http://…` MJPEG URL
4. Choose **GPU mode** — AUTO tries CUDA then DirectML then CPU
5. Adjust input size, confidence, class count
6. Click **Start** — active execution provider shown in sidebar
7. **MJPEG** — toggle the built-in server on port 8080 to share the annotated video
8. **Screenshot** — save current annotated frame as PNG

Export a compatible ONNX model:
```python
from ultralytics import YOLO
model = YOLO("yolov10n.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

---

## GitHub Actions (CI)

Three jobs run on every push:

| Job | Runner | Artifact |
|-----|--------|----------|
| `build-android` | ubuntu-latest | `app-debug.apk`, `app-release-unsigned.apk` |
| `build-desktop-linux` | ubuntu-latest | `YoloDetector_1.0.0_amd64.deb` |
| `build-desktop-windows` | windows-latest | `YoloDetector-1.0.0.msi` |

Artifacts are retained 30 days — downloadable from the **Actions** tab without building locally.

---

## Architecture notes

### Android inference pipeline
```
CameraX ImageAnalysis frame
  ├── streamExecutor   → composeFrame() + draw last known dets → MjpegServer.pushFrame()   [~30 fps]
  └── inferenceExecutor → YOLO / ONNX inference → update lastKnownDets                     [inference rate]
```
MJPEG output runs at full capture rate; inference rate is decoupled and does not throttle the stream.

### MJPEG server
`MjpegServer` runs a plain `ServerSocket`. Each client gets its own thread from a cached pool. `pushFrame()` encodes JPEG once into an `AtomicReference<ByteArray>`; every client thread reads the latest frame independently — zero per-client encoding overhead.

### Letterbox preprocessing
Frames are padded with gray (114) to a square matching the model's input size. Detection coordinates are un-letterboxed back to original pixel space before display.

---

## Permissions (Android)

| Permission | Reason |
|-----------|--------|
| `CAMERA` | Live camera capture |
| `WRITE_EXTERNAL_STORAGE` | Screenshots / videos on Android ≤ 9 |
| `READ_MEDIA_VIDEO` | Access saved videos on Android 13+ |
| `INTERNET` | MJPEG stream input and output |
| `ACCESS_NETWORK_STATE` / `ACCESS_WIFI_STATE` | Detect device IP for stream URL display |

---

## Model sources

- [nihui/ncnn-assets](https://github.com/nihui/ncnn-assets/tree/master/models) — ready-to-use NCNN models (downloadable in-app via model library)
- [Ultralytics](https://docs.ultralytics.com/) — export any model: `yolo export model=yolov8n.pt format=ncnn` or `format=onnx`
