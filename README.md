# YOLO Detector

Real-time object detection app using YOLOv5/v8/v10 models. Runs on **Android** (NCNN + ONNX Runtime) and **Desktop** (Linux/Windows/macOS via ONNX Runtime + Compose).

---

## Features

| Feature | Android | Desktop |
|---------|---------|---------|
| Live camera / webcam | ✓ | ✓ |
| MJPEG stream input | ✓ | ✓ |
| MJPEG stream output (server) | ✓ | ✓ |
| ONNX Runtime inference | ✓ | ✓ |
| NCNN inference | ✓ | — |
| Screenshot with boxes | ✓ | ✓ |
| Video recording (always / smart) | ✓ | — |
| Per-class object counter | ✓ | ✓ |
| Model library (auto-download) | ✓ | — |

---

## Supported Model Formats

### NCNN (Android only)
- `.param` + `.bin` file pair
- YOLOv5 (anchor-based), YOLOv8 (anchor-free), YOLOv10 (NMS-free)
- CPU or GPU (Vulkan) via NCNN native library

### ONNX (Android + Desktop)
- Single `.onnx` file
- NMS-free models (YOLOv10 style: output shape `[1, N, 6]`)
- Runs via ONNX Runtime; Android also supports NNAPI acceleration

---

## Android

### Requirements
- Android 8.0+ (API 26)
- Camera permission (for camera mode)
- Network permission (for stream mode)

### Build
```bash
# Clone and build debug APK
git clone https://github.com/dest1k/yolo.git
cd yolo
./gradlew assembleDebug
# APK: app/build/outputs/apk/debug/app-debug.apk
```

### Usage

1. **Load a model**
   - Tap **Библиотека моделей** to download a preset model automatically, or
   - Tap **Обзор** next to `.param` / `.bin` to pick NCNN files, or
   - Tap **Обзор** next to ONNX to pick a `.onnx` file

2. **Configure**
   - Tap **Настройки модели** to set:
     - YOLO version (v5 / v8 / v10)
     - Input size (e.g. 640)
     - Number of classes
     - Confidence & NMS thresholds
     - CPU threads / GPU toggle
     - Output layer names (or use **Определить выходы** for auto-detection)

3. **Stream input** (optional)
   - Enter an HTTP MJPEG URL (e.g. `http://192.168.1.100:8080/video`) in the stream field
   - The app will consume that stream instead of the phone camera

4. **Start**
   - Tap **Запустить камеру** (or **Запустить стрим** in stream mode)

### In-camera controls

| Button | Function |
|--------|----------|
| **MJPEG** | Toggle built-in MJPEG HTTP server on port 8080. Shows `http://DEVICE_IP:8080` |
| **720p** | Cycle camera resolution: 480p → 720p → 1080p |
| 📷 FAB | Save screenshot (frame + boxes) to gallery |
| 🎥 FAB | Start/stop video recording (MP4) |
| ⚙ FAB | Open settings mid-session |
| 🔍 FAB (bottom-left) | Toggle **Smart Record** mode (auto-starts on detection, stops 3 s after last box) |
| ↔ FAB | Flip front/back camera |

### MJPEG stream server
When **MJPEG** is active the app serves `multipart/x-mixed-replace` on:
```
http://<PHONE_IP>:8080
```
Open in any browser or VLC. The stream includes bounding boxes composited on each frame. FPS matches camera capture rate (~30 fps); inference runs independently and does not slow the stream.

---

## Desktop

### Requirements
- JDK 17+
- Linux x86-64, Windows x86-64, or macOS x86-64/arm64

### Build
```bash
# Run from source (current platform)
./gradlew :desktop:run

# Package installer for current OS
./gradlew :desktop:packageDeb      # Linux → desktop/build/compose/binaries/main/deb/
./gradlew :desktop:packageMsi      # Windows → desktop/build/compose/binaries/main/msi/
./gradlew :desktop:packageDmg      # macOS   → desktop/build/compose/binaries/main/dmg/
```

### Usage

1. **Select ONNX model** — click **Browse** and pick a `.onnx` file (NMS-free YOLOv10 format)
2. **Select source** — webcam index (0 = default) or paste an MJPEG HTTP URL
3. **Start** — frames appear in the left panel with boxes overlaid
4. **MJPEG server** — click **MJPEG** to start serving on port 8080
5. **Screenshot** — click **Screenshot** to save the current frame as PNG

---

## GitHub Actions (CI)

Three jobs run on every push:

| Job | Runner | Artifact |
|-----|--------|----------|
| `build-android` | ubuntu-latest | `app-debug.apk`, `app-release-unsigned.apk` |
| `build-desktop-linux` | ubuntu-latest | `YoloDetector_1.0.0_amd64.deb` |
| `build-desktop-windows` | windows-latest | `YoloDetector-1.0.0.msi` |

Artifacts are retained for 30 days and downloadable from the **Actions** tab.

---

## Model Sources

Pre-built NCNN models compatible with this app:
- [nihui/ncnn-assets](https://github.com/nihui/ncnn-assets/tree/master/models) — YOLOv5s, YOLOv7, YOLOv8s
- [ultralytics](https://docs.ultralytics.com/integrations/ncnn/) — export any Ultralytics model to NCNN

For ONNX models, export from Ultralytics:
```python
from ultralytics import YOLO
model = YOLO("yolov10n.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

---

## Architecture Notes

### Android inference pipeline
```
CameraX ImageAnalysis
  └─► streamExecutor  →  composeFrame()  →  MjpegServer.pushFrame()  (~30 fps)
  └─► inferenceExecutor  →  YoloDetector / OnnxDetector  →  lastKnownDets  (inference rate)
```

### Coordinate transform
Camera frames are letterboxed (gray padding, value 114) to the model's square input size. Detection coordinates are un-letterboxed and mapped from model space to view space accounting for `FILL_CENTER` scale/offset.

### MJPEG server internals
`MjpegServer` uses a plain `ServerSocket` with a `CopyOnWriteArrayList` of client sockets. Each client runs on a cached thread pool. `pushFrame()` encodes JPEG once and stores it in an `AtomicReference`; each client thread reads the latest frame independently — no per-client encoding.

---

## Permissions (Android)

| Permission | Reason |
|-----------|--------|
| `CAMERA` | Live camera capture |
| `RECORD_AUDIO` | (reserved for future audio recording) |
| `WRITE_EXTERNAL_STORAGE` | Save screenshots / videos (Android ≤ 9) |
| `READ_MEDIA_VIDEO` | Access saved videos (Android 13+) |
| `INTERNET` | MJPEG stream input/output |
| `ACCESS_NETWORK_STATE` / `ACCESS_WIFI_STATE` | Get device IP for stream URL display |
