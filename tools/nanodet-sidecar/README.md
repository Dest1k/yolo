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
| `train_nanodet_gui.py` | **graphical** front-end for the above — every field (incl. fine-tuning) in a window, live training log underneath (Windows-friendly, tkinter, zero extra deps) |
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
| `YOLO_FP16` | fp16 arithmetic + winograd/sgemm ncnn kernels (Pi 5 A76 = ARMv8.2 FP16; real FPS win). `0` = fp32 fallback | `1` |
| `YOLO_CV_THREADS` | OpenCV threads (1 = don't fight inference for cores) | `1` |
| `YOLO_TRACK_HOLD` | seconds a box lingers after it stops being detected (lower = tighter/less ghosting) | `0.3` |
| `YOLO_SOURCE` / `YOLO_LABELS` / `YOLO_CONF` / `YOLO_NMS` / `YOLO_FILTER` / `YOLO_PORT` / `YOLO_JPEG_Q` / `YOLO_CAM_*` / `YOLO_TRACK` / `YOLO_GIMBAL` | as the other sidecars | |

### Any video URL (ffmpeg + yt-dlp) — great for testing models

`YOLO_SOURCE` ruthlessly ingests almost anything. Besides `rpicam` and a webcam index
(`0`), give it **any URL or file** and it's piped through system **ffmpeg** (HLS `.m3u8`,
DASH, RTSP, RTMP, MJPEG, `.mp4`/`.mkv`, …). For a **site page** the stream URL is found in
three escalating steps: **yt-dlp** (YouTube/Twitch/…) → if that fails, the sidecar
**downloads the page HTML and scrapes** it for an `.m3u8`/`.mpd`/`.mp4`/rtmp/rtsp (digging
one `<iframe>` level deep, un-escaping JSON), passing the page as **Referer** (many webcams
hot-link-protect). Newest-frame-wins (low latency); ffmpeg auto-reconnects on drop.

If a page builds its URL in JavaScript (nothing in the HTML), open it in a browser, watch
the **Network tab** for the `.m3u8`/`.mp4` request and pass **that** URL directly.
`YOLO_SCRAPE=0` disables the HTML scrape.

### Detecting small objects in big/wide streams — tiling (`ND_TILES`)

A high-res or wide webcam squashed to the model's small input turns distant people/cars
into a few pixels → no detections. **Tiling** splits each frame into overlapping tiles,
detects on each (objects are now larger), and merges with NMS:

```bash
ND_TILES=auto  …    # pick a grid from the frame size (e.g. 1080p→3×2, 4K→4×3)
ND_TILES=3x2   …    # force a 3×2 grid
ND_TILES=off   …    # default (whole frame at once)
```
Tune `ND_TILE_OVERLAP` (0.2), `ND_TILE_SCALE` (auto aggressiveness, lower = more tiles),
`ND_TILE_NMS`. Cost: an N×M grid is ~N·M inferences per frame, so FPS drops accordingly —
it's for **accuracy/testing**, not max FPS. Also remember the model only finds classes it
was trained on (street cam → use a **COCO** model + `YOLO_FILTER=person,car`), and a high
`YOLO_CONF` can hide weak small-object hits — try `YOLO_CONF=0.25`.

```bash
sudo apt install -y ffmpeg          # required for URL ingestion
pip3 install -U yt-dlp              # only for site pages (YouTube/Twitch/…)

YOLO_SOURCE="https://youtu.be/XXXX"                ND_PARAM=… ND_BIN=… python3 nanodet_ncnn_sidecar.py
YOLO_SOURCE="https://cam/stream.m3u8"              …
YOLO_SOURCE="rtsp://192.168.1.10:554/h264"         …
YOLO_SOURCE="/home/dest/test_clip.mp4"             …
```
`YOLO_FFMPEG=0` falls back to OpenCV's own backend; `YOLO_YTDLP_FORMAT` overrides the
yt-dlp format (default `best`).

The decode mirrors RangiLyu/nanodet's `demo_ncnn`: per grid point, argmax class
score (sigmoid auto-applied if the export left logits), then each of the 4 box sides
is a softmax-integral over `reg_max+1` bins → distance from the cell centre. The
decode math is unit-tested against an independent reference (≤1e-14).

## 3. Train on your own data (one command)

YOLO-format dataset in, verified ncnn model out — same paradigm as
`train_yolofastest.py`.

```bash
# on your GPU box. Use Python 3.10 or 3.11 — torch/pytorch-lightning/pycocotools
# usually have NO wheels yet for 3.12+/3.14, which is the usual "No module named nanodet"
# / failed-install trap. Then:
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install opencv-python numpy onnx onnxsim ncnn pnnx

# edit the CONFIG block in train_nanodet.py (DATASET, CLASSES, INPUT, BATCH…), then:
python train_nanodet.py
```

You do **not** need to `pip install nanodet` yourself: the trainer clones the repo, puts it
on `PYTHONPATH` for the import, and auto-installs its requirements **minus torch** (so your
cu128 build is untouched) — getting the right pytorch-lightning/omegaconf/etc. versions.
`TRAIN_PIP=0` skips that auto-install.

Prefer a window? `python train_nanodet_gui.py` exposes every field — including
**fine-tuning / resume (дообучение)** — and streams the live training log underneath.
It just drives `train_nanodet.py` via `TRAIN_*` env vars, so both paths stay identical.

**Fine-tuning (дообучение):** point `WEIGHTS` (env `TRAIN_WEIGHTS`) at a finished
`.ckpt` to start from those weights and continue on new/expanded data, or `RESUME`
(env `TRAIN_RESUME`) to pick an interrupted run back up. In the GUI it's the
"Fine-tuning / resume" selector.
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

## Follow-me flight (Betaflight FC) — lock-on without a gimbal

Lock-on now works with **no gimbal at all** — a plain Pi camera. Press **Space**
(or the `track` button) and the sidecar locks the picked/clicked target and shows the
crosshair. With a **Betaflight** flight controller wired in (`flight_controller.py`),
that lock also **flies the drone**: it yaws to keep the person centred and pitches
forward/back to hold distance (box height = range proxy — person walks away ⇒ fly
forward, comes closer ⇒ back off).

**Multiple people in frame?** Click the one you want — the lock switches to that
detection (works with or without an FC); drag a box to lock an arbitrary object.

Use a **COCO model** so `person` is a class and add `YOLO_FILTER=person`. Enable the FC:

```bash
pip3 install pyserial
ND_PARAM=nanodet.param ND_BIN=nanodet.bin ND_INPUT=320 \
  YOLO_LABELS=coco.names YOLO_FILTER=person YOLO_SOURCE=rpicam \
  FC=betaflight FC_PORT=/dev/ttyAMA0 \
  python3 nanodet_ncnn_sidecar.py
# open http://<pi-ip>:8080 · Space = lock-on · click a person to pick · C = clear
```

| FC var | Meaning | Default |
|---|---|---|
| `FC` | `betaflight` to enable, else off | off |
| `FC_PORT` | serial port (`/dev/ttyAMA0` UART, `/dev/ttyACM0` USB) | `/dev/ttyAMA0` |
| `FC_BAUD` / `FC_RATE` | MSP baud / RC stream rate (Hz) | `115200` / `50` |
| `FC_MAX_YAW` / `FC_MAX_PITCH` / `FC_MAX_ROLL` | max stick offset from centre (µs); ROLL = manual strafe only | `150` / `120` / `120` |
| `FC_KP_YAW` / `FC_KP_PITCH` | P-gains (µs per unit error) | `360` / `500` |
| `FC_TARGET_FILL` | desired box-height / frame-height (distance setpoint) | `0.45` |
| `FC_YAW_DEADZONE` / `FC_FILL_DEADZONE` | no-move bands | `0.06` / `0.08` |
| `FC_INVERT_YAW` / `FC_INVERT_PITCH` | flip an axis if it moves the wrong way | off |
| `FC_CH_ROLL/PITCH/THROTTLE/YAW` | channel indices (Betaflight `map`, default AETR) | `0/1/2/3` |

### Panel ARM + manual control

Two safety interlocks gate the drone: the hardware **MSP Override** switch on the TX,
and a software **ARM** button on the web panel. Until ARM is on, the Pi streams centre
sticks only — for **both** autonomous follow and manual control. When an FC is present
the panel also shows a **d-pad** (▲/▼ forward/back = pitch, ⟲/⟳ = yaw, ◄/► = roll
strafe): held = move, released = recentre, and if the page is lost the sticks recentre
within ~0.4 s. A manual nudge briefly overrides the follower. HTTP API:
`/fcarm?on=1|0`, `/fcmove?yaw=&pitch=&roll=` (each -1..1), `/fcstop`.

**Transmitter sticks** (RadioMaster TX12 in USB joystick mode) drive the drone straight
from the panel via the browser **Gamepad API** — toggle **Sticks: ON**, set the axis/sign
mapping shown at the bottom of the panel. Endpoint `/fcsticks?roll=&pitch=&yaw=&throttle=`
(r/p/y -1..1, throttle 0..1 for full mode).

**Station-keeping (wind resist)** — the **HOLD** button turns on vision position-hold:
optical-flow (phase-correlation) estimates scene drift and commands opposite roll/pitch to
stay put. Experimental, mutually exclusive with follow, best with a **downward camera**.
Tune `FC_HOLD_KP`/`FC_HOLD_MAX`/`FC_HOLD_RES`/`FC_HOLD_LEAK`/`FC_HOLD_DEAD`,
flip `FC_HOLD_INVERT_X/Y`. Endpoint `/fchold?on=1|0`.

### Full RX=MSP mode (no transmitter) — `FC_MODE=full`

For boards without MSP Override. The Pi becomes the **whole** receiver and drives
roll/pitch/yaw **plus throttle and arm**; the panel ARM button arms the **motors** and
the panel gains **THR down/up** (a held throttle setpoint — you fly altitude). Three
safeties: software ARM, a ground-station watchdog (panel silent > `FC_LINK_TIMEOUT` =>
auto-descent + disarm via `FC_DESCENT_RATE`), and Betaflight's own failsafe if MSP
frames stop. Vars: `FC_MODE=full`, `FC_CH_ARM` (AUX1=4), `FC_ARM_US`/`FC_DISARM_US`,
`FC_THROTTLE_MIN`/`FC_THROTTLE_MAX`, `FC_THR_STEP`, `FC_LINK_TIMEOUT`, `FC_DESCENT_RATE`.
Extra endpoint: `/fcthrottle?d=+1|-1`. **Full step-by-step (Betaflight RX=MSP, arm
channel, Angle-always-on, props-off bench test, first flight) is in the main README**
([no MSP Override](../../README.md#если-в-прошивке-нет-msp-override)). Much higher risk —
no manual override; **bench-test props off.**

**Wiring (Pi 5 ↔ FC) and full Betaflight setup (MSP Override on an AUX switch, the
channel mask, failsafe) + a SAFETY checklist** are in the main README:
[Follow-me на дроне (Betaflight)](../../README.md#follow-me-дрон-следит-за-человеком-betaflight-fc).
The MSP framing, the follow controller and the ARM/manual gating are unit-tested; the
flight itself is not — **test props off first.**

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
