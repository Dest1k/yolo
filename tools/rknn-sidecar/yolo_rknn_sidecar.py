#!/usr/bin/env python3
"""
YOLO RKNN sidecar — NPU-accelerated detector for Rockchip boards (Orange Pi 5 /
RK3588, 6 TOPS NPU).

This is the NPU counterpart to the JVM headless runner (`:desktop:runHeadless`).
It runs a `.rknn` YOLO model on the NPU and broadcasts an annotated MJPEG stream
on the LAN — open `http://<board-ip>:8080` in a browser/VLC, exactly like the
Java app. Capture and inference are decoupled (stream at camera FPS, detection on
the latest frame only), and a small IoU tracker stabilises boxes — mirroring the
desktop pipeline so behaviour is consistent across the project.

Why a sidecar (not JVM): the RKNN runtime is a C library with no maintained Java
binding; driving the NPU from Python (rknn-toolkit-lite2) is the reliable path.

Config via environment variables (same names as the JVM app where they overlap):
  YOLO_MODEL    path to a .rknn model                         [required]
  YOLO_SOURCE   camera index "0"/"1", a GStreamer/V4L2 pipeline string,
                or an http MJPEG URL                          (default: "0")
  YOLO_INPUT    model input size (square)                     (default: 640)
  YOLO_CLASSES  number of classes                             (default: 80)
  YOLO_CONF     confidence threshold 0..1                     (default: 0.25)
  YOLO_NMS      NMS IoU threshold 0..1                         (default: 0.45)
  YOLO_PORT     MJPEG server port                             (default: 8080)
  YOLO_CAM_W / YOLO_CAM_H / YOLO_CAM_FPS  capture geometry    (default: 640x480x30)
  YOLO_TRACK    on | off  — IoU tracking / box persistence    (default: on)

Convert an ONNX model to .rknn on an x86 Linux host with rknn-toolkit2 — see the
README next to this file.
"""

import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2

COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]
PALETTE = [(80,80,255),(80,200,80),(255,120,80),(0,200,255),(200,0,200),(200,200,0)]


def env(key, default=None):
    v = os.environ.get(key)
    return v.strip() if v and v.strip() else default


def label_for(cls):
    return COCO[cls] if 0 <= cls < len(COCO) else f"cls{cls}"


# ── Letterbox preprocessing (aspect-preserving, gray pad) ────────────────────
def letterbox(img, size, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[dh:dh + nh, dw:dw + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, dw, dh


# ── Output decoding (mirrors the JVM OnnxDetector logic) ─────────────────────
def _looks_pixel(values):
    sample = values[:100]
    return bool(np.nanmax(sample) > 1.5) if sample.size else False


def decode(outputs, r, dw, dh, ow, oh, in_sz, conf_th, nms_th, num_classes):
    a = np.squeeze(outputs[0])
    if a.ndim == 1:
        return []
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1) if a.shape[0] < a.shape[-1] else a.reshape(-1, a.shape[-1])
    d0, d1 = a.shape

    # NMS-free [N,6] = x1,y1,x2,y2,score,cls (YOLOv10-style)
    if d1 == 6 and d0 != 6:
        return _decode_nmsfree(a, r, dw, dh, ow, oh, in_sz, conf_th, num_classes)
    if d0 == 6 and d1 != 6:
        return _decode_nmsfree(a.T, r, dw, dh, ow, oh, in_sz, conf_th, num_classes)

    # anchor-free [N, 4+nc] (attrs is the smaller axis) — needs NMS
    if d0 < d1:
        a = a.T
    return _decode_anchorfree(a, r, dw, dh, ow, oh, in_sz, conf_th, nms_th, num_classes)


def _back(x1, y1, x2, y2, dw, dh, r, ow, oh):
    x1 = np.clip((x1 - dw) / r, 0, ow)
    y1 = np.clip((y1 - dh) / r, 0, oh)
    x2 = np.clip((x2 - dw) / r, 0, ow)
    y2 = np.clip((y2 - dh) / r, 0, oh)
    return x1, y1, x2, y2


def _decode_nmsfree(a, r, dw, dh, ow, oh, in_sz, conf_th, num_classes):
    scores = a[:, 4]
    keep = scores >= conf_th
    a = a[keep]
    if a.shape[0] == 0:
        return []
    sc = in_sz if not _looks_pixel(a[:, 2]) else 1.0
    x1, y1, x2, y2 = _back(a[:, 0] * sc, a[:, 1] * sc, a[:, 2] * sc, a[:, 3] * sc, dw, dh, r, ow, oh)
    cls = np.clip(a[:, 5].astype(int), 0, num_classes - 1)
    out = []
    for i in range(a.shape[0]):
        if x2[i] > x1[i] and y2[i] > y1[i]:
            out.append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(a[i, 4]), int(cls[i])))
    return out


def _decode_anchorfree(a, r, dw, dh, ow, oh, in_sz, conf_th, nms_th, num_classes):
    nc = max(1, a.shape[1] - 4)
    cls = np.argmax(a[:, 4:4 + nc], axis=1)
    conf = np.max(a[:, 4:4 + nc], axis=1)
    keep = conf >= conf_th
    a, cls, conf = a[keep], cls[keep], conf[keep]
    if a.shape[0] == 0:
        return []
    sc = in_sz if not _looks_pixel(a[:, 2]) else 1.0
    cx, cy, bw, bh = a[:, 0] * sc, a[:, 1] * sc, a[:, 2] * sc, a[:, 3] * sc
    x1, y1, x2, y2 = _back(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, dw, dh, r, ow, oh)
    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(boxes, conf.tolist(), conf_th, nms_th)
    out = []
    for i in np.array(idxs).flatten().astype(int):
        out.append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(conf[i]), int(cls[i])))
    return out


# ── Simple IoU tracker (runtime stand-in for model.track()) ──────────────────
class Tracker:
    def __init__(self, hold_s=0.8, iou_th=0.3):
        self.hold = hold_s
        self.iou_th = iou_th
        self.tracks = []  # list of [box, last_seen]

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, dets, now):
        matched = [False] * len(self.tracks)
        for d in dets:
            best, best_iou = -1, self.iou_th
            for i, (box, _) in enumerate(self.tracks):
                if matched[i] or box[5] != d[5]:
                    continue
                v = self._iou(box, d)
                if v >= best_iou:
                    best, best_iou = i, v
            if best >= 0:
                self.tracks[best] = [d, now]
                matched[best] = True
            else:
                self.tracks.append([d, now])
        self.tracks = [t for t in self.tracks if now - t[1] <= self.hold]
        return [t[0] for t in self.tracks]


def draw(frame, dets, hud):
    for (x1, y1, x2, y2, conf, cls) in dets:
        color = PALETTE[cls % len(PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        text = f"{label_for(cls)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), color, -1)
        cv2.putText(frame, text, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if hud:
        cv2.rectangle(frame, (2, frame.shape[0] - 24), (2 + 9 * len(hud), frame.shape[0] - 2), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, frame.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (118, 230, 0), 2)
    return frame


# ── Shared state between capture / inference / stream threads ─────────────────
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None        # latest raw BGR frame
        self.dets = []           # latest detections
        self.stream_fps = 0
        self.det_fps = 0
        self.running = True


def open_source(src, w, h, fps):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap
    if src.startswith("http"):
        return cv2.VideoCapture(src)
    # otherwise treat as a GStreamer/V4L2 pipeline string
    return cv2.VideoCapture(src, cv2.CAP_GSTREAMER)


def capture_loop(state, cap):
    count, t0 = 0, time.time()
    while state.running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        with state.lock:
            state.frame = frame
        count += 1
        dt = time.time() - t0
        if dt >= 1.0:
            state.stream_fps = int(count / dt)
            count, t0 = 0, time.time()
    cap.release()


def inference_loop(state, rknn, in_sz, conf, nms, num_classes, track_on):
    tracker = Tracker() if track_on else None
    count, t0 = 0, time.time()
    while state.running:
        with state.lock:
            frame = None if state.frame is None else state.frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        oh, ow = frame.shape[:2]
        lb, r, dw, dh = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), in_sz)
        try:
            outputs = rknn.inference(inputs=[lb])
            dets = decode(outputs, r, dw, dh, ow, oh, in_sz, conf, nms, num_classes)
        except Exception as e:           # keep the stream alive on a bad frame
            sys.stderr.write(f"inference error: {e}\n")
            dets = []
        if tracker is not None:
            dets = tracker.update(dets, time.time())
        with state.lock:
            state.dets = dets
        count += 1
        dt = time.time() - t0
        if dt >= 1.0:
            state.det_fps = int(count / dt)
            count, t0 = 0, time.time()


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--mjpeg")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while state.running:
                    with state.lock:
                        frame = None if state.frame is None else state.frame.copy()
                        dets = list(state.dets)
                        hud = f"FPS {state.stream_fps}  |  det {state.det_fps}"
                    if frame is None:
                        time.sleep(0.02)
                        continue
                    draw(frame, dets, hud)
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if not ok:
                        continue
                    self.wfile.write(b"--mjpeg\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
                    time.sleep(0.005)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def lan_ips():
    ips = []
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips or ["<board-ip>"]


def main():
    model = env("YOLO_MODEL")
    if not model:
        sys.stderr.write("ERROR: set YOLO_MODEL to your .rknn model file\n")
        sys.exit(2)
    source = env("YOLO_SOURCE", "0")
    in_sz = int(env("YOLO_INPUT", "640"))
    num_classes = int(env("YOLO_CLASSES", "80"))
    conf = float(env("YOLO_CONF", "0.25"))
    nms = float(env("YOLO_NMS", "0.45"))
    port = int(env("YOLO_PORT", "8080"))
    cam_w = int(env("YOLO_CAM_W", "640"))
    cam_h = int(env("YOLO_CAM_H", "480"))
    cam_fps = int(env("YOLO_CAM_FPS", "30"))
    track_on = env("YOLO_TRACK", "on").lower() != "off"

    print("YOLO RKNN sidecar (NPU)")
    print(f"  model={model} source={source} input={in_sz} classes={num_classes} "
          f"conf={conf} nms={nms} port={port} track={'on' if track_on else 'off'}")

    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        sys.stderr.write("ERROR: rknn-toolkit-lite2 not installed. See the README.\n")
        sys.exit(1)

    rknn = RKNNLite()
    if rknn.load_rknn(model) != 0:
        sys.stderr.write("ERROR: failed to load .rknn model\n")
        sys.exit(1)
    if rknn.init_runtime() != 0:
        sys.stderr.write("ERROR: failed to init NPU runtime\n")
        sys.exit(1)
    print("  NPU runtime ready")

    cap = open_source(source, cam_w, cam_h, cam_fps)
    if not cap.isOpened():
        sys.stderr.write(f"ERROR: cannot open video source '{source}'\n")
        sys.exit(1)

    state = State()
    threading.Thread(target=capture_loop, args=(state, cap), daemon=True).start()
    threading.Thread(target=inference_loop,
                     args=(state, rknn, in_sz, conf, nms, num_classes, track_on),
                     daemon=True).start()

    for ip in lan_ips():
        print(f"  stream: http://{ip}:{port}")
    print("  (open a stream URL above in a browser or VLC on the same network)")

    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        rknn.release()


if __name__ == "__main__":
    main()
