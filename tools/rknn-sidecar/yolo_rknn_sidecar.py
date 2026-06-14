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
  YOLO_CLASSES  number of classes                  (default: labels count, else 80)
  YOLO_LABELS   path to labels.txt (one class per line) for custom models
  YOLO_FILTER   keep only these classes (comma-separated indices or names)
  YOLO_CONF     confidence threshold 0..1                     (default: 0.25)
  YOLO_NMS      NMS IoU threshold 0..1                         (default: 0.45)
  YOLO_PORT     MJPEG server / control panel port             (default: 8080)
  YOLO_JPEG_Q   MJPEG quality 1..100                          (default: 75)
  YOLO_CAM_W / YOLO_CAM_H / YOLO_CAM_FPS  capture geometry    (default: 640x480x30)
  YOLO_TRACK    on | off  — IoU tracking / box persistence    (default: on)

SIYI gimbal control (same as the JVM headless), served on the same port at "/":
  YOLO_GIMBAL   on | off  — enable control + web panel (auto-on for SIYI source)
  YOLO_GIMBAL_HOST / YOLO_GIMBAL_PORT          (default: 192.168.144.25 / 37260)
  YOLO_TRACK_SPEED                              max follow speed (default: 40)
  YOLO_TRACK_INVERT_YAW / YOLO_TRACK_INVERT_PITCH   flip an axis if it chases away
  Follow a target: Space toggles auto-follow, or click the video to lock a target.

Convert an ONNX model to .rknn on an x86 Linux host with rknn-toolkit2 — see the
README next to this file.
"""

import os
import sys
import time
import socket
import struct
import math
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

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


def label_for(cls, labels=None):
    if labels is not None and 0 <= cls < len(labels):
        return labels[cls]
    return COCO[cls] if 0 <= cls < len(COCO) else f"cls{cls}"


def parse_filter(spec, labels):
    """Set of class indices to keep, from comma-separated indices or names."""
    if not spec:
        return None
    names = labels if labels else COCO
    out = set()
    for tok in spec.split(","):
        t = tok.strip()
        if t.isdigit():
            out.add(int(t))
        else:
            for i, n in enumerate(names):
                if n.lower() == t.lower():
                    out.add(i); break
    return out or None


def load_labels(path):
    if not path:
        return None
    try:
        with open(path) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names or None
    except OSError as e:
        sys.stderr.write(f"WARNING: can't read labels '{path}': {e}\n")
        return None


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
    attrs = a.shape[1]
    # YOLOv5/v6 have an objectness channel (attrs = 5+nc); v8/v9/v11 don't (4+nc).
    has_obj = attrs == num_classes + 5
    cls_start = 5 if has_obj else 4
    nc = num_classes if has_obj else max(1, attrs - 4)
    cls = np.argmax(a[:, cls_start:cls_start + nc], axis=1)
    conf = np.max(a[:, cls_start:cls_start + nc], axis=1)
    if has_obj:
        conf = conf * a[:, 4]
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
        n = len(self.tracks)            # match only against pre-existing tracks
        matched = [False] * n
        for d in dets:
            best, best_iou = -1, self.iou_th
            for i in range(n):
                box = self.tracks[i][0]
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


# ── SIYI gimbal (UDP SDK) — mirrors the JVM SiyiGimbal ───────────────────────
class SiyiGimbal:
    def __init__(self, host="192.168.144.25", port=37260):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.seq = 0
        self._lock = threading.Lock()
        self.running = True
        self.yaw = self.pitch = self.roll = 0.0
        self.firmware = ""
        self.hardware_id = ""
        self.recording = False
        self.motion_mode = -1
        threading.Thread(target=self._rx, daemon=True).start()
        self.request_hardware_id(); self.request_firmware()
        self.request_config(); self.request_attitude()

    @staticmethod
    def _crc16(data):                       # CRC16/XMODEM (poly 0x1021, init 0)
        crc = 0
        for b in data:
            crc ^= (b << 8)
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
        return crc & 0xFFFF

    def _send(self, cmd, data=b""):
        with self._lock:
            s = self.seq & 0xFFFF
            self.seq += 1
        body = bytes([0x55, 0x66, 0x01]) + struct.pack("<H", len(data)) + \
            struct.pack("<H", s) + bytes([cmd & 0xFF]) + data
        frame = body + struct.pack("<H", self._crc16(body))
        try:
            self.sock.sendto(frame, self.addr)
        except OSError:
            pass

    @staticmethod
    def _i8(v):
        return max(-1, min(1, int(v)))

    @staticmethod
    def _sp(v):
        return max(-100, min(100, int(v)))

    def request_firmware(self):    self._send(0x01)
    def request_hardware_id(self): self._send(0x02)
    def autofocus(self):           self._send(0x04, bytes([1]))
    def manual_zoom(self, d):      self._send(0x05, struct.pack("b", self._i8(d)))
    def manual_focus(self, d):     self._send(0x06, struct.pack("b", self._i8(d)))
    def rotate(self, yaw, pitch):  self._send(0x07, struct.pack("bb", self._sp(yaw), self._sp(pitch)))
    def stop_rotation(self):       self.rotate(0, 0)
    def center(self):              self._send(0x08, bytes([1]))
    def request_config(self):      self._send(0x0A)
    def take_photo(self):          self._send(0x0C, bytes([0]))
    def toggle_hdr(self):          self._send(0x0C, bytes([1]))
    def toggle_record(self):       self._send(0x0C, bytes([2]))
    def set_lock(self):            self._send(0x0C, bytes([3]))
    def set_follow(self):          self._send(0x0C, bytes([4]))
    def set_fpv(self):             self._send(0x0C, bytes([5]))
    def request_attitude(self):    self._send(0x0D)

    def set_angle(self, yaw, pitch):
        y = int(max(-135.0, min(135.0, yaw)) * 10)
        p = int(max(-90.0, min(25.0, pitch)) * 10)
        self._send(0x0E, struct.pack("<hh", y, p))

    def absolute_zoom(self, x):
        x = max(1.0, x); ip = int(x); fr = int((x - ip) * 10)
        self._send(0x0F, bytes([ip & 0xFF, fr & 0xFF]))

    def _rx(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break
            self._parse(data)

    def _parse(self, b):
        if len(b) < 10 or b[0] != 0x55 or b[1] != 0x66:
            return
        dlen = b[3] | (b[4] << 8); cmd = b[7]; d = 8
        if 8 + dlen + 2 > len(b):
            return
        if cmd == 0x01:
            self.firmware = "".join(f"{b[d + i]:02x}" for i in range(min(dlen, 12)))
        elif cmd == 0x02:
            self.hardware_id = bytes(b[d:d + min(dlen, 16)]).decode("ascii", "ignore").strip()
        elif cmd == 0x0A and dlen >= 5:
            self.recording = b[d + 3] == 1
            self.motion_mode = b[d + 4]
        elif cmd == 0x0D and dlen >= 6:
            self.yaw = struct.unpack_from("<h", b, d)[0] / 10.0
            self.pitch = struct.unpack_from("<h", b, d + 2)[0] / 10.0
            self.roll = struct.unpack_from("<h", b, d + 4)[0] / 10.0

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass


# ── Visual-servoing follower — mirrors the JVM GimbalFollower ─────────────────
class GimbalFollower:
    def __init__(self, gimbal, max_speed=40, deadzone=0.05, stable_ticks=3,
                 invert_yaw=False, invert_pitch=False):
        self.g = gimbal
        self.max_speed = max_speed
        self.deadzone = deadzone
        self.stable_ticks = stable_ticks
        self.invert_yaw = invert_yaw
        self.invert_pitch = invert_pitch
        self.prev = None
        self.lock_count = 0
        self.moving = False
        self.pending = None          # (nx, ny) normalised click point for manual pick

    def request_pick(self, nx, ny):
        self.pending = (nx, ny)

    def step(self, dets, fw, fh):
        if fw <= 0 or fh <= 0 or not dets:
            self.stop(); self.prev = None; self.lock_count = 0; return None
        if self.pending is not None:
            px, py = self.pending[0] * fw, self.pending[1] * fh
            self.pending = None
            t = self._pick_at(dets, px, py)
            if t is not None:
                self.prev = t; self.lock_count = 0
        t = self._pick(dets, self.prev, fw)
        self.prev = t
        if t is None:
            self.stop(); self.lock_count = 0; return None
        self.lock_count += 1
        if self.lock_count < self.stable_ticks:
            self.stop(); return t
        cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
        ex, ey = cx / fw - 0.5, cy / fh - 0.5
        if abs(ex) < self.deadzone and abs(ey) < self.deadzone:
            self.stop(); return t
        gain = 2 * self.max_speed
        ys = int(max(-self.max_speed, min(self.max_speed, ex * gain)))
        ps = int(max(-self.max_speed, min(self.max_speed, -ey * gain)))
        if self.invert_yaw: ys = -ys
        if self.invert_pitch: ps = -ps
        self.g.rotate(ys, ps); self.moving = True
        return t

    def stop(self):
        if self.moving:
            self.g.stop_rotation(); self.moving = False

    @staticmethod
    def _pick_at(dets, px, py):
        inside = [d for d in dets if d[0] <= px <= d[2] and d[1] <= py <= d[3]]
        pool = inside if inside else dets
        return min(pool, key=lambda d: ((d[0] + d[2]) / 2 - px) ** 2 + ((d[1] + d[3]) / 2 - py) ** 2)

    def _pick(self, dets, prev, fw):
        if prev is not None:
            near = min(dets, key=lambda d: self._cdist(d, prev))
            if self._cdist(near, prev) < 0.3 * fw:
                return near
        return max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))

    @staticmethod
    def _cdist(a, b):
        return math.hypot(((a[0] + a[2]) - (b[0] + b[2])) / 2, ((a[1] + a[3]) - (b[1] + b[3])) / 2)


def draw(frame, dets, hud, labels=None, tracking=False, target=None):
    for (x1, y1, x2, y2, conf, cls) in dets:
        color = PALETTE[cls % len(PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        text = f"{label_for(cls, labels)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), color, -1)
        cv2.putText(frame, text, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if tracking:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 16, cy), (cx + 16, cy), (60, 60, 255), 2)
        cv2.line(frame, (cx, cy - 16), (cx, cy + 16), (60, 60, 255), 2)
        cv2.circle(frame, (cx, cy), 6, (60, 60, 255), 2)
        if target is not None:
            tx1, ty1, tx2, ty2 = int(target[0]), int(target[1]), int(target[2]), int(target[3])
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 230, 255), 3)
            cv2.line(frame, (cx, cy), ((tx1 + tx2) // 2, (ty1 + ty2) // 2), (60, 60, 255), 1)
        cv2.putText(frame, "TRACKING", (cx - 48, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
    if hud:
        cv2.rectangle(frame, (2, frame.shape[0] - 24), (2 + 9 * len(hud), frame.shape[0] - 2), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, frame.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (118, 230, 0), 2)
    return frame


# ── Shared state between capture / inference / stream threads ─────────────────
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None; self.frame_seq = 0        # latest raw BGR frame
        self.dets = []           # latest detections
        self.stream_fps = 0
        self.det_fps = 0
        self.labels = None
        self.jpeg_q = 75
        self.gimbal = None       # SiyiGimbal or None
        self.follower = None     # GimbalFollower or None
        self.tracking = False    # auto-follow on/off (Space / click)
        self.target = None       # currently tracked box (for drawing)
        self.running = True


def open_source(src, w, h, fps):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    if src.startswith(("http", "rtsp")):
        cap = cv2.VideoCapture(src); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    # otherwise treat as a GStreamer/V4L2 pipeline string
    return cv2.VideoCapture(src, cv2.CAP_GSTREAMER)


def follow_loop(state):
    """When tracking is on, steer the gimbal to keep the target centred."""
    while state.running:
        if state.tracking and state.follower is not None:
            with state.lock:
                frame = state.frame
                dets = list(state.dets)
            if frame is not None:
                h, w = frame.shape[:2]
                state.target = state.follower.step(dets, w, h)
            else:
                state.target = None
        else:
            if state.follower is not None:
                state.follower.stop()
            state.target = None
        time.sleep(0.066)


def capture_loop(state, cap):
    count, t0 = 0, time.time()
    while state.running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        with state.lock:
            state.frame = frame; state.frame_seq += 1
        count += 1
        dt = time.time() - t0
        if dt >= 1.0:
            state.stream_fps = int(count / dt)
            count, t0 = 0, time.time()
    cap.release()


def inference_loop(state, rknn, in_sz, conf, nms, num_classes, track_on, filter_set=None):
    tracker = Tracker(hold_s=float(env("YOLO_TRACK_HOLD", "0.3"))) if track_on else None
    count, t0 = 0, time.time(); last_seq = -1
    while state.running:
        with state.lock:
            seq = state.frame_seq
            frame = None if (state.frame is None or seq == last_seq) else state.frame.copy()
        last_seq = seq
        if frame is None:
            time.sleep(0.01)
            continue
        oh, ow = frame.shape[:2]
        lb, r, dw, dh = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), in_sz)
        try:
            outputs = rknn.inference(inputs=[lb])
            dets = decode(outputs, r, dw, dh, ow, oh, in_sz, conf, nms, num_classes)
            if filter_set is not None:
                dets = [d for d in dets if d[5] in filter_set]
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


def _fin(v):
    """Coerce to a JSON-safe finite float (NaN/Inf -> 0.0)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _status_json(state):
    g = state.gimbal
    mode = {0: "lock", 1: "follow", 2: "fpv"}.get(g.motion_mode if g else -1, "?")
    # Escape gimbal-reported text + guard floats so a stray byte can't yield invalid
    # JSON and hang the panel on "loading…".
    return json.dumps({
        "yaw": _fin(g.yaw) if g else 0.0,
        "pitch": _fin(g.pitch) if g else 0.0,
        "roll": _fin(g.roll) if g else 0.0,
        "recording": bool(g.recording) if g else False,
        "mode": mode,
        "tracking": bool(state.tracking),
        "streamFps": int(state.stream_fps), "detFps": int(state.det_fps), "ndet": len(state.dets),
        "firmware": (g.firmware if g else "") or "",
        "hardwareId": (g.hardware_id if g else "") or "",
    }, allow_nan=False)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else body.encode())

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--mjpeg")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_seq = -1
            try:
                while state.running:
                    with state.lock:
                        seq = state.frame_seq
                        frame = state.frame.copy() if (seq != last_seq and state.frame is not None) else None
                        dets = list(state.dets)
                        hud = f"FPS {state.stream_fps}  |  det {state.det_fps}"
                        tracking, target, q = state.tracking, state.target, state.jpeg_q
                    if frame is None:
                        time.sleep(0.003); continue   # only NEW frames -> low latency
                    last_seq = seq
                    draw(frame, dets, hud, state.labels, tracking, target)
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
                    if not ok:
                        continue
                    self.wfile.write(b"--mjpeg\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpg.tobytes()); self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            u = urlparse(self.path); path = u.path
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if path == "/stream" or (path == "/" and state.gimbal is None):
                self._stream(); return
            if path == "/":
                self._send(200, "text/html; charset=utf-8", PANEL); return

            g = state.gimbal
            fol = state.follower
            try:
                if path in ("/status", "/attitude"):
                    if path == "/attitude" and g: g.request_attitude()
                elif g is None:
                    pass
                elif path == "/rotate":  g.rotate(int(float(q.get("yaw", 0))), int(float(q.get("pitch", 0))))
                elif path == "/stop":    g.stop_rotation()
                elif path == "/angle":   g.set_angle(float(q.get("yaw", 0)), float(q.get("pitch", 0)))
                elif path == "/center":  g.center()
                elif path == "/zoom":
                    if "x" in q: g.absolute_zoom(float(q["x"]))
                    elif q.get("dir") == "in": g.manual_zoom(1)
                    elif q.get("dir") == "out": g.manual_zoom(-1)
                    else: g.manual_zoom(0)
                elif path == "/focus":
                    g.manual_focus(1 if q.get("dir") == "far" else -1 if q.get("dir") == "near" else 0)
                elif path == "/autofocus": g.autofocus()
                elif path == "/photo":   g.take_photo()
                elif path == "/record":  g.toggle_record()
                elif path == "/hdr":     g.toggle_hdr()
                elif path == "/mode":
                    m = q.get("m")
                    (g.set_lock if m == "lock" else g.set_follow if m == "follow"
                     else g.set_fpv if m == "fpv" else (lambda: None))()
                elif path == "/track":
                    on = q.get("on")
                    state.tracking = (on in ("1", "true")) if on is not None else (not state.tracking)
                elif path == "/pick":
                    if fol is not None and "nx" in q and "ny" in q:
                        fol.request_pick(float(q["nx"]), float(q["ny"])); state.tracking = True
                elif path == "/config":  g.request_config()
                elif path == "/version": g.request_firmware(); g.request_hardware_id()
                else:
                    self._send(404, "text/plain", "not found"); return
                self._send(200, "application/json", _status_json(state))
            except Exception as e:
                self._send(500, "text/plain", f"error: {e}")
    return Handler


PANEL = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SIYI Gimbal</title><style>
*{box-sizing:border-box}body{margin:0;background:#000;font-family:system-ui,sans-serif;color:#eee;overflow:hidden}
#vid{position:fixed;inset:0;width:100vw;height:100vh;object-fit:contain;background:#000;cursor:crosshair}
.ov{position:fixed;z-index:2}button{background:rgba(20,20,20,.6);color:#eee;border:1px solid #777;border-radius:10px;
padding:12px 14px;font-size:16px;margin:3px;cursor:pointer}button:active{background:#00e676;color:#000}
#st{top:8px;left:8px;font-family:monospace;white-space:pre;background:rgba(0,0,0,.5);padding:8px 10px;border-radius:8px;font-size:13px}
#trk{top:8px;left:50%;transform:translateX(-50%);padding:6px 12px;border-radius:8px;font-weight:bold;background:rgba(0,0,0,.5)}
#pad{bottom:12px;left:12px;display:grid;grid-template-columns:repeat(3,56px);gap:4px}
#side{bottom:12px;right:12px;display:flex;flex-direction:column;align-items:flex-end}
#top{top:8px;right:8px;display:flex;flex-wrap:wrap;justify-content:flex-end;max-width:60vw}
input{width:52px;background:rgba(0,0,0,.5);color:#eee;border:1px solid #777;border-radius:6px;padding:6px}
.grp{display:flex;align-items:center;margin:2px 0}
</style></head><body>
<img id="vid" alt="stream">
<div id="st" class="ov">loading...</div>
<div id="trk" class="ov">TRACK OFF <span style="opacity:.7">(Space / click)</span></div>
<div id="pad" class="ov">
<span></span><button data-y="0" data-p="60">^</button><span></span>
<button data-y="-60" data-p="0">&lt;</button><button onclick="g('/center')">o</button><button data-y="60" data-p="0">&gt;</button>
<span></span><button data-y="0" data-p="-60">v</button><span></span></div>
<div id="top" class="ov">
<button onclick="g('/track')">track</button>
<button onclick="g('/mode?m=lock')">lock</button><button onclick="g('/mode?m=follow')">follow</button><button onclick="g('/mode?m=fpv')">fpv</button>
<button onclick="g('/photo')">photo</button><button onclick="g('/record')">rec</button><button onclick="g('/hdr')">hdr</button></div>
<div id="side" class="ov">
<div class="grp"><button id="zin">+</button><button id="zout">-</button>x<input id="zx" value="2"><button onclick="g('/zoom?x='+zx.value)">set</button></div>
<div class="grp"><button onclick="g('/autofocus')">AF</button><button id="ff">focus+</button><button id="fn">focus-</button></div>
<div class="grp">yaw<input id="ay" value="0">pitch<input id="ap" value="0"><button onclick="g('/angle?yaw='+ay.value+'&pitch='+ap.value)">go</button></div></div>
<script>
var vid=document.getElementById('vid');vid.src='/stream';var gotStatus=false;
function g(u){return fetch(u).then(function(r){if(!r.ok)throw 0;return r.json()}).then(show).catch(fail)}
function fail(){if(!gotStatus)document.getElementById('st').textContent='disconnected — retrying…'}
function show(s){gotStatus=true;document.getElementById('st').textContent=
 'yaw '+s.yaw+'  pitch '+s.pitch+'  roll '+s.roll+'\\nmode '+s.mode+'  rec '+s.recording+'\\nfw '+s.firmware+' hw '+s.hardwareId;
 var t=document.getElementById('trk');t.firstChild.nodeValue=(s.tracking?'* TRACK ON ':'TRACK OFF ');
 t.style.background=s.tracking?'rgba(255,40,40,.75)':'rgba(0,0,0,.5)'}
function hold(el,dn,up){if(!el)return;el.onmousedown=dn;el.onmouseup=up;el.onmouseleave=up;
 el.ontouchstart=function(e){e.preventDefault();dn()};el.ontouchend=function(e){e.preventDefault();up()}}
document.querySelectorAll('#pad button[data-y]').forEach(function(b){
 hold(b,function(){g('/rotate?yaw='+b.dataset.y+'&pitch='+b.dataset.p)},function(){g('/stop')})});
hold(document.getElementById('zin'),function(){g('/zoom?dir=in')},function(){g('/zoom?dir=stop')});
hold(document.getElementById('zout'),function(){g('/zoom?dir=out')},function(){g('/zoom?dir=stop')});
hold(document.getElementById('ff'),function(){g('/focus?dir=far')},function(){g('/focus?dir=stop')});
hold(document.getElementById('fn'),function(){g('/focus?dir=near')},function(){g('/focus?dir=stop')});
document.addEventListener('keydown',function(e){if(e.code==='Space'||e.key===' '){e.preventDefault();g('/track')}});
vid.addEventListener('click',function(e){var nw=vid.naturalWidth,nh=vid.naturalHeight;if(!nw||!nh)return;
 var r=vid.getBoundingClientRect();var sc=Math.min(r.width/nw,r.height/nh);var dw=nw*sc,dh=nh*sc;
 var ox=r.left+(r.width-dw)/2,oy=r.top+(r.height-dh)/2;var x=(e.clientX-ox)/dw,y=(e.clientY-oy)/dh;
 if(x<0||x>1||y<0||y>1)return;g('/pick?nx='+x.toFixed(4)+'&ny='+y.toFixed(4))});
function poll(){fetch('/status').then(function(r){if(!r.ok)throw 0;return r.json()}).then(show).catch(fail)}
poll();setInterval(poll,1000);
</script></body></html>"""


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
    labels = load_labels(env("YOLO_LABELS"))
    num_classes = int(env("YOLO_CLASSES")) if env("YOLO_CLASSES") else (len(labels) if labels else 80)
    filter_set = parse_filter(env("YOLO_FILTER"), labels)
    conf = float(env("YOLO_CONF", "0.25"))
    nms = float(env("YOLO_NMS", "0.45"))
    port = int(env("YOLO_PORT", "8080"))
    cam_w = int(env("YOLO_CAM_W", "640"))
    cam_h = int(env("YOLO_CAM_H", "480"))
    cam_fps = int(env("YOLO_CAM_FPS", "30"))
    track_on = env("YOLO_TRACK", "on").lower() != "off"
    jpeg_q = int(env("YOLO_JPEG_Q", "75"))
    gimbal_env = (env("YOLO_GIMBAL", "") or "").lower()
    gimbal_on = gimbal_env == "on" or (gimbal_env != "off" and "192.168.144.25" in source)

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

    cv2.setNumThreads(int(env("YOLO_CV_THREADS", "1")))  # don't let OpenCV oversubscribe the inference cores
    state = State()
    state.labels = labels
    state.jpeg_q = jpeg_q
    if labels:
        print(f"  labels: {len(labels)} custom classes")
    if filter_set is not None:
        print(f"  filter: only classes {sorted(filter_set)}")

    if gimbal_on:
        g_host = env("YOLO_GIMBAL_HOST", "192.168.144.25")
        g_port = int(env("YOLO_GIMBAL_PORT", "37260"))
        state.gimbal = SiyiGimbal(g_host, g_port)
        state.follower = GimbalFollower(
            state.gimbal,
            max_speed=int(env("YOLO_TRACK_SPEED", "40")),
            invert_yaw=env("YOLO_TRACK_INVERT_YAW", "off").lower() == "on",
            invert_pitch=env("YOLO_TRACK_INVERT_PITCH", "off").lower() == "on")
        threading.Thread(target=follow_loop, args=(state,), daemon=True).start()
        print(f"  gimbal control: enabled ({g_host}:{g_port}) — Space/click to track")

    threading.Thread(target=capture_loop, args=(state, cap), daemon=True).start()
    threading.Thread(target=inference_loop,
                     args=(state, rknn, in_sz, conf, nms, num_classes, track_on, filter_set),
                     daemon=True).start()

    for ip in lan_ips():
        print(f"  open: http://{ip}:{port}   (video + gimbal control)" if gimbal_on
              else f"  stream: http://{ip}:{port}")
    print("  (open the URL above in a browser on the same network)")

    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if state.gimbal is not None:
            state.gimbal.close()
        rknn.release()


if __name__ == "__main__":
    main()
