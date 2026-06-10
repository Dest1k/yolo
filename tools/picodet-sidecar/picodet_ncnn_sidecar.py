#!/usr/bin/env python3
"""
PicoDet (NCNN) sidecar — fast CPU object detection for Raspberry Pi 5.

Runs a PicoDet model (PaddleDetection) on the NCNN runtime — PicoDet has
excellent NCNN support and is one of the faster detectors on ARM CPUs — and
broadcasts an annotated MJPEG stream + the same web panel as the other sidecars
(manual drag-to-lock capture, SIYI gimbal control). Open http://<board-ip>:8080.

Why NCNN (not the YOLO ONNX path): PicoDet's output is NOT YOLO-shaped — it's a
GFL head (per-stride class scores + a discrete box-distance distribution). NCNN
can't run PicoDet's baked-in NMS op, so this sidecar decodes the head itself
(softmax over the distance bins → distance2bbox → NMS), exactly like the canonical
ncnn picodet example. The decode is fully parametrised because blob names/strides
depend on how YOU exported the model — run with --inspect first (see README).

Config via environment variables (YOLO_* shared with the other sidecars):
  PICODET_PARAM   path to the .param file                       [required]
  PICODET_BIN     path to the .bin file                         [required]
  PICODET_INPUT   square input size (PicoDet-S 320/416, L 640)  (default 416)
  PICODET_STRIDES FPN strides                                   (default 8,16,32,64)
  PICODET_REG_MAX box distribution bins-1                       (default 7)
  PICODET_CELL_OFFSET grid-cell centre offset (try 0 if boxes are shifted) (default 0.5)
  PICODET_CLS_BLOBS comma list of class-score output blob names, per stride
  PICODET_REG_BLOBS comma list of box-distribution output blob names, per stride
                    (from --inspect; if unset, tries common ncnn-picodet names)
  PICODET_INPUT_BLOB model input blob name        (default: first input / "in0")
  PICODET_THREADS inference threads                              (default 4)
  YOLO_SOURCE   "0"/"1" USB, "rpicam" (Pi CSI), rtsp/http URL, GStreamer  (def 0)
  YOLO_LABELS   labels.txt (one per line)         (default: COCO 80)
  YOLO_CONF     score threshold                                 (default 0.4)
  YOLO_NMS      NMS IoU threshold                               (default 0.5)
  YOLO_FILTER   keep only these classes (names or indices)
  YOLO_PORT / YOLO_JPEG_Q / YOLO_CAM_W/H/FPS / YOLO_TRACK       (as other sidecars)
  YOLO_GIMBAL / YOLO_GIMBAL_HOST / YOLO_GIMBAL_PORT / YOLO_TRACK_SPEED …

  --inspect     load the model and print input/output blob names, then exit
                (use the output to set PICODET_CLS_BLOBS / PICODET_REG_BLOBS)

See the README for: install (ncnn), download a ready model, convert to ncnn, run.
"""

import os
import sys
import time
import socket
import struct
import math
import threading
import subprocess
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
    return v if v not in (None, "") else default


def label_for(cls, labels=None):
    if labels is not None and 0 <= cls < len(labels):
        return labels[cls]
    if 0 <= cls < len(COCO):
        return COCO[cls]
    return f"cls{cls}"


def load_labels(path):
    if not path:
        return None
    try:
        return [ln.strip() for ln in open(path) if ln.strip()]
    except OSError as e:
        sys.stderr.write(f"WARNING: can't read labels '{path}': {e}\n")
        return None


def parse_filter(spec, labels):
    if not spec:
        return None
    names = labels or COCO
    out = set()
    for t in spec.split(","):
        t = t.strip()
        if not t:
            continue
        if t.isdigit():
            out.add(int(t))
        else:
            i = next((k for k, n in enumerate(names) if n.lower() == t.lower()), None)
            if i is not None:
                out.add(i)
    return out or None


# ── PicoDet NCNN inference + GFL decode ───────────────────────────────────────
def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class PicoDetNCNN:
    # ImageNet BGR mean / (std*255) — PicoDet's standard normalisation.
    DEFAULT_MEAN = (103.53, 116.28, 123.675)
    DEFAULT_NORM = (0.017429, 0.017507, 0.017125)

    def __init__(self):
        import ncnn
        self.ncnn = ncnn
        self.input = int(env("PICODET_INPUT", "416"))
        self.strides = [int(s) for s in env("PICODET_STRIDES", "8,16,32,64").split(",")]
        self.reg_max = int(env("PICODET_REG_MAX", "7"))
        self.cell_offset = float(env("PICODET_CELL_OFFSET", "0.5"))
        self.conf = float(env("YOLO_CONF", "0.4"))
        self.nms = float(env("YOLO_NMS", "0.5"))

        self.net = ncnn.Net()
        self.net.opt.num_threads = int(env("PICODET_THREADS", "4"))
        self.net.opt.use_vulkan_compute = False
        param = env("PICODET_PARAM"); bin_ = env("PICODET_BIN")
        if not param or not bin_:
            sys.stderr.write("ERROR: set PICODET_PARAM and PICODET_BIN to your .param/.bin\n")
            sys.exit(2)
        if self.net.load_param(param) != 0 or self.net.load_model(bin_) != 0:
            sys.stderr.write(f"ERROR: failed to load ncnn model ({param} / {bin_})\n")
            sys.exit(1)

        self.in_name = env("PICODET_INPUT_BLOB") or (self._names("input") or ["in0"])[0]
        # Output blob names per stride. Must be supplied (via --inspect) for a custom
        # export; defaults match common ncnn-picodet exports.
        n = len(self.strides)
        self.cls_blobs = self._blob_list("PICODET_CLS_BLOBS", n,
            [f"cls_pred_stride_{s}" for s in self.strides])
        self.reg_blobs = self._blob_list("PICODET_REG_BLOBS", n,
            [f"dis_pred_stride_{s}" for s in self.strides])
        self.num_classes = None   # inferred from the first class blob on first frame

    def _names(self, kind):
        fn = getattr(self.net, f"{kind}_names", None)
        try:
            return list(fn()) if fn else None
        except Exception:
            return None

    def _blob_list(self, var, n, default):
        v = env(var)
        if v:
            names = [x.strip() for x in v.split(",") if x.strip()]
            if len(names) != n:
                sys.stderr.write(f"WARNING: {var} has {len(names)} names but {n} strides\n")
            return names
        return default

    def inspect(self):
        print("Model input blobs :", self._names("input"))
        print("Model output blobs:", self._names("output"))
        print("\nSet PICODET_CLS_BLOBS / PICODET_REG_BLOBS to the class-score and "
              "box-distribution outputs, ordered by stride", self.strides)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        ex = self.net.create_extractor()
        mat = self.ncnn.Mat.from_pixels_resize(
            bgr, self.ncnn.Mat.PixelType.PIXEL_BGR, w, h, self.input, self.input)
        mat.substract_mean_normalize(list(self.DEFAULT_MEAN), list(self.DEFAULT_NORM))
        ex.input(self.in_name, mat)

        dets = []
        for stride, cb, rb in zip(self.strides, self.cls_blobs, self.reg_blobs):
            try:
                _, cls_m = ex.extract(cb)
                _, reg_m = ex.extract(rb)
            except Exception as e:
                sys.stderr.write(f"extract failed for {cb}/{rb}: {e} — check PICODET_*_BLOBS\n")
                return []
            cls = np.array(cls_m); reg = np.array(reg_m)
            dets += self._decode_level(cls, reg, stride)
        if not dets:
            return []
        return self._nms(dets, w, h)

    def _decode_level(self, cls, reg, stride):
        fh = (self.input + stride - 1) // stride
        fw = fh
        g = fh * fw
        cls = cls.reshape(cls.shape[0], -1) if cls.ndim > 2 else cls
        reg = reg.reshape(reg.shape[0], -1) if reg.ndim > 2 else reg
        # Want [G, num_class] and [G, 4*(reg_max+1)] — transpose if it came [C, G].
        if cls.shape[0] != g and cls.shape[1] == g:
            cls = cls.T
        if reg.shape[0] != g and reg.shape[1] == g:
            reg = reg.T
        if cls.shape[0] != g:
            return []   # layout mismatch — wrong blob or stride/input config
        if self.num_classes is None:
            self.num_classes = cls.shape[1]

        scores_all = 1.0 / (1.0 + np.exp(-cls))         # sigmoid logits → scores
        scores = scores_all.max(1); labels = scores_all.argmax(1)
        keep = np.where(scores >= self.conf)[0]
        if keep.size == 0:
            return []
        ys = keep // fw; xs = keep % fw
        cx = (xs + self.cell_offset) * stride
        cy = (ys + self.cell_offset) * stride
        r = reg[keep].reshape(-1, 4, self.reg_max + 1)
        r = _softmax(r, axis=2)
        dist = (r * np.arange(self.reg_max + 1)).sum(2) * stride    # [n,4] = l,t,r,b
        x1 = cx - dist[:, 0]; y1 = cy - dist[:, 1]
        x2 = cx + dist[:, 2]; y2 = cy + dist[:, 3]
        return list(zip(x1, y1, x2, y2, scores[keep], labels[keep]))

    def _nms(self, dets, w, h):
        sx = w / self.input; sy = h / self.input
        boxes, scores, labels = [], [], []
        for (x1, y1, x2, y2, sc, lb) in dets:
            X1 = max(0, min(x1 * sx, w - 1)); Y1 = max(0, min(y1 * sy, h - 1))
            X2 = max(0, min(x2 * sx, w - 1)); Y2 = max(0, min(y2 * sy, h - 1))
            if X2 - X1 < 1 or Y2 - Y1 < 1:
                continue
            boxes.append([X1, Y1, X2 - X1, Y2 - Y1]); scores.append(float(sc)); labels.append(int(lb))
        if not boxes:
            return []
        out = []
        labels_np = np.array(labels)
        for cls in np.unique(labels_np):
            idx = np.where(labels_np == cls)[0]
            b = [boxes[i] for i in idx]; s = [scores[i] for i in idx]
            ind = cv2.dnn.NMSBoxes(b, s, self.conf, self.nms)
            for k in np.array(ind).reshape(-1):
                x, y, bw, bh = boxes[idx[k]]
                out.append((x, y, x + bw, y + bh, scores[idx[k]], int(cls)))
        return out


# ── Manual single-object tracker (OpenCV NCC) — mirrors the JVM ObjectTracker ──
class ObjectTracker:
    def __init__(self, track_ncc=0.42, adapt_ncc=0.72, reacq_ncc=0.55, max_miss=90):
        self.track_ncc, self.adapt_ncc, self.reacq_ncc, self.max_miss = track_ncc, adapt_ncc, reacq_ncc, max_miss
        self.locked = False; self.tmpl = self.anchor = None
        self.cx = self.cy = self.bw = self.bh = 0.0; self.miss = 0

    def lock(self, frame, x1, y1, x2, y2):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); H, W = g.shape
        x1 = max(0, min(int(x1), W - 2)); y1 = max(0, min(int(y1), H - 2))
        x2 = max(x1 + 2, min(int(x2), W)); y2 = max(y1 + 2, min(int(y2), H))
        roi = g[y1:y2, x1:x2]; rw, rh = x2 - x1, y2 - y1
        s = min(1.0, 64.0 / max(rw, rh))
        self.tw0 = max(8, int(rw * s)); self.th0 = max(8, int(rh * s))
        self.tmpl = cv2.resize(roi, (self.tw0, self.th0)); self.anchor = self.tmpl.copy()
        self.cx, self.cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.bw, self.bh = float(rw), float(rh); self.miss = 0; self.locked = True

    def reset(self):
        self.locked = False; self.miss = 0

    def update(self, frame):
        if not self.locked:
            return None
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best = self._search(g, max(8.0, max(self.bw, self.bh) * 0.5), (0.9, 1.0, 1.1))
        if best and best[0] >= self.track_ncc:
            self.cx, self.cy = best[1], best[2]
            self.bw += (best[3] - self.bw) * 0.5; self.bh += (best[4] - self.bh) * 0.5
            self._clamp(g.shape); self.miss = 0
            if best[0] >= self.adapt_ncc:
                self._adapt(g)
            return self._box()
        self.miss += 1
        if self.miss % 2 == 0:
            rr = min(250.0, max(self.bw, self.bh) * 1.5 * (1.0 + self.miss * 0.4))
            best = self._search(g, rr, (0.7, 0.85, 1.0, 1.2, 1.45))
            if best and best[0] >= self.reacq_ncc:
                self.cx, self.cy, self.bw, self.bh = best[1], best[2], best[3], best[4]
                self._clamp(g.shape); self.miss = 0
                return self._box()
        if self.miss > self.max_miss:
            self.locked = False; return None
        return self._box()

    def _box(self):
        return (self.cx - self.bw / 2, self.cy - self.bh / 2, self.cx + self.bw / 2, self.cy + self.bh / 2)

    def _clamp(self, shape):
        H, W = shape
        self.bw = float(np.clip(self.bw, 8, W)); self.bh = float(np.clip(self.bh, 8, H))
        self.cx = float(np.clip(self.cx, self.bw / 2, W - self.bw / 2))
        self.cy = float(np.clip(self.cy, self.bh / 2, H - self.bh / 2))

    def _search(self, g, radius, scales):
        H, W = g.shape
        x0 = int(max(0, self.cx - self.bw / 2 - radius)); y0 = int(max(0, self.cy - self.bh / 2 - radius))
        x1 = int(min(W, self.cx + self.bw / 2 + radius)); y1 = int(min(H, self.cy + self.bh / 2 + radius))
        roi = g[y0:y1, x0:x1]; rh, rw = roi.shape[:2]; best = None
        for tmpl in (self.tmpl, self.anchor):
            for s in scales:
                tw = max(8, int(self.bw * s)); th = max(8, int(self.bh * s))
                if tw >= rw or th >= rh:
                    continue
                res = cv2.matchTemplate(roi, cv2.resize(tmpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                if best is None or mx > best[0]:
                    best = (mx, x0 + loc[0] + tw / 2.0, y0 + loc[1] + th / 2.0, float(tw), float(th))
        return best

    def _adapt(self, g):
        x1 = max(0, int(self.cx - self.bw / 2)); y1 = max(0, int(self.cy - self.bh / 2))
        x2 = min(g.shape[1], int(self.cx + self.bw / 2)); y2 = min(g.shape[0], int(self.cy + self.bh / 2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return
        patch = cv2.resize(g[y1:y2, x1:x2], (self.tw0, self.th0)).astype(np.float32)
        self.tmpl = (self.tmpl.astype(np.float32) * 0.9 + patch * 0.1).astype(np.uint8)


# ── IoU tracker (box persistence) ─────────────────────────────────────────────
class Tracker:
    def __init__(self, hold_s=0.8, iou_th=0.3):
        self.hold = hold_s; self.iou_th = iou_th; self.tracks = []

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, dets, now):
        matched = [False] * len(self.tracks)
        for d in dets:
            best, bi = -1, self.iou_th
            for i, (box, _) in enumerate(self.tracks):
                if matched[i] or box[5] != d[5]:
                    continue
                v = self._iou(box, d)
                if v >= bi:
                    best, bi = i, v
            if best >= 0:
                self.tracks[best] = [d, now]; matched[best] = True
            else:
                self.tracks.append([d, now])
        self.tracks = [t for t in self.tracks if now - t[1] <= self.hold]
        return [t[0] for t in self.tracks]


# ── SIYI gimbal (UDP SDK) ─────────────────────────────────────────────────────
class SiyiGimbal:
    def __init__(self, host="192.168.144.25", port=37260):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.sock.settimeout(1.0)
        self.seq = 0; self._lock = threading.Lock(); self.running = True
        self.yaw = self.pitch = self.roll = 0.0; self.firmware = ""; self.hardware_id = ""
        self.recording = False; self.motion_mode = -1
        threading.Thread(target=self._rx, daemon=True).start()
        self.request_hardware_id(); self.request_firmware(); self.request_config(); self.request_attitude()

    @staticmethod
    def _crc16(data):
        crc = 0
        for b in data:
            crc ^= (b << 8)
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
        return crc & 0xFFFF

    def _send(self, cmd, data=b""):
        with self._lock:
            s = self.seq & 0xFFFF; self.seq += 1
        body = bytes([0x55, 0x66, 0x01]) + struct.pack("<H", len(data)) + struct.pack("<H", s) + bytes([cmd & 0xFF]) + data
        frame = body + struct.pack("<H", self._crc16(body))
        try:
            self.sock.sendto(frame, self.addr)
        except OSError:
            pass

    @staticmethod
    def _i8(v): return max(-1, min(1, int(v)))
    @staticmethod
    def _sp(v): return max(-100, min(100, int(v)))

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
        self._send(0x0E, struct.pack("<hh", int(max(-135, min(135, yaw)) * 10), int(max(-90, min(25, pitch)) * 10)))

    def absolute_zoom(self, x):
        x = max(1.0, x); ip = int(x); self._send(0x0F, bytes([ip & 0xFF, int((x - ip) * 10) & 0xFF]))

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
            self.recording = b[d + 3] == 1; self.motion_mode = b[d + 4]
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


class GimbalFollower:
    def __init__(self, gimbal, max_speed=40, deadzone=0.05, stable_ticks=3, invert_yaw=False, invert_pitch=False):
        self.g = gimbal; self.max_speed = max_speed; self.deadzone = deadzone; self.stable_ticks = stable_ticks
        self.invert_yaw = invert_yaw; self.invert_pitch = invert_pitch
        self.prev = None; self.lock_count = 0; self.moving = False; self.pending = None

    def request_pick(self, nx, ny):
        self.pending = (nx, ny)

    def step(self, dets, fw, fh):
        if fw <= 0 or fh <= 0 or not dets:
            self.stop(); self.prev = None; self.lock_count = 0; return None
        if self.pending is not None:
            px, py = self.pending[0] * fw, self.pending[1] * fh; self.pending = None
            t = self._pick_at(dets, px, py)
            if t is not None:
                self.prev = t; self.lock_count = 0
        t = self._pick(dets, self.prev, fw); self.prev = t
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
        return min(pool, key=lambda d: ((d[0]+d[2])/2-px)**2 + ((d[1]+d[3])/2-py)**2)

    def _pick(self, dets, prev, fw):
        if prev is not None:
            near = min(dets, key=lambda d: self._cdist(d, prev))
            if self._cdist(near, prev) < 0.3 * fw:
                return near
        return max(dets, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))

    @staticmethod
    def _cdist(a, b):
        return math.hypot(((a[0]+a[2])-(b[0]+b[2]))/2, ((a[1]+a[3])-(b[1]+b[3]))/2)


def draw(frame, dets, hud, labels=None, tracking=False, target=None, manual=None):
    for (x1, y1, x2, y2, conf, cls) in dets:
        color = PALETTE[cls % len(PALETTE)]; p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        text = f"{label_for(cls, labels)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), color, -1)
        cv2.putText(frame, text, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if tracking:
        h, w = frame.shape[:2]; cx, cy = w // 2, h // 2
        cv2.line(frame, (cx-16, cy), (cx+16, cy), (60, 60, 255), 2)
        cv2.line(frame, (cx, cy-16), (cx, cy+16), (60, 60, 255), 2)
        cv2.circle(frame, (cx, cy), 6, (60, 60, 255), 2)
        if target is not None:
            t = [int(v) for v in target[:4]]
            cv2.rectangle(frame, (t[0], t[1]), (t[2], t[3]), (0, 230, 255), 3)
            cv2.line(frame, (cx, cy), ((t[0]+t[2])//2, (t[1]+t[3])//2), (60, 60, 255), 1)
        cv2.putText(frame, "TRACKING", (cx-48, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
    if manual is not None:
        m = [int(v) for v in manual[:4]]; cyan = (230, 230, 0)
        cv2.rectangle(frame, (m[0], m[1]), (m[2], m[3]), cyan, 2)
        c = max(6, min(m[2]-m[0], m[3]-m[1]) // 4)
        for (px, py, dx, dy) in ((m[0], m[1], 1, 1), (m[2], m[1], -1, 1), (m[0], m[3], 1, -1), (m[2], m[3], -1, -1)):
            cv2.line(frame, (px, py), (px + dx*c, py), cyan, 4); cv2.line(frame, (px, py), (px, py + dy*c), cyan, 4)
        cv2.putText(frame, "LOCK", (m[0]+2, max(12, m[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 2)
    if hud:
        cv2.rectangle(frame, (2, frame.shape[0]-24), (2 + 9*len(hud), frame.shape[0]-2), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, frame.shape[0]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (118, 230, 0), 2)
    return frame


class State:
    def __init__(self):
        self.lock = threading.Lock(); self.frame = None; self.dets = []
        self.stream_fps = 0; self.det_fps = 0; self.labels = None; self.jpeg_q = 75
        self.gimbal = None; self.follower = None; self.tracking = False; self.target = None
        self.obj = ObjectTracker(); self.manual_req = None; self.manual_clear = False; self.manual_box = None
        self.running = True


class RpicamCapture:
    def __init__(self, w, h, fps):
        args = ["-t", "0", "--codec", "mjpeg", "--nopreview", "--width", str(w),
                "--height", str(h), "--framerate", str(fps), "-o", "-"]
        self.proc = None
        for b in ("rpicam-vid", "libcamera-vid"):
            try:
                self.proc = subprocess.Popen([b] + args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
                break
            except FileNotFoundError:
                continue
        self.buf = bytearray()

    def isOpened(self): return self.proc is not None

    def read(self):
        if self.proc is None:
            return False, None
        while True:
            s = self.buf.find(b"\xff\xd8"); e = self.buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
            if s >= 0 and e >= 0:
                jpg = bytes(self.buf[s:e + 2]); del self.buf[:e + 2]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return True, img
                continue
            chunk = self.proc.stdout.read(65536)
            if not chunk:
                return False, None
            self.buf.extend(chunk)

    def release(self):
        if self.proc is not None:
            self.proc.terminate()


def open_source(src, w, h, fps):
    if src.lower() in ("rpicam", "libcamera", "csi"):
        return RpicamCapture(w, h, fps)
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, fps)
        return cap
    if src.startswith(("http", "rtsp")):
        return cv2.VideoCapture(src)
    return cv2.VideoCapture(src, cv2.CAP_GSTREAMER)


def capture_loop(state, cap):
    count, t0 = 0, time.time()
    while state.running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue
        if state.manual_clear:
            state.manual_clear = False; state.obj.reset(); state.manual_box = None
        req = state.manual_req
        if req is not None:
            state.manual_req = None; h, w = frame.shape[:2]
            state.obj.lock(frame, req[0]*w, req[1]*h, req[2]*w, req[3]*h)
        state.manual_box = state.obj.update(frame) if state.obj.locked else None
        with state.lock:
            state.frame = frame
        count += 1; dt = time.time() - t0
        if dt >= 1.0:
            state.stream_fps = int(count / dt); count, t0 = 0, time.time()
    cap.release()


def follow_loop(state):
    while state.running:
        if state.tracking and state.follower is not None:
            with state.lock:
                frame = state.frame; dets = list(state.dets)
            mb = state.manual_box
            if mb is not None:
                dets = [(mb[0], mb[1], mb[2], mb[3], 1.0, -1)]
            if frame is not None:
                h, w = frame.shape[:2]; state.target = state.follower.step(dets, w, h)
            else:
                state.target = None
        else:
            if state.follower is not None:
                state.follower.stop()
            state.target = None
        time.sleep(0.066)


def inference_loop(state, detector, track_on, filter_set=None):
    tracker = Tracker() if track_on else None
    count, t0 = 0, time.time()
    while state.running:
        with state.lock:
            frame = None if state.frame is None else state.frame.copy()
        if frame is None:
            time.sleep(0.01); continue
        try:
            dets = detector.detect(frame)
            if filter_set is not None:
                dets = [d for d in dets if d[5] in filter_set]
        except Exception as e:
            sys.stderr.write(f"inference error: {e}\n"); dets = []
        if tracker is not None:
            dets = tracker.update(dets, time.time())
        with state.lock:
            state.dets = dets
        count += 1; dt = time.time() - t0
        if dt >= 1.0:
            state.det_fps = int(count / dt); count, t0 = 0, time.time()


def _status_json(state):
    g = state.gimbal
    mode = {0: "lock", 1: "follow", 2: "fpv"}.get(g.motion_mode if g else -1, "?")
    return ('{"hasGimbal":%s,"yaw":%s,"pitch":%s,"roll":%s,"recording":%s,"mode":"%s","tracking":%s,'
            '"firmware":"%s","hardwareId":"%s"}' % (
                str(g is not None).lower(), g.yaw if g else 0, g.pitch if g else 0, g.roll if g else 0,
                str(g.recording if g else False).lower(), mode, str(state.tracking).lower(),
                g.firmware if g else "", g.hardware_id if g else ""))


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def _send(self, code, ctype, body):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else body.encode())

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--mjpeg")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            try:
                while state.running:
                    with state.lock:
                        frame = None if state.frame is None else state.frame.copy()
                        dets = list(state.dets); hud = f"FPS {state.stream_fps}  |  det {state.det_fps}"
                        tracking, target, q = state.tracking, state.target, state.jpeg_q
                    if frame is None:
                        time.sleep(0.02); continue
                    draw(frame, dets, hud, state.labels, tracking, target, state.manual_box)
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
                    if not ok:
                        continue
                    self.wfile.write(b"--mjpeg\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpg.tobytes()); self.wfile.write(b"\r\n"); time.sleep(0.005)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            u = urlparse(self.path); path = u.path
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if path == "/stream":
                self._stream(); return
            if path == "/":
                self._send(200, "text/html; charset=utf-8", PANEL); return
            if path == "/lock":
                try:
                    a, b = min(float(q["x1"]), float(q["x2"])), min(float(q["y1"]), float(q["y2"]))
                    c, d = max(float(q["x1"]), float(q["x2"])), max(float(q["y1"]), float(q["y2"]))
                    if c - a > 0.01 and d - b > 0.01:
                        state.manual_req = (a, b, c, d)
                except (KeyError, TypeError, ValueError):
                    pass
                self._send(200, "application/json", _status_json(state)); return
            if path == "/unlock":
                state.manual_clear = True; self._send(200, "application/json", _status_json(state)); return
            g = state.gimbal; fol = state.follower
            try:
                if path in ("/status", "/attitude"):
                    if path == "/attitude" and g: g.request_attitude()
                elif path == "/track":
                    on = q.get("on"); state.tracking = (on in ("1", "true")) if on is not None else (not state.tracking)
                elif path == "/pick":
                    if fol is not None and "nx" in q and "ny" in q:
                        fol.request_pick(float(q["nx"]), float(q["ny"])); state.tracking = True
                elif g is None:
                    pass
                elif path == "/rotate": g.rotate(int(float(q.get("yaw", 0))), int(float(q.get("pitch", 0))))
                elif path == "/stop":   g.stop_rotation()
                elif path == "/angle":  g.set_angle(float(q.get("yaw", 0)), float(q.get("pitch", 0)))
                elif path == "/center": g.center()
                elif path == "/zoom":
                    if "x" in q: g.absolute_zoom(float(q["x"]))
                    elif q.get("dir") == "in": g.manual_zoom(1)
                    elif q.get("dir") == "out": g.manual_zoom(-1)
                    else: g.manual_zoom(0)
                elif path == "/focus":
                    g.manual_focus(1 if q.get("dir") == "far" else -1 if q.get("dir") == "near" else 0)
                elif path == "/autofocus": g.autofocus()
                elif path == "/photo":  g.take_photo()
                elif path == "/record": g.toggle_record()
                elif path == "/hdr":    g.toggle_hdr()
                elif path == "/mode":
                    m = q.get("m")
                    (g.set_lock if m == "lock" else g.set_follow if m == "follow" else g.set_fpv if m == "fpv" else (lambda: None))()
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
<title>PicoDet panel</title><style>
*{box-sizing:border-box}body{margin:0;background:#000;font-family:system-ui,sans-serif;color:#eee;overflow:hidden;user-select:none}
#vid{position:fixed;inset:0;width:100vw;height:100vh;object-fit:contain;background:#000;cursor:crosshair;touch-action:none}
.ov{position:fixed;z-index:2}button{background:rgba(20,20,20,.6);color:#eee;border:1px solid #777;border-radius:10px;
padding:12px 14px;font-size:16px;margin:3px;cursor:pointer}button:active{background:#00e676;color:#000}
#st{top:8px;left:8px;font-family:monospace;white-space:pre;background:rgba(0,0,0,.5);padding:8px 10px;border-radius:8px;font-size:13px}
#trk{top:8px;left:50%;transform:translateX(-50%);padding:6px 12px;border-radius:8px;font-weight:bold;background:rgba(0,0,0,.5)}
#hint{bottom:8px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.5);padding:6px 12px;border-radius:8px;font-size:12px;opacity:.85}
#sel{position:fixed;z-index:3;border:2px solid #00e6e6;background:rgba(0,230,230,.12);display:none;pointer-events:none}
#pad{bottom:12px;left:12px;display:grid;grid-template-columns:repeat(3,56px);gap:4px}
#side{bottom:12px;right:12px;display:flex;flex-direction:column;align-items:flex-end}
#top{top:8px;right:8px;display:flex;flex-wrap:wrap;justify-content:flex-end;max-width:60vw}
input{width:52px;background:rgba(0,0,0,.5);color:#eee;border:1px solid #777;border-radius:6px;padding:6px}
.grp{display:flex;align-items:center;margin:2px 0}
</style></head><body>
<img id="vid" alt="stream"><div id="sel"></div>
<div id="st" class="ov">loading...</div>
<div id="trk" class="ov">TRACK OFF <span style="opacity:.7">(Space)</span></div>
<div id="hint" class="ov">drag = lock target &middot; C = clear &middot; H = gimbal panel</div>
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
var vid=document.getElementById('vid'),sel=document.getElementById('sel');vid.src='/stream';
var HASGIMBAL=false,showG=false;
function g(u){return fetch(u).then(function(r){return r.json()}).then(show).catch(function(){})}
function applyG(){['pad','side','top','trk'].forEach(function(id){var el=document.getElementById(id);if(el)el.style.display=(HASGIMBAL&&showG)?'':'none'})}
function show(s){var st=document.getElementById('st');
 if(s.hasGimbal){st.textContent='yaw '+s.yaw+'  pitch '+s.pitch+'  roll '+s.roll+'\\nmode '+s.mode+'  rec '+s.recording}
 else{st.textContent='PicoDet (NCNN) — no gimbal'}
 if(s.hasGimbal!==HASGIMBAL){HASGIMBAL=s.hasGimbal;showG=s.hasGimbal;applyG()}
 var t=document.getElementById('trk');t.firstChild.nodeValue=(s.tracking?'* TRACK ON ':'TRACK OFF ');
 t.style.background=s.tracking?'rgba(255,40,40,.75)':'rgba(0,0,0,.5)'}
applyG();
document.addEventListener('keydown',function(e){
 if(e.key==='h'||e.key==='H'){showG=!showG;applyG()}
 else if(e.key==='c'||e.key==='C'||e.key==='Escape'){g('/unlock')}
 else if(e.code==='Space'||e.key===' '){e.preventDefault();if(HASGIMBAL)g('/track')}});
function mapPt(e){var nw=vid.naturalWidth,nh=vid.naturalHeight;if(!nw||!nh)return null;
 var r=vid.getBoundingClientRect();var sc=Math.min(r.width/nw,r.height/nh);var dw=nw*sc,dh=nh*sc;
 var ox=r.left+(r.width-dw)/2,oy=r.top+(r.height-dh)/2;var x=(e.clientX-ox)/dw,y=(e.clientY-oy)/dh;
 if(x<0||x>1||y<0||y>1)return null;return {x:x,y:y}}
var drag=null;
vid.addEventListener('pointerdown',function(e){var p=mapPt(e);if(!p)return;e.preventDefault();
 drag={x1:p.x,y1:p.y,sx:e.clientX,sy:e.clientY};if(vid.setPointerCapture)vid.setPointerCapture(e.pointerId)});
vid.addEventListener('pointermove',function(e){if(!drag)return;
 var x=Math.min(drag.sx,e.clientX),y=Math.min(drag.sy,e.clientY),w=Math.abs(e.clientX-drag.sx),h=Math.abs(e.clientY-drag.sy);
 sel.style.display='block';sel.style.left=x+'px';sel.style.top=y+'px';sel.style.width=w+'px';sel.style.height=h+'px'});
vid.addEventListener('pointerup',function(e){if(!drag)return;sel.style.display='none';
 var p=mapPt(e),moved=Math.abs(e.clientX-drag.sx)+Math.abs(e.clientY-drag.sy);
 if(p&&moved>8)g('/lock?x1='+drag.x1.toFixed(4)+'&y1='+drag.y1.toFixed(4)+'&x2='+p.x.toFixed(4)+'&y2='+p.y.toFixed(4));
 else if(p&&HASGIMBAL)g('/pick?nx='+p.x.toFixed(4)+'&ny='+p.y.toFixed(4));
 drag=null});
function hold(el,dn,up){if(!el)return;el.onmousedown=dn;el.onmouseup=up;el.onmouseleave=up;
 el.ontouchstart=function(e){e.preventDefault();dn()};el.ontouchend=function(e){e.preventDefault();up()}}
document.querySelectorAll('#pad button[data-y]').forEach(function(b){
 hold(b,function(){g('/rotate?yaw='+b.dataset.y+'&pitch='+b.dataset.p)},function(){g('/stop')})});
hold(document.getElementById('zin'),function(){g('/zoom?dir=in')},function(){g('/zoom?dir=stop')});
hold(document.getElementById('zout'),function(){g('/zoom?dir=out')},function(){g('/zoom?dir=stop')});
hold(document.getElementById('ff'),function(){g('/focus?dir=far')},function(){g('/focus?dir=stop')});
hold(document.getElementById('fn'),function(){g('/focus?dir=near')},function(){g('/focus?dir=stop')});
setInterval(function(){fetch('/status').then(function(r){return r.json()}).then(show).catch(function(){})},1000);
</script></body></html>"""


def lan_ips():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0]); s.close()
    except OSError:
        pass
    return ips or ["<board-ip>"]


def main():
    if "--inspect" in sys.argv:
        try:
            import ncnn  # noqa
        except ImportError:
            sys.stderr.write("ERROR: ncnn not installed.  pip install ncnn\n"); sys.exit(1)
        PicoDetNCNN().inspect(); return

    source = env("YOLO_SOURCE", "0")
    labels = load_labels(env("YOLO_LABELS"))
    filter_set = parse_filter(env("YOLO_FILTER"), labels)
    port = int(env("YOLO_PORT", "8080"))
    cam_w = int(env("YOLO_CAM_W", "1280")); cam_h = int(env("YOLO_CAM_H", "720")); cam_fps = int(env("YOLO_CAM_FPS", "30"))
    track_on = env("YOLO_TRACK", "on").lower() != "off"
    jpeg_q = int(env("YOLO_JPEG_Q", "75"))
    gimbal_env = (env("YOLO_GIMBAL", "") or "").lower()
    gimbal_on = gimbal_env == "on" or (gimbal_env != "off" and "192.168.144.25" in source)

    try:
        import ncnn  # noqa: F401
    except ImportError:
        sys.stderr.write("ERROR: ncnn not installed.  pip install ncnn   (see the README)\n"); sys.exit(1)

    print("YOLO PicoDet sidecar (NCNN / CPU)")
    detector = PicoDetNCNN()
    print(f"  model loaded: input={detector.input} strides={detector.strides} "
          f"reg_max={detector.reg_max} conf={detector.conf} nms={detector.nms}")
    print(f"  input blob: {detector.in_name}")
    print(f"  cls blobs:  {detector.cls_blobs}")
    print(f"  reg blobs:  {detector.reg_blobs}")
    print("  (if you get no detections, run with --inspect and set PICODET_CLS_BLOBS/"
          "PICODET_REG_BLOBS; try PICODET_CELL_OFFSET=0)")

    cap = open_source(source, cam_w, cam_h, cam_fps)
    if not cap.isOpened():
        sys.stderr.write(f"ERROR: cannot open video source '{source}'\n"); sys.exit(1)

    state = State(); state.labels = labels; state.jpeg_q = jpeg_q
    if filter_set is not None:
        print(f"  filter: only classes {sorted(filter_set)}")

    if gimbal_on:
        g_host = env("YOLO_GIMBAL_HOST", "192.168.144.25"); g_port = int(env("YOLO_GIMBAL_PORT", "37260"))
        state.gimbal = SiyiGimbal(g_host, g_port)
        state.follower = GimbalFollower(
            state.gimbal, max_speed=int(env("YOLO_TRACK_SPEED", "40")),
            invert_yaw=env("YOLO_TRACK_INVERT_YAW", "off").lower() == "on",
            invert_pitch=env("YOLO_TRACK_INVERT_PITCH", "off").lower() == "on")
        threading.Thread(target=follow_loop, args=(state,), daemon=True).start()
        print(f"  gimbal control: enabled ({g_host}:{g_port}) — Space to follow")

    threading.Thread(target=capture_loop, args=(state, cap), daemon=True).start()
    threading.Thread(target=inference_loop, args=(state, detector, track_on, filter_set), daemon=True).start()

    for ip in lan_ips():
        print(f"  panel: http://{ip}:{port}   "
              + ("(video + gimbal + manual capture)" if gimbal_on else "(video + manual capture)"))
    print("  (open the URL above; drag a box to lock a target)")

    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if state.gimbal is not None:
            state.gimbal.close()


if __name__ == "__main__":
    main()
