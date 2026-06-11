#!/usr/bin/env python3
"""
YOLO-FastestV2 (NCNN) sidecar — ultra-light, ultra-fast CPU detection for Pi 5.

Runs the tiny YOLO-FastestV2 detector (dog-qiuqiu/Yolo-FastestV2, ~0.25M params)
on NCNN and serves the same annotated MJPEG stream + web panel as the other
sidecars: manual drag-to-lock capture, IoU tracking, SIYI gimbal. Open
http://<board-ip>:8080. This is usually the highest-FPS option on a bare Pi 5 CPU.

YOLO-FastestV2 is anchor-based (ShuffleNetV2 backbone, 2 detection scales), with a
decoupled head: per-cell channels = reg(4*na) + obj(na) + cls(nc), where the class
scores are SHARED across the na anchors. NCNN gives the raw head, so the sidecar
decodes it (anchors → box, obj×class score, NMS). The exact box/score formulas and
blob names depend on the export, so they're env-tunable — use --inspect first.

Config (YOLO_* shared with the other sidecars):
  YF_PARAM / YF_BIN     ncnn model files                              [required]
  YF_INPUT              square input size                             (default 352)
  YF_STRIDES            detection strides                             (default 16,32)
  YF_ANCHORS_PER        anchors per cell                              (default 3)
  YF_ANCHORS_16 / YF_ANCHORS_32 …  per-stride anchors "w,h,w,h,w,h" (pixels @ input)
                        (defaults = repo COCO anchors)
  YF_BOX_DECODE         v5 | plain  (box formula — switch if boxes are wrong) (default v5)
  YF_SCORE              sqrt | mul  (final score = sqrt(obj*cls) or obj*cls)   (default sqrt)
  YF_OUTPUTS            comma list of the head output blob names, per stride
                        (from --inspect; defaults to common names)
  YF_INPUT_BLOB         model input blob name        (default: first input / "in0")
  YF_THREADS            inference threads                             (default 4)
  YOLO_SOURCE / YOLO_LABELS / YOLO_CONF / YOLO_NMS / YOLO_FILTER / YOLO_PORT /
  YOLO_JPEG_Q / YOLO_CAM_W/H/FPS / YOLO_TRACK / YOLO_GIMBAL …          (as other sidecars)

  --inspect     load the model and print input/output blob names, then exit.

See the README for: install (ncnn), the ready repo model, training, tuning.
"""

import os
import sys
import time
import socket
import struct
import math
import json
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sanity_score(dets, w, h):
    """Heuristic 'do these detections look sane' score, for --autotune. Rewards
    plausible in-frame boxes, penalises out-of-frame / absurd ones — the right
    box/score formula scores highest on a real scene."""
    score = 0.0
    for (x1, y1, x2, y2, conf, cls) in dets:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        if x1 < -2 or y1 < -2 or x2 > w + 2 or y2 > h + 2:
            score -= 1.0
            continue
        area = (bw * bh) / (w * h + 1e-6)
        ar = bw / max(1e-6, bh)
        score += float(conf) if (0.0005 <= area <= 0.6 and 0.1 <= ar <= 10) else -0.2
    return score


# ── YOLO-FastestV2 NCNN inference + anchor decode ─────────────────────────────
class YoloFastestV2NCNN:
    # Repo COCO anchors (pixels at the 352 input), per stride.
    DEFAULT_ANCHORS = {
        16: [12.64, 19.39, 37.88, 51.48, 55.71, 138.31],
        32: [126.91, 78.23, 131.57, 214.55, 279.92, 258.87],
    }

    def __init__(self):
        import ncnn
        self.ncnn = ncnn
        self.input = int(env("YF_INPUT", "352"))
        self.strides = [int(s) for s in env("YF_STRIDES", "16,32").split(",")]
        self.na = int(env("YF_ANCHORS_PER", "3"))
        self.conf = float(env("YOLO_CONF", "0.3"))
        self.nms = float(env("YOLO_NMS", "0.45"))
        self.box_mode = (env("YF_BOX_DECODE", "v5") or "v5").lower()
        self.score_mode = (env("YF_SCORE", "sqrt") or "sqrt").lower()
        self._act = None        # auto-detected: (obj needs sigmoid, cls needs softmax)
        self._dbg = True        # print decode diagnostics once
        self.anchors = {}
        for s in self.strides:
            ev = env(f"YF_ANCHORS_{s}")
            self.anchors[s] = [float(x) for x in ev.split(",")] if ev else self.DEFAULT_ANCHORS.get(s)

        self.net = ncnn.Net()
        self.net.opt.num_threads = int(env("YF_THREADS", "4"))
        self.net.opt.use_vulkan_compute = False
        param = env("YF_PARAM"); bin_ = env("YF_BIN")
        if not param or not bin_:
            sys.stderr.write("ERROR: set YF_PARAM and YF_BIN to your .param/.bin\n")
            sys.exit(2)
        if self.net.load_param(param) != 0 or self.net.load_model(bin_) != 0:
            sys.stderr.write(f"ERROR: failed to load ncnn model ({param} / {bin_})\n")
            sys.exit(1)

        self.in_name = env("YF_INPUT_BLOB") or (self._names("input") or ["in0"])[0]
        self.out_blobs = self._blob_list("YF_OUTPUTS", len(self.strides),
                                         [f"output_stride_{s}" for s in self.strides])
        self._last_fail_log = 0.0

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

    def _probe_outputs(self, names, input_size=None):
        """Run one forward at [input_size] (default self.input) and return
        {name: grid} for outputs that EXTRACT (ncnn ret==0). A failure (ncnn code
        -100) means a layer produced an empty blob — usually a wrong input size."""
        isz = int(input_size or self.input)
        out, fails = {}, {}
        try:
            ex = self.net.create_extractor()
            dummy = self.ncnn.Mat(isz, isz, 3); dummy.fill(0.0)
            ex.input(self.in_name, dummy)
            for nm in names:
                try:
                    ret, m = ex.extract(nm)
                except Exception as e:
                    fails[nm] = str(e); continue
                if ret != 0:
                    fails[nm] = f"code {ret}"; continue
                a = np.array(m)
                if a.ndim == 3:
                    g = a.shape[1]
                elif a.ndim == 2:
                    g = int(round(math.sqrt(max(a.shape))))
                else:
                    continue
                out[nm] = max(1, g)
        except Exception as e:
            print(f"  probe: forward failed ({type(e).__name__}: {e})")
        return out, fails

    def inspect(self):
        try:
            print("ncnn version:", self.ncnn.__version__)
        except Exception:
            pass
        ins = self._names("input"); outs = self._names("output")
        print("Model input blobs :", ins)
        print("Model output blobs:", outs)
        print(f"\nProbe forward at input={self.input} (input blob '{self.in_name}'):")
        ok, fails = self._probe_outputs(outs or [])
        for nm in (outs or []):
            if nm in ok:
                st = max(1, int(round(self.input / ok[nm])))
                print(f"  {nm}: OK  grid≈{ok[nm]}  → stride {st}")
            else:
                print(f"  {nm}: FAILED ({fails.get(nm, '?')})")
        if outs and not ok:
            # Sweep input sizes — if NONE work it's a conversion/ncnn problem, not size.
            print("\n  Sweeping input sizes to find one the model accepts…")
            found = []
            for s in (224, 256, 288, 320, 352, 384, 416, 448, 512, 576, 640):
                got, _ = self._probe_outputs(outs, s)
                if got:
                    found.append(s)
                    print(f"  YF_INPUT={s}: OK  " + ", ".join(f"{n}→stride{max(1, int(round(s / g)))}" for n, g in got.items()))
            if found:
                print(f"\n  → use YF_INPUT={found[0]}.")
            else:
                print("  No input size works → this is a CONVERSION/ncnn problem, not the size.\n"
                      "  Try: (1) the repo's bundled model Yolo-FastestV2/model/*.param to test the\n"
                      "  sidecar itself; (2) pip install -U ncnn; (3) re-run onnx2ncnn WITHOUT\n"
                      "  ncnnoptimize (or run onnxsim on the ONNX first).")
        if ok:
            print("\n  Outputs extract — just run without YF_OUTPUTS (autoconfig handles them). If\n"
                  "  boxes look wrong later, try YF_BOX_DECODE=plain / YF_SCORE=mul, or --autotune.")

    def autoconfig(self):
        """Detect strides + working output blobs by actually extracting them. Auto-
        corrects a bad YF_OUTPUTS, and flags a YF_INPUT mismatch if nothing extracts."""
        real = self._names("output") or []
        if not real:
            print("  auto: ncnn gave no output names — set YF_OUTPUTS/YF_INPUT (see --inspect)")
            return
        ok, fails = self._probe_outputs(real)
        if not ok:
            print(f"  auto: NO output extracted at input={self.input} ({fails}). This is almost\n"
                  f"        always YF_INPUT not matching the export size — try YF_INPUT=256/320/352/416.")
            return
        env_outs = env("YF_OUTPUTS")
        chosen = None
        if env_outs:
            names = [x.strip() for x in env_outs.split(",") if x.strip()]
            if all(n in ok for n in names):
                chosen = names
            else:
                bad = [n for n in names if n not in ok]
                print(f"  auto: YF_OUTPUTS {bad} don't extract — auto-detecting from {list(ok)}")
        if chosen is None:
            by_stride = {max(1, int(round(self.input / g))): nm for nm, g in ok.items()}
            self.strides = sorted(by_stride)
            chosen = [by_stride[s] for s in self.strides]
        else:
            self.strides = sorted({max(1, int(round(self.input / ok[n]))) for n in chosen})
            chosen = [c for _, c in sorted((max(1, int(round(self.input / ok[c]))), c) for c in chosen)]
        self.out_blobs = chosen
        for s in self.strides:
            self.anchors.setdefault(s, self.DEFAULT_ANCHORS.get(s, self.DEFAULT_ANCHORS[32]))
        print(f"  auto: strides={self.strides}  outputs={self.out_blobs}")

    def _safe_detect(self, f):
        try:
            return self.detect(f)
        except Exception:
            return []

    def autotune(self, frames):
        """Pick the box/score formula that makes boxes look sanest on real frames."""
        best = None
        for bm in ("v5", "plain"):
            for sm in ("sqrt", "mul"):
                self.box_mode, self.score_mode = bm, sm
                s = sum(_sanity_score(self._safe_detect(f), f.shape[1], f.shape[0]) for f in frames)
                if best is None or s > best[0]:
                    best = (s, bm, sm)
        self.box_mode, self.score_mode = best[1], best[2]
        print(f"  autotune: box={best[1]} score={best[2]} (score {best[0]:.1f})")

    def _fail(self, blob, why):
        """Rate-limited error so a broken model doesn't flood the log every frame."""
        now = time.time()
        if now - self._last_fail_log < 5:
            return
        self._last_fail_log = now
        sys.stderr.write(
            f"NCNN extract failed for '{blob}' ({why}). Real model outputs: "
            f"{self._names('output')}. ncnn code -100 usually means YF_INPUT ({self.input}) "
            f"doesn't match the export size — try YF_INPUT=256/320/352/416, or run --inspect.\n")

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        ex = self.net.create_extractor()
        mat = self.ncnn.Mat.from_pixels_resize(
            bgr, self.ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, self.input, self.input)
        mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        ex.input(self.in_name, mat)

        dets = []
        for stride, ob in zip(self.strides, self.out_blobs):
            try:
                ret, m = ex.extract(ob)
            except Exception as e:
                ret, m = -1, None
                self._fail(ob, e)
            if ret != 0 or m is None:
                self._fail(ob, f"ncnn code {ret}")
                time.sleep(0.05)            # don't spin a hot empty loop on a broken model
                return []
            out = np.array(m)   # [C, H, W]
            if out.ndim == 2:   # [C, H*W] → assume square grid
                g = int(round(math.sqrt(out.shape[1])))
                out = out.reshape(out.shape[0], g, g)
            dets += self._decode(out, stride, w, h)
        if not dets:
            return []
        return self._nms(dets)

    def _decode(self, out, stride, ow, oh):
        s = out.shape
        if len(s) != 3:
            return []
        # ncnn may hand outputs back as (C,H,W) or (H,W,C). The spatial grid is
        # square (two equal dims), so the channel axis is the odd one out. Flatten
        # to flat=[C, H*W] either way.
        if s[1] == s[2] and s[0] != s[1]:        # (C, H, W)
            C = s[0]; HW = s[1] * s[2]; flat = out.reshape(C, HW)
        elif s[0] == s[1] and s[2] != s[0]:      # (H, W, C)
            C = s[2]; HW = s[0] * s[1]; flat = np.ascontiguousarray(out.reshape(HW, C).T)
        else:
            C = s[0]; HW = s[1] * s[2]; flat = out.reshape(C, -1)
        W = int(round(math.sqrt(HW)))
        nc = C - 5 * self.na
        if nc <= 0:
            return []
        reg_block = flat[0:4 * self.na]
        reg = reg_block.reshape(self.na, 4, HW)
        obj_raw = flat[4 * self.na:5 * self.na]                      # [na, HW]
        cls_raw = flat[5 * self.na:5 * self.na + nc]                 # [nc, HW] shared
        # The export may already apply the activations (sigmoid on obj & box offsets,
        # softmax on cls). Detect it (values already in [0,1]) so we don't double-
        # activate — which squashes scores to ~0 AND blows box sizes up.
        if self._act is None:
            self._act = (bool(obj_raw.min() < -0.01 or obj_raw.max() > 1.01),     # obj needs sigmoid
                         bool(cls_raw.min() < -0.01 or cls_raw.max() > 1.01),     # cls needs softmax
                         bool(reg_block.min() < -0.01 or reg_block.max() > 1.01))  # reg needs sigmoid
        obj = _sigmoid(obj_raw) if self._act[0] else obj_raw
        cls = _softmax(cls_raw, axis=0) if self._act[1] else cls_raw
        cls_p = cls.max(0); cls_id = cls.argmax(0)                   # [HW]
        if self._dbg:
            self._dbg = False
            sc = (np.sqrt(obj * cls_p) if self.score_mode == "sqrt" else obj * cls_p)
            sys.stderr.write(
                f"decode dbg: C={C} nc={nc} obj[{obj_raw.min():.2f},{obj_raw.max():.2f}] "
                f"cls[{cls_raw.min():.2f},{cls_raw.max():.2f}] reg[{reg_block.min():.2f},{reg_block.max():.2f}] "
                f"act(obj={self._act[0]},cls={self._act[1]},reg={self._act[2]}) "
                f"max_score={float(sc.max()):.3f} #>={self.conf}:{int((sc >= self.conf).sum())}\n")
        gy, gx = np.divmod(np.arange(HW), W)
        anc = self.anchors[stride]
        sx, sy = ow / self.input, oh / self.input
        res = []
        for a in range(self.na):
            score = np.sqrt(obj[a] * cls_p) if self.score_mode == "sqrt" else obj[a] * cls_p
            keep = np.where(score >= self.conf)[0]
            if keep.size == 0:
                continue
            tx, ty, tw, th = reg[a, 0, keep], reg[a, 1, keep], reg[a, 2, keep], reg[a, 3, keep]
            aw, ah = anc[a * 2], anc[a * 2 + 1]
            cxg, cyg = gx[keep], gy[keep]
            # If reg is already activated in the model, don't sigmoid it again.
            sg = (lambda t: t) if not self._act[2] else _sigmoid
            sx_, sy_, sw_, sh_ = sg(tx), sg(ty), sg(tw), sg(th)
            if self.box_mode == "plain":
                bcx = (sx_ + cxg) * stride
                bcy = (sy_ + cyg) * stride
                bw = np.exp(tw) * aw
                bh = np.exp(th) * ah
            else:  # v5
                bcx = (sx_ * 2 - 0.5 + cxg) * stride
                bcy = (sy_ * 2 - 0.5 + cyg) * stride
                bw = (sw_ * 2) ** 2 * aw
                bh = (sh_ * 2) ** 2 * ah
            x1 = (bcx - bw / 2) * sx; y1 = (bcy - bh / 2) * sy
            x2 = (bcx + bw / 2) * sx; y2 = (bcy + bh / 2) * sy
            for i in range(keep.size):
                res.append((x1[i], y1[i], x2[i], y2[i], float(score[keep[i]]), int(cls_id[keep[i]])))
        return res

    def _nms(self, dets):
        boxes, scores, labels = [], [], []
        for (x1, y1, x2, y2, sc, lb) in dets:
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1]); scores.append(sc); labels.append(lb)
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


# ── Manual single-object tracker ──────────────────────────────────────────────
class ObjectTracker:
    """Robust manual lock for a MOVING camera. Runs OpenCV CSRT/KCF on a downscaled
    frame — much faster on a Pi, and the target shifts fewer pixels per frame, so a
    fast pan no longer shakes it off — and re-acquires the locked appearance in an
    expanding region around the last position when the tracker drops it (so it
    doesn't jump to a similar object elsewhere). NCC fallback if OpenCV lacks
    CSRT/KCF. Tune: MANUAL_TRACKER=csrt|kcf|ncc, MANUAL_TRACK_RES (px),
    MANUAL_REACQ (0..1), MANUAL_MAX_MISS. Lock stays active until cleared (C/Esc)."""

    def __init__(self):
        self.locked = False
        self.box = None
        self.miss = 0
        self.max_miss = int(env("MANUAL_MAX_MISS", "12"))
        self.reacq_ncc = float(env("MANUAL_REACQ", "0.6"))
        self.work = int(env("MANUAL_TRACK_RES", "480"))     # tracker runs at this max dim
        self._t = None
        self._ncc = None
        self.anchor = None                                  # grayscale template (work-frame scale)
        self.bw = self.bh = 0.0                             # box size, full-res px
        self.ds = 1.0
        self.kind = self._pick(env("MANUAL_TRACKER", "csrt").lower())
        print(f"  manual tracker: {self.kind.upper()} @ {self.work}px")

    @staticmethod
    def _ctor(kind):
        names = {"csrt": "TrackerCSRT_create", "kcf": "TrackerKCF_create"}.get(kind)
        if not names:
            return None
        if hasattr(cv2, names):
            return getattr(cv2, names)
        leg = getattr(cv2, "legacy", None)
        if leg is not None and hasattr(leg, names):
            return getattr(leg, names)
        return None

    def _pick(self, want):
        for k in (want, "csrt", "kcf"):
            if self._ctor(k) is not None:
                return k
        return "ncc"

    def _small(self, frame):
        H, W = frame.shape[:2]
        m = max(H, W)
        ds = (self.work / m) if m > self.work else 1.0
        if ds < 0.999:
            return cv2.resize(frame, (max(1, int(W * ds)), max(1, int(H * ds)))), ds
        return frame, 1.0

    def lock(self, frame, x1, y1, x2, y2):
        H, W = frame.shape[:2]
        x1 = max(0, min(int(x1), W - 2)); y1 = max(0, min(int(y1), H - 2))
        x2 = max(x1 + 2, min(int(x2), W)); y2 = max(y1 + 2, min(int(y2), H))
        self.box = (float(x1), float(y1), float(x2), float(y2))
        self.bw = float(x2 - x1); self.bh = float(y2 - y1); self.miss = 0
        small, ds = self._small(frame); self.ds = ds
        sx1, sy1 = int(x1 * ds), int(y1 * ds)
        sx2, sy2 = max(sx1 + 2, int(x2 * ds)), max(sy1 + 2, int(y2 * ds))
        try:
            if self.kind in ("csrt", "kcf"):
                self._t = self._ctor(self.kind)()
                self._t.init(small, (sx1, sy1, sx2 - sx1, sy2 - sy1))
                g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                roi = g[sy1:sy2, sx1:sx2]
                s = min(1.0, 48.0 / max(1, max(roi.shape)))
                self.anchor = cv2.resize(roi, (max(8, int(roi.shape[1] * s)), max(8, int(roi.shape[0] * s))))
            else:
                self._ncc = _NCCTracker(); self._ncc.lock(frame, x1, y1, x2, y2)
            self.locked = True
        except Exception as e:
            sys.stderr.write(f"manual lock failed: {e}\n"); self.locked = False

    def reset(self):
        self.locked = False; self.box = None; self._t = None; self._ncc = None
        self.anchor = None; self.miss = 0

    def update(self, frame):
        if not self.locked:
            return None
        if self.kind == "ncc":
            r = self._ncc.update(frame)
            self.locked = r is not None
            if r is not None:
                self.box = r
            return r
        small, ds = self._small(frame); self.ds = ds
        try:
            ok, b = self._t.update(small)
        except Exception:
            ok, b = False, None
        if ok and b is not None:
            x, y, w, h = b
            self.box = (x / ds, y / ds, (x + w) / ds, (y + h) / ds)
            self.bw = w / ds; self.bh = h / ds; self.miss = 0
            return self.box
        self.miss += 1
        if self.miss <= self.max_miss:
            return self.box                 # brief loss: hold the last box
        found = self._reacquire(small, ds)  # then search near the last position
        if found is not None:
            fx1, fy1, fx2, fy2 = found
            try:
                self._t = self._ctor(self.kind)()
                self._t.init(small, (int(fx1 * ds), int(fy1 * ds), int((fx2 - fx1) * ds), int((fy2 - fy1) * ds)))
                self.box = found; self.bw = fx2 - fx1; self.bh = fy2 - fy1; self.miss = 0
                return found
            except Exception:
                pass
        return None                          # still lost: stays locked, keeps searching

    def _reacquire(self, small, ds):
        if self.anchor is None:
            return None
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY); H, W = g.shape
        cx = (self.box[0] + self.box[2]) / 2 * ds; cy = (self.box[1] + self.box[3]) / 2 * ds
        bw_s = max(8.0, self.bw * ds); bh_s = max(8.0, self.bh * ds)
        grow = 2.0 + min(20, self.miss) * 0.5
        rx, ry = bw_s * grow, bh_s * grow
        x0 = int(max(0, cx - bw_s / 2 - rx)); y0 = int(max(0, cy - bh_s / 2 - ry))
        x1 = int(min(W, cx + bw_s / 2 + rx)); y1 = int(min(H, cy + bh_s / 2 + ry))
        roi = g[y0:y1, x0:x1]; rh, rw = roi.shape[:2]
        best = None
        for sc in (0.7, 0.85, 1.0, 1.2, 1.4):
            tw = max(8, int(bw_s * sc)); th = max(8, int(bh_s * sc))
            if tw >= rw or th >= rh:
                continue
            res = cv2.matchTemplate(roi, cv2.resize(self.anchor, (tw, th)), cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (mx, x0 + loc[0], y0 + loc[1], tw, th)
        if best and best[0] >= self.reacq_ncc:
            _, x, y, tw, th = best
            return (x / ds, y / ds, (x + tw) / ds, (y + th) / ds)
        return None


# ── NCC template-match fallback (used only when OpenCV has no CSRT/KCF) ────
class _NCCTracker:
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
        dets = [d for d in dets if len(d) >= 6]      # ignore malformed detections
        n = len(self.tracks)            # match only against pre-existing tracks
        matched = [False] * n
        for d in dets:
            best, bi = -1, self.iou_th
            for i in range(n):
                box = self.tracks[i][0]
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
            if tracker is not None:
                dets = tracker.update(dets, time.time())
        except Exception as e:
            sys.stderr.write(f"inference error: {e}\n"); dets = []
        with state.lock:
            state.dets = dets
        count += 1; dt = time.time() - t0
        if dt >= 1.0:
            state.det_fps = int(count / dt); count, t0 = 0, time.time()


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
    # Build via json.dumps so gimbal-reported text (firmware / hardwareId can hold
    # quotes, backslashes or control bytes) is escaped — a hand-built string would
    # produce invalid JSON and hang the panel on "loading…".
    return json.dumps({
        "hasGimbal": g is not None,
        "yaw": _fin(g.yaw) if g else 0.0,
        "pitch": _fin(g.pitch) if g else 0.0,
        "roll": _fin(g.roll) if g else 0.0,
        "recording": bool(g.recording) if g else False,
        "mode": mode,
        "tracking": bool(state.tracking),
        "firmware": (g.firmware if g else "") or "",
        "hardwareId": (g.hardware_id if g else "") or "",
    }, allow_nan=False)


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
<title>YOLO-FastestV2 panel</title><style>
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
var HASGIMBAL=false,showG=false,gotStatus=false;
function g(u){return fetch(u).then(function(r){if(!r.ok)throw 0;return r.json()}).then(show).catch(fail)}
function fail(){if(!gotStatus)document.getElementById('st').textContent='disconnected — retrying…'}
function applyG(){['pad','side','top','trk'].forEach(function(id){var el=document.getElementById(id);if(el)el.style.display=(HASGIMBAL&&showG)?'':'none'})}
function show(s){gotStatus=true;var st=document.getElementById('st');
 if(s.hasGimbal){st.textContent='yaw '+s.yaw+'  pitch '+s.pitch+'  roll '+s.roll+'\\nmode '+s.mode+'  rec '+s.recording}
 else{st.textContent='YOLO-FastestV2 (NCNN) — no gimbal'}
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
function poll(){fetch('/status').then(function(r){if(!r.ok)throw 0;return r.json()}).then(show).catch(fail)}
poll();setInterval(poll,1000);
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
        YoloFastestV2NCNN().inspect(); return

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

    print("YOLO-FastestV2 sidecar (NCNN / CPU)")
    detector = YoloFastestV2NCNN()
    detector.autoconfig()                                # auto-detect strides + output blobs
    print(f"  model loaded: input={detector.input} strides={detector.strides} "
          f"anchors/cell={detector.na} box={detector.box_mode} score={detector.score_mode} "
          f"conf={detector.conf} nms={detector.nms}")
    print(f"  input blob: {detector.in_name}   output blobs: {detector.out_blobs}")
    print("  (no detections? run with --inspect, set YF_OUTPUTS; try YF_BOX_DECODE=plain / YF_SCORE=mul)")

    cap = open_source(source, cam_w, cam_h, cam_fps)
    if not cap.isOpened():
        sys.stderr.write(f"ERROR: cannot open video source '{source}'\n"); sys.exit(1)

    if "--autotune" in sys.argv:
        print("  autotune: sampling frames (point the camera at your objects)…")
        frames, t = [], time.time()
        while len(frames) < 30 and time.time() - t < 8:
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
        if frames:
            detector.autotune(frames)
        else:
            print(f"  autotune: no frames from source '{source}' — keeping defaults. "
                  "(Pi CSI camera? use YOLO_SOURCE=rpicam)")

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
