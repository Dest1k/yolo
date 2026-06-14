#!/usr/bin/env python3
r"""
YOLO command center — a pseudographic (curses) mission-control TUI for the whole
headless stack. One screen to:

  • Control   live telemetry + gimbal/camera/track/record over the HTTP API (local
              or remote boards; Tab between views, 1-9 between boards)
  • Launch    start/stop any sidecar backend (fastestv2 / nanodet / picodet /
              mediapipe / rknn) with editable source/port/model fields
  • Models    fetch a stock COCO model for any backend (runs its get_model.py)
  • Train     kick off training (runs the backend's train_*.py; edit its CONFIG
              block first)

No dependencies (curses is stdlib). The optional ASCII video preview wants Pillow.
Run it from anywhere:  python tools/command-center/cc.py  [host[:port] ...]
Remote boards passed on the CLI show up in Control alongside ones you Launch here.
"""
import argparse
import curses
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from io import BytesIO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEF_PORT = 8080
RAMP = " .:-=+*#%@"
VIEWS = ["Control", "Launch", "Models", "Train"]

# Backend registry: where each sidecar lives, how to run it, and the launch fields.
_TRAIN_COMMON = [("TRAIN_DATASET", ""), ("TRAIN_CLASSES", ""), ("TRAIN_INPUT", "416"),
                 ("TRAIN_EPOCHS", "200"), ("TRAIN_BATCH", "96"), ("TRAIN_DEVICE", "gpu")]
BACKENDS = [
    dict(key="fastestv2", label="YOLO-FastestV2", dir="tools/yolo-fastestv2-sidecar",
         script="yolofastest_ncnn_sidecar.py", get_model="get_model.py",
         train="train_yolofastest.py",
         fields=[("YF_PARAM", "yolo-fastestv2-opt.param"), ("YF_BIN", "yolo-fastestv2-opt.bin"),
                 ("YF_INPUT", "352")],
         train_fields=[("TRAIN_DATASET", ""), ("TRAIN_CLASSES", ""), ("TRAIN_INPUT", "352"),
                       ("TRAIN_EPOCHS", "300"), ("TRAIN_BATCH", "192"), ("TRAIN_DEVICE", "gpu")]),
    dict(key="nanodet", label="NanoDet-Plus", dir="tools/nanodet-sidecar",
         script="nanodet_ncnn_sidecar.py", get_model="get_model.py",
         train="train_nanodet.py",
         fields=[("ND_PARAM", "nanodet.param"), ("ND_BIN", "nanodet.bin"), ("ND_INPUT", "416")],
         train_fields=list(_TRAIN_COMMON)),
    dict(key="picodet", label="PicoDet", dir="tools/picodet-sidecar",
         script="picodet_ncnn_sidecar.py", get_model="get_model.py",
         train="train_picodet.py",
         fields=[("PICODET_PARAM", "picodet_s_320.param"), ("PICODET_BIN", "picodet_s_320.bin")],
         train_fields=[("PD_DATASET", ""), ("PD_CLASSES", ""), ("PD_EPOCHS", "80"),
                       ("PD_BATCH", "24"), ("PD_DEVICE", "gpu")]),
    dict(key="mediapipe", label="MediaPipe", dir="tools/mediapipe-sidecar",
         script="yolo_mediapipe_sidecar.py", get_model="get_model.py",
         train="train_object_detector.py",
         fields=[("YOLO_MODEL", "efficientdet_lite0.tflite")],
         train_fields=[("MM_MODEL", "lite0"), ("MM_EPOCHS", "50"), ("MM_BATCH", "16")]),
    dict(key="rknn", label="RKNN (Rockchip)", dir="tools/rknn-sidecar",
         script="yolo_rknn_sidecar.py", get_model="get_model.py", train=None,
         fields=[("RKNN_MODEL", "yolov5s.rknn")], train_fields=[]),
]


def parse_target(s):
    if ":" in s:
        h, p = s.rsplit(":", 1)
        return h, int(p)
    return s, DEF_PORT


class Target:
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.status, self.ok, self.last = {}, False, 0.0

    @property
    def name(self):
        return f"{self.host}:{self.port}"

    def url(self, path):
        return f"http://{self.host}:{self.port}{path}"


def http_bytes(url, timeout=1.2):
    req = urllib.request.Request(url, headers={"User-Agent": "cc"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(t, path, timeout=1.0):
    try:
        return json.loads(http_bytes(t.url(path), timeout).decode("utf-8", "replace"))
    except Exception:
        return None


def read_one_jpeg(t, timeout=2.0):
    try:
        r = urllib.request.urlopen(urllib.request.Request(t.url("/stream")), timeout=timeout)
    except Exception:
        return None
    buf, deadline = b"", time.time() + timeout
    try:
        while time.time() < deadline:
            chunk = r.read(8192)
            if not chunk:
                break
            buf += chunk
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
            if s >= 0 and e >= 0:
                return buf[s:e + 2]
    except Exception:
        pass
    finally:
        try:
            r.close()
        except Exception:
            pass
    return None


class Proc:
    """A child process (sidecar / get_model / train) with captured output."""
    def __init__(self, cmd, cwd, env=None, label=""):
        self.label, self.log, self.cmd = label, deque(maxlen=1000), cmd
        e = dict(os.environ)
        if env:
            e.update(env)
        self.log.append(f"$ {' '.join(cmd)}  (cwd={cwd})")
        self.p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=1, text=True, errors="replace")
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for line in self.p.stdout:
                self.log.append(line.rstrip("\n"))
        except Exception:
            pass
        self.log.append(f"[exited {self.p.poll()}]")

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        try:
            self.p.terminate()
        except Exception:
            pass


class App:
    def __init__(self, targets):
        self.targets = targets
        self.active = 0 if targets else -1
        self.view = 0
        self.running = True
        self.log = deque(maxlen=300)
        self.lock = threading.Lock()
        # control
        self.speed = 40
        self.preview = False
        self.pv_lines, self.pv_w, self.pv_h = [], 0, 0
        self.move_dir, self.moving, self.last_move = None, False, 0.0
        # launch
        self.lb = 0
        self.lsel = 0
        self.lfields = self._mk_fields(0)
        # models / train
        self.mb = 0
        self.tb = 0
        self.tfields = self._mk_train_fields(0)
        self.tsel = 0
        # processes
        self.procs = {}          # "key:port" -> Proc (sidecars)
        self.task = None         # current get_model / train Proc
        try:
            from PIL import Image
            self.Image = Image
        except Exception:
            self.Image = None

    # ── helpers ───────────────────────────────────────────────────────────────
    def _mk_fields(self, bi):
        b = BACKENDS[bi]
        return [["YOLO_SOURCE", "rpicam"], ["YOLO_PORT", str(DEF_PORT)]] + [[k, v] for k, v in b["fields"]]

    def _mk_train_fields(self, bi):
        return [[k, v] for k, v in BACKENDS[bi]["train_fields"]]

    def at(self):
        return self.targets[self.active] if 0 <= self.active < len(self.targets) else None

    def logmsg(self, m):
        self.log.append((time.strftime("%H:%M:%S"), m))

    def add_target(self, host, port):
        for i, t in enumerate(self.targets):
            if t.host == host and t.port == port:
                return i
        self.targets.append(Target(host, port))
        return len(self.targets) - 1

    # ── networking / control ──────────────────────────────────────────────────
    def send(self, path, note=None):
        t = self.at()
        if not t:
            self.logmsg("no active board"); return
        threading.Thread(target=lambda: get_json(t, path, 1.0), daemon=True).start()
        self.logmsg(note or path)

    def poll_loop(self):
        while self.running:
            for i, t in enumerate(self.targets):
                if i == self.active or (time.time() - t.last) > 1.5:
                    st = get_json(t, "/status", 1.0)
                    with self.lock:
                        t.ok = st is not None
                        if st is not None:
                            t.status = st
                        t.last = time.time()
            time.sleep(0.15)

    def preview_loop(self):
        while self.running:
            t = self.at()
            if self.view == 0 and self.preview and self.Image and t and self.pv_w > 4 and self.pv_h > 2:
                jpg = read_one_jpeg(t)
                if jpg:
                    try:
                        im = self.Image.open(BytesIO(jpg)).convert("L").resize((self.pv_w, self.pv_h))
                        px = im.load(); n = len(RAMP) - 1
                        lines = ["".join(RAMP[px[x, y] * n // 255] for x in range(self.pv_w))
                                 for y in range(self.pv_h)]
                        with self.lock:
                            self.pv_lines = lines
                    except Exception:
                        pass
                time.sleep(0.05)
            else:
                time.sleep(0.2)

    def move(self, yaw, pitch, d):
        if self.move_dir != d or not self.moving:
            self.send(f"/rotate?yaw={yaw}&pitch={pitch}", f"rotate {d}")
            self.move_dir, self.moving = d, True
        self.last_move = time.time()

    def watchdog(self):
        if self.moving and (time.time() - self.last_move) > 0.18:
            self.send("/stop", "stop"); self.moving = False; self.move_dir = None

    def cycle_mode(self):
        order = ["lock", "follow", "fpv"]
        cur = (self.at().status.get("mode") if self.at() else "lock")
        nxt = order[(order.index(cur) + 1) % 3] if cur in order else "lock"
        self.send(f"/mode?m={nxt}", f"mode {nxt}")

    # ── process orchestration ─────────────────────────────────────────────────
    def launch_sidecar(self):
        b = BACKENDS[self.lb]
        env = {k: v for k, v in self.lfields}
        port = int(env.get("YOLO_PORT", str(DEF_PORT)) or DEF_PORT)
        key = f"{b['key']}:{port}"
        if key in self.procs and self.procs[key].alive():
            self.logmsg(f"{key} already running"); return
        cmd = [sys.executable, b["script"]]
        try:
            self.procs[key] = Proc(cmd, os.path.join(ROOT, b["dir"]), env, key)
        except Exception as e:
            self.logmsg(f"launch failed: {e}"); return
        idx = self.add_target("127.0.0.1", port)
        self.active = idx
        self.logmsg(f"launched {key}")

    def stop_sidecar(self):
        b = BACKENDS[self.lb]
        port = int(dict(self.lfields).get("YOLO_PORT", DEF_PORT) or DEF_PORT)
        key = f"{b['key']}:{port}"
        p = self.procs.get(key)
        if p and p.alive():
            p.stop(); self.logmsg(f"stopped {key}")
        else:
            self.logmsg(f"{key} not running")

    def run_task(self, bi, which, env=None):
        b = BACKENDS[bi]
        script = b.get(which)
        if not script:
            self.logmsg(f"{b['label']}: no {which}"); return
        if self.task and self.task.alive():
            self.logmsg("a task is already running"); return
        try:
            self.task = Proc([sys.executable, script], os.path.join(ROOT, b["dir"]), env,
                             f"{b['key']}:{which}")
            self.logmsg(f"{which} {b['label']} started")
        except Exception as e:
            self.logmsg(f"{which} failed: {e}")

    # ── keys ──────────────────────────────────────────────────────────────────
    def on_key(self, k):
        if k in (ord("\t"),):
            self.view = (self.view + 1) % len(VIEWS); return
        if k == curses.KEY_BTAB:
            self.view = (self.view - 1) % len(VIEWS); return
        if k == 27:
            self.running = False; return
        [self._k_control, self._k_launch, self._k_models, self._k_train][self.view](k)

    def _k_control(self, k):
        sp = self.speed
        if k == curses.KEY_UP: self.move(0, sp, "up")
        elif k == curses.KEY_DOWN: self.move(0, -sp, "down")
        elif k == curses.KEY_LEFT: self.move(-sp, 0, "left")
        elif k == curses.KEY_RIGHT: self.move(sp, 0, "right")
        elif k == ord("]"): self.send("/zoom?dir=in", "zoom in")
        elif k == ord("["): self.send("/zoom?dir=out", "zoom out")
        elif k == ord("."): self.send("/focus?dir=far", "focus far")
        elif k == ord(","): self.send("/focus?dir=near", "focus near")
        elif k == ord("o"): self.send("/autofocus", "autofocus")
        elif k == ord("c"): self.send("/center", "center")
        elif k == ord("m"): self.cycle_mode()
        elif k == ord("t"): self.send("/track", "track")
        elif k == ord("r"): self.send("/record", "record")
        elif k == ord("p"): self.send("/photo", "photo")
        elif k == ord("H"): self.send("/hdr", "hdr")
        elif k in (ord("="), ord("+")): self.speed = min(100, self.speed + 10)
        elif k == ord("-"): self.speed = max(10, self.speed - 10)
        elif k == ord("v"): self.preview = not self.preview
        elif ord("1") <= k <= ord("9") and (k - ord("1")) < len(self.targets):
            self.active = k - ord("1")
        elif k == ord("q"): self.running = False

    def _k_launch(self, k):
        nrows = 1 + len(self.lfields) + 2          # backend + fields + Start + Stop
        if k == curses.KEY_UP: self.lsel = (self.lsel - 1) % nrows
        elif k == curses.KEY_DOWN: self.lsel = (self.lsel + 1) % nrows
        elif self.lsel == 0 and k in (curses.KEY_LEFT, curses.KEY_RIGHT):
            self.lb = (self.lb + (1 if k == curses.KEY_RIGHT else -1)) % len(BACKENDS)
            self.lfields = self._mk_fields(self.lb); self.lsel = 0
        elif 1 <= self.lsel <= len(self.lfields):
            fv = self.lfields[self.lsel - 1]
            if k in (curses.KEY_BACKSPACE, 127, 8):
                fv[1] = fv[1][:-1]
            elif 32 <= k < 127:
                fv[1] += chr(k)
        elif k in (curses.KEY_ENTER, 10, 13):
            if self.lsel == 1 + len(self.lfields):
                self.launch_sidecar()
            elif self.lsel == 2 + len(self.lfields):
                self.stop_sidecar()

    def _k_models(self, k):
        if k in (curses.KEY_LEFT, curses.KEY_RIGHT):
            self.mb = (self.mb + (1 if k == curses.KEY_RIGHT else -1)) % len(BACKENDS)
        elif k in (curses.KEY_ENTER, 10, 13, ord("f")):
            self.run_task(self.mb, "get_model")
        elif k == ord("q"): self.running = False

    def _k_train(self, k):
        if k in (curses.KEY_LEFT, curses.KEY_RIGHT) and (self.tsel == 0 or not BACKENDS[self.tb].get("train")):
            self.tb = (self.tb + (1 if k == curses.KEY_RIGHT else -1)) % len(BACKENDS)
            self.tfields = self._mk_train_fields(self.tb); self.tsel = 0
            return
        if not BACKENDS[self.tb].get("train"):
            if k == ord("q"): self.running = False
            return
        nrows = 1 + len(self.tfields) + 1          # backend + fields + Start
        if k == curses.KEY_UP: self.tsel = (self.tsel - 1) % nrows
        elif k == curses.KEY_DOWN: self.tsel = (self.tsel + 1) % nrows
        elif 1 <= self.tsel <= len(self.tfields):
            fv = self.tfields[self.tsel - 1]
            if k in (curses.KEY_BACKSPACE, 127, 8):
                fv[1] = fv[1][:-1]
            elif 32 <= k < 127:
                fv[1] += chr(k)
        elif k in (curses.KEY_ENTER, 10, 13) and self.tsel == 1 + len(self.tfields):
            env = {kk: vv for kk, vv in self.tfields if vv != ""}
            self.run_task(self.tb, "train", env)


# ── drawing ─────────────────────────────────────────────────────────────────
def addstr(win, y, x, s, attr=0):
    h, w = win.getmaxyx()
    if 0 <= y < h and 0 <= x < w:
        try:
            win.addnstr(y, x, s, max(0, w - x - 1), attr)
        except curses.error:
            pass


def box(win, y, x, h, w, title="", attr=0):
    H, W = win.getmaxyx()
    if y < 0 or x < 0 or y + h > H or x + w > W or h < 2 or w < 2:
        return
    for i in range(1, w - 1):
        addstr(win, y, x + i, "─", attr); addstr(win, y + h - 1, x + i, "─", attr)
    for i in range(1, h - 1):
        addstr(win, y + i, x, "│", attr); addstr(win, y + i, x + w - 1, "│", attr)
    addstr(win, y, x, "┌", attr); addstr(win, y, x + w - 1, "┐", attr)
    addstr(win, y + h - 1, x, "└", attr); addstr(win, y + h - 1, x + w - 1, "┘", attr)
    if title:
        addstr(win, y, x + 2, f" {title} ", attr | curses.A_BOLD)


def bar(val, lo, hi, width):
    val = max(lo, min(hi, val))
    pos = int((val - lo) / (hi - lo) * (width - 1)) if hi > lo else 0
    return "".join("●" if i == pos else "─" for i in range(width))


def proc_log_box(scr, app, C, y, x, h, w, proc, title):
    box(scr, y, x, h, w, title, C["dim"])
    rows = list(proc.log)[-(h - 2):] if proc else ["(nothing yet)"]
    for i, ln in enumerate(rows):
        addstr(scr, y + 1 + i, x + 2, ln, C["fg"])


def draw_tabs(scr, app, C, W):
    addstr(scr, 0, 0, "═" * W, C["dim"])
    x = 2
    for i, v in enumerate(VIEWS):
        a = (C["title"] | curses.A_BOLD) if i == app.view else C["dim"]
        lab = f" {i+1 if False else ''}{v} "
        addstr(scr, 0, x, f"[{v}]", a)
        x += len(v) + 3
    hint = "Tab=view  Esc=quit"
    addstr(scr, 0, max(0, W - len(hint) - 1), hint, C["dim"])


def draw_control(scr, app, C, H, W):
    t = app.at()
    st = t.status if t else {}
    addstr(scr, 1, 1, "Boards:", C["dim"])
    x = 9
    for i, tg in enumerate(app.targets):
        col = C["ok"] if tg.ok else C["bad"]
        label = f"{'▶' if i == app.active else ' '}{i+1}:{tg.name}"
        addstr(scr, 1, x, label, col | (curses.A_BOLD if i == app.active else 0)); x += len(label) + 2
    if not t:
        addstr(scr, 3, 2, "No board. Launch one (Tab → Launch) or pass host on the CLI.", C["warn"]); return

    tw, th = min(46, W - 2), 13
    box(scr, 3, 1, th, tw, "TELEMETRY", C["dim"])
    addstr(scr, 4, 3, t.name, C["fg"] | curses.A_BOLD)
    addstr(scr, 4, tw - 10, "● ONLINE" if t.ok else "○ OFFLINE", C["ok"] if t.ok else C["bad"])
    if st.get("hasGimbal"):
        addstr(scr, 6, 3, f"yaw   {st.get('yaw',0):7.1f}°  [{bar(st.get('yaw',0),-135,135,18)}]", C["fg"])
        addstr(scr, 7, 3, f"pitch {st.get('pitch',0):7.1f}°  [{bar(st.get('pitch',0),-90,25,18)}]", C["fg"])
        addstr(scr, 8, 3, f"roll  {st.get('roll',0):7.1f}°  [{bar(st.get('roll',0),-45,45,18)}]", C["fg"])
    else:
        addstr(scr, 6, 3, "no gimbal — camera/track only", C["dim"])
    addstr(scr, 9, 3, f"mode {st.get('mode','?'):<7}", C["fg"])
    addstr(scr, 9, 16, "● REC" if st.get("recording") else "  rec", C["bad"] if st.get("recording") else C["dim"])
    addstr(scr, 9, 24, "◎ TRACK" if st.get("tracking") else "  track", C["warn"] if st.get("tracking") else C["dim"])
    fps, det, nd = st.get("streamFps"), st.get("detFps"), st.get("ndet")
    addstr(scr, 10, 3, f"stream {('—' if fps is None else fps)!s:>3} fps   infer "
                        f"{('—' if det is None else det)!s:>3} fps   dets {('—' if nd is None else nd)}", C["accent"])
    addstr(scr, 11, 3, f"speed {app.speed:3d}  (=/- )", C["dim"])

    rx, rw = tw + 2, W - (tw + 2) - 1
    if app.preview and rw > 8:
        box(scr, 3, rx, th, rw, "PREVIEW (v)", C["dim"])
        with app.lock:
            app.pv_w, app.pv_h = max(4, rw - 2), max(2, th - 2)
            lines = list(app.pv_lines)
        if not app.Image:
            addstr(scr, 5, rx + 2, "pip install pillow for preview", C["warn"])
        for i, ln in enumerate(lines[:th - 2]):
            addstr(scr, 4 + i, rx + 1, ln, C["fg"])
    elif rw > 22:
        box(scr, 3, rx, th, rw, "KEYS", C["dim"])
        for i, ln in enumerate([
                "↑↓←→ gimbal", "] [  zoom in/out", ". ,  focus far/near",
                "o autofocus   c center", "m mode  t track", "r record  p photo  H hdr",
                "v preview", "1-9 switch board  q quit"][:th - 2]):
            addstr(scr, 4 + i, rx + 2, ln, C["fg"])

    ly = 3 + th + 1
    if H - ly - 1 >= 3:
        box(scr, ly, 1, H - ly - 1, W - 2, "LOG", C["dim"])
        with app.lock:
            rows = list(app.log)[-(H - ly - 3):]
        for i, (ts, m) in enumerate(rows):
            addstr(scr, ly + 1 + i, 3, f"{ts}  {m}", C["dim"])


def draw_launch(scr, app, C, H, W):
    b = BACKENDS[app.lb]
    box(scr, 2, 1, 4 + len(app.lfields), min(50, W - 2), "LAUNCH", C["dim"])
    sel = app.lsel
    rowsel = lambda i: (C["title"] | curses.A_BOLD) if i == sel else C["fg"]
    addstr(scr, 3, 3, f"Backend ◄ {b['label']:<16} ►", rowsel(0))
    for i, (k, v) in enumerate(app.lfields):
        cur = "_" if sel == i + 1 else ""
        addstr(scr, 4 + i, 3, f"{k:<13} {v}{cur}", rowsel(i + 1))
    si = 1 + len(app.lfields)
    key = f"{b['key']}:{dict(app.lfields).get('YOLO_PORT', '?')}"
    running = key in app.procs and app.procs[key].alive()
    addstr(scr, 4 + len(app.lfields), 3, "[ Start ]", rowsel(si) | (C["ok"] if not running else 0))
    addstr(scr, 5 + len(app.lfields), 3, "[ Stop ]", rowsel(si + 1) | (C["bad"] if running else 0))
    addstr(scr, 5 + len(app.lfields), 16, ("● running" if running else "○ stopped"),
           C["ok"] if running else C["dim"])
    addstr(scr, 2, min(50, W - 2) - 26, "↑↓ field  ◄► change  ⏎ act", C["dim"])

    # log of this backend's process
    ly = 7 + len(app.lfields)
    if H - ly - 1 >= 4:
        proc = app.procs.get(key)
        proc_log_box(scr, app, C, ly, 1, H - ly - 1, W - 2, proc, f"OUTPUT  {key}")


def draw_models(scr, app, C, H, W):
    b = BACKENDS[app.mb]
    bw = min(64, W - 2)
    box(scr, 2, 1, 5, bw, "MODELS", C["dim"])
    addstr(scr, 3, 3, f"Backend ◄ {b['label']:<16} ►", C["title"] | curses.A_BOLD)
    addstr(scr, 4, 3, "⏎ or f = fetch a stock COCO model (runs get_model.py)", C["fg"])
    addstr(scr, 2, bw - 12, "◄► change", C["dim"])
    ly = 8
    if H - ly - 1 >= 4:
        proc_log_box(scr, app, C, ly, 1, H - ly - 1, W - 2, app.task, "TASK OUTPUT")


def draw_train(scr, app, C, H, W):
    b = BACKENDS[app.tb]
    nf = len(app.tfields)
    bw = min(66, W - 2)
    bh = 4 + max(1, nf) + 1
    box(scr, 2, 1, bh, bw, "TRAIN", C["dim"])
    rowsel = lambda i: (C["title"] | curses.A_BOLD) if i == app.tsel else C["fg"]
    addstr(scr, 3, 3, f"Backend ◄ {b['label']:<16} ►", rowsel(0))
    if not b.get("train"):
        addstr(scr, 5, 3, "no trainer for this backend", C["warn"])
    else:
        for i, (k, v) in enumerate(app.tfields):
            cur = "_" if app.tsel == i + 1 else ""
            addstr(scr, 4 + i, 3, f"{k:<15} {v}{cur}", rowsel(i + 1))
        addstr(scr, 4 + nf, 3, "[ Start training ]", rowsel(1 + nf) | C["ok"])
    addstr(scr, 2, bw - 28, "↑↓ field ◄► backend ⏎ act", C["dim"])
    addstr(scr, bh + 1, 3, "set fields here — no need to touch the CONFIG block", C["dim"])
    ly = bh + 2
    if H - ly - 1 >= 4:
        proc_log_box(scr, app, C, ly, 1, H - ly - 1, W - 2, app.task, "TASK OUTPUT")


def draw(scr, app, C):
    scr.erase()
    H, W = scr.getmaxyx()
    draw_tabs(scr, app, C, W)
    if app.view == 0:
        draw_control(scr, app, C, H, W)
    elif app.view == 1:
        draw_launch(scr, app, C, H, W)
    elif app.view == 2:
        draw_models(scr, app, C, H, W)
    else:
        draw_train(scr, app, C, H, W)
    scr.noutrefresh(); curses.doupdate()


def run(scr, app):
    curses.curs_set(0); scr.nodelay(True); scr.keypad(True)
    C = {}
    try:
        curses.start_color(); curses.use_default_colors()
        defs = {"title": (curses.COLOR_BLACK, curses.COLOR_CYAN), "ok": (curses.COLOR_GREEN, -1),
                "bad": (curses.COLOR_RED, -1), "warn": (curses.COLOR_YELLOW, -1),
                "accent": (curses.COLOR_CYAN, -1), "fg": (-1, -1), "dim": (curses.COLOR_WHITE, -1)}
        for i, (k, (f, b)) in enumerate(defs.items(), 1):
            curses.init_pair(i, f, b); C[k] = curses.color_pair(i)
        C["dim"] |= curses.A_DIM
    except Exception:
        C = {k: 0 for k in ("title", "ok", "bad", "warn", "accent", "fg", "dim")}

    threading.Thread(target=app.poll_loop, daemon=True).start()
    threading.Thread(target=app.preview_loop, daemon=True).start()
    while app.running:
        try:
            k = scr.getch()
        except KeyboardInterrupt:
            break
        if k != -1 and k != curses.KEY_RESIZE:
            app.on_key(k)
        app.watchdog()
        draw(scr, app, C)
        time.sleep(0.03)


def cli():
    ap = argparse.ArgumentParser(description="Pseudographic command center for the YOLO headless stack.")
    ap.add_argument("targets", nargs="*", help="host[:port] of remote boards to also show in Control")
    a = ap.parse_args()
    targets = [Target(*parse_target(s)) for s in a.targets]
    app = App(targets)
    try:
        curses.wrapper(run, app)
    finally:
        app.running = False
        for p in app.procs.values():
            p.stop()
        if app.task:
            app.task.stop()


if __name__ == "__main__":
    cli()
