#!/usr/bin/env python3
r"""
Betaflight flight-controller backend for the sidecars — make a Raspberry Pi the
"follow-me brain" of a Betaflight quad over MSP.

Two pieces:
  • BetaflightMSP — opens a serial link (UART or USB-VCP) to the FC and streams
    MSP_SET_RAW_RC (cmd 200) at a fixed rate. Used together with Betaflight's
    **MSP Override** mode so the Pi only takes the channels you allow (yaw + pitch),
    while the pilot keeps throttle, arm and failsafe on the transmitter.
  • DroneFollower — turns a locked YOLO target box into RC: yaw to keep the person
    centred, and forward/back pitch to hold distance (bounding-box height is the
    distance proxy — person shrinks ⇒ fly forward, grows ⇒ back off).

The RC math (`DroneFollower.compute`) and the MSP framing (`msp_frame`) are pure and
unit-tested; only the serial I/O needs hardware.

────────────────────────────────────────────────────────────────────────────────
⚠️  SAFETY — read before powering a drone
  • This flies a real aircraft toward a person. Test **props off** first, watch the
    RC values move the right way in Betaflight's Receiver tab, and only then fly,
    in an open area, away from people, at your own risk and within your local law.
  • Use **MSP Override on an AUX switch** (Modes tab). The Pi NEVER arms, never
    touches throttle, and the pilot can drop the switch to take back yaw+pitch
    instantly. Keep the transmitter on; configure failsafe.
  • If detection is lost or tracking is off, the follower sends **centre sticks**
    (no yaw, no forward) — the quad holds, it does not coast. If this process dies,
    MSP frames stop and Betaflight's failsafe/override-timeout takes over.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import struct
import sys
import threading
import time


# ── MSP v1 framing ────────────────────────────────────────────────────────────
MSP_SET_RAW_RC = 200


def msp_frame(cmd, payload=b""):
    r"""Encode one MSP v1 request frame: '$M<' + size + cmd + payload + XOR-crc.
    crc = size ^ cmd ^ payload bytes. Pure function — unit-tested."""
    size = len(payload)
    body = bytes([size & 0xFF, cmd & 0xFF]) + payload
    crc = 0
    for b in body:
        crc ^= b
    return b"$M<" + body + bytes([crc & 0xFF])


def rc_payload(channels):
    """Pack RC channel microseconds (1000..2000) as little-endian uint16s."""
    return b"".join(struct.pack("<H", int(max(1000, min(2000, c)))) for c in channels)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _envf(key, default):
    v = os.environ.get(key)
    try:
        return float(v) if v not in (None, "") else float(default)
    except ValueError:
        return float(default)


def _envi(key, default):
    return int(_envf(key, default))


def _envb(key, default=False):
    v = (os.environ.get(key) or "").strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "on")


# ── Serial MSP link to the FC ─────────────────────────────────────────────────
class BetaflightMSP:
    """Streams MSP_SET_RAW_RC to a Betaflight FC at a fixed rate from a background
    thread. The latest channel set is resent continuously (RX_MSP / MSP-Override
    needs a steady stream, and stopping it makes the FC fail safe).

    Channel order follows Betaflight's `map` (default AETR): index 0=Roll, 1=Pitch,
    2=Throttle, 3=Yaw, 4..=AUX. Override indices via FC_CH_* if your map differs.
    Throttle/AUX are left at neutral placeholders — keep them OUT of the FC's
    msp_override mask so the pilot owns them."""

    def __init__(self, port=None, baud=None, rate_hz=None, n_channels=8):
        import serial                                  # pyserial; lazy so import is cheap
        self.port = port or os.environ.get("FC_PORT", "/dev/ttyAMA0")
        self.baud = int(baud or _envi("FC_BAUD", 115200))
        self.rate = float(rate_hz or _envf("FC_RATE", 50))
        self.ch_roll = _envi("FC_CH_ROLL", 0)
        self.ch_pitch = _envi("FC_CH_PITCH", 1)
        self.ch_throttle = _envi("FC_CH_THROTTLE", 2)
        self.ch_yaw = _envi("FC_CH_YAW", 3)
        self.throttle_us = _envi("FC_THROTTLE_US", 1500)

        self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
        n = max(8, self.ch_yaw + 1)
        self.base = [1500] * n
        self.base[self.ch_throttle] = self.throttle_us
        self.lock = threading.Lock()
        self.channels = list(self.base)
        self.active = False                              # are we currently overriding?
        self.tx_count = 0
        self.running = True
        threading.Thread(target=self._tx_loop, daemon=True).start()

    def _tx_loop(self):
        period = 1.0 / max(1.0, self.rate)
        while self.running:
            with self.lock:
                ch = list(self.channels)
            try:
                self.ser.write(msp_frame(MSP_SET_RAW_RC, rc_payload(ch)))
                self.tx_count += 1
            except Exception:
                pass
            time.sleep(period)

    def set_control(self, roll=0.0, pitch=0.0, yaw=0.0):
        """Set signed microsecond offsets from centre for the controlled axes."""
        ch = list(self.base)
        ch[self.ch_roll] = int(clamp(1500 + roll, 1000, 2000))
        ch[self.ch_pitch] = int(clamp(1500 + pitch, 1000, 2000))
        ch[self.ch_yaw] = int(clamp(1500 + yaw, 1000, 2000))
        with self.lock:
            self.channels = ch
            self.active = (abs(roll) + abs(pitch) + abs(yaw)) > 0.5

    def neutral(self):
        """Centre the controlled axes (hold) — keeps streaming so the FC stays happy."""
        with self.lock:
            self.channels = list(self.base)
            self.active = False

    def status(self):
        with self.lock:
            return {"port": self.port, "active": self.active, "tx": self.tx_count}

    def close(self):
        self.running = False
        self.neutral()
        try:
            self.ser.write(msp_frame(MSP_SET_RAW_RC, rc_payload(self.base)))
            self.ser.close()
        except Exception:
            pass


# ── target picking (shared with the gimbal follower's idea of "stable target") ──
def _cdist(a, b):
    return ((((a[0] + a[2]) - (b[0] + b[2])) / 2) ** 2
            + (((a[1] + a[3]) - (b[1] + b[3])) / 2) ** 2) ** 0.5


def pick_target(dets, prev, fw):
    """Pick a stable target: stay on the one nearest the previous box if it's close,
    else take the largest (nearest/most prominent) detection."""
    if not dets:
        return None
    if prev is not None:
        near = min(dets, key=lambda d: _cdist(d, prev))
        if _cdist(near, prev) < 0.3 * fw:
            return near
    return max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))


# ── visual-servo follow controller (drives the FC) ────────────────────────────
class DroneFollower:
    """Locked target box → RC. Yaw keeps the target horizontally centred; pitch
    holds distance using the box height as a range proxy. Pure controller in
    `compute()`; `step()` adds target selection + FC output. Stick offsets are
    conservative by default and every gain/limit is env-tunable. The aircraft only
    moves while tracking is on AND a target exists — otherwise centre sticks."""

    def __init__(self, fc):
        self.fc = fc
        self.max_yaw = _envf("FC_MAX_YAW", 150)        # ±µs from centre on the yaw channel
        self.max_pitch = _envf("FC_MAX_PITCH", 120)    # ±µs from centre on the pitch channel
        self.kp_yaw = _envf("FC_KP_YAW", 360)          # µs per unit horizontal error (−0.5..0.5)
        self.kp_pitch = _envf("FC_KP_PITCH", 500)      # µs per unit fill error
        self.yaw_dz = _envf("FC_YAW_DEADZONE", 0.06)   # frac of frame width
        self.target_fill = _envf("FC_TARGET_FILL", 0.45)   # desired box-height / frame-height
        self.fill_dz = _envf("FC_FILL_DEADZONE", 0.08)
        self.invert_yaw = _envb("FC_INVERT_YAW", False)
        self.invert_pitch = _envb("FC_INVERT_PITCH", False)
        self.prev = None
        self.pending = None

    def request_pick(self, nx, ny):
        self.pending = (nx, ny)

    def compute(self, target, fw, fh):
        """Pure: (target box, frame w/h) → (roll, pitch, yaw) µs offsets from centre."""
        cx = (target[0] + target[2]) / 2.0
        bh = target[3] - target[1]
        ex = cx / fw - 0.5                              # −0.5 (left) .. +0.5 (right)
        yaw = 0.0 if abs(ex) < self.yaw_dz else clamp(self.kp_yaw * ex, -self.max_yaw, self.max_yaw)
        fill = bh / fh
        derr = self.target_fill - fill                 # >0 ⇒ target too small ⇒ fly forward
        pitch = 0.0 if abs(derr) < self.fill_dz else clamp(self.kp_pitch * derr, -self.max_pitch, self.max_pitch)
        if self.invert_yaw:
            yaw = -yaw
        if self.invert_pitch:
            pitch = -pitch
        return 0.0, pitch, yaw

    def step(self, dets, fw, fh):
        if self.pending is not None and dets:
            px, py = self.pending[0] * fw, self.pending[1] * fh
            self.pending = None
            inside = [d for d in dets if d[0] <= px <= d[2] and d[1] <= py <= d[3]]
            pool = inside if inside else dets
            self.prev = min(pool, key=lambda d: ((d[0] + d[2]) / 2 - px) ** 2 + ((d[1] + d[3]) / 2 - py) ** 2)
        target = pick_target(dets, self.prev, fw)
        self.prev = target
        if target is None or fw <= 0 or fh <= 0:
            self.fc.neutral()
            return None
        roll, pitch, yaw = self.compute(target, fw, fh)
        self.fc.set_control(roll=roll, pitch=pitch, yaw=yaw)
        return target

    def stop(self):
        self.fc.neutral()
        self.prev = None


def make_follower():
    """Build a (BetaflightMSP, DroneFollower) pair from env, or raise with a clear
    install/why message. Returns None if FC control isn't enabled."""
    backend = (os.environ.get("FC", "") or "").strip().lower()
    if backend in ("", "off", "0", "none"):
        return None
    if backend not in ("betaflight", "bf", "msp"):
        sys.stderr.write(f"WARNING: unknown FC backend '{backend}' — only 'betaflight' is supported.\n")
        return None
    try:
        fc = BetaflightMSP()
    except ImportError:
        sys.stderr.write("ERROR: pyserial not installed — needed for FC control.  pip3 install pyserial\n")
        return None
    except Exception as e:
        sys.stderr.write(f"ERROR: cannot open FC serial port ({e}). Check FC_PORT / wiring / that the "
                         "FC UART runs MSP.\n")
        return None
    return fc, DroneFollower(fc)


def print_safety_banner(fc):
    st = fc.status()
    sys.stderr.write(
        "\n" + "=" * 70 + "\n"
        "  ⚠️  BETAFLIGHT FOLLOW MODE ENABLED — autonomous flight toward a target\n"
        f"     port={st['port']}  channels: yaw+pitch overridden, throttle/arm = pilot\n"
        "     • MUST use Betaflight 'MSP Override' on an AUX switch (Modes tab).\n"
        "     • Test PROPS OFF; verify the Receiver tab moves the right way first.\n"
        "     • Keep the transmitter on; pilot owns throttle/arm/failsafe.\n"
        "     • Tracking off or target lost ⇒ centre sticks (hold), never coast.\n"
        + "=" * 70 + "\n")
