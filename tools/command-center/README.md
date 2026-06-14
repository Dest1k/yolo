# YOLO Command Center (TUI)

One pseudographic screen to drive the **whole** headless stack — no walking through
config files. Pure `curses` (stdlib); the optional ASCII video preview wants Pillow.

```bash
python tools/command-center/cc.py                     # local
python tools/command-center/cc.py 192.168.1.50 192.168.1.51:8080   # + remote boards
```

`Tab` switches views, `Esc` quits. Remote boards passed on the CLI appear in
**Control** next to ones you launch here.

## Views

**Control** — live telemetry + manual control over the HTTP API (works against the
Pi sidecars *and* the desktop headless runner):
- gimbal attitude bars (yaw/pitch/roll), mode, ● REC, ◎ TRACK, stream/infer FPS,
  detection count, firmware/hw.
- `↑↓←→` gimbal (auto-stop on release) · `] [` zoom · `. ,` focus · `o` autofocus ·
  `c` center · `m` cycle mode · `t` track · `r` record · `p` photo · `H` hdr ·
  `=/-` speed · `v` ASCII video preview · `1`-`9` switch board.
- `d` **discover** boards on your LAN (scans the local /24 for sidecars answering
  `/status` and adds them) · `s` **snapshot** the current frame to `cc_snapshots/`.

**Launch** — start/stop any backend sidecar (fastestv2 / nanodet / picodet /
mediapipe / rknn). Edit `YOLO_SOURCE`, `YOLO_PORT` and the model fields right in the
form (`↑↓` field, `◄►` backend, type to edit), then select `[ Start ]` / `[ Stop ]`
and press `⏎`. The launched board is added to **Control** automatically; its stdout
streams in the OUTPUT pane.

**Models** — fetch a stock COCO model for the selected backend (`◄►` to choose,
`⏎`/`f` to run its `get_model.py`). Output streams live. fastestv2 + mediapipe are
instant downloads; picodet/nanodet download + convert.

**Train** — kick off training without touching any config: edit `TRAIN_DATASET`,
`TRAIN_CLASSES`, input/epochs/batch/device in the form and press `⏎` on
`[ Start training ]`. The fields are passed as env to the trainer (the trainers read
`TRAIN_*` / `PD_*` / `MM_*` overriding their in-file CONFIG), and the run (incl. the
auto-export) streams in the OUTPUT pane.

## Editable fields, the easy way

Every form field (source, port, input size, epochs, device, model paths…) is a
**combo box**: press `◄►` to cycle through sensible presets, or just type a custom
value. Whatever you set is **remembered between runs** (saved to `~/.yolo_cc.json`),
and any custom value you type becomes a preset you can cycle back to next time. The
last-used backend per view is restored on launch too. A running task can be killed
with `Del` from any view; the top bar shows how many sidecars/tasks are live.

## How it fits together

The command center is a thin orchestrator: it **launches the existing scripts as
child processes** (capturing their logs) and **talks to their HTTP control API** for
telemetry/control. Nothing new on the device side — it drives exactly the endpoints
the web panel uses, plus the same `get_model.py` / `train_*.py` you'd run by hand.

Optional preview: `pip install pillow` (one frame is pulled from `/stream` and
rendered as ASCII). Everything else needs only the Python standard library.
