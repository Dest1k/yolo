#!/usr/bin/env python3
r"""
NanoDet-Plus trainer — graphical front-end (Windows-friendly, zero extra deps).

A window for every knob in train_nanodet.py — dataset, classes, input size,
epochs, batch, workers, device, learning rate, and **fine-tuning / resume
(дообучение)** — with a live log pane underneath that streams everything the
trainer prints (data conversion → training → ncnn export → VERIFY OK).

    python train_nanodet_gui.py

It doesn't reimplement training: it just sets the TRAIN_* environment variables
that train_nanodet.py already understands and runs it as a child process, so the
GUI and the plain `python train_nanodet.py` path stay in lock-step. Field values
are remembered between runs (~/.nanodet_trainer_gui.json).

Only the standard library is used (tkinter ships with python on Windows/macOS and
is `apt install python3-tk` on Linux). torch/nanodet are needed by the trainer it
launches, not by this window.
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception as e:                                       # headless box / no Tk
    sys.stderr.write(
        "ERROR: tkinter is not available — this is the GUI launcher.\n"
        f"  ({type(e).__name__}: {e})\n"
        "  Linux:   sudo apt install python3-tk\n"
        "  Or just run the trainer without a GUI:  python train_nanodet.py\n")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "train_nanodet.py")
STATE_PATH = os.path.expanduser("~/.nanodet_trainer_gui.json")

INPUT_CHOICES = ["256", "320", "352", "416", "512"]
DEVICE_CHOICES = ["gpu", "cpu"]


def load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
    except Exception:
        pass


class TrainerGUI:
    def __init__(self, root):
        self.root = root
        self.proc = None
        self.q = queue.Queue()
        self.state = load_state()
        root.title("NanoDet-Plus Trainer")
        root.minsize(760, 620)

        self._build_form()
        self._build_log()
        self._restore()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(80, self._drain)

    # ── layout ────────────────────────────────────────────────────────────────
    def _build_form(self):
        f = ttk.Frame(self.root, padding=10)
        f.pack(fill="x", side="top")
        for c in range(4):
            f.columnconfigure(c, weight=(1 if c in (1, 3) else 0))
        self.vars = {}
        r = 0

        # Dataset + browse (spans the row)
        ttk.Label(f, text="Dataset (YOLO root)").grid(row=r, column=0, sticky="w", pady=3)
        self.vars["TRAIN_DATASET"] = v = tk.StringVar()
        ttk.Entry(f, textvariable=v).grid(row=r, column=1, columnspan=2, sticky="ew", padx=6)
        ttk.Button(f, text="Browse…", command=self._pick_dataset).grid(row=r, column=3, sticky="ew")
        r += 1

        # Classes (spans the row)
        ttk.Label(f, text="Classes (id order, comma-sep)").grid(row=r, column=0, sticky="w", pady=3)
        self.vars["TRAIN_CLASSES"] = v = tk.StringVar()
        ttk.Entry(f, textvariable=v).grid(row=r, column=1, columnspan=3, sticky="ew", padx=6)
        r += 1

        # Two-up numeric / combo fields
        def pair(label_l, key_l, kind_l, label_r, key_r, kind_r):
            nonlocal r
            self._field(f, r, 0, label_l, key_l, kind_l)
            self._field(f, r, 2, label_r, key_r, kind_r)
            r += 1

        pair("Input size", "TRAIN_INPUT", INPUT_CHOICES, "Epochs", "TRAIN_EPOCHS", "int")
        pair("Batch size", "TRAIN_BATCH", "int", "Dataloader workers", "TRAIN_WORKERS", "int")
        pair("Device", "TRAIN_DEVICE", DEVICE_CHOICES, "GPU ids (comma-sep)", "TRAIN_GPU_IDS", "str")
        pair("reg_max (DFL bins−1)", "TRAIN_REG_MAX", "int",
             "Learning rate (blank=stock)", "TRAIN_LR", "str")

        # Export checkbox
        self.export_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Auto-export a verified NCNN model after training",
                        variable=self.export_var).grid(row=r, column=0, columnspan=4, sticky="w", pady=(6, 2))
        r += 1

        # ── Fine-tuning / resume (дообучение) ──
        lf = ttk.LabelFrame(f, text="Fine-tuning / resume (дообучение)", padding=8)
        lf.grid(row=r, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        lf.columnconfigure(1, weight=1)
        self.ft_mode = tk.StringVar(value="scratch")
        modes = ttk.Frame(lf); modes.grid(row=0, column=0, columnspan=3, sticky="w")
        for text, val in (("Train from scratch", "scratch"),
                          ("Fine-tune from weights", "weights"),
                          ("Resume interrupted run", "resume")):
            ttk.Radiobutton(modes, text=text, value=val, variable=self.ft_mode,
                            command=self._sync_ft).pack(side="left", padx=(0, 12))
        ttk.Label(lf, text="Checkpoint (.ckpt)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ckpt_var = tk.StringVar()
        self.ckpt_entry = ttk.Entry(lf, textvariable=self.ckpt_var)
        self.ckpt_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        self.ckpt_btn = ttk.Button(lf, text="Browse…", command=self._pick_ckpt)
        self.ckpt_btn.grid(row=1, column=2, sticky="ew", pady=(6, 0))
        r += 1

        # Action buttons
        bar = ttk.Frame(f); bar.grid(row=r, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        self.start_btn = ttk.Button(bar, text="▶  Start training", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="■  Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Clear log", command=self._clear_log).pack(side="left", padx=6)
        self.status = ttk.Label(bar, text="idle"); self.status.pack(side="right")

    def _field(self, parent, row, col, label, key, kind):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", pady=3)
        v = tk.StringVar()
        self.vars[key] = v
        if isinstance(kind, list):
            ttk.Combobox(parent, textvariable=v, values=kind, width=14).grid(
                row=row, column=col + 1, sticky="ew", padx=6)
        else:
            ttk.Entry(parent, textvariable=v, width=16).grid(
                row=row, column=col + 1, sticky="ew", padx=6)

    def _build_log(self):
        frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Training log").pack(anchor="w")
        wrap = ttk.Frame(frame); wrap.pack(fill="both", expand=True)
        self.log = tk.Text(wrap, wrap="none", bg="#101418", fg="#d6e2ec",
                           insertbackground="#d6e2ec", font=("Consolas", 10), height=18)
        ys = ttk.Scrollbar(wrap, orient="vertical", command=self.log.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.log.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        xs.pack(fill="x")
        self.log.tag_config("err", foreground="#ff9b9b")
        self.log.tag_config("ok", foreground="#8ce6a0")

    # ── persistence ────────────────────────────────────────────────────────────
    def _restore(self):
        saved = self.state.get("fields", {})
        defaults = {"TRAIN_INPUT": "416", "TRAIN_EPOCHS": "200", "TRAIN_BATCH": "96",
                    "TRAIN_WORKERS": "20", "TRAIN_DEVICE": "gpu", "TRAIN_GPU_IDS": "0",
                    "TRAIN_REG_MAX": "7", "TRAIN_LR": "", "TRAIN_DATASET": "", "TRAIN_CLASSES": ""}
        for k, var in self.vars.items():
            var.set(saved.get(k, defaults.get(k, "")))
        self.export_var.set(self.state.get("export", True))
        self.ft_mode.set(self.state.get("ft_mode", "scratch"))
        self.ckpt_var.set(self.state.get("ckpt", ""))
        self._sync_ft()

    def _collect(self):
        return {k: var.get().strip() for k, var in self.vars.items()}

    def _persist(self):
        self.state["fields"] = self._collect()
        self.state["export"] = bool(self.export_var.get())
        self.state["ft_mode"] = self.ft_mode.get()
        self.state["ckpt"] = self.ckpt_var.get().strip()
        save_state(self.state)

    # ── small handlers ──────────────────────────────────────────────────────────
    def _pick_dataset(self):
        d = filedialog.askdirectory(title="Select the YOLO dataset root")
        if d:
            self.vars["TRAIN_DATASET"].set(d)

    def _pick_ckpt(self):
        p = filedialog.askopenfilename(title="Select a checkpoint",
                                       filetypes=[("Checkpoints", "*.ckpt"), ("All files", "*.*")])
        if p:
            self.ckpt_var.set(p)

    def _sync_ft(self):
        on = self.ft_mode.get() != "scratch"
        st = "normal" if on else "disabled"
        self.ckpt_entry.configure(state=st)
        self.ckpt_btn.configure(state=st)

    def _clear_log(self):
        self.log.delete("1.0", "end")

    def _append(self, text, tag=None):
        at_end = self.log.yview()[1] >= 0.999
        self.log.insert("end", text, tag or ())
        if at_end:
            self.log.see("end")

    # ── run / stop ───────────────────────────────────────────────────────────────
    def _validate(self, env):
        ds = env.get("TRAIN_DATASET", "")
        if not ds or not os.path.isdir(ds):
            messagebox.showerror("Dataset", f"Dataset folder not found:\n{ds or '(empty)'}")
            return False
        if not env.get("TRAIN_CLASSES", ""):
            messagebox.showerror("Classes", "Enter at least one class name (comma-separated, id order).")
            return False
        if self.ft_mode.get() != "scratch":
            ckpt = self.ckpt_var.get().strip()
            if not ckpt or not os.path.isfile(ckpt):
                messagebox.showerror("Checkpoint", f"Checkpoint not found:\n{ckpt or '(empty)'}")
                return False
        return True

    def _build_env(self):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"                       # stream the child's prints live
        fields = self._collect()
        for k, v in fields.items():
            if v:
                env[k] = v
            else:
                env.pop(k, None)
        env["TRAIN_EXPORT"] = "1" if self.export_var.get() else "0"
        ckpt = self.ckpt_var.get().strip()
        env.pop("TRAIN_WEIGHTS", None); env.pop("TRAIN_RESUME", None)
        if self.ft_mode.get() == "weights" and ckpt:
            env["TRAIN_WEIGHTS"] = ckpt
        elif self.ft_mode.get() == "resume" and ckpt:
            env["TRAIN_RESUME"] = ckpt
        return env

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        env = self._build_env()
        if not self._validate(env):
            return
        if not os.path.isfile(TRAINER):
            messagebox.showerror("Trainer", f"train_nanodet.py not found next to this GUI:\n{TRAINER}")
            return
        self._persist()
        self._append(f"$ {sys.executable} -u train_nanodet.py   (cwd={HERE})\n", "ok")
        creflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", TRAINER], cwd=HERE, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, errors="replace",
                creationflags=creflags,
                start_new_session=(os.name != "nt"))
        except Exception as e:
            self._append(f"failed to launch: {e}\n", "err")
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.configure(text="running…")

    def _reader(self, proc):
        try:
            for line in proc.stdout:
                self.q.put(line)
        except Exception:
            pass
        finally:
            self.q.put(("__exit__", proc.poll()))

    def _drain(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__exit__":
                    self._on_exit(item[1])
                else:
                    low = item.lower()
                    tag = "err" if ("error" in low or "fail" in low or "traceback" in low) \
                        else "ok" if ("verify ok" in low or "✅" in item or "done" in low) else None
                    self._append(item, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    def _on_exit(self, code):
        self._append(f"\n[trainer exited with code {code}]\n", "ok" if code == 0 else "err")
        self.status.configure(text="done" if code == 0 else f"exited ({code})")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.proc = None

    def stop(self):
        p = self.proc
        if not p or p.poll() is not None:
            return
        self.status.configure(text="stopping…")
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True)
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception as e:
            self._append(f"stop failed: {e}\n", "err")

    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Quit", "Training is still running. Stop it and quit?"):
                return
            self.stop()
        self._persist()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    TrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
