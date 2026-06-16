#!/usr/bin/env python3
r"""
Synthetic dataset generator — graphical front-end (Windows-friendly, zero extra deps).

Every knob of generate_dataset.py in one tabbed window, with a live log underneath:

  • Run       paths (browse), target image count, which phases to run, Start/Stop
  • LLM       endpoint, api key, model id, temperature, max tokens, LM Studio toggle
  • Image     FLUX quant mode, batch/micro-batch/super-chunk, steps, guidance, size, JPEG q
  • Prompts   FULL vocabulary editor — drone types, materials, backgrounds, weather,
              states, camera angles, the scale mix (weight + phrase), and the raw
              system-prompt template. This is where the "maximum prompt customization"
              lives: edit any list, one item per line, and it feeds the LLM verbatim.
  • Labeling  YOLO-World weights, class synonyms, conf/iou, output class id + name

It writes a JSON config and runs `generate_dataset.py <config>` as a child process,
streaming its output here. The plain CLI path stays identical, so a config saved from
this window also runs head-less on the GPU box. State is remembered between sessions
(~/.dataset_generator_gui.json).

Only the standard library + the sibling generate_dataset.py are needed to open this
window; torch/diffusers/openai/ultralytics are imported by the engine when it runs.
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception as e:
    sys.stderr.write(
        "ERROR: tkinter is not available — this is the GUI launcher.\n"
        f"  ({type(e).__name__}: {e})\n"
        "  Linux:   sudo apt install python3-tk\n"
        "  Or run the engine directly:  python generate_dataset.py [config.json]\n")
    sys.exit(1)

try:
    import generate_dataset as engine         # sibling; only stdlib at import time
except ModuleNotFoundError:
    msg = ("generate_dataset.py не найден рядом с этим окном.\n\n"
           "Положи ОБА файла в одну папку и запусти оттуда:\n"
           "    generate_dataset.py\n"
           "    generate_dataset_gui.py\n\n"
           "Они всегда должны лежать вместе — окно само запускает движок.\n\n"
           f"Сейчас окно запущено из:\n    {HERE}")
    try:
        messagebox.showerror("Нет generate_dataset.py", msg)
    except Exception:
        pass
    sys.stderr.write("ERROR: " + msg.replace("\n", "\n  ") + "\n")
    sys.exit(1)

ENGINE = os.path.join(HERE, "generate_dataset.py")
STATE_PATH = os.path.expanduser("~/.dataset_generator_gui.json")
RUN_CONFIG = os.path.join(HERE, ".gen_run_config.json")     # what we hand the child

QUANT_CHOICES = ["torchao", "nf4", "layerwise"]

# Plain (label, config-path, kind) fields. config-path is a dotted key into the cfg.
POOL_KEYS = [
    ("Drone types", "prompts.drone_types"),
    ("Body materials / textures", "prompts.drone_materials"),
    ("Backgrounds / environments", "prompts.backgrounds"),
    ("Weather & lighting", "prompts.conditions"),
    ("Physical states", "prompts.states"),
    ("Camera perspectives", "prompts.perspectives"),
]


# ── dotted-key helpers on a nested dict ───────────────────────────────────────
def dget(cfg, path):
    cur = cfg
    for k in path.split("."):
        cur = cur[k]
    return cur


def dset(cfg, path, value):
    keys = path.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return engine.deep_merge(engine.DEFAULTS, json.load(f))
    except Exception:
        import copy
        return copy.deepcopy(engine.DEFAULTS)


class GeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.proc = None
        self.q = queue.Queue()
        self.cfg = load_state()
        self.widgets = {}          # config-path -> tk var / Text
        root.title("Synthetic Dataset Generator")
        root.minsize(900, 720)

        nb = ttk.Notebook(root)
        nb.pack(fill="x", side="top", padx=8, pady=(8, 4))
        self._tab_run(nb)
        self._tab_llm(nb)
        self._tab_image(nb)
        self._tab_prompts(nb)
        self._tab_labeling(nb)
        self._build_log()
        self._restore()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(80, self._drain)

    # ── generic field builders ───────────────────────────────────────────────
    def _entry(self, parent, row, label, path, width=20, choices=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 6))
        v = tk.StringVar()
        self.widgets[path] = v
        if choices:
            ttk.Combobox(parent, textvariable=v, values=choices, width=width - 2).grid(
                row=row, column=1, sticky="w")
        else:
            ttk.Entry(parent, textvariable=v, width=width).grid(row=row, column=1, sticky="ew")
        parent.columnconfigure(1, weight=1)
        return v

    def _check(self, parent, row, label, path, colspan=2):
        v = tk.BooleanVar()
        self.widgets[path] = v
        ttk.Checkbutton(parent, text=label, variable=v).grid(
            row=row, column=0, columnspan=colspan, sticky="w", pady=2)
        return v

    def _textbox(self, parent, height=6):
        wrap = ttk.Frame(parent)
        t = tk.Text(wrap, height=height, wrap="word", bg="#101418", fg="#d6e2ec",
                    insertbackground="#d6e2ec", font=("Consolas", 10), undo=True)
        ys = ttk.Scrollbar(wrap, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=ys.set)
        t.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        return wrap, t

    # ── tabs ──────────────────────────────────────────────────────────────────
    def _tab_run(self, nb):
        f = ttk.Frame(nb, padding=12); nb.add(f, text="Run")
        f.columnconfigure(1, weight=1)
        rows = [
            ("FLUX model dir", "paths.flux_dir", "dir"),
            ("Transformer dir", "paths.transformer_path", "dir"),
            ("Images output dir", "paths.images_dir", "dir"),
            ("YOLO output dir", "paths.output_yolo_dir", "dir"),
            ("Prompts .jsonl file", "paths.prompts_file", "file"),
        ]
        r = 0
        for label, path, kind in rows:
            self._entry(f, r, label, path, width=48)
            ttk.Button(f, text="Browse…",
                       command=lambda p=path, k=kind: self._browse(p, k)).grid(row=r, column=2, padx=6)
            r += 1
        self._entry(f, r, "Target image count", "total_images", width=12); r += 1

        box = ttk.LabelFrame(f, text="Phases to run", padding=8)
        box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(8, 4)); r += 1
        self._check(box, 0, "1 · Generate prompts (LLM)", "run.phase1_prompts")
        self._check(box, 1, "2 · Render images (FLUX)", "run.phase2_images")
        self._check(box, 2, "3 · Auto-label (YOLO-World)", "run.phase3_label")

        bar = ttk.Frame(f); bar.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.start_btn = ttk.Button(bar, text="▶  Start", command=self.start); self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="■  Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Save config…", command=self.save_config_as).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear log", command=lambda: self.log.delete("1.0", "end")).pack(side="left", padx=6)
        self.status = ttk.Label(bar, text="idle"); self.status.pack(side="right")

    def _tab_llm(self, nb):
        f = ttk.Frame(nb, padding=12); nb.add(f, text="LLM")
        f.columnconfigure(1, weight=1)
        self._entry(f, 0, "OpenAI-compatible base URL", "llm.base_url", width=40)
        self._entry(f, 1, "API key", "llm.api_key", width=40)
        self._entry(f, 2, "Model identifier", "llm.model", width=40)
        self._entry(f, 3, "Temperature", "llm.temperature", width=10)
        self._entry(f, 4, "Max tokens", "llm.max_tokens", width=10)
        self._check(f, 5, "Drive LM Studio CLI (lms load/unload) to (un)load the model", "llm.use_lms")
        ttk.Label(f, text="Tip: point base URL at any OpenAI-compatible server "
                          "(LM Studio, llama.cpp, vLLM, OpenAI itself).",
                  foreground="#888").grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _tab_image(self, nb):
        f = ttk.Frame(nb, padding=12); nb.add(f, text="Image (FLUX)")
        f.columnconfigure(1, weight=1); f.columnconfigure(3, weight=1)

        def two(row, l1, p1, c1, l2, p2, c2):
            self._entry(f, row, l1, p1, width=14, choices=c1)
            ttk.Label(f, text=l2).grid(row=row, column=2, sticky="w", padx=(16, 6))
            v = tk.StringVar(); self.widgets[p2] = v
            if c2:
                ttk.Combobox(f, textvariable=v, values=c2, width=12).grid(row=row, column=3, sticky="w")
            else:
                ttk.Entry(f, textvariable=v, width=14).grid(row=row, column=3, sticky="w")

        two(0, "Quant mode", "generation.quant_mode", QUANT_CHOICES, "Prompts per LLM call", "generation.batch_size", None)
        two(1, "Micro-batch (imgs/fwd)", "generation.micro_batch", None, "Super-chunk", "generation.super_chunk", None)
        two(2, "Text-encode batch", "generation.encode_batch", None, "Inference steps", "generation.num_inference_steps", None)
        two(3, "Guidance scale", "generation.guidance_scale", None, "JPEG quality", "generation.jpeg_quality", None)
        two(4, "Width", "generation.width", None, "Height", "generation.height", None)
        self._entry(f, 5, "HF endpoint", "generation.hf_endpoint", width=40)
        self._entry(f, 6, "Prompt suffix (appended to every prompt)", "generation.prompt_suffix", width=48)
        self._check(f, 7, "Allow TF32 matmul (faster on Ampere+)", "generation.allow_tf32", colspan=4)

    def _tab_prompts(self, nb):
        outer = ttk.Frame(nb, padding=6); nb.add(outer, text="Prompts")
        # scrollable canvas so all the pools fit
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        ttk.Label(inner, text="One item per line. These feed the LLM's per-batch focus verbatim — "
                              "add/remove/rewrite freely.", foreground="#888").pack(anchor="w", pady=(4, 6))
        self.pool_texts = {}
        for label, path in POOL_KEYS:
            ttk.Label(inner, text=label).pack(anchor="w", pady=(6, 0))
            wrap, t = self._textbox(inner, height=5)
            wrap.pack(fill="x", expand=False)
            self.pool_texts[path] = t

        ttk.Label(inner, text="Object-scale mix  —  one per line:  weight | phrase  "
                              "(weights are proportions; the planner holds them exactly)",
                  foreground="#888").pack(anchor="w", pady=(10, 0))
        wrap, self.scales_text = self._textbox(inner, height=5); wrap.pack(fill="x")

        ttk.Label(inner, text="System-prompt template  —  placeholders: {batch_size} {drone_type} "
                              "{material} {background} {condition} {state} {perspective}",
                  foreground="#888").pack(anchor="w", pady=(10, 0))
        wrap, self.template_text = self._textbox(inner, height=9); wrap.pack(fill="both", expand=True)

    def _tab_labeling(self, nb):
        f = ttk.Frame(nb, padding=12); nb.add(f, text="Labeling")
        f.columnconfigure(1, weight=1)
        self._entry(f, 0, "YOLO-World weights", "labeling.weights", width=30)
        self._entry(f, 1, "Confidence threshold", "labeling.conf", width=10)
        self._entry(f, 2, "IoU (NMS)", "labeling.iou", width=10)
        self._entry(f, 3, "Output class id", "labeling.class_index", width=10)
        self._entry(f, 4, "Output class name", "labeling.class_name", width=20)
        ttk.Label(f, text="Class synonyms (one per line — all map to the single output class):")\
            .grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 2))
        wrap, self.classes_text = self._textbox(f, height=7)
        wrap.grid(row=6, column=0, columnspan=2, sticky="ew")

    def _build_log(self):
        frame = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Log").pack(anchor="w")
        wrap = ttk.Frame(frame); wrap.pack(fill="both", expand=True)
        self.log = tk.Text(wrap, wrap="none", bg="#101418", fg="#d6e2ec",
                           insertbackground="#d6e2ec", font=("Consolas", 10), height=14)
        ys = ttk.Scrollbar(wrap, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=ys.set)
        self.log.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self.log.tag_config("err", foreground="#ff9b9b")
        self.log.tag_config("ok", foreground="#8ce6a0")

    # ── populate widgets from cfg / read widgets into cfg ─────────────────────
    def _restore(self):
        for path, var in self.widgets.items():
            try:
                val = dget(self.cfg, path)
            except KeyError:
                continue
            if isinstance(var, tk.BooleanVar):
                var.set(bool(val))
            else:
                var.set("" if val is None else str(val))
        for path, t in self.pool_texts.items():
            t.delete("1.0", "end")
            t.insert("1.0", "\n".join(dget(self.cfg, path)))
        self.scales_text.delete("1.0", "end")
        self.scales_text.insert("1.0", "\n".join(
            f"{w} | {phrase}" for phrase, w in self.cfg["prompts"]["object_scales"]))
        self.template_text.delete("1.0", "end")
        self.template_text.insert("1.0", self.cfg["prompts"]["system_template"])
        self.classes_text.delete("1.0", "end")
        self.classes_text.insert("1.0", "\n".join(self.cfg["labeling"]["classes"]))

    @staticmethod
    def _lines(text_widget):
        return [ln.strip() for ln in text_widget.get("1.0", "end").splitlines() if ln.strip()]

    def _collect(self):
        import copy
        cfg = copy.deepcopy(self.cfg)
        int_paths = {"total_images", "generation.batch_size", "generation.micro_batch",
                     "generation.super_chunk", "generation.encode_batch",
                     "generation.num_inference_steps", "generation.jpeg_quality",
                     "generation.width", "generation.height", "llm.max_tokens",
                     "labeling.class_index"}
        float_paths = {"generation.guidance_scale", "llm.temperature",
                       "labeling.conf", "labeling.iou"}
        for path, var in self.widgets.items():
            if isinstance(var, tk.BooleanVar):
                dset(cfg, path, bool(var.get()))
                continue
            raw = var.get().strip()
            try:
                if path in int_paths:
                    dset(cfg, path, int(float(raw)))
                elif path in float_paths:
                    dset(cfg, path, float(raw))
                else:
                    dset(cfg, path, raw)
            except ValueError:
                raise ValueError(f"'{path}' expects a number, got {raw!r}")
        for path, t in self.pool_texts.items():
            dset(cfg, path, self._lines(t))
        # scales: "weight | phrase"
        scales = []
        for ln in self._lines(self.scales_text):
            if "|" in ln:
                w, phrase = ln.split("|", 1)
            else:
                w, phrase = "1", ln
            try:
                weight = float(w.strip())
            except ValueError:
                raise ValueError(f"scale weight not a number in line: {ln!r}")
            phrase = phrase.strip()
            if phrase:
                scales.append([phrase, weight])
        if not scales:
            raise ValueError("Object-scale mix is empty — add at least one 'weight | phrase' line.")
        cfg["prompts"]["object_scales"] = scales
        cfg["prompts"]["system_template"] = self.template_text.get("1.0", "end").strip("\n")
        cfg["labeling"]["classes"] = self._lines(self.classes_text)
        return cfg

    # ── browse / save ─────────────────────────────────────────────────────────
    def _browse(self, path, kind):
        var = self.widgets[path]
        cur = var.get().strip()
        if kind == "dir":
            p = filedialog.askdirectory(title="Select folder",
                                        initialdir=cur if os.path.isdir(cur) else None)
        else:
            p = filedialog.asksaveasfilename(title="Select / name the prompts .jsonl",
                                             defaultextension=".jsonl",
                                             initialfile=os.path.basename(cur) or "prompts.jsonl")
        if p:
            var.set(p)

    def save_config_as(self):
        try:
            cfg = self._collect()
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e)); return
        p = filedialog.asksaveasfilename(title="Save config", defaultextension=".json",
                                         initialfile="dataset_config.json")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            self._append(f"saved config → {p}\n", "ok")

    def _persist(self, cfg):
        self.cfg = cfg
        try:
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=1, ensure_ascii=False)
        except Exception:
            pass

    # ── log streaming ────────────────────────────────────────────────────────
    def _append(self, text, tag=None):
        at_end = self.log.yview()[1] >= 0.999
        self.log.insert("end", text, tag or ())
        if at_end:
            self.log.see("end")

    # ── run / stop ────────────────────────────────────────────────────────────
    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        try:
            cfg = self._collect()
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e)); return
        if not any(cfg["run"].values()):
            messagebox.showwarning("Nothing to do", "Enable at least one phase on the Run tab.")
            return
        self._persist(cfg)
        try:
            with open(RUN_CONFIG, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=1, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Config", f"Could not write run config:\n{e}"); return
        env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
        self._append(f"$ {sys.executable} -u generate_dataset.py {RUN_CONFIG}\n", "ok")
        creflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", ENGINE, RUN_CONFIG], cwd=HERE, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, errors="replace",
                creationflags=creflags, start_new_session=(os.name != "nt"))
        except Exception as e:
            self._append(f"failed to launch: {e}\n", "err"); return
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
                    tag = "err" if ("error" in low or "[!]" in item or "traceback" in low) \
                        else "ok" if ("success" in low or "done" in low or "✅" in item) else None
                    self._append(item, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    def _on_exit(self, code):
        self._append(f"\n[engine exited with code {code}]\n", "ok" if code == 0 else "err")
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
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception as e:
            self._append(f"stop failed: {e}\n", "err")

    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Quit", "Generation is still running. Stop it and quit?"):
                return
            self.stop()
        try:
            self._persist(self._collect())
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    GeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
