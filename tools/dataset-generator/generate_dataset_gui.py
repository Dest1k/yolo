#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Генератор синтетических датасетов — графическое окно (дружелюбно к Windows, без лишних
зависимостей у самого окна).

Каждая ручка generate_dataset.py во вкладках, с живым логом и полосой прогресса снизу:

  • Запуск     — объект, кол-во картинок, движок (FLUX / свои картинки), фазы, авто-установка,
                 пути (с обзором), Старт/Стоп.
  • LLM        — эндпойнт, ключ, модель, температура, токены, тумблер LM Studio.
  • Картинки   — режим квантования FLUX, батчи, шаги, размер, JPEG, префикс файлов, repo для
                 авто-скачивания.
  • Промпты    — УНИВЕРСАЛЬНЫЙ редактор словаря: объект и любые свои категории вариативности
                 (тип, материал, фон, погода, состояние, ракурс — или что угодно своё), микс
                 масштабов и шаблон системного промпта. Под любой объект, не только дроны.
  • Разметка   — ВЫБОР движка (YOLO-World / Grounding DINO / OWLv2) с пояснениями, классы,
                 conf/iou, id и имена классов.

Окно пишет JSON-конфиг и запускает `generate_dataset.py <config>` дочерним процессом, стримя
его вывод сюда. CLI-путь идентичен — сохранённый конфиг так же гоняется без окна на GPU-машине.
Состояние запоминается между запусками (~/.dataset_generator_gui.json).
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
        "ОШИБКА: tkinter недоступен — а это графический лаунчер.\n"
        f"  ({type(e).__name__}: {e})\n"
        "  Linux:   sudo apt install python3-tk\n"
        "  Или запусти движок напрямую:  python generate_dataset.py [config.json]\n")
    sys.exit(1)

try:
    import generate_dataset as engine
except ModuleNotFoundError:
    msg = ("generate_dataset.py не найден рядом с этим окном.\n\n"
           "Положи ОБА файла в одну папку и запусти оттуда.")
    try:
        messagebox.showerror("Нет generate_dataset.py", msg)
    except Exception:
        pass
    sys.stderr.write("ОШИБКА: " + msg + "\n")
    sys.exit(1)

ENGINE = os.path.join(HERE, "generate_dataset.py")
STATE_PATH = os.path.expanduser("~/.dataset_generator_gui.json")
RUN_CONFIG = os.path.join(HERE, ".gen_run_config.json")

QUANT_CHOICES = ["torchao", "nf4", "layerwise"]
BACKEND_CHOICES = ["flux", "own"]
LABEL_CHOICES = ["yoloworld", "groundingdino", "owlv2"]
LABEL_HELP = (
    "YOLO-World — быстрый, лёгкий, ставится из ultralytics. Хорош для большинства классов (по умолчанию).\n"
    "Grounding DINO — точнее на редких/необычных классах и сложных описаниях, но тяжелее и медленнее.\n"
    "OWLv2 — ещё один open-vocab (Google), иногда лучше ловит мелкие объекты. Любой движок ставится сам."
)


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
        self.widgets = {}
        root.title("Генератор датасетов")
        root.minsize(920, 760)

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

    # ── строители полей ─────────────────────────────────────────────────────────
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

    # ── вкладки ─────────────────────────────────────────────────────────────────
    def _tab_run(self, nb):
        f = ttk.Frame(nb, padding=12); nb.add(f, text="  Запуск  ")
        f.columnconfigure(1, weight=1)
        rows = [
            ("Папка FLUX", "paths.flux_dir", "dir"),
            ("Папка трансформера", "paths.transformer_path", "dir"),
            ("Папка вывода картинок", "paths.images_dir", "dir"),
            ("Папка вывода YOLO", "paths.output_yolo_dir", "dir"),
            ("Файл промптов .jsonl", "paths.prompts_file", "file"),
        ]
        r = 0
        for label, path, kind in rows:
            self._entry(f, r, label, path, width=48)
            ttk.Button(f, text="Обзор…",
                       command=lambda p=path, k=kind: self._browse(p, k)).grid(row=r, column=2, padx=6)
            r += 1
        self._entry(f, r, "Сколько картинок", "total_images", width=12); r += 1
        self._entry(f, r, "Движок генерации", "generation.backend", width=12, choices=BACKEND_CHOICES); r += 1
        ttk.Label(f, text="flux — рендерить картинки FLUX;  own — взять свои готовые из папки картинок",
                  foreground="#888").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        box = ttk.LabelFrame(f, text="Какие фазы запускать", padding=8)
        box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(8, 4)); r += 1
        self._check(box, 0, "1 · Сгенерировать промпты (LLM)", "run.phase1_prompts")
        self._check(box, 1, "2 · Отрендерить картинки (FLUX)", "run.phase2_images")
        self._check(box, 2, "3 · Авторазметка", "run.phase3_label")
        self._check(f, r, "Сам ставить недостающие пакеты и качать модели (pip / HuggingFace)",
                    "auto_install", colspan=3); r += 1

        bar = ttk.Frame(f); bar.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.start_btn = ttk.Button(bar, text="▶  Старт", command=self.start); self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="■  Стоп", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Сохранить конфиг…", command=self.save_config_as).pack(side="left", padx=6)
        ttk.Button(bar, text="Очистить лог", command=lambda: self.log.delete("1.0", "end")).pack(side="left", padx=6)
        self.status = ttk.Label(bar, text="простаивает"); self.status.pack(side="right")

    def _tab_llm(self, nb):
        f = ttk.Frame(nb, padding=12); nb.add(f, text="  LLM  ")
        f.columnconfigure(1, weight=1)
        self._entry(f, 0, "Базовый URL (OpenAI-совместимый)", "llm.base_url", width=40)
        self._entry(f, 1, "API-ключ", "llm.api_key", width=40)
        self._entry(f, 2, "Идентификатор модели", "llm.model", width=40)
        self._entry(f, 3, "Температура", "llm.temperature", width=10)
        self._entry(f, 4, "Макс. токенов", "llm.max_tokens", width=10)
        self._check(f, 5, "Управлять LM Studio CLI (lms load/unload) для (вы)грузки модели", "llm.use_lms")
        ttk.Label(f, text="Подсказка: базовый URL может указывать на любой OpenAI-совместимый сервер "
                          "(LM Studio, llama.cpp, vLLM, сам OpenAI).",
                  foreground="#888").grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _tab_image(self, nb):
        f = ttk.Frame(nb, padding=12); nb.add(f, text="  Картинки (FLUX)  ")
        f.columnconfigure(1, weight=1); f.columnconfigure(3, weight=1)

        def two(row, l1, p1, c1, l2, p2, c2):
            self._entry(f, row, l1, p1, width=14, choices=c1)
            ttk.Label(f, text=l2).grid(row=row, column=2, sticky="w", padx=(16, 6))
            v = tk.StringVar(); self.widgets[p2] = v
            if c2:
                ttk.Combobox(f, textvariable=v, values=c2, width=12).grid(row=row, column=3, sticky="w")
            else:
                ttk.Entry(f, textvariable=v, width=14).grid(row=row, column=3, sticky="w")

        two(0, "Квантование", "generation.quant_mode", QUANT_CHOICES, "Промптов на запрос LLM", "generation.batch_size", None)
        two(1, "Микро-батч (карт/проход)", "generation.micro_batch", None, "Суперчанк", "generation.super_chunk", None)
        two(2, "Батч текст-энкодера", "generation.encode_batch", None, "Шагов инференса", "generation.num_inference_steps", None)
        two(3, "Guidance scale", "generation.guidance_scale", None, "Качество JPEG", "generation.jpeg_quality", None)
        two(4, "Ширина", "generation.width", None, "Высота", "generation.height", None)
        self._entry(f, 5, "HF endpoint", "generation.hf_endpoint", width=40)
        self._entry(f, 6, "Префикс имён файлов", "generation.file_prefix", width=20)
        self._entry(f, 7, "Repo FLUX для авто-скачивания", "generation.flux_repo", width=40)
        self._entry(f, 8, "Суффикс к каждому промпту", "generation.prompt_suffix", width=48)
        self._check(f, 9, "Разрешить TF32 matmul (быстрее на Ampere+)", "generation.allow_tf32", colspan=4)

    def _tab_prompts(self, nb):
        outer = ttk.Frame(nb, padding=6); nb.add(outer, text="  Промпты  ")
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        ttk.Label(inner, text="Объект генерации (что это — любой объект, не только дрон):",
                  foreground="#bbb").pack(anchor="w", pady=(4, 0))
        self.object_noun = tk.StringVar(); self.widgets["prompts.object_noun"] = self.object_noun
        ttk.Entry(inner, textvariable=self.object_noun).pack(fill="x", pady=(0, 8))

        ttk.Label(inner, text="СЛОВАРЬ КАТЕГОРИЙ. Заголовок категории — строкой «## Имя», ниже её варианты "
                              "по одному в строке. Категории любые свои; LLM фокусируется на случайном "
                              "варианте из каждой.", foreground="#888", wraplength=820,
                  justify="left").pack(anchor="w", pady=(2, 4))
        wrap, self.cat_text = self._textbox(inner, height=16); wrap.pack(fill="both", expand=True)

        ttk.Label(inner, text="Микс масштабов объекта  —  по одной в строке:  вес | фраза  "
                              "(веса это пропорции; планировщик держит их точно)",
                  foreground="#888").pack(anchor="w", pady=(10, 0))
        wrap, self.scales_text = self._textbox(inner, height=5); wrap.pack(fill="x")

        ttk.Label(inner, text="Шаблон системного промпта  —  плейсхолдеры: {batch_size} {object_noun} {config_block}",
                  foreground="#888").pack(anchor="w", pady=(10, 0))
        wrap, self.template_text = self._textbox(inner, height=9); wrap.pack(fill="both", expand=True)

    def _tab_labeling(self, nb):
        f = ttk.Frame(nb, padding=12); nb.add(f, text="  Разметка  ")
        f.columnconfigure(1, weight=1)
        self._entry(f, 0, "Движок авторазметки", "labeling.backend", width=16, choices=LABEL_CHOICES)
        ttk.Label(f, text=LABEL_HELP, foreground="#888", justify="left",
                  wraplength=820).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 8))
        self._entry(f, 2, "Веса YOLO-World", "labeling.yoloworld_weights", width=30)
        self._entry(f, 3, "Модель Grounding DINO", "labeling.groundingdino_model", width=36)
        self._entry(f, 4, "Модель OWLv2", "labeling.owlv2_model", width=36)
        self._entry(f, 5, "Порог уверенности (conf)", "labeling.conf", width=10)
        self._entry(f, 6, "IoU (NMS)", "labeling.iou", width=10)
        ttk.Label(f, text="КЛАССЫ (мультикласс). Заголовок класса — строкой «## Имя», ниже его "
                          "синонимы по одному в строке. id класса = порядок (0,1,2…). Разные классы "
                          "НЕ смешиваются — у каждого своя группа синонимов.",
                  foreground="#888", justify="left", wraplength=820)\
            .grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 2))
        wrap, self.cls_text = self._textbox(f, height=12)
        wrap.grid(row=8, column=0, columnspan=2, sticky="ew")
        f.rowconfigure(8, weight=1)

    def _build_log(self):
        frame = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        frame.pack(fill="both", expand=True)
        pb = ttk.Frame(frame); pb.pack(fill="x", pady=(0, 4))
        self.prog = ttk.Progressbar(pb, orient="horizontal", mode="determinate", maximum=1000)
        self.prog.pack(side="left", fill="x", expand=True)
        self.prog_pct = ttk.Label(pb, text="", width=6, anchor="e"); self.prog_pct.pack(side="left", padx=(6, 6))
        self.prog_txt = ttk.Label(frame, text="", foreground="#9fb3c8", anchor="w"); self.prog_txt.pack(fill="x")
        ttk.Label(frame, text="Лог").pack(anchor="w")
        wrap = ttk.Frame(frame); wrap.pack(fill="both", expand=True)
        self.log = tk.Text(wrap, wrap="none", bg="#101418", fg="#d6e2ec",
                           insertbackground="#d6e2ec", font=("Consolas", 10), height=14)
        ys = ttk.Scrollbar(wrap, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=ys.set)
        self.log.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self.log.tag_config("err", foreground="#ff9b9b")
        self.log.tag_config("ok", foreground="#8ce6a0")
        self.log.tag_config("step", foreground="#7fd1ff")

    def _set_progress(self, frac, text):
        try:
            frac = max(0.0, min(1.0, float(frac)))
        except (TypeError, ValueError):
            return
        self.prog["value"] = frac * 1000
        self.prog_pct.configure(text="%d%%" % round(frac * 100))
        if text:
            self.prog_txt.configure(text=text)

    # ── категории <-> текст ───────────────────────────────────────────────────
    @staticmethod
    def _cats_to_text(cats):
        blocks = []
        for name, items in cats.items():
            blocks.append("## " + name + "\n" + "\n".join(items))
        return "\n\n".join(blocks)

    @staticmethod
    def _text_to_cats(text):
        cats, cur = {}, None
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("## "):
                cur = s[3:].strip()
                cats.setdefault(cur, [])
            elif s:
                if cur is None:
                    cur = "Категория"; cats.setdefault(cur, [])
                cats[cur].append(s)
        return {k: v for k, v in cats.items() if v}

    # ── классы разметки (мультикласс) <-> текст ────────────────────────────────
    @staticmethod
    def _classes_to_text(lab):
        """labeling.classes ([{name,synonyms}] или старый плоский список) -> текст блоками."""
        raw = lab.get("classes", [])
        blocks = []
        if raw and isinstance(raw[0], dict):
            for c in raw:
                nm = c.get("name", "object")
                syns = c.get("synonyms") or [nm]
                blocks.append("## " + nm + "\n" + "\n".join(str(s) for s in syns))
        else:                                            # старый формат: один класс
            nm = (lab.get("class_names") or ["object"])[0]
            blocks.append("## " + nm + "\n" + "\n".join(str(s) for s in raw))
        return "\n\n".join(blocks)

    @staticmethod
    def _text_to_classes(text):
        """Текст блоками «## Имя» + синонимы -> [{name, synonyms}] (id = порядок)."""
        classes, cur = [], None
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("## "):
                cur = {"name": s[3:].strip(), "synonyms": []}
                classes.append(cur)
            elif s:
                if cur is None:
                    cur = {"name": "object", "synonyms": []}
                    classes.append(cur)
                cur["synonyms"].append(s)
        # отбрасываем пустые группы; если у класса нет синонимов — берём само имя
        out = []
        for c in classes:
            if not c["synonyms"]:
                c["synonyms"] = [c["name"]]
            if c["name"]:
                out.append(c)
        return out

    # ── заполнение/чтение ─────────────────────────────────────────────────────
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
        self.cat_text.delete("1.0", "end")
        self.cat_text.insert("1.0", self._cats_to_text(self.cfg["prompts"].get("categories", {})))
        self.scales_text.delete("1.0", "end")
        self.scales_text.insert("1.0", "\n".join(
            f"{w} | {phrase}" for phrase, w in self.cfg["prompts"]["object_scales"]))
        self.template_text.delete("1.0", "end")
        self.template_text.insert("1.0", self.cfg["prompts"]["system_template"])
        self.cls_text.delete("1.0", "end")
        self.cls_text.insert("1.0", self._classes_to_text(self.cfg.get("labeling", {})))

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
                dset(cfg, path, bool(var.get())); continue
            raw = var.get().strip()
            try:
                if path in int_paths:
                    dset(cfg, path, int(float(raw)))
                elif path in float_paths:
                    dset(cfg, path, float(raw))
                else:
                    dset(cfg, path, raw)
            except ValueError:
                raise ValueError(f"'{path}' ожидает число, а получено {raw!r}")
        cats = self._text_to_cats(self.cat_text.get("1.0", "end"))
        if not cats:
            raise ValueError("Словарь категорий пуст — добавь хотя бы одну «## Категория» с вариантами.")
        cfg["prompts"]["categories"] = cats
        scales = []
        for ln in self._lines(self.scales_text):
            w, phrase = (ln.split("|", 1) if "|" in ln else ("1", ln))
            try:
                weight = float(w.strip())
            except ValueError:
                raise ValueError(f"вес масштаба не число в строке: {ln!r}")
            if phrase.strip():
                scales.append([phrase.strip(), weight])
        if not scales:
            raise ValueError("Микс масштабов пуст — добавь хотя бы одну строку «вес | фраза».")
        cfg["prompts"]["object_scales"] = scales
        cfg["prompts"]["system_template"] = self.template_text.get("1.0", "end").strip("\n")
        classes = self._text_to_classes(self.cls_text.get("1.0", "end"))
        if not classes:
            raise ValueError("Классы пусты — добавь хотя бы один «## Имя» с синонимами на вкладке «Разметка».")
        cfg["labeling"]["classes"] = classes
        cfg["labeling"]["class_names"] = [c["name"] for c in classes]   # для совместимости
        return cfg

    # ── обзор / сохранение ────────────────────────────────────────────────────
    def _browse(self, path, kind):
        var = self.widgets[path]
        cur = var.get().strip()
        if kind == "dir":
            p = filedialog.askdirectory(title="Выбери папку",
                                        initialdir=cur if os.path.isdir(cur) else None)
        else:
            p = filedialog.asksaveasfilename(title="Файл промптов .jsonl",
                                             defaultextension=".jsonl",
                                             initialfile=os.path.basename(cur) or "prompts.jsonl")
        if p:
            var.set(p)

    def save_config_as(self):
        try:
            cfg = self._collect()
        except ValueError as e:
            messagebox.showerror("Неверное значение", str(e)); return
        p = filedialog.asksaveasfilename(title="Сохранить конфиг", defaultextension=".json",
                                         initialfile="dataset_config.json")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            self._append(f"конфиг сохранён → {p}\n", "ok")

    def _persist(self, cfg):
        self.cfg = cfg
        try:
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=1, ensure_ascii=False)
        except Exception:
            pass

    def _append(self, text, tag=None):
        at_end = self.log.yview()[1] >= 0.999
        self.log.insert("end", text, tag or ())
        if at_end:
            self.log.see("end")

    # ── запуск / стоп ──────────────────────────────────────────────────────────
    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        try:
            cfg = self._collect()
        except ValueError as e:
            messagebox.showerror("Неверное значение", str(e)); return
        if not any(cfg["run"].values()):
            messagebox.showwarning("Нечего делать", "Включи хотя бы одну фазу на вкладке «Запуск».")
            return
        self._persist(cfg)
        try:
            with open(RUN_CONFIG, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=1, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Конфиг", f"Не смог записать конфиг запуска:\n{e}"); return
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONUTF8"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        self._append(f"$ {sys.executable} -u generate_dataset.py {RUN_CONFIG}\n", "ok")
        self._set_progress(0, "")
        creflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", ENGINE, RUN_CONFIG], cwd=HERE, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, encoding="utf-8", errors="replace",
                creationflags=creflags, start_new_session=(os.name != "nt"))
        except Exception as e:
            self._append(f"не удалось запустить: {e}\n", "err"); return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.configure(text="выполняется…")

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
                    self._on_exit(item[1]); continue
                if item.startswith("@@PB@@\t"):
                    parts = item.rstrip("\n").split("\t")
                    if len(parts) >= 3:
                        self._set_progress(parts[1], parts[2])
                    continue
                low = item.lower()
                tag = "err" if ("ошибк" in low or "error" in low or "[!]" in item or "traceback" in low) \
                    else "ok" if ("готово" in low or "success" in low or "[успех" in low) \
                    else "step" if item.lstrip().startswith(("[", "===", "##")) else None
                self._append(item, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    def _on_exit(self, code):
        self._append(f"\n[движок завершился с кодом {code}]\n", "ok" if code == 0 else "err")
        self.status.configure(text="готово" if code == 0 else f"вышел ({code})")
        if code == 0:
            self._set_progress(1.0, "готово")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.proc = None

    def stop(self):
        p = self.proc
        if not p or p.poll() is not None:
            return
        self.status.configure(text="останавливаю…")
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception as e:
            self._append(f"не смог остановить: {e}\n", "err")

    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Выход", "Генерация ещё идёт. Остановить и выйти?"):
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
