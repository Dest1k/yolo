#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Тренер NanoDet-Plus — графическое окно (дружелюбно к Windows, без лишних зависимостей).

Две вкладки:
  • «Обучение на своих данных» — каждая ручка из train_nanodet.py: датасет, классы,
    тяжесть модели, входное разрешение, эпохи, батч, воркеры, устройство, learning rate
    и дообучение / возобновление.
  • «Готовые модели» — собрать официальную COCO-модель (80 классов) в ОДИН клик: выбери
    вариант (вес × разрешение) и нажми «Создать» — скрипт всё скачает и сконвертирует.

Снизу — общий лог, куда живьём льётся всё, что печатают скрипты (конвертация данных ->
обучение -> экспорт ncnn -> ПРОВЕРКА OK).

    python train_nanodet_gui.py

Окно ничего не переизобретает: для обучения оно выставляет переменные TRAIN_*, которые
понимает train_nanodet.py, а для готовых моделей зовёт get_model.py — и запускает их
дочерним процессом. Значения полей запоминаются между запусками (~/.nanodet_trainer_gui.json).

Используется только стандартная библиотека (tkinter идёт с python на Windows/macOS, на
Linux — `sudo apt install python3-tk`). torch/nanodet нужны запускаемым скриптам, не этому окну.
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
except Exception as e:                                       # машина без графики / без Tk
    sys.stderr.write(
        "ОШИБКА: tkinter недоступен — а это графический лаунчер.\n"
        f"  ({type(e).__name__}: {e})\n"
        "  Linux:   sudo apt install python3-tk\n"
        "  Или запусти тренер без окна:  python train_nanodet.py\n"
        "  Или собери готовую модель:    python get_model.py\n")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "train_nanodet.py")
GETMODEL = os.path.join(HERE, "get_model.py")
STATE_PATH = os.path.expanduser("~/.nanodet_trainer_gui.json")

INPUT_CHOICES = ["256", "320", "352", "416", "512"]
DEVICE_CHOICES = ["gpu", "cpu"]
SIZE_CHOICES = ["1.0x", "1.5x"]

# Готовые COCO-варианты (зеркало каталога из get_model.py) — для вкладки «Готовые модели».
READY_VARIANTS = [
    ("m-320",      "вес 1.0x · вход 320 — самая БЫСТРАЯ"),
    ("m-416",      "вес 1.0x · вход 416 — БАЛАНС (рекомендую)"),
    ("m-1.5x-320", "вес 1.5x · вход 320 — точнее, чуть медленнее"),
    ("m-1.5x-416", "вес 1.5x · вход 416 — самая ТОЧНАЯ"),
]


def load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, ensure_ascii=False)
    except Exception:
        pass


class TrainerGUI:
    def __init__(self, root):
        self.root = root
        self.proc = None
        self.q = queue.Queue()
        self.state = load_state()
        root.title("Тренер NanoDet-Plus")
        root.minsize(800, 680)

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="x", side="top")
        self.tab_train = ttk.Frame(self.nb)
        self.tab_ready = ttk.Frame(self.nb)
        self.nb.add(self.tab_train, text="  Обучение на своих данных  ")
        self.nb.add(self.tab_ready, text="  Готовые модели (1 клик)  ")

        self._build_train_form(self.tab_train)
        self._build_ready_form(self.tab_ready)
        self._build_log()
        self._restore()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(80, self._drain)

    # ── вкладка обучения ────────────────────────────────────────────────────────
    def _build_train_form(self, parent):
        f = ttk.Frame(parent, padding=10)
        f.pack(fill="x", side="top")
        for c in range(4):
            f.columnconfigure(c, weight=(1 if c in (1, 3) else 0))
        self.vars = {}
        r = 0

        # Датасет + кнопка обзора (на всю строку)
        ttk.Label(f, text="Папка датасета (корень YOLO)").grid(row=r, column=0, sticky="w", pady=3)
        self.vars["TRAIN_DATASET"] = v = tk.StringVar()
        ttk.Entry(f, textvariable=v).grid(row=r, column=1, columnspan=2, sticky="ew", padx=6)
        ttk.Button(f, text="Обзор…", command=self._pick_dataset).grid(row=r, column=3, sticky="ew")
        r += 1

        # Классы (на всю строку)
        ttk.Label(f, text="Классы (в порядке id, через запятую)").grid(row=r, column=0, sticky="w", pady=3)
        self.vars["TRAIN_CLASSES"] = v = tk.StringVar()
        ttk.Entry(f, textvariable=v).grid(row=r, column=1, columnspan=3, sticky="ew", padx=6)
        r += 1

        # Парные числовые / выпадающие поля
        def pair(label_l, key_l, kind_l, label_r, key_r, kind_r):
            nonlocal r
            self._field(f, r, 0, label_l, key_l, kind_l)
            self._field(f, r, 2, label_r, key_r, kind_r)
            r += 1

        pair("Тяжесть модели", "TRAIN_MODEL_SIZE", SIZE_CHOICES,
             "Входное разрешение", "TRAIN_INPUT", INPUT_CHOICES)
        pair("Эпохи", "TRAIN_EPOCHS", "int", "Размер батча", "TRAIN_BATCH", "int")
        pair("Воркеры загрузчика", "TRAIN_WORKERS", "int", "Устройство", "TRAIN_DEVICE", DEVICE_CHOICES)
        pair("ID видеокарт (через запятую)", "TRAIN_GPU_IDS", "str",
             "reg_max (бинов DFL−1)", "TRAIN_REG_MAX", "int")
        pair("Learning rate (пусто = из конфига)", "TRAIN_LR", "str", "", "_pad", "str")

        # Чекбокс экспорта
        self.export_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="После обучения авто-экспортировать проверенную NCNN-модель",
                        variable=self.export_var).grid(row=r, column=0, columnspan=4, sticky="w", pady=(6, 2))
        r += 1

        # ── Дообучение / возобновление ──
        lf = ttk.LabelFrame(f, text="Дообучение / возобновление", padding=8)
        lf.grid(row=r, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        lf.columnconfigure(1, weight=1)
        self.ft_mode = tk.StringVar(value="scratch")
        modes = ttk.Frame(lf); modes.grid(row=0, column=0, columnspan=3, sticky="w")
        for text, val in (("Обучать с нуля", "scratch"),
                          ("Дообучить от весов", "weights"),
                          ("Продолжить прерванный запуск", "resume")):
            ttk.Radiobutton(modes, text=text, value=val, variable=self.ft_mode,
                            command=self._sync_ft).pack(side="left", padx=(0, 12))
        ttk.Label(lf, text="Чекпойнт (.ckpt)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ckpt_var = tk.StringVar()
        self.ckpt_entry = ttk.Entry(lf, textvariable=self.ckpt_var)
        self.ckpt_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        self.ckpt_btn = ttk.Button(lf, text="Обзор…", command=self._pick_ckpt)
        self.ckpt_btn.grid(row=1, column=2, sticky="ew", pady=(6, 0))
        r += 1

        # Кнопки действий
        bar = ttk.Frame(f); bar.grid(row=r, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        self.start_btn = ttk.Button(bar, text="▶  Начать обучение", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="■  Стоп", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Очистить лог", command=self._clear_log).pack(side="left", padx=6)
        self.status = ttk.Label(bar, text="простаивает"); self.status.pack(side="right")

    def _field(self, parent, row, col, label, key, kind):
        if key == "_pad":                               # пустышка-заполнитель колонки
            return
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", pady=3)
        v = tk.StringVar()
        self.vars[key] = v
        if isinstance(kind, list):
            ttk.Combobox(parent, textvariable=v, values=kind, width=14).grid(
                row=row, column=col + 1, sticky="ew", padx=6)
        else:
            ttk.Entry(parent, textvariable=v, width=16).grid(
                row=row, column=col + 1, sticky="ew", padx=6)

    # ── вкладка готовых моделей ──────────────────────────────────────────────────
    def _build_ready_form(self, parent):
        f = ttk.Frame(parent, padding=12)
        f.pack(fill="x", side="top")
        f.columnconfigure(0, weight=1)
        ttk.Label(f, text="Готовая модель на 80 классах COCO — собирается в один клик:\n"
                          "скачиваем официальный чекпойнт и конвертируем в проверенную "
                          "fp16-модель для сайдкара.", justify="left").grid(
            row=0, column=0, sticky="w", pady=(0, 8))

        self.ready_var = tk.StringVar(value="m-416")
        box = ttk.LabelFrame(f, text="Выбери вариант (тяжесть × разрешение)", padding=8)
        box.grid(row=1, column=0, sticky="ew")
        for i, (ключ, описание) in enumerate(READY_VARIANTS):
            ttk.Radiobutton(box, text=f"{ключ:<12}  —  {описание}",
                            value=ключ, variable=self.ready_var).grid(
                row=i, column=0, sticky="w", pady=2)

        bar = ttk.Frame(f); bar.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.ready_btn = ttk.Button(bar, text="⬇  Создать модель", command=self.build_ready)
        self.ready_btn.pack(side="left")
        self.ready_all_btn = ttk.Button(bar, text="Создать все четыре", command=self.build_ready_all)
        self.ready_all_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Очистить лог", command=self._clear_log).pack(side="left", padx=6)
        ttk.Label(f, text="Первый запуск дольше: качаются репозиторий и чекпойнт. "
                          "Прогресс — в логе ниже.", justify="left").grid(
            row=3, column=0, sticky="w", pady=(10, 0))

    def _build_log(self):
        frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Лог").pack(anchor="w")
        wrap = ttk.Frame(frame); wrap.pack(fill="both", expand=True)
        self.log = tk.Text(wrap, wrap="none", bg="#101418", fg="#d6e2ec",
                           insertbackground="#d6e2ec", font=("Consolas", 10), height=16)
        ys = ttk.Scrollbar(wrap, orient="vertical", command=self.log.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.log.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        xs.pack(fill="x")
        self.log.tag_config("err", foreground="#ff9b9b")
        self.log.tag_config("ok", foreground="#8ce6a0")

    # ── сохранение состояния ─────────────────────────────────────────────────────
    def _restore(self):
        saved = self.state.get("fields", {})
        # Дефолты под мощную машину: RTX 5090 32GB + Ultra 9 285K + 128GB RAM, Windows.
        defaults = {"TRAIN_MODEL_SIZE": "1.0x", "TRAIN_INPUT": "416", "TRAIN_EPOCHS": "200",
                    "TRAIN_BATCH": "96", "TRAIN_WORKERS": "20", "TRAIN_DEVICE": "gpu",
                    "TRAIN_GPU_IDS": "0", "TRAIN_REG_MAX": "7", "TRAIN_LR": "",
                    "TRAIN_DATASET": "", "TRAIN_CLASSES": ""}
        for k, var in self.vars.items():
            var.set(saved.get(k, defaults.get(k, "")))
        self.export_var.set(self.state.get("export", True))
        self.ft_mode.set(self.state.get("ft_mode", "scratch"))
        self.ckpt_var.set(self.state.get("ckpt", ""))
        self.ready_var.set(self.state.get("ready", "m-416"))
        self._sync_ft()

    def _collect(self):
        return {k: var.get().strip() for k, var in self.vars.items()}

    def _persist(self):
        self.state["fields"] = self._collect()
        self.state["export"] = bool(self.export_var.get())
        self.state["ft_mode"] = self.ft_mode.get()
        self.state["ckpt"] = self.ckpt_var.get().strip()
        self.state["ready"] = self.ready_var.get()
        save_state(self.state)

    # ── мелкие обработчики ───────────────────────────────────────────────────────
    def _pick_dataset(self):
        d = filedialog.askdirectory(title="Выбери корень YOLO-датасета")
        if d:
            self.vars["TRAIN_DATASET"].set(d)

    def _pick_ckpt(self):
        p = filedialog.askopenfilename(title="Выбери чекпойнт",
                                       filetypes=[("Чекпойнты", "*.ckpt"), ("Все файлы", "*.*")])
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

    # ── запуск / стоп ────────────────────────────────────────────────────────────
    def _validate(self, env):
        ds = env.get("TRAIN_DATASET", "")
        if not ds or not os.path.isdir(ds):
            messagebox.showerror("Датасет", f"Папка датасета не найдена:\n{ds or '(пусто)'}")
            return False
        if not env.get("TRAIN_CLASSES", ""):
            messagebox.showerror("Классы", "Укажи хотя бы один класс (через запятую, в порядке id).")
            return False
        if self.ft_mode.get() != "scratch":
            ckpt = self.ckpt_var.get().strip()
            if not ckpt or not os.path.isfile(ckpt):
                messagebox.showerror("Чекпойнт", f"Чекпойнт не найден:\n{ckpt or '(пусто)'}")
                return False
        return True

    def _build_env(self):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"                       # лить вывод дочернего процесса живьём
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

    def _busy(self):
        return self.proc and self.proc.poll() is None

    def _launch(self, cmd, env, заголовок):
        """Общий запуск дочернего процесса с потоковым логом."""
        if self._busy():
            messagebox.showinfo("Занято", "Процесс уже выполняется — дождись завершения или нажми «Стоп».")
            return
        self._persist()
        self._append(f"$ {' '.join(заголовок)}   (cwd={HERE})\n", "ok")
        creflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=HERE, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, errors="replace",
                creationflags=creflags,
                start_new_session=(os.name != "nt"))
        except Exception as e:
            self._append(f"не удалось запустить: {e}\n", "err")
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self._set_running(True)

    def _set_running(self, on):
        st_run = "disabled" if on else "normal"
        for b in (self.start_btn, self.ready_btn, self.ready_all_btn):
            b.configure(state=st_run)
        self.stop_btn.configure(state="normal" if on else "disabled")
        self.status.configure(text="выполняется…" if on else self.status.cget("text"))

    def start(self):
        if self._busy():
            return
        env = self._build_env()
        if not self._validate(env):
            return
        if not os.path.isfile(TRAINER):
            messagebox.showerror("Тренер", f"train_nanodet.py не найден рядом с этим окном:\n{TRAINER}")
            return
        self._launch([sys.executable, "-u", TRAINER], env,
                     [os.path.basename(sys.executable), "-u", "train_nanodet.py"])

    def build_ready(self, variants=None):
        if self._busy():
            return
        if not os.path.isfile(GETMODEL):
            messagebox.showerror("Генератор", f"get_model.py не найден рядом с этим окном:\n{GETMODEL}")
            return
        env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
        if variants == "all":
            cmd = [sys.executable, "-u", GETMODEL, "--все"]
            заг = [os.path.basename(sys.executable), "-u", "get_model.py", "--все"]
        else:
            v = self.ready_var.get()
            cmd = [sys.executable, "-u", GETMODEL, "--вариант", v]
            заг = [os.path.basename(sys.executable), "-u", "get_model.py", "--вариант", v]
        self._launch(cmd, env, заг)

    def build_ready_all(self):
        self.build_ready(variants="all")

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
                    tag = "err" if ("ошибк" in low or "провал" in low or "error" in low
                                    or "fail" in low or "traceback" in low) \
                        else "ok" if ("проверка ok" in low or "verify ok" in low
                                      or "готово" in low or "done" in low) else None
                    self._append(item, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    def _on_exit(self, code):
        self._append(f"\n[процесс завершился с кодом {code}]\n", "ok" if code == 0 else "err")
        self._set_running(False)
        self.status.configure(text="готово" if code == 0 else f"вышел ({code})")
        self.proc = None

    def stop(self):
        p = self.proc
        if not p or p.poll() is not None:
            return
        self.status.configure(text="останавливаю…")
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True)
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception as e:
            self._append(f"не удалось остановить: {e}\n", "err")

    def _on_close(self):
        if self._busy():
            if not messagebox.askyesno("Выход", "Процесс ещё выполняется. Остановить и выйти?"):
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
