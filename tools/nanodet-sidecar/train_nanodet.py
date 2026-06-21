#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Обучение NanoDet-Plus на своём датасете И автоматический экспорт оптимизированной
NCNN-модели — одной командой, в той же логике, что и train_yolofastest.py.

>>> Поправь блок НАСТРОЙКИ ниже и запусти:   python train_nanodet.py
    (Никаких переменных окружения не нужно. Дружелюбно к Windows.)
    Любишь кликать мышкой? Запусти окно:      python train_nanodet_gui.py
    (там же — вкладка «Готовые модели»: COCO-модель в один клик, без обучения.)

Дообучение: укажи WEIGHTS на готовый .ckpt, чтобы продолжить модель на новых/расширенных
данных, или RESUME на прерванный запуск. И то, и другое доступно из окна.

Что делает: конвертирует твой YOLO-датасет в COCO JSON (без копирования картинок),
клонирует RangiLyu/nanodet, пишет кастомный конфиг на основе штатного nanodet-plus-m,
обучает, затем запускает export_ncnn.py -> проверенную fp16 .param/.bin для сайдкара.

Почему NanoDet-Plus, а не YOLO-FastestV2: FPN по 3-4 страйдам (8/16/32/64) даёт намного
лучшее обнаружение МЕЛКИХ объектов, оставаясь реалтаймом на CPU Pi 5. Это чистый
PyTorch + PyTorch-Lightning, поэтому nightly cu128 обучает его на GPU Blackwell (в отличие
от PaddleDetection/PicoDet, которые на Blackwell вообще не пойдут).

Установка (Windows, в твоём venv):
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install opencv-python numpy onnx onnxsim ncnn pnnx pytorch-lightning pycocotools omegaconf

⚠️ Это обёртка-оркестратор поверх апстрим-репозитория. Схема конфига NanoDet стабильна,
   но не заморожена — тренер печатает каждое поле, которое патчит; если предупредит, что
   ключ не найден — открой сгенерированный конфиг и поправь руками.
"""

import os
import re
import sys
import glob
import json
import shutil
import subprocess

# Консоли Windows по умолчанию в legacy-кодировке (cp1251), которая не может закодировать
# стрелки/эмодзи -> UnicodeEncodeError. Просим заменять непечатаемые символы вместо краха.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

# ========================= НАСТРОЙКИ — ПРАВЬ ЗДЕСЬ ===========================
DATASET    = r"C:\Users\dest\Desktop\test\merged_dataset"   # корень YOLO (images/{train,val} + labels/{train,val})
CLASSES    = ["Birds", "Drones", "Dron2"]                   # в порядке id из YOLO

MODEL_SIZE = "1.0x"        # тяжесть модели: "1.0x" (легче/быстрее) или "1.5x" (тяжелее/точнее)
INPUT      = 416           # квадратный вход (у nanodet-plus-m по умолчанию 416; 320 = быстрее, хуже мелочь)
EPOCHS     = 200           # NanoDet сходится быстрее, чем YOLO с нуля; обычно 100-300
REG_MAX    = 7             # число бинов DFL на сторону − 1 (по умолчанию 7) — не трогай, если не менял голову
LR         = None          # learning rate; None = оставить из штатного конфига. Для дообучения берут поменьше (напр. 0.0005)

# ── Дообучение / возобновление ────────────────────────────────────────────────
#   WEIGHTS — загрузить эти веса как СТАРТОВУЮ точку и обучаться заново на твоих данных
#             (transfer-learning / продолжить готовую модель на новых/расширенных данных).
#   RESUME  — продолжить ПРЕРВАННЫЙ запуск, сохранив состояние оптимизатора + счётчик эпох.
#   Передай путь к .ckpt в одно из двух (оставь "" — обучение с нуля). WEIGHTS — обычный
#   рычаг «дообучения»; RESUME — чтобы подхватить упавший запуск.
WEIGHTS    = ""            # напр. r"C:\...\nanodet\workspace\custom\model_best\model_best.ckpt"
RESUME     = ""            # напр. r"C:\...\nanodet\workspace\custom\model_last.ckpt"

# ── Железо (Ultra 9 285K + RTX 5090 32GB + 128GB RAM, Windows) ────────────────
DEVICE     = "gpu"         # "gpu" (RTX 5090, nightly torch cu128) или "cpu"
GPU_IDS    = [0]
BATCH      = 96            # nanodet-plus-m небольшой; 96-160 влезает в 32GB. OOM? уменьши.
WORKERS    = 20            # воркеров загрузчика на GPU (у 285K — 24 ядра)

# ── Служебное ──────────────────────────────────────────────────────────────────
OUT        = "nd_data"                 # куда писать COCO .json
REPO_DIR   = "nanodet"                 # куда клонируется RangiLyu/nanodet
BASE_CFG   = ""                        # базовый конфиг; "" = выбрать автоматически по MODEL_SIZE+INPUT
EXPORT     = True                      # после обучения — авто-экспорт проверенной ncnn-модели
OUT_STEM   = "nanodet"                 # имя-основа для экспортируемых .param/.bin
# =============================================================================

# ── Переопределения через окружение — чтобы командный центр (или любой запускатель)
#    управлял обучением без правки этого файла. Одни и те же ключи TRAIN_* у всех тренеров.
def _envc(name, cur, cast=str):
    v = os.environ.get(name)
    if v in (None, ""):
        return cur
    try:
        return cast(v)
    except (ValueError, TypeError):
        # Повреждённое значение (например мусор из состояния окна) — оставляем дефолт.
        print(f"  [!] {name}={v[:60]!r} не подходит — использую значение по умолчанию ({cur})")
        return cur
DATASET = _envc("TRAIN_DATASET", DATASET)
if os.environ.get("TRAIN_CLASSES"):
    CLASSES = [c.strip() for c in os.environ["TRAIN_CLASSES"].split(",") if c.strip()]
MODEL_SIZE = _envc("TRAIN_MODEL_SIZE", MODEL_SIZE)
INPUT   = _envc("TRAIN_INPUT", INPUT, int)
EPOCHS  = _envc("TRAIN_EPOCHS", EPOCHS, int)
REG_MAX = _envc("TRAIN_REG_MAX", REG_MAX, int)
BATCH   = _envc("TRAIN_BATCH", BATCH, int)
WORKERS = _envc("TRAIN_WORKERS", WORKERS, int)
DEVICE  = _envc("TRAIN_DEVICE", DEVICE)
WEIGHTS = _envc("TRAIN_WEIGHTS", WEIGHTS)
RESUME  = _envc("TRAIN_RESUME", RESUME)
BASE_CFG = _envc("TRAIN_BASE_CFG", BASE_CFG)
_lr_env = os.environ.get("TRAIN_LR", "").strip()
if _lr_env:
    try:
        LR = float(_lr_env)
    except ValueError:
        # Поле learning rate содержит мусор (бывает из-за повреждённого состояния окна) —
        # не валим обучение, просто берём LR из штатного конфига.
        print(f"  [!] TRAIN_LR не похоже на число ({_lr_env[:60]!r}…) — беру learning rate из конфига")
        LR = None
if os.environ.get("TRAIN_GPU_IDS"):
    GPU_IDS = [int(x) for x in re.split(r"[,\s]+", os.environ["TRAIN_GPU_IDS"].strip()) if x]
if os.environ.get("TRAIN_EXPORT"):
    EXPORT = os.environ["TRAIN_EXPORT"].lower() not in ("0", "false", "no", "off")

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def выбрать_базовый_конфиг():
    """Базовый конфиг под выбранную тяжесть модели (1.0x/1.5x) и вход. У NanoDet-Plus есть
    штатные конфиги на 320 и 416; для нестандартного входа берём ближайший и патчим
    input_size. BASE_CFG, заданный явно, имеет приоритет."""
    if BASE_CFG:
        return BASE_CFG
    size = "-1.5x" if str(MODEL_SIZE).strip().lower() in ("1.5x", "1.5", "m-1.5x") else ""
    base_res = 320 if INPUT <= 352 else 416     # какой штатный конфиг ближе по разрешению
    return f"config/nanodet-plus-m{size}_{base_res}.yml"


def sh(cmd, cwd=None, extra_env=None):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    e = dict(os.environ)
    if extra_env:
        e.update(extra_env)
    return subprocess.call([str(c) for c in cmd], cwd=cwd, env=e)


def setup_repo_env():
    """Ставим зависимости склонированного nanodet — БЕЗ затрагивания твоего torch — чтобы
    следующие шаги не упали на отсутствующем подмодуле или не той версии pytorch-lightning.

    `tools/train.py` делает `import nanodet`; это решается через PYTHONPATH=<repo> (он задаётся
    на вызове обучения), поэтому сам пакет pip-ом не ставим (избегаем сборки setup.py). Тут
    ставим только зависимости из requirements репозитория, КРОМЕ torch/vision/audio — это тянет
    правильные версии pytorch-lightning/omegaconf/tabulate/…, не трогая твой nightly cu128 torch.
    Пропустить всё это можно через TRAIN_PIP=0."""
    if os.environ.get("TRAIN_PIP", "1").lower() in ("0", "false", "no", "off"):
        print("  (TRAIN_PIP=0 — пропускаю авто-установку зависимостей; полагаюсь на PYTHONPATH + твоё окружение)")
        return
    req = os.path.join(os.path.abspath(REPO_DIR), "requirements.txt")
    if not os.path.isfile(req):
        return
    pkgs = []
    for line in open(req, encoding="utf-8"):
        spec = line.strip()
        if not spec or spec.startswith(("#", "-")):
            continue
        name = re.split(r"[<>=!~;\[ ]", spec)[0].strip().lower()
        if name in ("torch", "torchvision", "torchaudio"):
            continue                                  # сохраняем nightly cu128 сборку пользователя
        pkgs.append(spec)
    if pkgs:
        print(f"[2b/4] Ставлю зависимости nanodet (без torch): {', '.join(pkgs)}")
        sh([sys.executable, "-m", "pip", "install", *pkgs])


_WRAPPER_SRC = '''\
# -*- coding: utf-8 -*-
# Авто-сгенерировано train_nanodet.py. Возвращает API, которые свежий PyTorch убрал, чтобы
# (старый) код nanodet / pytorch-lightning работал на сборке torch 2.x / cu128, затем
# запускает nanodet/tools/train.py.
import os, sys, types, runpy
import collections.abc as _abc
sys.path.insert(0, os.getcwd())                     # чтобы `import nanodet` нашёл склонированный репозиторий
try:
    import torch
    if not hasattr(torch, "_six"):                  # убрано в torch 2.0 — старый код всё ещё импортирует
        m = types.ModuleType("torch._six")
        m.string_classes = (str, bytes)
        m.int_classes = int
        m.container_abcs = _abc
        m.PY3 = True
        m.PY37 = sys.version_info >= (3, 7)
        torch._six = m
        sys.modules["torch._six"] = m
        print("  [совместимость] подменил torch._six для старого кода обучения")
    # torch 2.6+: weights_only по умолчанию True -> дообучение с .ckpt (объекты Lightning)
    # иначе падает. Чекпойнты — твои/официальные, источник доверенный.
    _orig_load = torch.load
    def _load_compat(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)
    torch.load = _load_compat
except Exception as e:
    print("  [совместимость] шим torch пропущен:", e)
cfg = sys.argv[1]
sys.argv = [os.path.join("tools", "train.py"), cfg]
runpy.run_path(os.path.join("tools", "train.py"), run_name="__main__")
'''


def write_train_wrapper():
    """Пишем крошечный лаунчер, который подменяет torch._six (и torch.load) до nanodet/
    tools/train.py — обычная ошибка 'No module named torch._six' на современном torch.
    Возвращает абсолютный путь; запускается с cwd=<repo>."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.abspath(os.path.join(OUT, "_nd_train_wrapper.py"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(_WRAPPER_SRC)
    return path


def _has_imgs(d):
    return os.path.isdir(d) and any(
        glob.glob(os.path.join(d, e)) or glob.glob(os.path.join(d, "**", e), recursive=True)
        for e in IMG_EXT)


def _split_dir(split):
    """Находим папку с картинками сплита для частых раскладок YOLO."""
    for c in (os.path.join(DATASET, "images", split),
              os.path.join(DATASET, split, "images"),
              os.path.join(DATASET, split)):
        if _has_imgs(c):
            return c
    return None


def _list_images(d):
    files = []
    for e in IMG_EXT:
        files += glob.glob(os.path.join(d, e))
    return sorted(set(os.path.abspath(f) for f in files))


def _img_size(path):
    """(w, h) без полного декодирования, когда можно (PIL), иначе OpenCV."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        import cv2
        im = cv2.imread(path)
        if im is None:
            return None
        return im.shape[1], im.shape[0]


def _label_path(img_path):
    return os.path.splitext(img_path)[0].replace("images", "labels") + ".txt"


def build_coco(split):
    """YOLO -> COCO JSON (без копирования картинок). Возвращает (img_dir, json_path, n_imgs)."""
    d = _split_dir(split)
    if not d:
        return None, None, 0
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
    img_id, ann_id, n_obj = 1, 1, 0
    for img in _list_images(d):
        wh = _img_size(img)
        if not wh:
            continue
        w, h = wh
        coco["images"].append({"id": img_id, "file_name": os.path.basename(img), "width": w, "height": h})
        lp = _label_path(img)
        if os.path.isfile(lp):
            for line in open(lp):
                p = line.split()
                if len(p) < 5:
                    continue
                cls = int(float(p[0])); xc, yc, bw, bh = (float(x) for x in p[1:5])
                x = (xc - bw / 2) * w; y = (yc - bh / 2) * h; bw *= w; bh *= h
                if bw <= 1 or bh <= 1:
                    continue
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cls,
                    "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
                ann_id += 1; n_obj += 1
        img_id += 1
    os.makedirs(OUT, exist_ok=True)
    jp = os.path.abspath(os.path.join(OUT, f"instances_{split}.json"))
    json.dump(coco, open(jp, "w"))
    print(f"  {split}: {img_id - 1} картинок, {n_obj} объектов  (img_path={d})")
    return os.path.abspath(d), jp, img_id - 1


def write_config(base_cfg, train_img, train_json, val_img, val_json):
    """Копируем штатный конфиг nanodet-plus и патчим только поля датасета/классов/расписания.
    Сообщаем о каждом патче; предупреждаем о ключе, который не нашли (дрейф схемы)."""
    src = os.path.join(REPO_DIR, *base_cfg.split("/"))
    if not os.path.isfile(src):
        sys.exit(f"ОШИБКА: базовый конфиг не найден: {src}\n  (проверь MODEL_SIZE/INPUT/BASE_CFG и клон)")
    s = open(src, encoding="utf-8").read()

    def sub(pattern, repl, n, what, flags=0):
        nonlocal s
        s, c = re.subn(pattern, repl, s, count=n, flags=flags)
        print(f"  патч {what}: {c} мест" + ("  [!] НЕ НАЙДЕНО" if c == 0 else ""))

    names = ", ".join(f"'{c}'" for c in CLASSES)
    sub(r"(?m)^save_dir:.*$", "save_dir: workspace/custom", 1, "save_dir")
    sub(r"num_classes:\s*\d+", f"num_classes: {len(CLASSES)}", 0, "num_classes")
    # class_names в COCO-конфигах — это МНОГОСТРОЧНЫЙ список (≈80 имён на 13 строк).
    # Заменяем весь блок целиком (от '[' до ']'), а не первую строку, иначе остаются
    # хвостовые имена и len(class_names) != num_classes -> падение обучения.
    sub(r"class_names:\s*\[.*?\]", f"class_names: [{names}]", 1, "class_names", flags=re.DOTALL)
    # img_path / ann_path идут train-затем-val; меняем первые два каждого по порядку.
    img_it = iter([train_img, val_img]); ann_it = iter([train_json, val_json])
    sub(r"(?m)^(\s*img_path:\s*).*$", lambda m: m.group(1) + next(img_it), 2, "img_path (train,val)")
    sub(r"(?m)^(\s*ann_path:\s*).*$", lambda m: m.group(1) + next(ann_it), 2, "ann_path (train,val)")
    sub(r"input_size:\s*\[\s*\d+\s*,\s*\d+\s*\]", f"input_size: [{INPUT}, {INPUT}]", 0, "input_size")
    sub(r"gpu_ids:\s*\[.*?\]", f"gpu_ids: {GPU_IDS if DEVICE == 'gpu' else []}", 0, "gpu_ids")
    sub(r"workers_per_gpu:\s*\d+", f"workers_per_gpu: {WORKERS}", 0, "workers_per_gpu")
    sub(r"batchsize_per_gpu:\s*\d+", f"batchsize_per_gpu: {BATCH}", 0, "batchsize_per_gpu")
    sub(r"total_epochs:\s*\d+", f"total_epochs: {EPOCHS}", 0, "total_epochs")
    if LR is not None:                       # первый lr: под schedule.optimizer
        sub(r"(?m)^(\s*lr:\s*).*$", lambda m: m.group(1) + repr(float(LR)), 1, "optimizer lr")

    # Дообучение / возобновление: nanodet читает schedule.load_model (перенос весов, свежее
    # расписание) и schedule.resume (продолжить запуск). В штатном конфиге это закомментированные
    # заглушки; удаляем любую копию и вставляем заново под `schedule:`.
    def set_schedule_key(key, value):
        nonlocal s
        s = re.sub(rf"(?m)^[ \t]*#?[ \t]*{key}:.*$\n?", "", s)          # вырезаем старое/закомментированное
        ins = f'\\g<1>\n  {key}: "{value}"'                             # кавычки: безопасные для YAML пути
        s, c = re.subn(r"(?m)^(schedule:)[ \t]*$", ins, s, count=1)     # вставляем прямо под schedule:
        print(f"  патч schedule.{key}: {value}" + ("  [!] 'schedule:' не найдено" if c == 0 else ""))
    if WEIGHTS:
        set_schedule_key("load_model", os.path.abspath(WEIGHTS).replace("\\", "/"))
    if RESUME:
        set_schedule_key("resume", os.path.abspath(RESUME).replace("\\", "/"))

    os.makedirs(OUT, exist_ok=True)
    out_cfg = os.path.abspath(os.path.join(OUT, "custom.yml"))
    open(out_cfg, "w", encoding="utf-8").write(s)
    print(f"  конфиг -> {out_cfg}")
    return out_cfg


def preflight_gpu():
    """Быстрая проверка, что выбранная видеокарта реально умеет считать на этом torch.
    RTX 5090 (Blackwell, sm_120) требует torch собранный под CUDA 12.8 (cu128); со старым
    torch любой GPU-запуск падает 'no kernel image is available' — но только глубоко внутри
    обучения. Ловим это здесь, до конвертации датасета, и даём точную инструкцию."""
    if DEVICE != "gpu":
        return
    try:
        import torch
    except Exception:
        return  # torch ещё не поставлен — обычный путь установки разберётся сам
    if not torch.cuda.is_available():
        sys.exit("ОШИБКА: DEVICE=gpu, но torch не видит CUDA-устройств.\n"
                 "  Поставь GPU-сборку torch или переключись на CPU (TRAIN_DEVICE=cpu).")
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "?"
    try:
        x = torch.randn(8, device="cuda")          # реальный запуск ядра на карте
        torch.cuda.synchronize()
        _ = (x + 1).sum().item()
    except Exception as e:
        py = sys.executable
        sys.exit(
            f"ОШИБКА: видеокарта '{name}' несовместима с установленным PyTorch.\n"
            f"  {type(e).__name__}: {e}\n"
            "  Скорее всего это новая карта (RTX 50xx, Blackwell, sm_120), а у тебя torch без ядер под неё.\n"
            "  Поставь сборку под CUDA 12.8:\n"
            f'    "{py}" -m pip uninstall -y torch torchvision torchaudio\n'
            f'    "{py}" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio\n'
            "  Либо обучай на CPU: задай TRAIN_DEVICE=cpu (медленно).")


def main():
    if sys.version_info >= (3, 13):
        print(f"  [!] У тебя Python {sys.version_info.major}.{sys.version_info.minor}. У стека обучения "
              "(torch, pytorch-lightning, pycocotools) часто ещё НЕТ колёс под такой свежий\n"
              "      Python, поэтому установка тихо попадает не туда / не собирается. Если что-то ниже\n"
              "      упадёт на импортах — используй для машины обучения Python 3.11 (см. README).")
    preflight_gpu()
    print(f"[1/4] Конвертирую датасет в COCO... (модель {MODEL_SIZE}, вход {INPUT})")
    if not os.path.isdir(DATASET):
        sys.exit(f"ОШИБКА: DATASET не найден: {DATASET}")
    tr_img, tr_json, n_tr = build_coco("train")
    va_img, va_json, n_va = build_coco("val")
    if n_tr == 0:
        sys.exit(f"ОШИБКА: нет картинок для обучения в {DATASET} (искал в images/train, train/images, train)")
    if n_va == 0:                       # NanoDet нужен val-сет; откатываемся на train
        print("  ВНИМАНИЕ: val-сплит не найден — использую train как val (метрики будут оптимистичными)")
        va_img, va_json = tr_img, tr_json

    print("[2/4] Беру репозиторий (клонирую, если нужно)...")
    if not os.path.isdir(REPO_DIR):
        ok = False
        for попытка in range(1, 5):
            if sh(["git", "clone", "--depth", "1", "https://github.com/RangiLyu/nanodet.git", REPO_DIR]) == 0:
                ok = True; break
            print(f"  клонирование не удалось (попытка {попытка}/4) — повтор…")
        if not ok:
            sys.exit("ОШИБКА: git clone не удался (проверь интернет/доступ к github.com)")
    setup_repo_env()                         # делаем `import nanodet` рабочим + зависимости (torch не трогаем)
    base_cfg = выбрать_базовый_конфиг()
    print(f"  базовый конфиг: {base_cfg}")
    cfg = write_config(base_cfg, tr_img, tr_json, va_img, va_json)

    mode = (f"дообучение от {os.path.basename(WEIGHTS)}" if WEIGHTS
            else f"возобновление {os.path.basename(RESUME)}" if RESUME else "с нуля")
    print(f"[3/4] Обучение на {DEVICE.upper()} ({mode}; batch={BATCH}, workers={WORKERS}, epochs={EPOCHS})...")
    # PYTHONPATH=<repo> гарантирует, что `import nanodet` укажет на склонированный репозиторий,
    # даже если pip-шаг пропущен/офлайн или другой 'nanodet' установлен где-то ещё.
    train_env = {"PYTHONPATH": os.path.abspath(REPO_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    wrapper = write_train_wrapper()          # шимит убранный в torch 2.x torch._six + torch.load, затем обучает
    if sh([sys.executable, wrapper, cfg], cwd=REPO_DIR, extra_env=train_env):
        sys.exit("ОШИБКА: обучение не удалось (см. вывод выше). Если это ошибка API pytorch-\n"
                 "  lightning — авто-установленные зависимости nanodet ставят правильную версию PL,\n"
                 "  перезапусти; либо pip install -r nanodet/requirements.txt (сохрани свой torch).\n"
                 "  Проверь предупреждения о патчах конфига.")

    if EXPORT:
        print("[4/4] Экспортирую оптимизированную, проверенную NCNN-модель...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import export_ncnn
            res = export_ncnn.run_export(REPO_DIR, cfg, OUT_STEM, INPUT, REG_MAX, classes=len(CLASSES))
        except Exception as e:
            res = None; print(f"  шаг экспорта упал: {e}")
        if res:
            param, binf = res
            names = os.path.abspath(os.path.join(OUT, "classes.txt"))
            open(names, "w", encoding="utf-8").write("\n".join(CLASSES) + "\n")
            print("\n[ГОТОВО] Обучено И экспортировано. Запуск:")
            print(f"  Сайдкар на Pi:  ND_PARAM={param} ND_BIN={binf} ND_INPUT={INPUT} \\")
            print(f"                  YOLO_LABELS={names} python3 nanodet_ncnn_sidecar.py --inspect")
            print(f"  Телефон:        NanoDet нужен Android-декодер (GFL/DFL) — пока только Pi.")
            return
        print("  Авто-экспорт не завершился — запусти вручную:")
    else:
        print("[4/4] Экспорт (EXPORT=False) — запусти вручную:")
    print(f"  python export_ncnn.py --repo {REPO_DIR} --cfg {cfg} --out {OUT_STEM} "
          f"--input {INPUT} --reg-max {REG_MAX} --classes {len(CLASSES)}")


if __name__ == "__main__":
    main()
