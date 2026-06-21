#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Генератор готовых моделей NanoDet-Plus (COCO, 80 классов) -> NCNN.

«Нажал и готово»: запусти без аргументов

    python get_model.py

и скрипт сам выберет вариант (или покажет меню в терминале), клонирует репозиторий
RangiLyu/nanodet, скачает официальный чекпойнт, сконвертирует его в проверенную
fp16-модель .param/.bin для сайдкара и напишет рядом coco.names + готовую команду
запуска. Никаких ручных скачиваний.

ВАРИАНТЫ (тяжесть модели x входное разрешение) — выбирай под свою задачу:

    m-320        вес 1.0x, вход 320  — самая БЫСТРАЯ, точность пониже
    m-416        вес 1.0x, вход 416  — БАЛАНС скорость/точность (рекомендую)
    m-1.5x-320   вес 1.5x, вход 320  — точнее m-320, чуть медленнее
    m-1.5x-416   вес 1.5x, вход 416  — самая ТОЧНАЯ, самая тяжёлая

Примеры:
    python get_model.py                       # меню / вариант по умолчанию (m-416)
    python get_model.py --вариант m-320       # конкретный вариант
    python get_model.py --все                 # сразу все четыре варианта
    python get_model.py --список              # показать таблицу вариантов и выйти

Что нужно на машине: torch (для экспорта в ONNX) и любой конвертер ONNX->ncnn
(`pip install pnnx` — проще всего, один пакет). Чего не хватает — скрипт честно
скажет и даст точную команду установки. Своё уже есть? Переопредели --ckpt/--cfg/--repo.
"""
import argparse
import os
import sys
import subprocess
import urllib.request

# Старые консоли Windows (cp1251 и т.п.) не умеют печатать стрелки/эмодзи -> падение.
# Просим поток заменять непечатаемые символы вместо краха.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/RangiLyu/nanodet.git"
RELEASE = "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1"

# ── Каталог официальных COCO-вариантов NanoDet-Plus ───────────────────────────
# Для каждого: понятное имя, конфиг в репозитории, имя чекпойнта в релизе,
# входное разрешение и reg_max (у всех плюс-моделей он равен 7).
VARIANTS = {
    "m-320": {
        "вес": "1.0x", "вход": 320, "reg_max": 7,
        "cfg": "config/nanodet-plus-m_320.yml",
        "ckpt": "nanodet-plus-m_320_checkpoint.ckpt",
        "описание": "самая БЫСТРАЯ, точность пониже",
    },
    "m-416": {
        "вес": "1.0x", "вход": 416, "reg_max": 7,
        "cfg": "config/nanodet-plus-m_416.yml",
        "ckpt": "nanodet-plus-m_416_checkpoint.ckpt",
        "описание": "БАЛАНС скорость/точность (рекомендую)",
    },
    "m-1.5x-320": {
        "вес": "1.5x", "вход": 320, "reg_max": 7,
        "cfg": "config/nanodet-plus-m-1.5x_320.yml",
        "ckpt": "nanodet-plus-m-1.5x_320_checkpoint.ckpt",
        "описание": "точнее m-320, чуть медленнее",
    },
    "m-1.5x-416": {
        "вес": "1.5x", "вход": 416, "reg_max": 7,
        "cfg": "config/nanodet-plus-m-1.5x_416.yml",
        "ckpt": "nanodet-plus-m-1.5x_416_checkpoint.ckpt",
        "описание": "самая ТОЧНАЯ, самая тяжёлая",
    },
}
DEFAULT_VARIANT = "m-416"

# Принимаем «человеческие» написания варианта (с пробелами, в верхнем регистре, и т.п.).
ALIASES = {
    "m320": "m-320", "m_320": "m-320", "320": "m-320",
    "m416": "m-416", "m_416": "m-416", "416": "m-416",
    "m-1.5x_320": "m-1.5x-320", "1.5x-320": "m-1.5x-320", "1.5x320": "m-1.5x-320",
    "m-1.5x_416": "m-1.5x-416", "1.5x-416": "m-1.5x-416", "1.5x416": "m-1.5x-416",
}

# 80 классов COCO в том порядке, в котором их отдаёт официальная модель.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def норм_вариант(имя):
    """Приводит написанное пользователем имя варианта к ключу каталога."""
    if not имя:
        return None
    к = имя.strip().lower().replace(" ", "")
    return к if к in VARIANTS else ALIASES.get(к)


def печать_таблицы():
    print("Доступные варианты готовых моделей (COCO, 80 классов):\n")
    print(f"  {'вариант':<14}{'вес':<7}{'вход':<7}описание")
    print("  " + "-" * 60)
    for ключ, v in VARIANTS.items():
        пометка = "  <- по умолчанию" if ключ == DEFAULT_VARIANT else ""
        print(f"  {ключ:<14}{v['вес']:<7}{v['вход']:<7}{v['описание']}{пометка}")
    print("\nПримеры:")
    print("  python get_model.py --вариант m-320")
    print("  python get_model.py --все")


def выбор_в_меню():
    """Интерактивное меню в терминале — «нажал и готово»."""
    ключи = list(VARIANTS)
    print("Какой готовый вариант модели создать?\n")
    for i, ключ in enumerate(ключи, 1):
        v = VARIANTS[ключ]
        пометка = " (по умолчанию)" if ключ == DEFAULT_VARIANT else ""
        print(f"  {i}. {ключ:<12} — вес {v['вес']}, вход {v['вход']}, {v['описание']}{пометка}")
    print(f"  {len(ключи) + 1}. все четыре сразу")
    while True:
        try:
            ответ = input(f"\nНомер [Enter = {DEFAULT_VARIANT}]: ").strip()
        except EOFError:
            return [DEFAULT_VARIANT]
        if ответ == "":
            return [DEFAULT_VARIANT]
        if ответ.isdigit():
            n = int(ответ)
            if 1 <= n <= len(ключи):
                return [ключи[n - 1]]
            if n == len(ключи) + 1:
                return ключи[:]
        вр = норм_вариант(ответ)
        if вр:
            return [вр]
        print("  Не понял выбор, попробуй ещё раз (введи номер).")


def скачать(url, dst, timeout=600):
    """Качаем чекпойнт с индикатором прогресса. Уже есть — пропускаем."""
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  уже скачан: {os.path.basename(dst)} ({os.path.getsize(dst) // (1<<20)} МБ)")
        return dst
    print(f"  качаю {os.path.basename(dst)} …")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dst + ".part"
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        всего = int(r.headers.get("Content-Length", 0))
        получено = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            получено += len(chunk)
            if всего:
                pct = получено * 100 // всего
                print(f"\r    {pct:3d}%  ({получено // (1<<20)}/{всего // (1<<20)} МБ)",
                      end="", flush=True)
    if всего:
        print()
    if not os.path.getsize(tmp):
        os.remove(tmp)
        raise RuntimeError(f"скачалось 0 байт из {url}")
    os.replace(tmp, dst)
    print(f"    -> {dst} ({os.path.getsize(dst) // (1<<20)} МБ)")
    return dst


def проверка_окружения():
    """Проверяем torch и конвертер ДО долгого скачивания. Конвертер, если можем,
    ставим сами (pnnx — один wheel). Возвращает список понятных проблем (пустой = ок)."""
    import shutil
    проблемы = []
    try:
        import torch  # noqa: F401
    except Exception:
        проблемы.append(
            "Не найден PyTorch (нужен, чтобы превратить чекпойнт в ONNX).\n"
            "    Поставь его так:\n"
            "      pip install torch torchvision\n"
            "    (или nightly cu128 для свежих GPU — см. README сайдкара).")
    есть_конвертер = shutil.which("onnx2ncnn") or shutil.which("pnnx")
    if not есть_конвертер:
        try:
            import pnnx  # noqa: F401
            есть_конвертер = True
        except Exception:
            есть_конвертер = False
    if not есть_конвертер:
        print("  конвертер ONNX->ncnn не найден — пробую поставить pnnx автоматически…")
        # --no-deps: сам бинарь pnnx самодостаточен; иначе pip тянет весь CUDA-torch.
        код = subprocess.call([sys.executable, "-m", "pip", "install", "--no-deps", "pnnx"])
        if код == 0 and (shutil.which("pnnx") or _импортируется("pnnx")):
            print("  pnnx установлен.")
        else:
            проблемы.append(
                "Нет конвертера ONNX->ncnn, и автоустановка pnnx не удалась.\n"
                "    Поставь вручную:\n"
                "      pip install pnnx\n"
                "    либо собери ncnn и положи onnx2ncnn / ncnnoptimize в PATH.")
    return проблемы


def _импортируется(имя):
    try:
        __import__(имя)
        return True
    except Exception:
        return False


def обеспечить_репозиторий(repo):
    if os.path.isdir(repo):
        return
    print("Клонирую RangiLyu/nanodet …")
    for попытка in range(1, 5):
        if subprocess.call(["git", "clone", "--depth", "1", REPO_URL, repo]) == 0:
            return
        print(f"  клонирование не удалось (попытка {попытка}/4) — повтор…")
    sys.exit("ОШИБКА: не удалось склонировать репозиторий nanodet.\n"
             "  Проверь интернет/доступ к github.com и запусти снова.")


def установить_зависимости_репо(repo):
    """Доставить зависимости самого nanodet (matplotlib, pycocotools, tabulate и т.п.) —
    БЕЗ torch/vision/audio, чтобы не сломать уже установленную сборку (например cu128).
    Без них код модели не импортируется при экспорте. Отключается GET_MODEL_PIP=0."""
    import re
    if os.environ.get("GET_MODEL_PIP", "1").lower() in ("0", "false", "no", "off"):
        return
    req = os.path.join(os.path.abspath(repo), "requirements.txt")
    if not os.path.isfile(req):
        return
    pkgs = []
    for line in open(req, encoding="utf-8"):
        spec = line.strip()
        if not spec or spec.startswith(("#", "-")):
            continue
        name = re.split(r"[<>=!~;\[ ]", spec)[0].strip().lower()
        if name in ("torch", "torchvision", "torchaudio"):
            continue                                  # не трогаем уже установленный torch
        pkgs.append(spec)
    if pkgs:
        print(f"  доустанавливаю зависимости nanodet (без torch): {', '.join(pkgs)}")
        subprocess.call([sys.executable, "-m", "pip", "install", *pkgs])


def создать_вариант(ключ, repo, общий_ckpt=None, общий_cfg=None,
                    override_input=None, override_reg=None):
    """Скачать чекпойнт варианта и собрать из него проверенную ncnn-модель.
    Возвращает (param, bin) или None."""
    v = VARIANTS[ключ]
    cfg = общий_cfg or os.path.join(repo, *v["cfg"].split("/"))
    if not os.path.isfile(cfg):
        print(f"  [{ключ}] ОШИБКА: не найден конфиг {cfg}")
        return None
    вход = override_input or v["вход"]
    reg = override_reg if override_reg is not None else v["reg_max"]

    ckpt = общий_ckpt
    if not ckpt:
        try:
            ckpt = скачать(f"{RELEASE}/{v['ckpt']}", os.path.join(HERE, v["ckpt"]))
        except Exception as e:
            print(f"  [{ключ}] ОШИБКА: не смог скачать чекпойнт\n    {e}\n"
                  f"    Скачай вручную {RELEASE}/{v['ckpt']} и передай --ckpt.")
            return None

    stem = os.path.join(HERE, f"nanodet-{ключ}")
    sys.path.insert(0, HERE)
    import export_ncnn
    res = export_ncnn.run_export(repo, cfg, stem, вход, reg, classes=80, ckpt=ckpt)
    if not res:
        return None
    # Кладём рядом coco.names, чтобы сайдкар сразу заработал.
    names = os.path.join(HERE, "coco.names")
    if not os.path.isfile(names):
        with open(names, "w", encoding="utf-8") as f:
            f.write("\n".join(COCO_NAMES) + "\n")
    return res + (вход,)


def main():
    ap = argparse.ArgumentParser(
        description="Генератор готовых COCO-моделей NanoDet-Plus -> NCNN (нажал и готово).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--вариант", "--variant", dest="вариант", default=None,
                    help="какой вариант собрать: m-320 / m-416 / m-1.5x-320 / m-1.5x-416")
    ap.add_argument("--все", "--all", dest="все", action="store_true",
                    help="собрать сразу все четыре варианта")
    ap.add_argument("--список", "--list", dest="список", action="store_true",
                    help="показать таблицу вариантов и выйти")
    ap.add_argument("--repo", default=os.path.join(HERE, "nanodet"),
                    help="папка с репозиторием nanodet (по умолчанию рядом со скриптом)")
    ap.add_argument("--ckpt", default=None,
                    help="свой чекпойнт .ckpt (по умолчанию качаем официальный)")
    ap.add_argument("--cfg", default=None,
                    help="свой конфиг .yml (по умолчанию берём из репозитория под вариант)")
    ap.add_argument("--input", type=int, default=None, help="переопределить входное разрешение")
    ap.add_argument("--reg-max", type=int, default=None, help="переопределить reg_max")
    a = ap.parse_args()

    if a.список:
        печать_таблицы()
        return

    # Какие варианты делаем.
    if a.все:
        варианты = list(VARIANTS)
    elif a.вариант:
        ключ = норм_вариант(a.вариант)
        if not ключ:
            print(f"ОШИБКА: неизвестный вариант '{a.вариант}'.\n")
            печать_таблицы()
            sys.exit(1)
        варианты = [ключ]
    elif sys.stdin is not None and sys.stdin.isatty():
        варианты = выбор_в_меню()
    else:
        варианты = [DEFAULT_VARIANT]
        print(f"Вариант не задан — беру по умолчанию: {DEFAULT_VARIANT} "
              f"(см. --список, чтобы выбрать другой).")

    print("\n[1/3] Проверяю окружение (torch + конвертер ONNX->ncnn)…")
    проблемы = проверка_окружения()
    if проблемы:
        print("\nНе хватает инструментов для сборки модели:\n")
        for i, p in enumerate(проблемы, 1):
            print(f"  {i}) {p}")
        print("\nУстрани это и запусти снова — дальше всё произойдёт само.")
        sys.exit(1)

    print("[2/3] Готовлю репозиторий nanodet (клонирую, если нужно)…")
    обеспечить_репозиторий(a.repo)
    установить_зависимости_репо(a.repo)

    print(f"[3/3] Собираю {'варианты' if len(варианты) > 1 else 'вариант'}: "
          f"{', '.join(варианты)}\n")
    готовые = []
    for ключ in варианты:
        print(f"=== {ключ}  (вес {VARIANTS[ключ]['вес']}, вход {VARIANTS[ключ]['вход']}) ===")
        res = создать_вариант(ключ, a.repo, a.ckpt, a.cfg, a.input, a.reg_max)
        if res:
            param, binf, вход = res
            готовые.append((ключ, param, binf, вход))
        print()

    if not готовые:
        sys.exit("Ни один вариант не собрался — смотри сообщения об ошибках выше.")

    names = os.path.join(HERE, "coco.names")
    print("Готово! Собрано моделей:", len(готовые))
    for ключ, param, binf, вход in готовые:
        print(f"\n  [{ключ}]  модель готова:")
        print(f"    .param: {param}")
        print(f"    .bin:   {binf}")
        print(f"    Запуск на Pi (COCO, 80 классов):")
        print(f"      ND_PARAM={param} ND_BIN={binf} ND_INPUT={вход} \\")
        print(f"        YOLO_LABELS={names} YOLO_SOURCE=rpicam python3 nanodet_ncnn_sidecar.py")
    print("\n  (USB-камера: YOLO_SOURCE=0 · первый запуск с --inspect покажет формы выходов)")


if __name__ == "__main__":
    main()
