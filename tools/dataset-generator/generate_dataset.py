#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Генератор синтетических датасетов для детекции — картинки (FLUX.1-schnell) + промпты (LLM)
+ авторазметка (YOLO-World / Grounding DINO / OWLv2), весь конвейер одним запуском по конфигу.

Три фазы, каждую можно включать/выключать отдельно:
  [1] Промпты — LLM (LM Studio или любой OpenAI-совместимый эндпойнт) пишет большой набор
                уникальных промптов сцены, сбалансированный по масштабам объекта.
  [2] Картинки — FLUX.1-schnell рендерит их (nf4 / torchao-fp8 / layerwise), с потоковой
                записью на диск (прогон на 30k переживает прерывания и продолжается).
                Либо режим «свои картинки» — генерация пропускается, берётся готовая папка.
  [3] Разметка — open-vocabulary детектор размечает картинки в YOLO-txt + dataset.yaml.

ВСЁ универсально и задаётся конфигом: объект может быть любым (не только дроны) — словарь
промптов состоит из произвольных категорий. Инструменты (FLUX, ultralytics, transformers)
ставятся и скачиваются сами, если их нет.

    python generate_dataset.py                 # встроенные дефолты ниже
    python generate_dataset.py my_config.json  # файл конфига
    GEN_CONFIG=my_config.json python generate_dataset.py

Окно generate_dataset_gui.py пишет этот JSON за тебя, даёт редактор словаря промптов и
выбор движков, и показывает лог живьём.
"""

import os
import sys
import json
import re
import time
import gc
import copy
import random
import subprocess

# Мгновенный небуферизованный вывод (чтобы GUI стримил лог построчно) и UTF-8, чтобы русский
# не падал на старой кодовой странице Windows и не превращался в крокозябры при чтении в GUI.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


# =====================================================================
# ДЕФОЛТНЫЙ КОНФИГ — любое значение переопределяется из JSON-конфига
# =====================================================================
DEFAULTS = {
    "paths": {
        "flux_dir":         r"C:\Users\dest\Desktop\test\FLUX.1-schnell",
        "transformer_path": r"C:\Users\dest\Desktop\test\FLUX.1-schnell\transformer",
        "images_dir":       r"C:\Users\dest\Desktop\test\synthetic_dataset\images",
        "output_yolo_dir":  r"C:\Users\dest\Desktop\test\synthetic_dataset_yolo",
        "prompts_file":     r"C:\Users\dest\Desktop\test\synthetic_dataset\prompts.jsonl",
    },
    "run": {"phase1_prompts": True, "phase2_images": True, "phase3_label": True},
    "total_images": 30000,
    "val_split": 0.15,              # доля в val (авто-сплит train/val в фазе 3); 0 = без сплита
    "auto_install": True,            # сам ставить недостающие пакеты (pip) и качать модели
    "generation": {
        "backend": "flux",          # "flux" (рендерить) | "own" (свои картинки, генерацию пропустить)
        "flux_repo": "black-forest-labs/FLUX.1-schnell",  # для авто-скачивания, если flux_dir пуст
        "hf_token": "",             # токен HF (FLUX закрыт гейтом — нужен токен + принять условия)
        "file_prefix": "synth",      # имя файлов: <prefix>_00001.jpg
        "batch_size": 20,            # промптов за один запрос к LLM (меньше = меньше токенов/обрывов)
        "quant_mode": "torchao",     # nf4 | torchao | layerwise (torchao fp8 — нативно быстр на Blackwell)
        # Дефолты рассчитаны на RTX 5090 (32 ГБ) + 128 ГБ RAM + Ultra 9 285K (24 потока):
        "micro_batch": 16,           # картинок за проход FLUX (32 ГБ тянет; при OOM — уменьшить)
        "super_chunk": 2000,         # промптов на загрузку трансформера (меньше перезагрузок; RAM хватает)
        "encode_batch": 128,         # под-батч текст-энкодера (грузим GPU плотнее)
        "save_workers": 12,          # потоков на сохранение JPEG (чтобы GPU не ждал диск)
        "num_inference_steps": 4,    # FLUX.1-schnell — few-step модель
        "guidance_scale": 0.0,
        "width": 640,
        "height": 640,
        "jpeg_quality": 92,
        "prompt_suffix": ", realistic photo, high resolution",
        # Официальный HF (если у тебя он недоступен — поставь рабочее зеркало, напр. https://hf-mirror.com).
        "hf_endpoint": "https://huggingface.co",
        "allow_tf32": True,
    },
    "llm": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model": "qwen2.5-coder-14b-instruct",
        "temperature": 0.9,
        "max_tokens": 8192,
        "use_lms": True,             # дёргать `lms load/unload` для (вы)грузки модели в LM Studio
        "no_think": True,            # отключать «размышления» reasoning-моделей (Qwen3 /no_think) — экономит токены
    },
    # ── УНИВЕРСАЛЬНЫЙ СЛОВАРЬ ПРОМПТОВ ──
    # object_noun — что генерируем (любой объект). categories — произвольные категории
    # вариативности: имя -> список вариантов. Шаблон подставит по одному случайному из каждой.
    "prompts": {
        # Краткое описание датасета на любом языке — кнопка «Заполнить из описания» в окне
        # попросит LLM развернуть его в object_noun/objects/categories/классы.
        "brief": "",
        "object_noun": "drone (UAV / quadcopter)",
        # МУЛЬТИОБЪЕКТ: если objects непусто и в нём ≥2 объектов, часть сцен будет содержать
        # 2+ разных объекта вместе (доля = multi_object_prob, максимум = multi_object_max).
        # Объекты — естественные фразы для генерации; их имена обычно совпадают с классами
        # разметки. Пусто => используется один object_noun (как раньше).
        "objects": [],
        "multi_object_prob": 0.3,
        "multi_object_max": 2,
        # НЕГАТИВЫ: доля сцен БЕЗ объектов (чистый фон) — у них будут пустые метки, что снижает
        # ложные срабатывания детектора. 0 = выключено.
        "empty_scene_prob": 0.0,
        "dedup_near": True,          # отсекать почти-дубликаты (одинаковый набор значимых слов)
        "categories": {
            "Тип/форм-фактор": [
                "commercial DJI Mavic style quadcopter with folded arms",
                "custom DIY FPV racing drone with visible carbon frame and colorful wires",
                "heavy industrial hexacopter with 6 carbon rotors and large landing gear",
                "agricultural octocopter with 8 motors and dual heavy batteries",
                "fixed-wing hybrid VTOL surveillance drone with aerodynamic wings",
                "tiny micro-whoop drone with full plastic propeller guards",
            ],
            "Материал/текстура": [
                "matte carbon fiber chassis", "glossy white plastic body",
                "3D-printed TPU parts in bright neon orange", "scratched anodized aluminum arms",
                "camo-painted military composite shell", "weathered gray industrial polymer",
            ],
            "Фон/окружение": [
                "dense forest canopy from above", "cloudy grey open sky",
                "urban concrete jungle with skyscrapers", "abandoned factory ruins",
                "green corn field", "rocky mountain range", "snowy winter fields",
                "asphalt airport runway", "brick wall background",
            ],
            "Погода/освещение": [
                "bright sunny day with harsh shadows", "heavy pouring rain with droplets",
                "thick autumn fog, low visibility", "sunset golden hour dramatic lighting",
                "pitch black night with searchlight", "overcast flat diffuse light",
            ],
            "Состояние": [
                "flying steadily mid-air", "fast aggressive maneuver with slight motion blur",
                "crashed on the ground, inverted", "broken arms, exposed wires, shattered props",
                "completely disassembled into parts on the ground",
            ],
            "Ракурс": [
                "eye-level view", "top-down bird-eye view", "low angle looking up",
                "high angle looking down", "three-quarter front view", "side profile view",
                "dynamic tilted dutch angle",
            ],
        },
        # [фраза, вес] — веса это пропорции (суммировать в 100 не обязательно).
        "object_scales": [
            # нижняя граница ~3% — при ресайзе обучения до 416px это ≈12px, минимум для головы NanoDet
            ["the object is small and distant, ~3-6% of the frame, far away, lots of empty scene", 30],
            ["the object is small in the frame, ~10-15%, plenty of surrounding background", 30],
            ["the object is medium-sized, ~30-40% of the frame", 25],
            ["the object is large and close, filling most of the frame, edges may be cropped", 15],
        ],
        # Плейсхолдеры: {batch_size} {object_noun} {config_block}
        "system_template": (
            "Generate a raw JSON array of exactly {batch_size} highly detailed, completely unique "
            "image-generation prompts in English depicting: {object_noun}. "
            "For this specific batch you MUST heavily focus on the following configuration:\n"
            "{config_block}\n"
            "Do NOT describe how large the object appears, its distance, or how much of the frame it "
            "fills — that is appended separately. Focus on the object, the scene, the action and the "
            "lighting. Slightly vary micro-details inside the batch. Return ONLY the raw JSON array of "
            "strings. No markdown, no triple backticks, no explanations."
        ),
    },
    "labeling": {
        "backend": "yoloworld",      # yoloworld | groundingdino | owlv2
        # МУЛЬТИКЛАСС: каждый класс — отдельная группа со своими синонимами и своим id
        # (id = порядок в списке). Разные классы НЕ валятся в одну кучу.
        "classes": [
            {"name": "drone", "synonyms": ["drone", "quadcopter", "uav", "fpv drone",
                                           "hexacopter", "octocopter", "multirotor"]},
        ],
        "conf": 0.05,
        "iou": 0.5,
        "imgsz": 0,                  # разрешение инференса разметки (0=дефолт 640; 1280+ ловит мелочь)
        "batch": 16,                 # картинок за один проход модели разметки (грузим GPU плотно)
        "resume": True,              # пропускать картинки, у которых метка уже есть (докачка после обрыва)
        # веса под каждый бэкенд (скачиваются сами на первом запуске)
        "yoloworld_weights": "yolov8x-worldv2.pt",
        "groundingdino_model": "IDEA-Research/grounding-dino-base",
        "owlv2_model": "google/owlv2-base-patch16-ensemble",
    },
}


# =====================================================================
# ЗАГРУЗКА КОНФИГА
# =====================================================================
def deep_merge(base, override):
    """Рекурсивно накладывает override на копию base."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# Ниже этого порога объект при ресайзе обучения до 416px становится < ~12px — голова NanoDet
# (страйд 8) такие почти не учит. Старые сохранённые конфиги (в т.ч. из GUI) могли иметь
# масштабы ~1-3% / ~2-5% — поднимаем их нижнюю границу автоматически при загрузке.
_SCALE_MIN_PCT = 3


def _bump_scale_phrase(phrase):
    """Если в фразе масштаба нижняя граница процента меньше _SCALE_MIN_PCT — поднимаем её
    (текст в остальном сохраняем). Понимает '~A-B%' и '~A%'. Идемпотентно."""
    s = str(phrase)
    m = re.search(r"~\s*(\d+)\s*-\s*(\d+)\s*%", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a < _SCALE_MIN_PCT:
            nb = max(b, _SCALE_MIN_PCT + 3)
            return s[:m.start()] + f"~{_SCALE_MIN_PCT}-{nb}%" + s[m.end():]
        return s
    m = re.search(r"~\s*(\d+)\s*%", s)
    if m and int(m.group(1)) < _SCALE_MIN_PCT:
        return s[:m.start()] + f"~{_SCALE_MIN_PCT}%" + s[m.end():]
    return s


# Ключи-«коллекции», которые НЕЛЬЗЯ мёржить с дефолтами по-словарно — их задаёт пользователь
# и они должны заменяться ЦЕЛИКОМ (иначе дефолтные дрон-категории подмешиваются обратно).
def merge_user_config(user):
    """deep_merge с DEFAULTS, но prompts.categories заменяется целиком из пользовательского конфига."""
    cfg = deep_merge(DEFAULTS, user or {})
    up = (user or {}).get("prompts", {})
    if isinstance(up, dict) and isinstance(up.get("categories"), dict):
        cfg["prompts"]["categories"] = copy.deepcopy(up["categories"])   # замена, не объединение
    # авто-миграция мелких масштабов из старых сохранённых конфигов под вход 416
    scales = cfg.get("prompts", {}).get("object_scales")
    if isinstance(scales, list):
        new = []
        bumped = False
        for item in scales:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                ph = _bump_scale_phrase(item[0])
                bumped = bumped or (ph != str(item[0]))
                new.append([ph, item[1]])
            else:
                new.append(item)
        cfg["prompts"]["object_scales"] = new
        if bumped:
            print("[КОНФИГ] масштабы объектов подняты до 416-безопасного минимума "
                  f"(нижняя граница >= {_SCALE_MIN_PCT}%).")
    return cfg


def load_config():
    path = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        path = sys.argv[1]
    path = path or os.environ.get("GEN_CONFIG")
    if path:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        print(f"[КОНФИГ] загружен {path}")
        return merge_user_config(user)
    print("[КОНФИГ] встроенные дефолты (файл конфига не задан)")
    return copy.deepcopy(DEFAULTS)


# =====================================================================
# САМО-УСТАНОВКА ЗАВИСИМОСТЕЙ И МОДЕЛЕЙ
# =====================================================================
_AUTO = True   # переопределяется из cfg в main()


def _pip_install(*pkgs):
    """pip install с ретраями — на медленном/рвущемся канале одиночная попытка падает на
    IncompleteRead/Connection broken; pip докачивает кэш колёс, поэтому повтор обычно проходит."""
    print(f"  [setup] устанавливаю: {', '.join(pkgs)} …")
    cmd = [sys.executable, "-m", "pip", "install",
           "--retries", "10", "--timeout", "120", *pkgs]
    for attempt in range(1, 5):
        if subprocess.call(cmd) == 0:
            return True
        print(f"  [setup] установка не удалась (попытка {attempt}/4) — обрыв канала? повтор…")
        time.sleep(2 * attempt)
    print(f"  [setup] не удалось установить {', '.join(pkgs)} за 4 попытки.")
    return False


def ensure(import_name, pip_name=None):
    """Импорт с авто-установкой через pip, если пакета нет (и auto_install включён)."""
    try:
        return __import__(import_name)
    except ImportError:
        if not _AUTO:
            raise
        _pip_install(pip_name or import_name)
        return __import__(import_name)


def _dir_size(path):
    """Суммарный размер файлов в папке (для heartbeat-прогресса скачивания)."""
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def _has_incomplete(flux_dir):
    """Есть ли недокачанные файлы (huggingface_hub оставляет *.incomplete при обрыве)."""
    for root, _, files in os.walk(flux_dir):
        for fn in files:
            if fn.endswith(".incomplete"):
                return True
    return False


def ensure_flux(cfg):
    """Если папки FLUX нет ИЛИ скачана не до конца — (до)качиваем веса с HuggingFace (~24 ГБ).
    Прогресс по размеру идёт в лог; при гейте/недоступности — понятная диагностика."""
    flux_dir = cfg["paths"]["flux_dir"]
    tr = cfg["paths"]["transformer_path"]
    if os.path.isdir(flux_dir) and os.path.isdir(tr) and not _has_incomplete(flux_dir):
        return True
    if os.path.isdir(flux_dir) and _has_incomplete(flux_dir):
        print("[setup] FLUX скачан НЕ полностью (есть .incomplete) — докачиваю недостающее…")
    if not _AUTO:
        print(f"[!] FLUX не найден в {flux_dir}, а авто-скачивание выключено.")
        return False
    ensure("huggingface_hub", "huggingface_hub")
    import huggingface_hub
    from huggingface_hub import snapshot_download
    # Зеркало HF (в части сетей huggingface.co недоступен). Форсим и через env, и через
    # уже импортированную константу — иначе при раннем импорте hub зеркало не подхватится.
    ep = (cfg["generation"].get("hf_endpoint") or "").strip()
    if ep:
        os.environ["HF_ENDPOINT"] = ep
        try:
            huggingface_hub.constants.ENDPOINT = ep
        except Exception:
            pass
    repo = cfg["generation"].get("flux_repo", "black-forest-labs/FLUX.1-schnell")
    token = ((cfg["generation"].get("hf_token") or "").strip()
             or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN") or None)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")   # терпимее к медленному каналу

    def _gated_help():
        print("[!] Репозиторий FLUX закрыт ГЕЙТОМ (GatedRepoError): нужен доступ + токен HF.")
        print("    Что сделать (один раз):")
        print(f"    1) Залогинься на huggingface.co и на странице модели прими условия:")
        print(f"       https://huggingface.co/{repo}  (кнопка Agree / Access repository);")
        print("    2) Создай токен: https://huggingface.co/settings/tokens (тип Read);")
        print("    3) Дай токен генератору одним из способов:")
        print("       • поле «HF токен» на вкладке «Картинки» в окне; ИЛИ")
        print("       • выполни в консоли:  huggingface-cli login   (вставь токен); ИЛИ")
        print("       • переменная окружения HF_TOKEN=hf_xxx;")
        print("    4) Запусти генерацию снова.")
        print("    Альтернатива: переключи «Движок генерации» на «own» — FLUX не нужен, только разметка.")

    # Качаем ТОЛЬКО diffusers-формат (подпапки), пропуская дубль-чекпойнт в корне
    # (flux1-schnell.safetensors ~24 ГБ для ComfyUI) и прочие single-file/оннх — FluxPipeline
    # их не использует, иначе вышло бы ~57 ГБ вместо нужных ~33.
    import fnmatch
    ignore = ["flux1-schnell.safetensors", "ae.safetensors", "*.gguf", "*.onnx"]

    def _ignored(name):
        return any(fnmatch.fnmatch(name, p) for p in ignore)

    # Реальный размер того, что СКАЧАЕМ, для прогресса (tqdm от hub в GUI не виден — он пишет \r).
    total_bytes = 0
    try:
        info = huggingface_hub.HfApi().model_info(repo, files_metadata=True, token=token)
        total_bytes = sum((getattr(s, "size", 0) or 0) for s in (info.siblings or [])
                          if not _ignored(getattr(s, "rfilename", "")))
    except Exception:
        pass

    print(f"[setup] FLUX не найден — качаю {repo} (только нужный diffusers-формат)"
          + (f" (~{total_bytes/1e9:.1f} ГБ)" if total_bytes else " (~33 ГБ)") + ", надолго.")
    print(f"[setup] эндпойнт: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')} | "
          f"токен: {'есть' if token else 'нет'}")
    print("[setup] прогресс печатаю каждые 5с (heartbeat); при обрыве повторяю с ДОКАЧКОЙ…")
    t0 = time.perf_counter()
    last = None

    def _heartbeat(stop_evt):
        while not stop_evt.wait(5):
            got = _dir_size(flux_dir)
            sp = got / max(1e-6, time.perf_counter() - t0) / 1e6   # МБ/с
            if total_bytes:
                eta = (total_bytes - got) / max(1e-6, got / max(1e-6, time.perf_counter() - t0))
                _pb(min(0.999, got / total_bytes), f"FLUX: {got/1e9:.1f}/{total_bytes/1e9:.1f} ГБ")
                print(f"[setup] FLUX: {got/1e9:.2f}/{total_bytes/1e9:.2f} ГБ "
                      f"({100*got/total_bytes:4.1f}%) | {sp:.1f} МБ/с | осталось ~{fmt_hms(eta)}", flush=True)
            else:
                print(f"[setup] FLUX: скачано {got/1e9:.2f} ГБ | {sp:.1f} МБ/с…", flush=True)

    import threading
    for attempt in range(1, 7):
        stop_evt = threading.Event()
        hb = threading.Thread(target=_heartbeat, args=(stop_evt,), daemon=True)
        hb.start()
        try:
            snapshot_download(repo, local_dir=flux_dir, max_workers=8, token=token,
                              ignore_patterns=ignore)
            stop_evt.set()
            if not os.path.isdir(tr):
                cfg["paths"]["transformer_path"] = os.path.join(flux_dir, "transformer")
            print(f"[setup] FLUX скачан в {flux_dir} за {fmt_hms(time.perf_counter()-t0)}")
            return True
        except Exception as e:
            stop_evt.set()
            last = e
            name = type(e).__name__
            # Гейт / отказ авторизации — ретраи бессмысленны, сразу даём инструкцию.
            if "Gated" in name or "401" in str(e) or "403" in str(e) or "Unauthorized" in name:
                _gated_help()
                return False
            print(f"[setup] обрыв скачивания FLUX (попытка {attempt}/6): {name}. Повтор с докачкой…")
            time.sleep(min(30, 3 * attempt))
    print(f"[!] Не удалось скачать FLUX за 6 попыток: {type(last).__name__}: {last}")
    print("    Похоже, рвётся канал на больших файлах. Что помогает:")
    print("    1) Просто ЗАПУСТИ ЕЩЁ РАЗ — докачается с места обрыва (готовые файлы пропускаются);")
    print("    2) Либо скачай FLUX.1-schnell вручную (hf download / git lfs) и укажи путь в «Папка FLUX»")
    print("       и «Папка трансформера»;")
    print("    3) Либо переключи «Движок генерации» на «own» — генерация не нужна, только разметка.")
    return False


# =====================================================================
# ХЕЛПЕРЫ
# =====================================================================
def fmt_hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}ч {m:02d}м {s:02d}с" if h else f"{m:d}м {s:02d}с"


def _pb(frac, text=""):
    """Маркер прогресса для GUI (рисует полосу). В консоли — просто строка."""
    print("@@PB@@\t%.4f\t%s" % (max(0.0, min(1.0, frac)), text), flush=True)


def vram_gb(torch):
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3)


def extract_prompts(raw_text):
    """Стойкий разбор ответа LLM: терпит markdown-заборы, болтовню, хвостовые запятые, обрыв."""
    txt = (raw_text or "").strip()
    # reasoning-модели (Qwen3 и т.п.) могут вернуть <think>…</think> перед ответом — срезаем.
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    start, end = txt.find("["), txt.rfind("]")
    candidate = txt[start:end + 1] if (start != -1 and end > start) else txt
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
    try:
        data = json.loads(candidate)
        if isinstance(data, list) and data:
            return data
    except json.JSONDecodeError:
        pass
    salvaged = []
    for s in re.findall(r'"((?:[^"\\]|\\.)*)"', txt):
        try:
            salvaged.append(json.loads(f'"{s}"'))
        except json.JSONDecodeError:
            salvaged.append(s)
    return salvaged


def normalize_prompt(p):
    if isinstance(p, dict):
        p = p.get("prompt") or p.get("text") or (list(p.values())[0] if p else "")
    return str(p).strip()


class ScalePlanner:
    """Планировщик-дефицит: каждый вызов возвращает фразу масштаба, сильнее всего отстающую
    от своей доли — так пропорции держатся точно при любом размере датасета."""

    def __init__(self, object_scales):
        self.phrases = [p for p, _ in object_scales]
        weights = [max(0.0, float(w)) for _, w in object_scales]
        total = sum(weights) or 1.0
        self.targets = [w / total for w in weights]
        self.counts = [0] * len(self.phrases)

    def next(self):
        total = sum(self.counts) + 1
        deficits = [self.targets[i] * total - self.counts[i] for i in range(len(self.phrases))]
        i = max(range(len(deficits)), key=lambda k: deficits[k])
        self.counts[i] += 1
        return self.phrases[i]

    def seed(self, existing_prompts):
        for p in existing_prompts:
            for i, ph in enumerate(self.phrases):
                if p.endswith(ph):
                    self.counts[i] += 1
                    break


def parse_weighted(items):
    """Список строк «фраза» или «фраза | вес» -> (phrases, weights). Вес необязателен (=1)."""
    phrases, weights = [], []
    for it in items:
        s = str(it).strip()
        if not s or s.startswith("##"):        # пропускаем заголовки категорий, если попали в объекты
            continue
        if "|" in s:
            ph, w = s.rsplit("|", 1)
            try:
                wt = float(w.strip())
            except ValueError:
                ph, wt = s, 1.0
        else:
            ph, wt = s, 1.0
        ph = ph.strip()
        if ph:
            phrases.append(ph); weights.append(max(0.0, wt))
    if not any(weights):
        weights = [1.0] * len(phrases)
    return phrases, weights


def weighted_sample(phrases, weights, k):
    """k различных фраз с учётом весов (без повторов)."""
    pairs = list(zip(phrases, weights))
    out = []
    for _ in range(min(k, len(pairs))):
        i = random.choices(range(len(pairs)), weights=[p[1] for p in pairs], k=1)[0]
        out.append(pairs[i][0]); pairs.pop(i)
    return out


def choose_scene_objects(pr):
    """Выбирает объект(ы) сцены для текущего батча с учётом ВЕСОВ. Иногда (multi_object_prob)
    берёт 2+ разных объекта вместе. Возвращает строку для подстановки в {object_noun}."""
    items = [s for s in (pr.get("objects") or []) if str(s).strip()]
    if not items:
        return pr.get("object_noun", "the target object")
    phrases, weights = parse_weighted(items)
    if not phrases:
        return pr.get("object_noun", "the target object")
    p = float(pr.get("multi_object_prob", 0.0) or 0.0)
    mx = max(2, min(int(pr.get("multi_object_max", 2) or 2), len(phrases)))
    if len(phrases) >= 2 and random.random() < p:
        k = random.randint(2, mx)
        chosen = weighted_sample(phrases, weights, k)
        return (" AND ".join(chosen) +
                " — ALL of them clearly visible together in the SAME single scene")
    return random.choices(phrases, weights=weights, k=1)[0]


def _llm_chat(client, model, messages, max_tokens, temperature, no_think):
    """Вызов чата с НАДЁЖНЫМ отключением мышления: extra_body с флагами разных серверов
    (Qwen3/vLLM/SGLang/LM Studio), с откатом если сервер их не принимает. Текстовый /no_think
    и срез <think> в парсерах дополняют это для любых моделей (вкл. Gemma)."""
    base = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if not no_think:
        return client.chat.completions.create(**base)
    extra = {
        "chat_template_kwargs": {"enable_thinking": False},   # Qwen3 на vLLM/SGLang
        "enable_thinking": False,                             # часть OpenAI-совместимых/LM Studio
        "reasoning_effort": "none",                           # o-style серверы
    }
    try:
        return client.chat.completions.create(extra_body=extra, **base)
    except Exception:
        # сервер отверг неизвестные поля — текстовый /no_think + срез <think> всё равно сработают
        return client.chat.completions.create(**base)


def _parse_json_object(text):
    """Достаёт первый JSON-объект из ответа LLM (терпит markdown-заборы и болтовню)."""
    t = (text or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return None
    chunk = re.sub(r",\s*([\]}])", r"\1", t[s:e + 1])
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None


def autoconfig_from_brief(cfg):
    """Разворачивает короткое описание (prompts.brief) в поля промптов и классов через LLM.
    Заполняет object_noun, objects, categories и labeling.classes."""
    brief = (cfg["prompts"].get("brief") or "").strip()
    if not brief:
        print("[autoconfig] поле описания пустое — заполнять нечего.")
        return cfg
    OpenAI = ensure("openai", "openai").OpenAI
    model = cfg["llm"]["model"]
    use_lms = cfg["llm"].get("use_lms", True)
    if use_lms:
        print(f"[autoconfig] загружаю LLM {model}…")
        subprocess.run(f"lms load {model}", shell=True, stdout=subprocess.DEVNULL)
    client = OpenAI(base_url=cfg["llm"]["base_url"], api_key=cfg["llm"]["api_key"])
    sys_prompt = (
        "Ты — помощник по проектированию датасетов для детекции объектов. По краткому описанию "
        "верни ТОЛЬКО один JSON-объект (без markdown, без пояснений) со схемой:\n"
        '{\n'
        '  "object_noun": "короткая фраза, что снимаем в целом",\n'
        '  "objects": ["english phrase per distinct object, e.g. a person", "a car"],\n'
        '  "categories": {"Тип/форм-фактор": ["...english..."], "Материал/текстура": ["..."], '
        '"Фон/окружение": ["..."], "Погода/освещение": ["..."], "Состояние": ["..."], "Ракурс": ["..."]},\n'
        '  "classes": [{"name": "person", "synonyms": ["person","pedestrian","human"]}, '
        '{"name": "car", "synonyms": ["car","vehicle","automobile"]}]\n'
        '}\n'
        "Значения в objects/categories — на АНГЛИЙСКОМ (для модели картинок). Имена классов — как удобно. "
        "Категорий 4-7, в каждой 5-10 разнообразных вариантов, отражающих описание (например ракурс "
        "'top-down drone view', если сказано 'вид с дрона'). Классы — реальные объекты для разметки.\n"
        f"Описание: {brief}"
    )
    no_think = cfg["llm"].get("no_think", True)
    if no_think:
        sys_prompt += "\n/no_think"
    print("[autoconfig] прошу LLM собрать настройки из описания…")
    try:
        resp = _llm_chat(
            client, model,
            [{"role": "system", "content": "Ты возвращаешь ТОЛЬКО один JSON-объект, без markdown и размышлений."},
             {"role": "user", "content": sys_prompt}],
            cfg["llm"]["max_tokens"], 0.5, no_think)
        data = _parse_json_object(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"[autoconfig] ошибка LLM: {e}")
        data = None
    if use_lms:
        subprocess.run("lms unload --all", shell=True, stdout=subprocess.DEVNULL)
    if not data:
        print("[autoconfig] не удалось разобрать ответ LLM — поля не изменены.")
        return cfg

    pr = cfg["prompts"]
    if data.get("object_noun"):
        pr["object_noun"] = str(data["object_noun"])
    if isinstance(data.get("objects"), list) and data["objects"]:
        pr["objects"] = [str(x).strip() for x in data["objects"] if str(x).strip()]
    if isinstance(data.get("categories"), dict) and data["categories"]:
        pr["categories"] = {str(k): [str(v) for v in vals if str(v).strip()]
                            for k, vals in data["categories"].items() if vals}
    norm = []
    for c in (data.get("classes") or []):
        if isinstance(c, dict) and c.get("name"):
            syns = [str(s).strip() for s in (c.get("synonyms") or [c["name"]]) if str(s).strip()]
            norm.append({"name": str(c["name"]).strip(), "synonyms": syns or [str(c["name"]).strip()]})
        elif isinstance(c, str) and c.strip():
            norm.append({"name": c.strip(), "synonyms": [c.strip()]})
    if norm:
        cfg["labeling"]["classes"] = norm
    # если объектов 2+, а мультисцены были выключены — включим разумную долю
    if len(pr.get("objects", [])) >= 2 and not float(pr.get("multi_object_prob", 0) or 0):
        pr["multi_object_prob"] = 0.3

    # Аэро-съёмка: если описание про дрон/вид сверху/с высоты — ДЕТЕРМИНИРОВАННО ставим
    # вид сверху на КАЖДОМ кадре (ракурс + суффикс) и мелкий/далёкий масштаб объектов
    # (иначе FLUX по умолчанию снимает сбоку и крупно).
    blow = brief.lower()
    aerial = any(k in blow for k in (
        "дрон", "drone", "uav", "сверху", "с высоты", "высоты", "с воздуха", "воздуха",
        "aerial", "top-down", "top down", "overhead", "bird", "птичьего", "квадрокоптер"))
    if aerial:
        cats = pr.setdefault("categories", {})
        # убираем любые «ракурсные» категории (как бы их ни назвал LLM) и ставим одну аэро
        for k in [k for k in cats if any(t in k.lower()
                  for t in ("ракурс", "angle", "perspective", "camera", "view", "ракурс"))]:
            del cats[k]
        cats["Ракурс (вид сверху)"] = [
            "top-down bird's-eye view looking straight down (near-nadir)",
            "high-altitude aerial drone view from far above",
            "overhead drone shot, camera pointing down at the ground",
            "oblique aerial view from a high-flying drone",
        ]
        cfg["generation"]["prompt_suffix"] = (
            ", aerial top-down drone photo taken from high altitude, looking down at the ground, "
            "objects appear small and far below, overhead perspective, realistic photo, high resolution")
        # Нижняя граница ~4%: при рендере 640px и ресайзе обучения до 416 это ≈12px —
        # минимум, который голова NanoDet (страйд 8, вход 416) ещё уверенно учит/детектит.
        # Мельче делать нельзя: метки станут нетренируемыми «пикселями».
        pr["object_scales"] = [
            ["the object is small, seen from high above, ~4-8% of the frame, lots of ground around it", 45],
            ["the object is small-to-medium, seen from above, ~8-15% of the frame, surrounded by ground", 40],
            ["the object is medium-small from an overhead view, ~15-25% of the frame", 15],
        ]
        cfg["labeling"]["imgsz"] = 1280   # мелкие объекты сверху — размечаем в высоком разрешении
        print("[autoconfig] описание про съёмку с высоты — выставил вид сверху + мелкий масштаб + разметку 1280px.")

    print("[autoconfig] готово:")
    print("   object_noun:", pr.get("object_noun"))
    print("   объекты:", pr.get("objects"))
    print("   категории:", list(pr.get("categories", {}).keys()))
    print("   классы:", [c["name"] for c in cfg["labeling"]["classes"]])
    return cfg


def build_system_prompt(cfg, empty=False):
    """Системный промпт. empty=True -> просим ФОНОВУЮ сцену без объектов (негатив)."""
    pr = cfg["prompts"]
    cats = pr.get("categories", {})
    bs = cfg["generation"]["batch_size"]
    if empty:
        objs = parse_weighted(pr.get("objects") or [])[0] or [pr.get("object_noun", "the target object")]
        bg = []
        for name, options in cats.items():
            low = name.lower()
            if options and any(k in low for k in ("фон", "окруж", "background", "environment",
                                                  "погод", "свет", "weather", "light",
                                                  "ракурс", "perspective", "angle")):
                bg.append(f"- {name}: {random.choice(list(options))}")
        block = ("\nДля этого батча:\n" + "\n".join(bg)) if bg else ""
        return (f"Generate a raw JSON array of exactly {bs} highly detailed, unique English "
                f"image-generation prompts of realistic BACKGROUND / EMPTY scenes that contain "
                f"NO {', '.join(objs)} and no similar objects at all — only the environment."
                f"{block}\nVary places, weather, lighting and viewpoint. Return ONLY a raw JSON "
                f"array of strings. No markdown, no explanations.")
    lines = [f"- {name}: {random.choice(list(options))}" for name, options in cats.items() if options]
    template = pr.get("system_template", "") or ""
    # Авто-миграция: старый дрон-шаблон (с {drone_type} и т.п.) не содержит новых плейсхолдеров —
    # берём актуальный универсальный, иначе .format упал бы на KeyError('drone_type').
    if "{config_block}" not in template or "{object_noun}" not in template:
        template = DEFAULTS["prompts"]["system_template"]
    # format_map с «безопасным» словарём: любые посторонние плейсхолдеры -> пустая строка,
    # чтобы кастомный шаблон пользователя никогда не валил генерацию.
    class _Safe(dict):
        def __missing__(self, k):
            return ""
    return template.format_map(_Safe(
        batch_size=bs, object_noun=choose_scene_objects(pr), config_block="\n".join(lines)))


# =====================================================================
# ФАЗА 1 — ПРОМПТЫ (LLM загружается один раз)
# =====================================================================
def _valid_prompt(p):
    """Отсев мусора: артефакты редактора категорий (## …, - …), эхо инструкции, JSON, многострочное."""
    s = (p or "").strip()
    if len(s) < 12 or "\n" in s or "\r" in s:
        return False
    if s.startswith(("##", "- ", "•", "*", "{", "[", "}")):
        return False
    if "## " in s:
        return False
    low = s.lower()
    if any(b in low for b in ("json array", "raw json", "return only", "no markdown")):
        return False
    return True


def _norm_sig(s):
    """Сигнатура почти-дубля: множество значимых слов (без регистра/пунктуации/коротких слов).
    Два промпта с ОДИНАКОВЫМ набором значимых слов считаем одинаковыми (порядок/пунктуация/
    мелкие слова игнорируются). Дёшево и масштабируемо (O(1) на промпт)."""
    s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
    return frozenset(w for w in s.split() if len(w) >= 4)


def load_existing_prompts(prompts_file):
    out = []
    seen = set()
    if os.path.exists(prompts_file):
        with open(prompts_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = normalize_prompt(p)
                p = _bump_scale_phrase(p)                   # лечим вшитый мелкий масштаб под 416
                if _valid_prompt(p) and p not in seen:     # чистим мусор и дубли
                    seen.add(p); out.append(p)
    return out


def generate_all_prompts(cfg):
    OpenAI = ensure("openai", "openai").OpenAI

    total = cfg["total_images"]
    prompts_file = cfg["paths"]["prompts_file"]
    planner = ScalePlanner(cfg["prompts"]["object_scales"])

    all_prompts = load_existing_prompts(prompts_file)
    # Перезаписываем файл вычищенным списком (мусор/дубли + поднятый масштаб — навсегда).
    try:
        raw = []
        if os.path.exists(prompts_file):
            with open(prompts_file, encoding="utf-8") as rf:
                for ln in rf:
                    ln = ln.strip()
                    if ln:
                        try:
                            raw.append(json.loads(ln))
                        except json.JSONDecodeError:
                            raw.append(None)               # мусор -> точно перезапишем
        if raw != all_prompts:                             # изменился состав/текст -> переписать
            with open(prompts_file, "w", encoding="utf-8") as wf:
                for p in all_prompts:
                    wf.write(json.dumps(p, ensure_ascii=False) + "\n")
            print(f"[ФАЗА 1] почищен prompts.jsonl: было строк {len(raw)}, "
                  f"валидных уникальных {len(all_prompts)} (мусор/дубли убраны, масштаб подтянут под 416).")
    except Exception as e:
        print(f"[ФАЗА 1] не смог переписать prompts.jsonl: {e}")
    planner.seed(all_prompts)
    seen = set(all_prompts)
    near_dedup = bool(cfg["prompts"].get("dedup_near", True))
    sigs = set(_norm_sig(p) for p in all_prompts) if near_dedup else set()
    if len(all_prompts) >= total:
        print(f"\n[ФАЗА 1] Промпты уже готовы: {len(all_prompts)} >= {total}. Пропускаю LLM.")
        return all_prompts

    model = cfg["llm"]["model"]
    use_lms = cfg["llm"].get("use_lms", True)
    print(f"\n[ФАЗА 1] Есть {len(all_prompts)} промптов, нужно {total}.")
    if use_lms:
        print(f" -> Загружаю LLM {model} через LM Studio CLI…")
        subprocess.run(f"lms load {model}", shell=True, stdout=subprocess.DEVNULL)

    client = OpenAI(base_url=cfg["llm"]["base_url"], api_key=cfg["llm"]["api_key"])
    t0 = time.perf_counter()
    consecutive_fail = 0
    with open(prompts_file, "a", encoding="utf-8") as f:
        empty_prob = float(cfg["prompts"].get("empty_scene_prob", 0) or 0)
        while len(all_prompts) < total:
            try:
                empty = empty_prob > 0 and random.random() < empty_prob
                no_think = cfg["llm"].get("no_think", True)
                instr = build_system_prompt(cfg, empty=empty)
                if no_think:
                    instr += "\n/no_think"            # текстовый переключатель (Qwen3 и пр.)
                resp = _llm_chat(
                    client, model,
                    # отдельный user-ход надёжнее: на «только system» некоторые модели молчат
                    [{"role": "system",
                      "content": "You output ONLY a raw JSON array of image-prompt strings. No prose, no markdown, no thinking."},
                     {"role": "user", "content": instr}],
                    cfg["llm"]["max_tokens"], cfg["llm"]["temperature"], no_think)
                raw = (resp.choices[0].message.content or "").strip()
                valid = [p for p in (normalize_prompt(x) for x in extract_prompts(raw)) if _valid_prompt(p)]
                # негативам масштаб объекта не добавляем (объекта в кадре нет)
                if not empty:
                    valid = [f"{p}, {planner.next()}" for p in valid]
                # точный + почти-дубль дедуп (LLM часто повторяется)
                batch = []
                for p in valid:
                    if p in seen:
                        continue
                    if near_dedup:
                        sig = _norm_sig(p)
                        if sig in sigs:
                            continue
                        sigs.add(sig)
                    seen.add(p); batch.append(p)

                if not valid:                       # модель реально ничего не выдала
                    consecutive_fail += 1
                    print(f"   [!] Пустой/битый ответ LLM ({consecutive_fail} подряд). Сырое: {raw[:150]!r}")
                    if consecutive_fail >= 8 and use_lms:
                        print("   [!] Слишком много пустых ответов. Перезагружаю LLM…")
                        subprocess.run("lms unload --all", shell=True, stdout=subprocess.DEVNULL)
                        time.sleep(2)
                        subprocess.run(f"lms load {model}", shell=True, stdout=subprocess.DEVNULL)
                        consecutive_fail = 0
                    continue
                if not batch:                       # выдала, но всё — дубли (не вина модели)
                    print("   все промпты батча оказались дублями — продолжаю…")
                    continue

                consecutive_fail = 0
                for p in batch:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                f.flush()
                all_prompts.extend(batch)

                done = len(all_prompts)
                elapsed = time.perf_counter() - t0
                rate = done / max(1e-6, elapsed)
                eta = (total - done) / max(1e-6, rate)
                _pb(done / total, f"Фаза 1 — промпты: {done}/{total}")
                print(f"   [ФАЗА 1] промпты {done}/{total} (+{len(batch)}) | "
                      f"{rate*60:5.1f}/мин | осталось ~{fmt_hms(eta)}")
            except Exception as e:
                consecutive_fail += 1
                print(f"   [!] Ошибка LLM: {e}. Пауза 3с…")
                time.sleep(3)
                continue

    if use_lms:
        print(" -> Выгружаю LLM, освобождаю VRAM…")
        subprocess.run("lms unload --all", shell=True, stdout=subprocess.DEVNULL)
    print(f"[ФАЗА 1] Готово: {len(all_prompts)} промптов сохранено в {prompts_file}")
    return all_prompts


# =====================================================================
# ФАЗА 2 — КАРТИНКИ (FLUX)
# =====================================================================
def build_fast_transformer(cfg, torch):
    """Грузит трансформер в выбранном режиме квантования сразу в cuda. При ошибке —
    откат на layerwise fp8. Возвращает (transformer, имя_режима)."""
    from diffusers import FluxTransformer2DModel
    mode = cfg["generation"]["quant_mode"]
    path = cfg["paths"]["transformer_path"]

    if mode == "nf4":
        try:
            from diffusers import BitsAndBytesConfig
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16)
            tr = FluxTransformer2DModel.from_pretrained(path, quantization_config=qcfg,
                                                        torch_dtype=torch.bfloat16)
            try:
                tr = tr.to("cuda")
            except Exception:
                pass
            return tr, "NF4 4-bit (bitsandbytes, ~7 ГБ)"
        except Exception as e:
            print(f"    [QUANT] NF4 недоступен ({type(e).__name__}: {e}). Откат на layerwise.")
    elif mode == "torchao":
        try:
            from torchao.quantization import quantize_
            try:
                from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
                qcfg = Float8DynamicActivationFloat8WeightConfig()
            except Exception:
                from torchao.quantization import float8_dynamic_activation_float8_weight
                qcfg = float8_dynamic_activation_float8_weight()
            tr = FluxTransformer2DModel.from_pretrained(path, torch_dtype=torch.bfloat16)
            quantize_(tr, qcfg)
            tr = tr.to("cuda")
            return tr, "torchao fp8 (нативный fp8 matmul)"
        except Exception as e:
            print(f"    [QUANT] torchao недоступен ({type(e).__name__}: {e}). Откат на layerwise.")

    tr = FluxTransformer2DModel.from_pretrained(path, torch_dtype=torch.bfloat16)
    tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    tr = tr.to("cuda")
    return tr, "layerwise fp8 (медленно, может не влезть в 16 ГБ)"


def encode_chunk(cfg, torch, pipe, chunk_prompts):
    """Кодирует промпты под-батчами; эмбеддинги держим на CPU, освобождая VRAM под трансформер."""
    enc_batch = cfg["generation"]["encode_batch"]
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    pe_parts, pp_parts = [], []
    total = len(chunk_prompts)
    t0 = time.perf_counter()
    with torch.no_grad():
        for j in range(0, total, enc_batch):
            sub = chunk_prompts[j:j + enc_batch]
            ti = pipe.tokenizer(sub, padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt").to("cuda")
            pooled = pipe.text_encoder(ti.input_ids, output_hidden_states=False).pooler_output
            ti2 = pipe.tokenizer_2(sub, padding="max_length", max_length=256,
                                   truncation=True, return_tensors="pt").to("cuda")
            emb = pipe.text_encoder_2(ti2.input_ids, output_hidden_states=False).last_hidden_state
            pe_parts.append(emb.detach().to("cpu", dtype=torch.bfloat16))
            pp_parts.append(pooled.detach().to("cpu", dtype=torch.bfloat16))
            done = min(j + enc_batch, total)
            if done == total or (j // enc_batch) % 4 == 0:
                eta = (time.perf_counter() - t0) / done * (total - done)
                print(f"    [энкод] {done}/{total} промптов закодировано (осталось ~{fmt_hms(eta)})", flush=True)
    prompt_embeds = torch.cat(pe_parts)
    pooled_embeds = torch.cat(pp_parts)
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    gc.collect()
    torch.cuda.empty_cache()
    return prompt_embeds, pooled_embeds


def _save_jpg(img, path, quality):
    img.save(path, quality=quality)


def generate_all_images(cfg, prompts):
    from concurrent.futures import ThreadPoolExecutor
    ensure("torch", "torch")
    import torch
    # FluxPipeline тянет за собой text-энкодеры -> нужны transformers + токенайзер T5
    # (sentencepiece/protobuf) + accelerate. Ставим заранее, иначе падает на from_pretrained.
    for imp, pip in (("transformers", "transformers"), ("accelerate", "accelerate"),
                     ("sentencepiece", "sentencepiece"), ("google.protobuf", "protobuf")):
        ensure(imp, pip)
    ensure("diffusers", "diffusers")
    from diffusers import FluxPipeline
    if not ensure_flux(cfg):
        print("[ФАЗА 2] FLUX недоступен — пропускаю генерацию.")
        return 0

    g = cfg["generation"]
    # Сохранение JPEG — в пуле потоков, чтобы GPU не ждал диск (CPU Ultra 9 берёт это на себя).
    save_pool = ThreadPoolExecutor(max_workers=int(g.get("save_workers", 8)))
    total = cfg["total_images"]
    images_dir = cfg["paths"]["images_dir"]
    flux_dir = cfg["paths"]["flux_dir"]
    prefix = g.get("file_prefix", "synth")
    pad = max(5, len(str(max(0, total - 1))))
    super_chunk = min(int(g["super_chunk"]), total)
    micro_batch = int(g["micro_batch"])
    suffix = g["prompt_suffix"]

    prompts = [normalize_prompt(p) for p in prompts[:total]]
    limit = min(len(prompts), total)
    # Резюме по ИНДЕКСАМ: парсим уже готовые файлы и генерим только недостающие — так дозаполнение
    # корректно закрывает любые «дыры» после обрыва (а не просто продолжает с конца).
    existing_idx = set()
    pat = re.compile(re.escape(prefix) + r"_(\d+)\.jpg$")
    for fn in os.listdir(images_dir):
        m = pat.match(fn)
        if m:
            existing_idx.add(int(m.group(1)))
    todo = [i for i in range(limit) if i not in existing_idx]
    done_total = len(existing_idx)
    print(f"\n[ФАЗА 2] Картинок готово: {done_total}/{total}; докачать: {len(todo)}")
    if not todo:
        print("[ФАЗА 2] Всё уже сгенерировано.")
        save_pool.shutdown(wait=True)
        return done_total

    session_start = time.perf_counter()
    session_done = 0

    for c in range(0, len(todo), super_chunk):
        chunk_idx = todo[c:c + super_chunk]                       # глобальные индексы этого чанка
        chunk_final = [prompts[i] + suffix for i in chunk_idx]

        print(f"\n=== [ЧАНК {c}..{c + len(chunk_idx) - 1} из недостающих, "
              f"индексы {chunk_idx[0]}..{chunk_idx[-1]}] ===")
        print(f" -> Загружаю пайплайн (без трансформера), кодирую {len(chunk_final)} промптов…")
        try:
            pipe = FluxPipeline.from_pretrained(flux_dir, transformer=None, torch_dtype=torch.bfloat16)
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            pipe.set_progress_bar_config(disable=True)

            prompt_embeds, pooled_embeds = encode_chunk(cfg, torch, pipe, chunk_final)
            a, r = vram_gb(torch)
            print(f"    [VRAM] после кодирования + выгрузки энкодеров: занято {a:.2f} / резерв {r:.2f} ГБ")

            print(" -> Поднимаю квантованный трансформер (раз на чанк)…")
            transformer, quant_mode = build_fast_transformer(cfg, torch)
            pipe.transformer = transformer
            pipe.vae.to("cuda")
            gc.collect()
            torch.cuda.empty_cache()
            a, r = vram_gb(torch)
            print(f"    [QUANT] {quant_mode} | exec={pipe._execution_device} | VRAM {a:.1f}/{r:.1f} ГБ")
        except Exception as e:
            print(f"[!] Ошибка инициализации FLUX на чанке {c}: {e}. Пропускаю чанк.")
            torch.cuda.empty_cache()
            continue

        torch.cuda.reset_peak_memory_stats()

        for start in range(0, len(chunk_final), micro_batch):
            end = min(start + micro_batch, len(chunk_final))
            cur = end - start
            try:
                t_mb = time.perf_counter()
                images = pipe(
                    prompt_embeds=prompt_embeds[start:end].to("cuda", dtype=torch.bfloat16),
                    pooled_prompt_embeds=pooled_embeds[start:end].to("cuda", dtype=torch.bfloat16),
                    num_inference_steps=int(g["num_inference_steps"]),
                    guidance_scale=float(g["guidance_scale"]),
                    width=int(g["width"]),
                    height=int(g["height"]),
                ).images

                for k, img in enumerate(images):
                    gi = chunk_idx[start + k]                     # настоящий глобальный индекс
                    save_pool.submit(_save_jpg, img,
                                     os.path.join(images_dir, f"{prefix}_{gi:0{pad}d}.jpg"),
                                     int(g["jpeg_quality"]))
                    done_total += 1
                    session_done += 1

                dt = time.perf_counter() - t_mb
                sess = time.perf_counter() - session_start
                avg = sess / max(1, session_done)
                eta = avg * (total - done_total)
                _, rr = vram_gb(torch)
                _pb(done_total / total, f"Фаза 2 — картинки: {done_total}/{total}")
                print(f"    [+] {done_total:05d}/{total} | +{cur} за {dt:4.1f}с "
                      f"({dt / cur:4.1f}с/шт) | сред {avg:4.1f}с/шт | VRAM {rr:4.1f}ГБ | осталось ~{fmt_hms(eta)}")
            except torch.cuda.OutOfMemoryError:
                print(f"    [!] CUDA OOM на {start}-{end}. Уменьши micro_batch (сейчас {micro_batch}).")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"    [!] Ошибка микро-батча {start}-{end}: {e}")
                torch.cuda.empty_cache()
                continue

        peak = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        print(f" -> Чанк готов. Пик VRAM {peak:.2f} ГБ. Всего готово: {done_total}/{total}")

        del transformer
        pipe.transformer = None
        del pipe, prompt_embeds, pooled_embeds
        gc.collect()
        torch.cuda.empty_cache()

    save_pool.shutdown(wait=True)      # дождаться, пока все JPEG допишутся на диск
    print(f"\n[ФАЗА 2] Готово: {done_total}/{total} картинок.")
    return done_total


# =====================================================================
# ФАЗА 3 — АВТОРАЗМЕТКА (выбор бэкенда)
# =====================================================================
def _list_jpgs(images_dir):
    return sorted([fn for fn in os.listdir(images_dir) if fn.lower().endswith((".jpg", ".jpeg", ".png"))])


def resolve_classes(lab):
    """Разворачивает мультиклассовую разметку в (flat, syn2id, names):
      flat   — плоский список ВСЕХ синонимов (это словарь для open-vocab модели);
      syn2id — для каждого синонима его id класса (параллельно flat);
      names  — имена классов в порядке id (для dataset.yaml).
    Понимает и новый формат [{name, synonyms}], и старый плоский [синонимы] (1 класс)."""
    raw = lab.get("classes", [])
    flat, syn2id, names = [], [], []
    if raw and isinstance(raw[0], dict):                 # новый: список классов
        for cid, c in enumerate(raw):
            nm = (c.get("name") or f"class{cid}").strip()
            names.append(nm)
            for s in (c.get("synonyms") or [nm]):
                s = str(s).strip()
                if s:
                    flat.append(s); syn2id.append(cid)
    else:                                                # старый: плоский список -> один класс
        names = lab.get("class_names") or ["object"]
        for s in raw:
            s = str(s).strip()
            if s:
                flat.append(s); syn2id.append(0)
    if not names:
        names = ["object"]
    if not flat:                                         # на всякий случай — имена как синонимы
        flat = list(names); syn2id = list(range(len(names)))
    return flat, syn2id, names


def _match_label_id(text, flat, syn2id):
    """Текстовую метку (Grounding DINO) -> id класса по вхождению синонима."""
    t = str(text).lower()
    for i, s in enumerate(flat):
        if s.lower() in t or t in s.lower():
            return syn2id[i]
    return 0


def _write_label(labels_dir, stem, boxes):
    """boxes — список (cx,cy,w,h,cls_id) нормированных. Пишет YOLO-txt."""
    lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (cx, cy, w, h, cid) in boxes]
    with open(os.path.join(labels_dir, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(lines)


def _link_or_copy(src, dst):
    """Жёсткая ссылка (без дублирования места), с откатом на копию (другой том / Windows / FS без хардлинков)."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        import shutil
        shutil.copy2(src, dst)


def _write_yaml(cfg, names, images):
    """Материализует датасет в раскладке, которую напрямую ест train_nanodet.py:

        <output_yolo_dir>/
            images/train/*.jpg   images/val/*.jpg
            labels/train/*.txt   labels/val/*.txt
            dataset.yaml

    Картинки/метки кладутся ЖЁСТКИМИ ССЫЛКАМИ на рабочую плоскую папку (images_dir + labels/),
    поэтому место не дублируется, а рабочая папка остаётся для дозапуска/возобновления.
    Сплит детерминированный (seed 1337). train_nanodet.py читает images/<split> и labels/<split>,
    а dataset.yaml — для ultralytics/совместимости."""
    import random as _r
    images_dir = cfg["paths"]["images_dir"]
    labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
    out_dir = cfg["paths"]["output_yolo_dir"]
    val_split = float(cfg.get("val_split", 0.0) or 0.0)

    # размечаем только то, у чего реально есть .txt (пустой = «фон/негатив», тоже валиден)
    labeled = [fn for fn in images
               if os.path.isfile(os.path.join(labels_dir, os.path.splitext(fn)[0] + ".txt"))]
    if not labeled:
        labeled = list(images)

    order = list(labeled); _r.Random(1337).shuffle(order)
    if val_split > 0 and len(order) > 1:
        k = max(1, int(len(order) * val_split))
    else:
        k = 0
    val = set(order[:k])

    splits = {"train": [fn for fn in labeled if fn not in val],
              "val":   [fn for fn in labeled if fn in val]}
    if not splits["val"]:                       # без сплита: val=train, чтобы трейнеру было что валидировать
        splits["val"] = splits["train"]

    for split, files in (("train", splits["train"]), ("val", splits["val"])):
        img_out = os.path.join(out_dir, "images", split)
        lab_out = os.path.join(out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lab_out, exist_ok=True)
        for fn in files:
            stem = os.path.splitext(fn)[0]
            _link_or_copy(os.path.join(images_dir, fn), os.path.join(img_out, fn))
            src_lbl = os.path.join(labels_dir, stem + ".txt")
            if os.path.isfile(src_lbl):
                _link_or_copy(src_lbl, os.path.join(lab_out, stem + ".txt"))
            else:                                # негатив без меток — пустой .txt
                open(os.path.join(lab_out, stem + ".txt"), "w").close()

    n_tr = len(set(splits["train"])); n_va = len(set(splits["val"]))
    print(f"  раскладка для обучения: {out_dir}\\images\\{{train,val}} + labels\\{{train,val}}")
    print(f"  сплит: train {n_tr} / val {n_va}"
          + ("  (val=train — сплит выключен)" if k == 0 else f"  ({int(val_split*100)}% в val)"))

    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Датасет детекции — авторазметка генератора (раскладка под NanoDet/YOLO)\n")
        f.write(f"path: {out_dir}\n")
        f.write("train: images/train\nval: images/val\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names: [%s]\n" % ", ".join(f"'{n}'" for n in names))
    return yaml_path


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _label_loop(cfg, predict_batch, banner, names):
    """Батчевый цикл разметки (грузит GPU плотно): predict_batch(paths, sizes) -> список
    результатов параллельно paths, каждый — список (cx,cy,w,h,cls_id) нормированных."""
    from PIL import Image
    images_dir = cfg["paths"]["images_dir"]
    lab = cfg["labeling"]
    batch = max(1, int(lab.get("batch", 16)))
    labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
    os.makedirs(labels_dir, exist_ok=True)

    all_images = _list_jpgs(images_dir)
    total = len(all_images)
    # Возобновление: пропускаем картинки, у которых метка уже есть (включая пустую — это
    # валидный «размечено, объектов нет»). Выключается labeling.resume=false (перелейблить всё).
    resume = str(lab.get("resume", True)).lower() not in ("0", "false", "no", "off")
    if resume:
        images = [fn for fn in all_images
                  if not os.path.isfile(os.path.join(labels_dir, os.path.splitext(fn)[0] + ".txt"))]
    else:
        images = all_images
    skipped = total - len(images)
    print(f" -> {banner}: {total} картинок (уже размечено и пропущено: {skipped}), "
          f"классов: {len(names)} ({', '.join(names)}) (conf={lab['conf']}, батч={batch})…")
    detected = empty = total_boxes = done = 0
    per_class = [0] * len(names)
    t0 = time.perf_counter()
    for group in _chunks(images, batch):
        paths = [os.path.join(images_dir, fn) for fn in group]
        sizes = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
            except Exception:
                sizes.append((0, 0))
        try:
            results = predict_batch(paths, sizes)
        except Exception as e:
            print(f"    [!] батч: {e}")
            results = [[] for _ in paths]
        for fn, boxes in zip(group, results):
            n = _write_label(labels_dir, os.path.splitext(fn)[0], boxes)
            for (_, _, _, _, cid) in boxes:
                if 0 <= cid < len(per_class):
                    per_class[cid] += 1
            if n:
                detected += 1; total_boxes += n
            else:
                empty += 1
            done += 1
        shown = skipped + done
        rem = len(images) - done
        eta = (time.perf_counter() - t0) / max(1, done) * rem
        _pb(shown / max(1, total), f"Фаза 3 — разметка: {shown}/{total}")
        print(f"  [{shown}/{total}] (сессия {done}/{len(images)}) с рамками: {detected} | "
              f"пустых: {empty} | боксов: {total_boxes} | осталось ~{fmt_hms(eta)}")

    yaml_path = _write_yaml(cfg, names, all_images)        # сплит/yaml — по ВСЕМ картинкам
    print(f"\n{'='*60}")
    print(f"[ФАЗА 3] Разметка завершена за {fmt_hms(time.perf_counter()-t0)}")
    print(f"  обработано в этой сессии: {len(images)} (пропущено готовых: {skipped}) из {total}")
    print(f"  с рамками: {detected}    пустых меток: {empty}    боксов: {total_boxes}")
    print("  по классам (сессия): " + ", ".join(f"{names[c]}={per_class[c]}" for c in range(len(names))))
    print(f"  метки: {labels_dir}\n  dataset.yaml: {yaml_path}")
    print(f"{'='*60}")


def label_yoloworld(cfg):
    """YOLO-World (ultralytics) — быстрый open-vocab. Веса качаются на первом запуске."""
    ensure("ultralytics", "ultralytics")
    from ultralytics import YOLO
    lab = cfg["labeling"]
    flat, syn2id, names = resolve_classes(lab)
    print(f" -> Загружаю YOLO-World ({lab['yoloworld_weights']})…")
    model = YOLO(lab["yoloworld_weights"])
    model.set_classes(flat)                              # словарь = все синонимы всех классов
    conf, iou = float(lab["conf"]), float(lab["iou"])
    imgsz = int(lab.get("imgsz", 0) or 0)               # 0 = дефолт модели (640); больше = лучше мелочь
    extra = {"imgsz": imgsz} if imgsz > 0 else {}
    if imgsz > 0:
        print(f"    (инференс на {imgsz}px — для мелких объектов)")

    def predict_batch(paths, sizes):
        res = model.predict(source=paths, conf=conf, iou=iou, verbose=False, save=False,
                            batch=len(paths), **extra)
        out = []
        for r in res:
            b = []
            if r.boxes is not None and len(r.boxes):
                for (cx, cy, w, h), c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist()):
                    ci = int(c)
                    b.append((cx, cy, w, h, syn2id[ci] if 0 <= ci < len(syn2id) else 0))
            out.append(b)
        return out
    _label_loop(cfg, predict_batch, "YOLO-World разметка", names)


def label_groundingdino(cfg):
    """Grounding DINO (transformers) — точнее на редких/необычных классах."""
    ensure("torch", "torch"); ensure("transformers", "transformers")
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    lab = cfg["labeling"]
    flat, syn2id, names = resolve_classes(lab)
    name = lab.get("groundingdino_model", "IDEA-Research/grounding-dino-base")
    print(f" -> Загружаю Grounding DINO ({name})…")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(dev)
    text = ". ".join(s.lower() for s in flat) + "."
    conf = float(lab["conf"])
    from PIL import Image

    def predict_batch(paths, sizes):
        imgs = [Image.open(p).convert("RGB") for p in paths]
        inp = proc(images=imgs, text=[text] * len(imgs), return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = model(**inp)
        targets = [(h, w) for (w, h) in sizes]          # post_process ждёт (H, W)
        res = proc.post_process_grounded_object_detection(
            out, inp["input_ids"], box_threshold=conf, text_threshold=conf, target_sizes=targets)
        out_boxes = []
        for r, (W, H) in zip(res, sizes):
            labels = r.get("labels") or r.get("text_labels") or [""] * len(r["boxes"])
            b = []
            if W > 0 and H > 0:
                for (x0, y0, x1, y1), lab_txt in zip(r["boxes"].tolist(), labels):
                    cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
                    bw, bh = (x1 - x0) / W, (y1 - y0) / H
                    if bw > 0 and bh > 0:
                        b.append((cx, cy, bw, bh, _match_label_id(lab_txt, flat, syn2id)))
            out_boxes.append(b)
        return out_boxes
    _label_loop(cfg, predict_batch, "Grounding DINO разметка", names)


def label_owlv2(cfg):
    """OWLv2 (transformers) — ещё один open-vocab, иногда лучше ловит мелочь."""
    ensure("torch", "torch"); ensure("transformers", "transformers")
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    lab = cfg["labeling"]
    flat, syn2id, names = resolve_classes(lab)
    name = lab.get("owlv2_model", "google/owlv2-base-patch16-ensemble")
    print(f" -> Загружаю OWLv2 ({name})…")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained(name)
    model = Owlv2ForObjectDetection.from_pretrained(name).to(dev)
    conf = float(lab["conf"])
    from PIL import Image

    def predict_batch(paths, sizes):
        imgs = [Image.open(p).convert("RGB") for p in paths]
        inp = proc(text=[flat] * len(imgs), images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
        targets = torch.tensor([(h, w) for (w, h) in sizes]).to(dev)   # (H, W)
        res = proc.post_process_object_detection(out, threshold=conf, target_sizes=targets)
        out_boxes = []
        for r, (W, H) in zip(res, sizes):
            b = []
            if W > 0 and H > 0:
                for (x0, y0, x1, y1), li in zip(r["boxes"].tolist(), r["labels"].tolist()):
                    cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
                    bw, bh = (x1 - x0) / W, (y1 - y0) / H
                    if bw > 0 and bh > 0:
                        b.append((cx, cy, bw, bh, syn2id[li] if 0 <= li < len(syn2id) else 0))
            out_boxes.append(b)
        return out_boxes
    _label_loop(cfg, predict_batch, "OWLv2 разметка", names)


def label_dataset(cfg):
    backend = cfg["labeling"].get("backend", "yoloworld").lower()
    print(f"\n[ФАЗА 3] Авторазметка, движок: {backend}")
    if backend in ("groundingdino", "dino", "grounding-dino"):
        label_groundingdino(cfg)
    elif backend in ("owlv2", "owl"):
        label_owlv2(cfg)
    else:
        label_yoloworld(cfg)


# =====================================================================
# ТОЧКА ВХОДА
# =====================================================================
def setup_torch(cfg):
    ep = (cfg["generation"].get("hf_endpoint") or "").strip()
    if ep:                                     # пустой эндпойнт -> официальный HF по умолчанию
        os.environ["HF_ENDPOINT"] = ep
    if not (cfg["run"]["phase2_images"] or cfg["run"]["phase3_label"]):
        return
    try:
        import torch
        if cfg["generation"].get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception as e:
        print(f"[setup] torch не настроен ({e})")


def preprod_check(cfg):
    """Предпрод-проверка готовой раскладки тем же чекером, что стоит перед обучением
    (tools/nanodet-sidecar/check_dataset.py) — РЕЖИМ АНАЛИЗА, ничего не меняет.
    Печатает сводку: битые боксы, баланс классов, доля пустых, наличие val. Падать
    не должна никогда — это диагностика, а не часть конвейера."""
    out_dir = cfg["paths"]["output_yolo_dir"]
    if not os.path.isdir(os.path.join(out_dir, "images")):
        return                                            # раскладки нет (фаза 3 не гонялась) — пропускаем
    _, _, names = resolve_classes(cfg["labeling"])
    checker = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "nanodet-sidecar", "check_dataset.py")
    print("\n" + "=" * 60)
    print("[ПРЕДПРОД] Проверяю готовый датасет чекером тренера (анализ, без правок)…")
    if not os.path.isfile(checker):
        print(f"  чекер не найден: {checker} — пропускаю.")
        return
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        subprocess.run(
            [sys.executable, "-u", checker, "--dataset", out_dir, "--classes", ",".join(names)],
            env=env, check=False)
    except Exception as e:
        print(f"  не смог запустить чекер ({e}) — это не влияет на сам датасет.")
    print("  (нашлись правки? почини так: python check_dataset.py --dataset "
          f"\"{out_dir}\" --classes {','.join(names)} --fix)")
    print("=" * 60)


def main():
    global _AUTO
    # Режим авто-конфига: разворачиваем описание в поля и перезаписываем тот же конфиг, выходим.
    if "--autoconfig" in sys.argv:
        paths = [a for a in sys.argv[1:] if not a.startswith("-")]
        path = paths[0] if paths else os.environ.get("GEN_CONFIG")
        if not path or not os.path.isfile(path):
            print("[autoconfig] не передан путь к конфигу."); return
        with open(path, encoding="utf-8") as f:
            cfg = merge_user_config(json.load(f))
        _AUTO = bool(cfg.get("auto_install", True))
        cfg = autoconfig_from_brief(cfg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1, ensure_ascii=False)
        print(f"[autoconfig] конфиг обновлён: {path}")
        return

    print("[СТАРТ] Генератор синтетических датасетов")
    cfg = load_config()
    _AUTO = bool(cfg.get("auto_install", True))

    for p in (cfg["paths"]["images_dir"], os.path.dirname(cfg["paths"]["prompts_file"])):
        if p:
            os.makedirs(p, exist_ok=True)
    print(f"[+] Папки готовы. Картинки: {cfg['paths']['images_dir']}")

    setup_torch(cfg)
    g = cfg["generation"]
    own = g.get("backend", "flux") == "own"
    objs = parse_weighted(cfg['prompts'].get('objects') or [])[0] or [cfg['prompts'].get('object_noun', '?')]
    multi = (len(objs) >= 2 and float(cfg['prompts'].get('multi_object_prob', 0) or 0) > 0)
    print("\n" + "=" * 60)
    print(f"[ПЛАН] Цель: {cfg['total_images']} картинок | объект(ы): {', '.join(objs)}"
          + (f" | мультиобъект ~{int(float(cfg['prompts']['multi_object_prob'])*100)}% сцен" if multi else ""))
    print(f"       Генерация: {'СВОИ картинки (без рендера)' if own else 'FLUX'} | разметка: {cfg['labeling'].get('backend')}")
    print(f"       Фазы: промпты={cfg['run']['phase1_prompts']} картинки={cfg['run']['phase2_images']} "
          f"разметка={cfg['run']['phase3_label']}")
    print("=" * 60)

    prompts = []
    if cfg["run"]["phase1_prompts"] and not own:
        prompts = generate_all_prompts(cfg)
    elif not own:
        prompts = load_existing_prompts(cfg["paths"]["prompts_file"])
        print(f"\n[ФАЗА 1] Пропущена — загружено {len(prompts)} готовых промптов.")
    else:
        print("\n[ФАЗА 1] Пропущена (режим «свои картинки»).")

    if cfg["run"]["phase2_images"] and not own:
        if not prompts:
            print("[ФАЗА 2] Нет промптов — сначала запусти фазу 1 или укажи prompts_file.")
        else:
            generate_all_images(cfg, prompts)
    elif own:
        n = len(_list_jpgs(cfg["paths"]["images_dir"]))
        print(f"[ФАЗА 2] Режим «свои картинки» — беру готовые из папки ({n} шт).")
    else:
        print("[ФАЗА 2] Пропущена.")

    if cfg["run"]["phase3_label"]:
        label_dataset(cfg)
        preprod_check(cfg)
    else:
        print("[ФАЗА 3] Пропущена.")

    print("\n[ГОТОВО] Конвейер завершён.")


if __name__ == "__main__":
    main()
