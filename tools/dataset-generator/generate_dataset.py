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
    "auto_install": True,            # сам ставить недостающие пакеты (pip) и качать модели
    "generation": {
        "backend": "flux",          # "flux" (рендерить) | "own" (свои картинки, генерацию пропустить)
        "flux_repo": "black-forest-labs/FLUX.1-schnell",  # для авто-скачивания, если flux_dir пуст
        "file_prefix": "synth",      # имя файлов: <prefix>_00001.jpg
        "batch_size": 40,            # промптов за один запрос к LLM
        "quant_mode": "torchao",     # nf4 | torchao | layerwise
        "micro_batch": 4,            # картинок за один проход FLUX
        "super_chunk": 1000,         # промптов на одну загрузку трансформера (не больше total)
        "encode_batch": 64,          # под-батч текст-энкодера
        "num_inference_steps": 4,    # FLUX.1-schnell — few-step модель
        "guidance_scale": 0.0,
        "width": 640,
        "height": 640,
        "jpeg_quality": 92,
        "prompt_suffix": ", realistic photo, high resolution",
        "hf_endpoint": "https://hf-mirror.com",
        "allow_tf32": True,
    },
    "llm": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model": "qwen2.5-coder-14b-instruct",
        "temperature": 0.9,
        "max_tokens": 8192,
        "use_lms": True,             # дёргать `lms load/unload` для (вы)грузки модели в LM Studio
    },
    # ── УНИВЕРСАЛЬНЫЙ СЛОВАРЬ ПРОМПТОВ ──
    # object_noun — что генерируем (любой объект). categories — произвольные категории
    # вариативности: имя -> список вариантов. Шаблон подставит по одному случайному из каждой.
    "prompts": {
        "object_noun": "drone (UAV / quadcopter)",
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
            ["the object is a tiny distant speck, ~2-5% of the frame, far away, lots of empty scene", 30],
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
        "classes": ["drone", "quadcopter", "uav", "fpv drone", "hexacopter", "octocopter", "multirotor"],
        "conf": 0.05,
        "iou": 0.5,
        "class_index": 0,            # все находки маппятся в этот id (синонимы -> один класс)
        "class_names": ["drone"],    # имена классов для dataset.yaml
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


def load_config():
    path = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        path = sys.argv[1]
    path = path or os.environ.get("GEN_CONFIG")
    if path:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        print(f"[КОНФИГ] загружен {path}")
        return deep_merge(DEFAULTS, user)
    print("[КОНФИГ] встроенные дефолты (файл конфига не задан)")
    return copy.deepcopy(DEFAULTS)


# =====================================================================
# САМО-УСТАНОВКА ЗАВИСИМОСТЕЙ И МОДЕЛЕЙ
# =====================================================================
_AUTO = True   # переопределяется из cfg в main()


def _pip_install(*pkgs):
    print(f"  [setup] устанавливаю: {', '.join(pkgs)} …")
    return subprocess.call([sys.executable, "-m", "pip", "install", *pkgs]) == 0


def ensure(import_name, pip_name=None):
    """Импорт с авто-установкой через pip, если пакета нет (и auto_install включён)."""
    try:
        return __import__(import_name)
    except ImportError:
        if not _AUTO:
            raise
        _pip_install(pip_name or import_name)
        return __import__(import_name)


def ensure_flux(cfg):
    """Если папки FLUX нет — скачиваем веса с HuggingFace (большие, ~24 ГБ)."""
    flux_dir = cfg["paths"]["flux_dir"]
    tr = cfg["paths"]["transformer_path"]
    if os.path.isdir(flux_dir) and os.path.isdir(tr):
        return True
    if not _AUTO:
        print(f"[!] FLUX не найден в {flux_dir}, а авто-скачивание выключено.")
        return False
    ensure("huggingface_hub", "huggingface_hub")
    from huggingface_hub import snapshot_download
    repo = cfg["generation"].get("flux_repo", "black-forest-labs/FLUX.1-schnell")
    print(f"[setup] FLUX не найден — качаю {repo} (это большие веса, ~24 ГБ, надолго)…")
    snapshot_download(repo, local_dir=flux_dir, resume_download=True)
    if not os.path.isdir(tr):
        cfg["paths"]["transformer_path"] = os.path.join(flux_dir, "transformer")
    print(f"[setup] FLUX скачан в {flux_dir}")
    return True


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


def build_system_prompt(cfg):
    """Собирает системный промпт: по одному случайному варианту из каждой категории словаря."""
    pr = cfg["prompts"]
    cats = pr.get("categories", {})
    lines = []
    for name, options in cats.items():
        if options:
            lines.append(f"- {name}: {random.choice(list(options))}")
    config_block = "\n".join(lines)
    return pr["system_template"].format(
        batch_size=cfg["generation"]["batch_size"],
        object_noun=pr.get("object_noun", "the target object"),
        config_block=config_block,
    )


# =====================================================================
# ФАЗА 1 — ПРОМПТЫ (LLM загружается один раз)
# =====================================================================
def load_existing_prompts(prompts_file):
    out = []
    if os.path.exists(prompts_file):
        with open(prompts_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def generate_all_prompts(cfg):
    OpenAI = ensure("openai", "openai").OpenAI

    total = cfg["total_images"]
    prompts_file = cfg["paths"]["prompts_file"]
    planner = ScalePlanner(cfg["prompts"]["object_scales"])

    all_prompts = load_existing_prompts(prompts_file)
    planner.seed(all_prompts)
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
        while len(all_prompts) < total:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": build_system_prompt(cfg)}],
                    temperature=cfg["llm"]["temperature"],
                    max_tokens=cfg["llm"]["max_tokens"],
                )
                raw = (resp.choices[0].message.content or "").strip()
                batch = [normalize_prompt(p) for p in extract_prompts(raw)]
                batch = [p for p in batch if len(p) > 10]
                batch = [f"{p}, {planner.next()}" for p in batch]

                if not batch:
                    consecutive_fail += 1
                    print(f"   [!] Пустой/битый ответ LLM ({consecutive_fail} подряд). Сырое: {raw[:150]!r}")
                    if consecutive_fail >= 8 and use_lms:
                        print("   [!] Слишком много пустых ответов. Перезагружаю LLM…")
                        subprocess.run("lms unload --all", shell=True, stdout=subprocess.DEVNULL)
                        time.sleep(2)
                        subprocess.run(f"lms load {model}", shell=True, stdout=subprocess.DEVNULL)
                        consecutive_fail = 0
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
    with torch.no_grad():
        for j in range(0, len(chunk_prompts), enc_batch):
            sub = chunk_prompts[j:j + enc_batch]
            ti = pipe.tokenizer(sub, padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt").to("cuda")
            pooled = pipe.text_encoder(ti.input_ids, output_hidden_states=False).pooler_output
            ti2 = pipe.tokenizer_2(sub, padding="max_length", max_length=256,
                                   truncation=True, return_tensors="pt").to("cuda")
            emb = pipe.text_encoder_2(ti2.input_ids, output_hidden_states=False).last_hidden_state
            pe_parts.append(emb.detach().to("cpu", dtype=torch.bfloat16))
            pp_parts.append(pooled.detach().to("cpu", dtype=torch.bfloat16))
    prompt_embeds = torch.cat(pe_parts)
    pooled_embeds = torch.cat(pp_parts)
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    gc.collect()
    torch.cuda.empty_cache()
    return prompt_embeds, pooled_embeds


def generate_all_images(cfg, prompts):
    ensure("torch", "torch")
    import torch
    ensure("diffusers", "diffusers")
    from diffusers import FluxPipeline
    if not ensure_flux(cfg):
        print("[ФАЗА 2] FLUX недоступен — пропускаю генерацию.")
        return 0

    g = cfg["generation"]
    total = cfg["total_images"]
    images_dir = cfg["paths"]["images_dir"]
    flux_dir = cfg["paths"]["flux_dir"]
    prefix = g.get("file_prefix", "synth")
    pad = max(5, len(str(max(0, total - 1))))
    super_chunk = min(int(g["super_chunk"]), total)
    micro_batch = int(g["micro_batch"])
    suffix = g["prompt_suffix"]

    prompts = [normalize_prompt(p) for p in prompts[:total]]
    existing = [fn for fn in os.listdir(images_dir) if fn.endswith(".jpg")]
    generated_count = len(existing)
    print(f"\n[ФАЗА 2] Картинок готово: {generated_count}/{total}")
    if generated_count >= min(total, len(prompts)):
        print("[ФАЗА 2] Всё уже сгенерировано.")
        return generated_count

    session_start = time.perf_counter()
    session_done = 0
    idx = generated_count

    while idx < min(len(prompts), total):
        chunk = prompts[idx: idx + super_chunk]
        chunk_final = [p + suffix for p in chunk]

        print(f"\n=== [СУПЕРЧАНК idx={idx}..{idx + len(chunk) - 1}] ===")
        print(f" -> Загружаю пайплайн (без трансформера), кодирую {len(chunk_final)} промптов…")
        try:
            pipe = FluxPipeline.from_pretrained(flux_dir, transformer=None, torch_dtype=torch.bfloat16)
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            pipe.set_progress_bar_config(disable=True)

            prompt_embeds, pooled_embeds = encode_chunk(cfg, torch, pipe, chunk_final)
            a, r = vram_gb(torch)
            print(f"    [VRAM] после кодирования + выгрузки энкодеров: занято {a:.2f} / резерв {r:.2f} ГБ")

            print(" -> Поднимаю квантованный трансформер (раз на суперчанк)…")
            transformer, quant_mode = build_fast_transformer(cfg, torch)
            pipe.transformer = transformer
            pipe.vae.to("cuda")
            gc.collect()
            torch.cuda.empty_cache()
            a, r = vram_gb(torch)
            print(f"    [QUANT] {quant_mode} | exec={pipe._execution_device} | VRAM {a:.1f}/{r:.1f} ГБ")
        except Exception as e:
            print(f"[!] Ошибка инициализации FLUX на суперчанке idx={idx}: {e}. Пропускаю суперчанк.")
            torch.cuda.empty_cache()
            idx += len(chunk)
            continue

        torch.cuda.reset_peak_memory_stats()

        for start in range(0, len(chunk_final), micro_batch):
            if generated_count >= total:
                break
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
                    gi = idx + start + k
                    if gi >= total:
                        break
                    img.save(os.path.join(images_dir, f"{prefix}_{gi:0{pad}d}.jpg"),
                             quality=int(g["jpeg_quality"]))
                    generated_count = max(generated_count, gi + 1)
                    session_done += 1

                dt = time.perf_counter() - t_mb
                sess = time.perf_counter() - session_start
                avg = sess / max(1, session_done)
                eta = avg * (total - generated_count)
                _, rr = vram_gb(torch)
                _pb(generated_count / total, f"Фаза 2 — картинки: {generated_count}/{total}")
                print(f"    [+] {generated_count:05d}/{total} | +{cur} за {dt:4.1f}с "
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
        print(f" -> Суперчанк готов. Пик VRAM {peak:.2f} ГБ. Всего готово: {generated_count}/{total}")

        del transformer
        pipe.transformer = None
        del pipe, prompt_embeds, pooled_embeds
        gc.collect()
        torch.cuda.empty_cache()
        idx += len(chunk)

    print(f"\n[ФАЗА 2] Готово: {generated_count}/{total} картинок.")
    return generated_count


# =====================================================================
# ФАЗА 3 — АВТОРАЗМЕТКА (выбор бэкенда)
# =====================================================================
def _list_jpgs(images_dir):
    return sorted([fn for fn in os.listdir(images_dir) if fn.lower().endswith((".jpg", ".jpeg", ".png"))])


def _write_label(labels_dir, stem, boxes_xywhn, cls_idx):
    lines = [f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (cx, cy, w, h) in boxes_xywhn]
    with open(os.path.join(labels_dir, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(lines)


def _write_yaml(cfg):
    images_dir = cfg["paths"]["images_dir"]
    output_yolo_dir = cfg["paths"]["output_yolo_dir"]
    names = cfg["labeling"].get("class_names", ["object"])
    os.makedirs(output_yolo_dir, exist_ok=True)
    yaml_path = os.path.join(output_yolo_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Датасет детекции — авторазметка генератора\n")
        f.write(f"path: {os.path.dirname(images_dir)}\n")
        f.write("train: images\nval: images\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names: [%s]\n" % ", ".join(f"'{n}'" for n in names))
    return yaml_path


def _label_loop(cfg, predict_one, banner):
    """Общий цикл разметки: predict_one(path, (W,H)) -> список (cx,cy,w,h) нормированных."""
    from PIL import Image
    images_dir = cfg["paths"]["images_dir"]
    lab = cfg["labeling"]
    cls_idx = int(lab["class_index"])
    labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
    os.makedirs(labels_dir, exist_ok=True)

    images = _list_jpgs(images_dir)
    total = len(images)
    print(f" -> {banner}: {total} картинок (conf={lab['conf']})…")
    detected = empty = total_boxes = 0
    t0 = time.perf_counter()
    for i, fn in enumerate(images, 1):
        path = os.path.join(images_dir, fn)
        try:
            with Image.open(path) as im:
                wh = im.size
            boxes = predict_one(path, wh)
        except Exception as e:
            boxes = []
            print(f"    [!] {fn}: {e}")
        n = _write_label(labels_dir, os.path.splitext(fn)[0], boxes, cls_idx)
        if n:
            detected += 1; total_boxes += n
        else:
            empty += 1
        if i % 50 == 0 or i == total:
            eta = (time.perf_counter() - t0) / i * (total - i)
            _pb(i / total, f"Фаза 3 — разметка: {i}/{total}")
            print(f"  [{i}/{total}] с рамками: {detected} ({100*detected/i:4.1f}%) | "
                  f"пустых: {empty} | боксов всего: {total_boxes} | осталось ~{fmt_hms(eta)}")

    yaml_path = _write_yaml(cfg)
    print(f"\n{'='*60}")
    print(f"[ФАЗА 3] Разметка завершена за {fmt_hms(time.perf_counter()-t0)}")
    print(f"  картинок с рамками: {detected}/{total} ({100*detected/max(1,total):.1f}%)")
    print(f"  пустых меток: {empty}    боксов всего: {total_boxes}")
    print(f"  метки: {labels_dir}\n  dataset.yaml: {yaml_path}")
    print(f"{'='*60}")


def label_yoloworld(cfg):
    """YOLO-World (ultralytics) — быстрый open-vocab. Веса качаются на первом запуске."""
    ensure("ultralytics", "ultralytics")
    from ultralytics import YOLO
    lab = cfg["labeling"]
    print(f" -> Загружаю YOLO-World ({lab['yoloworld_weights']})…")
    model = YOLO(lab["yoloworld_weights"])
    model.set_classes(list(lab["classes"]))
    conf, iou = float(lab["conf"]), float(lab["iou"])
    # ultralytics эффективнее батчем по папке — но ради единого цикла идём по одной картинке
    def predict_one(path, wh):
        r = model.predict(source=path, conf=conf, iou=iou, verbose=False, save=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for cx, cy, w, h in r.boxes.xywhn.tolist():
                out.append((cx, cy, w, h))
        return out
    _label_loop(cfg, predict_one, "YOLO-World разметка")


def label_groundingdino(cfg):
    """Grounding DINO (transformers) — точнее на редких/необычных классах."""
    ensure("torch", "torch"); ensure("transformers", "transformers")
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    lab = cfg["labeling"]
    name = lab.get("groundingdino_model", "IDEA-Research/grounding-dino-base")
    print(f" -> Загружаю Grounding DINO ({name})…")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(dev)
    text = ". ".join(str(c).lower() for c in lab["classes"]) + "."
    conf = float(lab["conf"])
    from PIL import Image

    def predict_one(path, wh):
        W, H = wh
        with Image.open(path) as im:
            im = im.convert("RGB")
            inp = proc(images=im, text=text, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp["input_ids"], box_threshold=conf, text_threshold=conf,
            target_sizes=[(H, W)])[0]
        boxes = []
        for (x0, y0, x1, y1) in res["boxes"].tolist():
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            if bw > 0 and bh > 0:
                boxes.append((cx, cy, bw, bh))
        return boxes
    _label_loop(cfg, predict_one, "Grounding DINO разметка")


def label_owlv2(cfg):
    """OWLv2 (transformers) — ещё один open-vocab, иногда лучше ловит мелочь."""
    ensure("torch", "torch"); ensure("transformers", "transformers")
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    lab = cfg["labeling"]
    name = lab.get("owlv2_model", "google/owlv2-base-patch16-ensemble")
    print(f" -> Загружаю OWLv2 ({name})…")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained(name)
    model = Owlv2ForObjectDetection.from_pretrained(name).to(dev)
    queries = [str(c) for c in lab["classes"]]
    conf = float(lab["conf"])
    from PIL import Image

    def predict_one(path, wh):
        W, H = wh
        with Image.open(path) as im:
            im = im.convert("RGB")
            inp = proc(text=[queries], images=im, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inp)
        res = proc.post_process_object_detection(
            out, threshold=conf, target_sizes=torch.tensor([(H, W)]).to(dev))[0]
        boxes = []
        for (x0, y0, x1, y1) in res["boxes"].tolist():
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            if bw > 0 and bh > 0:
                boxes.append((cx, cy, bw, bh))
        return boxes
    _label_loop(cfg, predict_one, "OWLv2 разметка")


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
    os.environ["HF_ENDPOINT"] = cfg["generation"]["hf_endpoint"]
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


def main():
    global _AUTO
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
    print("\n" + "=" * 60)
    print(f"[ПЛАН] Цель: {cfg['total_images']} картинок | объект: {cfg['prompts'].get('object_noun','?')}")
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
    else:
        print("[ФАЗА 3] Пропущена.")

    print("\n[ГОТОВО] Конвейер завершён.")


if __name__ == "__main__":
    main()
