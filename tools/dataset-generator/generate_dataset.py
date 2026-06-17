#!/usr/bin/env python3
r"""
Synthetic detection-dataset generator — FLUX images + LLM prompts + YOLO-World
auto-labels, the whole pipeline in one config-driven run.

Three phases, each independently toggleable:
  [1] Prompts  — an LLM (LM Studio / any OpenAI-compatible endpoint) writes a huge,
                 deduplicated set of scene prompts, balanced across object scales
                 by a deficit planner (exact proportions at any dataset size).
  [2] Images   — FLUX.1-schnell renders them (nf4 / torchao-fp8 / layerwise quant),
                 streaming to disk so a 30k run survives interruptions and resumes.
  [3] Labels   — YOLO-World zero-shot boxes → YOLO txt labels + a ready dataset.yaml.

Everything — paths, counts, quant mode, sampler params, the LLM connection AND the
full prompt vocabulary (drone types, materials, backgrounds, weather, states,
camera angles, scale mix, system-prompt template) — comes from a JSON config so
nothing is hard-coded. Drive it three ways:

    python generate_dataset.py                 # built-in defaults below
    python generate_dataset.py my_config.json  # a config file
    GEN_CONFIG=my_config.json python generate_dataset.py

The companion GUI (generate_dataset_gui.py) writes that JSON for you, exposes every
field — including a dedicated prompt-vocabulary editor — and streams this log live.
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

# Instant, unbuffered console (so the GUI's log streams line-by-line), and don't let a
# legacy Windows code page (cp1251 etc.) crash on any non-ASCII we print.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace", line_buffering=True)
    except Exception:
        pass


# =====================================================================
# DEFAULT CONFIG — every value here is overridable from the JSON config
# =====================================================================
DEFAULTS = {
    "paths": {
        "flux_dir":        r"C:\Users\dest\Desktop\test\FLUX.1-schnell",
        "transformer_path": r"C:\Users\dest\Desktop\test\FLUX.1-schnell\transformer",
        "images_dir":      r"C:\Users\dest\Desktop\test\synthetic_dataset\images",
        "output_yolo_dir": r"C:\Users\dest\Desktop\test\drone_dataset_yolo",
        "prompts_file":    r"C:\Users\dest\Desktop\test\synthetic_dataset\prompts.jsonl",
    },
    "run": {"phase1_prompts": True, "phase2_images": True, "phase3_label": True},
    "total_images": 30000,
    "generation": {
        "batch_size": 40,            # prompts requested from the LLM per call
        "quant_mode": "torchao",     # nf4 | torchao | layerwise
        "micro_batch": 4,            # images per FLUX forward
        "super_chunk": 1000,         # prompts encoded+rendered per transformer load (capped to total)
        "encode_batch": 64,          # text-encoder sub-batch
        "num_inference_steps": 4,    # FLUX.1-schnell is a few-step model
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
        "use_lms": True,             # drive `lms load/unload` to (un)load the model in LM Studio
    },
    "prompts": {
        "drone_types": [
            "commercial DJI Mavic style quadcopter with folded arms",
            "custom build DIY FPV racing drone with visible carbon frame, speed controllers, and colorful loose wires",
            "heavy industrial hexacopter with 6 carbon fiber rotors and large landing gear",
            "massive agricultural octocopter drone with 8 motors and dual heavy batteries",
            "fixed-wing hybrid VTOL surveillance drone with aerodynamic wings",
            "tiny cetus-style micro whoop drone with full plastic propeller guards",
            "military-style matte black quadcopter with dual thermal camera gimbal",
            "cinewhoop drone with thick protective foam ducts around its propellers",
        ],
        "drone_materials": [
            "matte carbon fiber chassis", "glossy white plastic body",
            "3D-printed TPU parts in bright neon orange", "scratched anodized aluminum arms",
            "camo-painted military composite shell", "weathered dusty gray industrial polymer",
        ],
        "backgrounds": [
            "dense forest canopy from above", "cloudy grey open sky",
            "urban concrete jungle with skyscrapers", "abandoned industrial factory ruins",
            "green corn field", "rocky mountain range", "snowy winter fields",
            "asphalt airport runway", "thick mud and swamp area", "brick wall background",
        ],
        "conditions": [
            "bright sunny day with harsh dynamic shadows",
            "heavy pouring rain with visible water droplets on surface",
            "thick autumn fog with low visibility", "sunset golden hour dramatic lighting",
            "pitch black nighttime with sharp searchlight illumination",
            "overcast weather with flat diffuse light",
        ],
        "states": [
            "flying steadily mid-air", "performing fast aggressive maneuver with slight motion blur",
            "crashed hard on the ground, inverted",
            "broken frame arms, exposed wires, shattered propellers",
            "burning lithium battery with thick black toxic smoke",
            "completely disassembled into separate parts on the ground",
        ],
        "perspectives": [
            "eye-level view", "top-down bird-eye view looking straight down",
            "low angle shot looking up at the object", "high angle shot looking down at the object",
            "three-quarter front view", "side profile view", "dynamic tilted dutch angle",
        ],
        # [phrase, weight] — weights are proportions (need not sum to 100).
        "object_scales": [
            ["the drone is a tiny distant speck, occupying only about 2-5% of the frame, very far away, lots of empty scene around it", 30],
            ["the drone is small in the frame, occupying roughly 10-15%, with plenty of surrounding background", 30],
            ["the drone is medium-sized, occupying about 30-40% of the frame", 25],
            ["the drone is large and close, filling most of the frame, some parts may be cropped at the edges", 15],
        ],
        # Placeholders: {batch_size} {drone_type} {material} {background} {condition} {state} {perspective}
        "system_template": (
            "Generate a raw JSON array of exactly {batch_size} highly detailed, completely unique image generation prompts in English "
            "for unmanned aerial vehicles. For this specific batch, you MUST heavily focus on the following configuration:\n"
            "- Drone Type and Form-factor: {drone_type}\n"
            "- Body Material/Texture: {material}\n"
            "- Scene Environment/Background: {background}\n"
            "- Weather and Lighting: {condition}\n"
            "- Physical State: {state}\n"
            "- Camera Perspective/Angle: {perspective}\n"
            "Do NOT describe how large the drone appears or how much of the frame it fills or its distance — "
            "that will be appended separately. Focus on the drone, the scene, the action and the lighting. "
            "Slightly vary micro-details inside the batch. Return ONLY the raw JSON array of strings. "
            "No markdown, no triple backticks, no explanations."
        ),
    },
    "labeling": {
        "weights": "yolov8x-worldv2.pt",
        "classes": ["drone", "quadcopter", "uav", "fpv drone", "hexacopter", "octocopter", "multirotor"],
        "conf": 0.05,
        "iou": 0.5,
        "class_index": 0,
        "class_name": "drone",
    },
}


# =====================================================================
# CONFIG LOADING
# =====================================================================
def deep_merge(base, override):
    """Recursively overlay `override` onto a copy of `base`."""
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
        print(f"[CONFIG] loaded {path}")
        return deep_merge(DEFAULTS, user)
    print("[CONFIG] using built-in defaults (no config file given)")
    return copy.deepcopy(DEFAULTS)


# =====================================================================
# HELPERS
# =====================================================================
def fmt_hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"


def vram_gb(torch):
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3)


def extract_prompts(raw_text):
    """Robust parse of an LLM reply: tolerates markdown fences, chatter, trailing
    commas and truncation."""
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
    """Deficit planner: every call returns whichever scale phrase is furthest below
    its target share, so the requested proportions hold exactly at any dataset size
    and scales come interleaved rather than in blocks."""

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
        """Restore counts from already-saved prompts (scale appended at line end)."""
        for p in existing_prompts:
            for i, ph in enumerate(self.phrases):
                if p.endswith(ph):
                    self.counts[i] += 1
                    break


def build_system_prompt(cfg):
    pr = cfg["prompts"]
    return pr["system_template"].format(
        batch_size=cfg["generation"]["batch_size"],
        drone_type=random.choice(pr["drone_types"]),
        material=random.choice(pr["drone_materials"]),
        background=random.choice(pr["backgrounds"]),
        condition=random.choice(pr["conditions"]),
        state=random.choice(pr["states"]),
        perspective=random.choice(pr["perspectives"]),
    )


# =====================================================================
# PHASE 1 — PROMPTS (LLM loaded once)
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
    from openai import OpenAI

    total = cfg["total_images"]
    prompts_file = cfg["paths"]["prompts_file"]
    planner = ScalePlanner(cfg["prompts"]["object_scales"])

    all_prompts = load_existing_prompts(prompts_file)
    planner.seed(all_prompts)
    if len(all_prompts) >= total:
        print(f"\n[PHASE 1] Prompts already complete: {len(all_prompts)} >= {total}. Skipping LLM.")
        return all_prompts

    model = cfg["llm"]["model"]
    use_lms = cfg["llm"].get("use_lms", True)
    print(f"\n[PHASE 1] Have {len(all_prompts)} prompts, need {total}.")
    if use_lms:
        print(f" -> Loading LLM {model} once via LM Studio CLI…")
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
                    print(f"   [!] Empty/garbled LLM reply ({consecutive_fail} in a row). Raw: {raw[:150]!r}")
                    if consecutive_fail >= 8 and use_lms:
                        print("   [!] Too many empty replies. Reloading LLM…")
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
                print(f"   [PHASE 1] prompts {done}/{total} (+{len(batch)}) | "
                      f"{rate*60:5.1f}/min | ETA {fmt_hms(eta)}")
            except Exception as e:
                consecutive_fail += 1
                print(f"   [!] LLM error: {e}. Pausing 3s…")
                time.sleep(3)
                continue

    if use_lms:
        print(" -> Unloading LLM, freeing VRAM…")
        subprocess.run("lms unload --all", shell=True, stdout=subprocess.DEVNULL)
    print(f"[PHASE 1] Done: {len(all_prompts)} prompts saved to {prompts_file}")
    return all_prompts


# =====================================================================
# PHASE 2 — IMAGES (FLUX, no LLM thrash)
# =====================================================================
def build_fast_transformer(cfg, torch):
    """Load the transformer in the chosen quant mode, straight to cuda. Falls back to
    layerwise fp8 on any error. Returns (transformer, mode_name)."""
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
            return tr, "NF4 4-bit (bitsandbytes, ~7 GB)"
        except Exception as e:
            print(f"    [QUANT] NF4 unavailable ({type(e).__name__}: {e}). Falling back to layerwise.")
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
            return tr, "torchao fp8 (native fp8 matmul)"
        except Exception as e:
            print(f"    [QUANT] torchao unavailable ({type(e).__name__}: {e}). Falling back to layerwise.")

    tr = FluxTransformer2DModel.from_pretrained(path, torch_dtype=torch.bfloat16)
    tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    tr = tr.to("cuda")
    return tr, "layerwise fp8 (slow, may overflow 16 GB)"


def encode_chunk(cfg, torch, pipe, chunk_prompts):
    """Encode prompts in sub-batches; keep embeddings on CPU to free VRAM for the
    transformer."""
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
    import torch
    from diffusers import FluxPipeline

    g = cfg["generation"]
    total = cfg["total_images"]
    images_dir = cfg["paths"]["images_dir"]
    flux_dir = cfg["paths"]["flux_dir"]
    pad = max(5, len(str(max(0, total - 1))))
    super_chunk = min(int(g["super_chunk"]), total)
    micro_batch = int(g["micro_batch"])
    suffix = g["prompt_suffix"]

    prompts = [normalize_prompt(p) for p in prompts[:total]]
    existing = [fn for fn in os.listdir(images_dir) if fn.endswith(".jpg")]
    generated_count = len(existing)
    print(f"\n[PHASE 2] Images ready: {generated_count}/{total}")
    if generated_count >= min(total, len(prompts)):
        print("[PHASE 2] Everything already generated.")
        return generated_count

    session_start = time.perf_counter()
    session_done = 0
    idx = generated_count

    while idx < min(len(prompts), total):
        chunk = prompts[idx: idx + super_chunk]
        chunk_final = [p + suffix for p in chunk]

        print(f"\n=== [SUPERCHUNK idx={idx}..{idx + len(chunk) - 1}] ===")
        print(f" -> Loading pipeline (no transformer), encoding {len(chunk_final)} prompts…")
        try:
            pipe = FluxPipeline.from_pretrained(flux_dir, transformer=None, torch_dtype=torch.bfloat16)
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            pipe.set_progress_bar_config(disable=True)

            prompt_embeds, pooled_embeds = encode_chunk(cfg, torch, pipe, chunk_final)
            a, r = vram_gb(torch)
            print(f"    [VRAM] after encode + encoder unload: alloc {a:.2f} / reserved {r:.2f} GB")

            print(" -> Bringing up the quantized transformer (once per superchunk)…")
            transformer, quant_mode = build_fast_transformer(cfg, torch)
            pipe.transformer = transformer
            pipe.vae.to("cuda")
            gc.collect()
            torch.cuda.empty_cache()
            a, r = vram_gb(torch)
            print(f"    [QUANT] {quant_mode} | exec={pipe._execution_device} | VRAM {a:.1f}/{r:.1f} GB")
        except Exception as e:
            print(f"[!] FLUX init error on superchunk idx={idx}: {e}. Skipping superchunk.")
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
                    img.save(os.path.join(images_dir, f"drone_synth_{gi:0{pad}d}.jpg"),
                             quality=int(g["jpeg_quality"]))
                    generated_count = max(generated_count, gi + 1)
                    session_done += 1

                dt = time.perf_counter() - t_mb
                sess = time.perf_counter() - session_start
                avg = sess / max(1, session_done)
                eta = avg * (total - generated_count)
                _, rr = vram_gb(torch)
                print(f"    [+] {generated_count:05d}/{total} | +{cur} in {dt:4.1f}s "
                      f"({dt / cur:4.1f}s/img) | avg {avg:4.1f}s/img | VRAM {rr:4.1f}GB | ETA {fmt_hms(eta)}")
            except torch.cuda.OutOfMemoryError:
                print(f"    [!] CUDA OOM on {start}-{end}. Lower micro_batch (now {micro_batch}).")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"    [!] Micro-batch {start}-{end} error: {e}")
                torch.cuda.empty_cache()
                continue

        peak = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        print(f" -> Superchunk done. Peak VRAM {peak:.2f} GB. Total ready: {generated_count}/{total}")

        del transformer
        pipe.transformer = None
        del pipe, prompt_embeds, pooled_embeds
        gc.collect()
        torch.cuda.empty_cache()
        idx += len(chunk)

    print(f"\n[PHASE 2] Done: {generated_count}/{total} images.")
    return generated_count


# =====================================================================
# PHASE 3 — YOLO-WORLD AUTO-LABELING
# =====================================================================
def label_with_yolo_world(cfg):
    """Zero-shot labels via YOLO-World → Ultralytics-standard images/ + labels/ +
    dataset.yaml. pip install ultralytics; weights download on first run."""
    from ultralytics import YOLO as YOLOModel

    images_dir = cfg["paths"]["images_dir"]
    output_yolo_dir = cfg["paths"]["output_yolo_dir"]
    lab = cfg["labeling"]
    cls_idx = int(lab["class_index"])
    cls_name = lab["class_name"]

    labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(output_yolo_dir, exist_ok=True)

    print(f" -> Loading YOLO-World ({lab['weights']})…")
    model = YOLOModel(lab["weights"])
    model.set_classes(lab["classes"])      # synonyms; all detections map to class cls_idx

    images = sorted([fn for fn in os.listdir(images_dir) if fn.endswith(".jpg")])
    total = len(images)
    print(f" -> Labeling {total} images (stream=True, conf={lab['conf']})…")

    detected = empty = total_boxes = 0
    t0 = time.perf_counter()
    results = model.predict(source=images_dir, conf=float(lab["conf"]), iou=float(lab["iou"]),
                            stream=True, verbose=False, save=False)

    for i, r in enumerate(results, 1):
        stem = os.path.splitext(os.path.basename(r.path))[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        lines = []
        if r.boxes is not None and len(r.boxes):
            for cx, cy, w, h in r.boxes.xywhn.tolist():
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        if lines:
            detected += 1
            total_boxes += len(lines)
        else:
            empty += 1
        if i % 100 == 0 or i == total:
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (total - i)
            avg_boxes = total_boxes / max(1, detected)
            print(f"  [{i:>{len(str(total))}}/{total}] with boxes: {detected} ({100*detected/i:4.1f}%) | "
                  f"empty: {empty} | avg boxes/img: {avg_boxes:.2f} | ETA {fmt_hms(eta)}")

    dataset_root = os.path.dirname(images_dir)
    yaml_path = os.path.join(output_yolo_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Detection dataset — YOLO-World auto-labels\n")
        f.write(f"path: {dataset_root}\n")
        f.write("train: images\n")
        f.write("val: images\n\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{cls_name}']\n")

    elapsed = time.perf_counter() - t0
    pct = 100 * detected / max(1, total)
    empty_note = "fine — tiny/far objects" if empty else "all images labeled"
    print(f"\n{'='*60}")
    print(f"[PHASE 3] Labeling done in {fmt_hms(elapsed)}")
    print(f"  Images with boxes : {detected}/{total} ({pct:.1f}%)")
    print(f"  Empty labels      : {empty}  ({empty_note})")
    print(f"  Total boxes       : {total_boxes}")
    print(f"  Labels (labels/)  : {labels_dir}")
    print(f"  dataset.yaml      : {yaml_path}")
    print("\n  Train on it:")
    print(f'  yolo train data="{yaml_path}" model=yolov8n.pt imgsz=640 epochs=100 batch=16')
    print(f"{'='*60}")


# =====================================================================
# ENTRY POINT
# =====================================================================
def setup_torch(cfg):
    """Apply env + torch perf switches (only when a render/label phase will run)."""
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
        print(f"[setup] torch not configured ({e})")


def main():
    print("[START] Synthetic dataset generator")
    cfg = load_config()

    for p in (cfg["paths"]["images_dir"], os.path.dirname(cfg["paths"]["prompts_file"])):
        if p:
            os.makedirs(p, exist_ok=True)
    print(f"[+] Directories ready. Images: {cfg['paths']['images_dir']}")

    setup_torch(cfg)
    g = cfg["generation"]
    pad = max(5, len(str(max(0, cfg["total_images"] - 1))))
    super_chunk = min(int(g["super_chunk"]), cfg["total_images"])
    print("\n" + "=" * 60)
    print(f"[PLAN] Target: {cfg['total_images']} images")
    print(f"       Superchunks: ~{(cfg['total_images'] + super_chunk - 1) // max(1, super_chunk)} "
          f"(of {super_chunk}; that many transformer loads)")
    print(f"       File names: drone_synth_{0:0{pad}d}.jpg .. drone_synth_{max(0, cfg['total_images']-1):0{pad}d}.jpg")
    print(f"       Quant: {g['quant_mode']} | micro_batch={g['micro_batch']}")
    print(f"       Phases: prompts={cfg['run']['phase1_prompts']} images={cfg['run']['phase2_images']} "
          f"label={cfg['run']['phase3_label']}")
    print("=" * 60)

    prompts = []
    if cfg["run"]["phase1_prompts"]:
        prompts = generate_all_prompts(cfg)
    else:
        prompts = load_existing_prompts(cfg["paths"]["prompts_file"])
        print(f"\n[PHASE 1] Skipped — loaded {len(prompts)} existing prompts.")

    if cfg["run"]["phase2_images"]:
        if not prompts:
            print("[PHASE 2] No prompts available — run phase 1 first or point prompts_file at a file.")
        else:
            generate_all_images(cfg, prompts)
    else:
        print("[PHASE 2] Skipped.")

    if cfg["run"]["phase3_label"]:
        print("\n[PHASE 3] YOLO-World auto-labeling…")
        label_with_yolo_world(cfg)
    else:
        print("[PHASE 3] Skipped.")

    print("\n[SUCCESS] Pipeline finished.")


if __name__ == "__main__":
    main()
