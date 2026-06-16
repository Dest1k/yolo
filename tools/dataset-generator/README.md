# Генератор синтетического датасета (FLUX + LLM + YOLO-World)

Собирает детекторный датасет с нуля: картинки рисует **FLUX.1-schnell**, промпты к
ним пишет локальная **LLM** (LM Studio или любой OpenAI-совместимый эндпоинт), а
боксы расставляет **YOLO-World** (zero-shot). На выходе — готовый YOLO-датасет
(`images/`, `labels/`, `dataset.yaml`), который сразу скармливается обучалкам из
соседних сайдкаров.

```
tools/dataset-generator/
  generate_dataset.py       движок (config-driven, без интерактива) — можно гонять headless
  generate_dataset_gui.py   окно: все параметры + редактор словаря промптов + живой лог
```

## Три фазы (включаются по отдельности)

1. **Промпты** — LLM пишет большой набор уникальных сцен, сбалансированных по
   масштабу объекта **дефицит-планировщиком** (точные пропорции при любом размере
   датасета, масштабы идут вперемешку). Пишутся в `prompts.jsonl` построчно, прогон
   возобновляется.
2. **Картинки** — FLUX.1-schnell рендерит их с квантизацией трансформера
   (`torchao` fp8 / `nf4` 4-bit / `layerwise`), суперчанками — один заход = одна
   загрузка трансформера; результат стримится на диск.
3. **Разметка** — YOLO-World даёт боксы по списку синонимов класса → YOLO `.txt` +
   `dataset.yaml`.

## Графический запуск (рекомендуется)

```bash
python tools/dataset-generator/generate_dataset_gui.py
```
Вкладки и живой лог снизу:

| Вкладка | Что задаёт |
|---|---|
| **Run** | пути (FLUX, трансформер, выход картинок/YOLO, файл промптов) с «Browse», число картинок, какие фазы запускать, Start/Stop, «Save config…» |
| **LLM** | base URL, API-ключ, идентификатор модели, температура, max tokens, тумблер драйва LM Studio (`lms load/unload`) |
| **Image** | quant mode, prompts-per-call, micro-batch, super-chunk, encode-batch, число шагов, guidance, размер, JPEG-качество, суффикс промпта, HF endpoint, TF32 |
| **Prompts** | **полный редактор словаря**: типы дронов, материалы, фоны, погода, состояния, ракурсы; **микс масштабов** (`вес \| фраза`); сырой шаблон system-prompt |
| **Labeling** | веса YOLO-World, синонимы класса, conf/iou, id и имя выходного класса |

Окно только пишет JSON-конфиг и запускает движок — поэтому конфиг, сохранённый из
окна, точно так же гоняется headless на GPU-машине. Значения запоминаются между
сессиями (`~/.dataset_generator_gui.json`).

> Окну нужен только Python с **tkinter** (на Windows из коробки; Linux —
> `sudo apt install python3-tk`). Тяжёлые зависимости импортирует движок при запуске.

## Запуск из консоли

```bash
python generate_dataset.py                 # встроенные дефолты
python generate_dataset.py my_config.json  # свой конфиг
GEN_CONFIG=my_config.json python generate_dataset.py
```

## Зависимости движка

```bash
pip install torch diffusers transformers accelerate openai ultralytics pillow
# для quant-режимов: nf4 → bitsandbytes; torchao → torchao
```
Нужна **FLUX.1-schnell** локально (укажи папку в `paths.flux_dir`/`transformer_path`)
и запущенная LLM на `llm.base_url`. YOLO-World веса (`yolov8x-worldv2.pt`) скачаются
автоматически при первой разметке.

## Кастомизация промптов

Вся «соль» — на вкладке **Prompts** (или в секции `prompts` конфига):

- списки `drone_types` / `drone_materials` / `backgrounds` / `conditions` /
  `states` / `perspectives` — по одному элементу на строку, подставляются в шаблон
  как `{drone_type}`, `{material}`, `{background}`, `{condition}`, `{state}`,
  `{perspective}`;
- `object_scales` — строки `вес | фраза`; веса это пропорции (не обязаны давать 100),
  планировщик держит их точно при любом N;
- `system_template` — сырой шаблон запроса к LLM с плейсхолдерами выше плюс
  `{batch_size}`.

Не дрон, а свой объект? Перепиши словарь и `labeling.classes`/`class_name` под свою
задачу — пайплайн от предметной области не зависит.
