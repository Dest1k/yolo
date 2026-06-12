# От нуля до рабочей модели — пошагово

Сквозной мануал: свой датасет → обученная модель → запуск на Raspberry Pi 5,
в desktop-headless и на телефоне. Парадигма одна для всех детекторов: **отредактировал
`CONFIG`-блок и запустил один скрипт** — он сам соберёт данные, склонирует репозиторий,
обучит и **экспортирует проверенную ncnn-модель**.

Платформы из коробки: обучение на **Windows + RTX 5090 (Blackwell)**, инференс на
**Raspberry Pi 5** (CPU). Та же `.param`/`.bin` работает и в desktop-headless, и в
Android-приложении.

---

## 0. Что понадобится

- **Машина для обучения** с NVIDIA GPU (у тебя — Ultra 9 285K + RTX 5090 32GB + 128GB).
- **Raspberry Pi 5** (боевой инференс) — опционально, можно гонять и на ПК.
- **Python 3.10–3.11** в виртуальном окружении (venv) на машине обучения.
- Датасет в **YOLO-формате** (см. шаг 2).

---

## 1. Окружение на машине обучения (Windows, в venv)

```bash
python -m venv yolo-venv
yolo-venv\Scripts\activate

# PyTorch для Blackwell (RTX 50xx) — обязательно сборка cu128:
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# общие инструменты экспорта/проверки:
pip install opencv-python numpy onnx onnxsim ncnn pnnx
```

Проверь, что GPU виден:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True NVIDIA GeForce RTX 5090
```
Если `False` — стоит не та сборка torch (нужна **cu128 nightly**), переустанови.

---

## 2. Датасет (YOLO-формат)

Структура (стандартный Ultralytics-layout; поддерживается и `train/images/…`):
```
<dataset>/
  images/train/*.jpg      images/val/*.jpg
  labels/train/*.txt      labels/val/*.txt
```
Каждый `.txt` — по строке на объект: `class_id  xc yc w h` (всё нормировано 0..1,
центр + размеры). Имена классов задаются **списком в порядке id** (`0,1,2,…`) в
`CONFIG`-блоке тренера.

> Нужен и `val`-набор (без него NanoDet ругается; FastestV2 обучится, но без честных
> метрик). Хватает 10–20% данных в `val`.

---

## 3. Выбери модель

| | Когда брать | FPS @Pi5 | Мелкие объекты | Где работает |
|---|---|---|---|---|
| **YOLO-FastestV2** | нужен максимальный FPS | ★★★ (~78 @320) | ★ | Pi + headless + телефон |
| **NanoDet-Plus** | нужна точность по мелочи | ★★ (≥30) | ★★★ | Pi + headless + телефон |

Рекомендация: **начни с FastestV2** (быстрее всех, уже проверен). Если теряет мелкие
объекты — переходи на **NanoDet-Plus** (FPN, страйд-8). Обе обучаются на твоём 5090 и
экспортируются в ncnn одной командой.

---

## 4. Настрой и запусти обучение (одна команда)

### Вариант A — YOLO-FastestV2
Открой `tools/yolo-fastestv2-sidecar/train_yolofastest.py`, в `CONFIG`-блоке задай:
```python
DATASET = r"C:\путь\к\датасету"      # корень YOLO-датасета
CLASSES = ["Birds", "Drones", "Dron2"]  # имена в порядке id 0,1,2
INPUT   = 416         # 320 = быстрее, 416 = лучше по мелочи
EPOCHS  = 300
# железо уже выставлено под 285K/5090: BATCH=192, WORKERS=20, DEVICE="gpu"
```
Запуск:
```bash
cd tools/yolo-fastestv2-sidecar
python train_yolofastest.py
```

### Вариант B — NanoDet-Plus
Доустанови тренировочные зависимости:
```bash
pip install pytorch-lightning pycocotools omegaconf tensorboard
```
Открой `tools/nanodet-sidecar/train_nanodet.py`, в `CONFIG`:
```python
DATASET = r"C:\путь\к\датасету"
CLASSES = ["Birds", "Drones", "Dron2"]
INPUT   = 416         # под него уже подобран nanodet-plus-m_416
EPOCHS  = 200
# железо: BATCH=96, WORKERS=20, GPU_IDS=[0]
```
Запуск:
```bash
cd tools/nanodet-sidecar
python train_nanodet.py
```

**Что делает скрипт (оба варианта):**
`[1/4]` готовит данные · `[2/4]` клонирует upstream-репозиторий · `[3/4]` обучает
(GPU) · `[4/4]` **экспортирует и проверяет** ncnn-модель.

За чем следить:
- падает loss, по эпохам растут метрики (mAP);
- GPU загружен (если недогружен — подними `BATCH`; OOM — опусти);
- NanoDet печатает, какие поля конфига пропатчил — если предупреждает `⚠️ NOT FOUND`,
  открой `nd_data/custom.yml` и поправь руками.

---

## 5. Результат экспорта

В конце увидишь **`VERIFY OK`** и пути к файлам, например:
```
✅ Done — trained AND exported. Run it:
  Pi sidecar:  YF_PARAM=…/yolofastestv2.param YF_BIN=…/yolofastestv2.bin YF_INPUT=416 …
```
Получаешь:
- `*.param` + `*.bin` — оптимизированная (fp16) ncnn-модель;
- `custom.names` / `classes.txt` — имена классов.

`VERIFY OK` означает, что модель реально загружается и выходные блобы извлекаются —
её уже можно нести на Pi. Если вместо этого `VERIFY FAILED` / `EXPORT FAILED` — см.
шаг 8.

> **Важно про размер входа и якоря:** инференс должен идти на **том же `INPUT`**, что
> и обучение (для FastestV2 якоря — абсолютные пиксели, не масштабируются). Поставишь
> на Pi другой размер — боксы поедут.

---

## 6. Запуск на Raspberry Pi 5

Скопируй `*.param`, `*.bin` и файл имён классов на Pi. Установи рантайм:
```bash
pip3 install ncnn numpy opencv-python
```

### FastestV2
```bash
YF_PARAM=yolofastestv2.param YF_BIN=yolofastestv2.bin YF_INPUT=416 \
  YOLO_LABELS=custom.names YOLO_SOURCE=rpicam \
  python3 tools/yolo-fastestv2-sidecar/yolofastest_ncnn_sidecar.py
```
### NanoDet-Plus
```bash
ND_PARAM=nanodet.param ND_BIN=nanodet.bin ND_INPUT=416 \
  YOLO_LABELS=classes.txt YOLO_SOURCE=rpicam \
  python3 tools/nanodet-sidecar/nanodet_ncnn_sidecar.py
```
Открой `http://<ip-пи>:8080` — увидишь поток с боксами. **Перетащи рамку** — захват
цели; **H** — панель подвеса (для SIYI ZR10); **Space** — авто-следование.
`YOLO_SOURCE`: `rpicam` (CSI), `0`/`1` (USB), либо `rtsp://…` (например ZR10).

Если боксов нет / они кривые — сначала `--inspect` (шаг 8).

Автозапуск на Pi — через systemd (примеры юнитов в README сайдкаров).

---

## 7. Запуск в desktop-headless и на телефоне

**Desktop headless** (JVM-раннер, ONNX/PT). Использует промежуточный `*.onnx` из
экспорта (`*-sim.onnx`):
```bash
# FastestV2-онниксы декодятся автоматически; для NanoDet укажи декодер:
YOLO_MODEL=nanodet-sim.onnx YOLO_DECODE=nanodet ND_REG_MAX=7 YOLO_INPUT=416 \
  YOLO_LABELS=classes.txt YOLO_SOURCE=rtsp://192.168.144.25:8554/main.264 \
  <запуск desktop Headless>
```

**Телефон (Android-приложение):**
1. Перекинь `*.param` + `*.bin` на телефон.
2. В приложении выбери файлы → «Определить выходы модели».
3. В настройках: версия = **FastestV2** или **NanoDet**, **размер входа = твой `INPUT`**,
   число классов = под модель.
4. Запусти камеру.

---

## 8. Если что-то не так

- **Нет детекций / `-100` на Pi** → `--inspect`: он перебирает размеры входа и печатает
  форму выхода. Чаще всего `YF_INPUT`/`ND_INPUT` ≠ размеру экспорта — поставь совпадающий.
  ```bash
  YF_PARAM=… YF_BIN=… python3 …_sidecar.py --inspect
  ```
- **`EXPORT FAILED` при обучении** → не найден конвертер ncnn. Поставь `pip install pnnx`
  (одно колесо, конвертит и оптимизирует) — экспорт сам его подхватит.
- **Боксы есть, но смещены/неправильного размера** (FastestV2) → проверь, что `YF_INPUT`
  == обучающему `INPUT`; при своих якорях (`GENANCHORS=True`) пропиши `YF_ANCHORS_16/32`.
- **Мелкие объекты теряются** → подними `INPUT` (416/512) или перейди на NanoDet-Plus.
- **Низкий FPS** → опусти `INPUT` (320), увеличь `*_THREADS` (`YF_THREADS`/`ND_THREADS=4`),
  модель уже fp16; рантайн добавляет fp16/int8 сам.

---

### Кратко
```
venv + cu128 torch  →  датасет в YOLO-формате  →  правишь CONFIG  →
python train_*.py  →  VERIFY OK + .param/.bin  →  на Pi/headless/телефон.
```
Детали по каждому детектору — в `tools/yolo-fastestv2-sidecar/README.md` и
`tools/nanodet-sidecar/README.md`.
