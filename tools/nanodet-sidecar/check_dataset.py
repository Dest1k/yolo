#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Чекер YOLO-датасета для тренера NanoDet-Plus — проверяет разметку «от и до» и при желании
безопасно чинит найденное.

Проверяет ровно то, что потом читает train_nanodet.py (те же раскладки и формат меток):
  раскладка:  images/<split> | <split>/images | <split>
  метка:      путь картинки, 'images'->'labels', расширение .txt
  строка:     "cls xc yc bw bh", нормировано в 0..1, id класса 0..N-1

Что находит:
  • битые / нечитаемые картинки
  • картинки без метки (фон) и метки-сироты без картинки
  • строки не из 5 чисел, мусор, разделители-запятые
  • id класса вне диапазона [0, N-1] или не целое
  • координаты вне [0..1], коробки за краем кадра -> подрезает
  • вырожденные коробки (нулевая ширина/высота)
  • дубликаты коробок в одном файле
  • похоже-на-пиксели координаты (не нормированные) — только сообщает
  • сегментацию (полигоны) вместо рамок -> конвертирует в bbox
  • пустые классы, перекос по классам, отсутствие val-сплита, конфликты имён

Что чинит (только с --fix и ТОЛЬКО после бэкапа):
  подрезка координат, удаление вырожденных/невалидных строк и дублей, конвертация
  полигонов в bbox, разбор запятых, удаление меток-сирот.

ОБЯЗАТЕЛЬНО: перед любыми изменениями делается полная резервная копия папки датасета
ВНУТРИ самой папки датасета: <датасет>/_dataset_backup_<дата_время>/ (прошлые бэкапы
в копию не попадают). Без --fix скрипт ничего не меняет — только отчёт.

Запуск:
  python check_dataset.py --dataset <путь> --classes Birds,Drones        # только анализ
  python check_dataset.py --dataset <путь> --classes Birds,Drones --fix  # анализ + починка
Датасет и классы также читаются из окружения TRAIN_DATASET / TRAIN_CLASSES (как у тренера).
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
SPLIT_NAMES = ("train", "val", "valid", "test")
BACKUP_PREFIX = "_dataset_backup_"
EPS = 1e-6


def _log(msg=""):
    print(msg, flush=True)


# ── обнаружение раскладки ────────────────────────────────────────────────────────
def _has_imgs(d):
    return os.path.isdir(d) and any(glob.glob(os.path.join(d, e)) for e in IMG_EXT)


def _split_dir(dataset, split):
    for c in (os.path.join(dataset, "images", split),
              os.path.join(dataset, split, "images"),
              os.path.join(dataset, split)):
        if _has_imgs(c):
            return c
    return None


def find_splits(dataset):
    """Возвращает {имя_сплита: папка_картинок}. Если сплитов нет — пробуем плоскую раскладку."""
    found = {}
    for s in SPLIT_NAMES:
        d = _split_dir(dataset, s)
        if d:
            found[s] = d
    if not found:                                   # плоско: images/ + labels/ или просто картинки в корне
        for c in (os.path.join(dataset, "images"), dataset):
            if _has_imgs(c):
                found["all"] = c
                break
    return found


def _list_images(d):
    files = []
    for e in IMG_EXT:
        files += glob.glob(os.path.join(d, e))
    return sorted(set(os.path.abspath(f) for f in files))


def _label_path(img_path):
    # ровно как в train_nanodet.py (для совпадения с тем, что реально читает тренер)
    return os.path.splitext(img_path)[0].replace("images", "labels") + ".txt"


def _img_ok(path):
    """Картинка читается? Возвращает (ok, (w,h)|None). Если ни PIL, ни cv2 нет — (None, None)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return True, im.size
    except ImportError:
        pass
    except Exception:
        return False, None
    try:
        import cv2
        im = cv2.imread(path)
        if im is None:
            return False, None
        return True, (im.shape[1], im.shape[0])
    except Exception:
        return None, None                           # нет декодера — проверку картинок пропускаем


# ── разбор/починка одной строки метки ────────────────────────────────────────────
def fix_line(raw, ncls):
    """raw -> (новые_части[5] | None, тег). Запятые трактуем как разделители."""
    parts = raw.replace(",", " ").split()
    if len(parts) < 5:
        return None, "malformed"
    try:
        cls = int(float(parts[0]))
    except ValueError:
        return None, "malformed"
    coords = parts[1:]
    tag = "ok"
    if len(coords) > 4:
        if len(coords) % 2 == 0:                    # класс + чётное число точек -> полигон (сегментация)
            try:
                pts = [float(x) for x in coords]
            except ValueError:
                return None, "malformed"
            xs, ys = pts[0::2], pts[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            tag = "poly"
        else:
            try:
                xc, yc, bw, bh = (float(x) for x in coords[:4])
            except ValueError:
                return None, "malformed"
            tag = "trunc"
    else:
        try:
            xc, yc, bw, bh = (float(x) for x in coords[:4])
        except ValueError:
            return None, "malformed"
    if any(v > 1.5 or v < -0.5 for v in (xc, yc, bw, bh)):
        return None, "not_normalized"               # похоже на пиксели — безопасно не починить
    if ncls is not None and not (0 <= cls < ncls):
        return None, "bad_class"
    x0, y0, x1, y1 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
    cx0, cy0 = min(max(x0, 0.0), 1.0), min(max(y0, 0.0), 1.0)
    cx1, cy1 = min(max(x1, 0.0), 1.0), min(max(y1, 0.0), 1.0)
    nbw, nbh = cx1 - cx0, cy1 - cy0
    if nbw <= EPS or nbh <= EPS:
        return None, "degenerate"
    nxc, nyc = (cx0 + cx1) / 2, (cy0 + cy1) / 2
    clamped = any(abs(a - b) > 1e-9 for a, b in ((nxc, xc), (nyc, yc), (nbw, bw), (nbh, bh)))
    new = [str(cls), "%.6f" % nxc, "%.6f" % nyc, "%.6f" % nbw, "%.6f" % nbh]
    if tag in ("poly", "trunc"):
        return new, tag
    return new, ("clamped" if clamped else "ok")


# ── бэкап ─────────────────────────────────────────────────────────────────────────
def make_backup(dataset):
    """Полная копия датасета ВНУТРИ него: <датасет>/_dataset_backup_<дата_время>/.
    Прошлые бэкапы исключаются (без рекурсии и разрастания)."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(dataset, BACKUP_PREFIX + ts)

    def _ignore(d, names):
        return [n for n in names if n.startswith(BACKUP_PREFIX)]

    _log("  делаю резервную копию (картинки + метки целиком) -> %s" % dst)
    _log("  это может занять время и место на диске; не прерывай…")
    shutil.copytree(dataset, dst, ignore=_ignore)
    _log("  бэкап готов: %s" % dst)
    return dst


# ── нейро-проверка разметки (YOLO-World, как в генераторе датасетов) ───────────────
def _iou(a, b):
    """IoU двух нормированных коробок (xc,yc,w,h)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def _load_yolo_world(weights):
    """Загружаем YOLO-World через ultralytics (как генератор). При отсутствии — ставим pip-ом."""
    try:
        from ultralytics import YOLO
    except ImportError:
        _log("  ultralytics не установлен — ставлю (pip install ultralytics)…")
        subprocess.call([sys.executable, "-m", "pip", "install", "ultralytics"])
        try:
            from ultralytics import YOLO
        except ImportError:
            _log("  не удалось установить ultralytics — нейро-проверку пропускаю.")
            return None
    try:
        return YOLO(weights)                        # веса скачаются при первом запуске
    except Exception as e:
        _log("  не удалось загрузить модель %s: %s" % (weights, e))
        return None


def nn_audit(splits, classes, weights, conf, add_conf, iou_match):
    """Прогоняем YOLO-World по картинкам (классы датасета = словарь) и сверяем с разметкой.
    Возвращает (additions, stat, details):
      additions: {label_path: [[cls,xc,yc,w,h]...]} — уверенные находки БЕЗ существующей метки
                 (кандидаты на дозаполнение пропусков — «выжать максимум из картинок»);
      stat: счётчики; details: человекочитаемые примеры расхождений."""
    if not classes:
        _log("\n  нейро-проверка пропущена: не заданы классы (нужны как словарь для модели).")
        return None
    model = _load_yolo_world(weights)
    if model is None:
        return None
    try:
        model.set_classes(classes)                  # open-vocabulary: имена классов датасета
    except Exception as e:
        _log("  модель не поддерживает set_classes (%s) — нужна YOLO-World (yolov8x-worldv2.pt)." % e)
        return None

    additions, details = {}, []
    stat = dict(imgs=0, preds=0, missing=0, unsupported=0, mismatch=0)
    for split, img_dir in splits.items():
        n = len(_list_images(img_dir))
        if not n:
            continue
        _log("\n  [нейро-проверка: %s] %d картинок моделью %s (conf>=%.2f, дозаполнять conf>=%.2f)…"
             % (split, n, os.path.basename(weights), conf, add_conf))
        results = model.predict(source=img_dir, conf=conf, iou=0.5, stream=True, verbose=False)
        for r in results:
            stat["imgs"] += 1
            preds = []
            if r.boxes is not None and len(r.boxes):
                for c, cc, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                               r.boxes.xywhn.tolist()):
                    preds.append((int(c), float(cc), x, y, w, h))
            stat["preds"] += len(preds)
            lp = _label_path(os.path.abspath(r.path))
            labels = []
            if os.path.isfile(lp):
                for ln in open(lp, encoding="utf-8", errors="replace").read().splitlines():
                    f, _t = fix_line(ln, len(classes))
                    if f:
                        labels.append(tuple(int(f[0]) if i == 0 else float(f[i]) for i in range(5)))
            # 1) уверенная находка без метки того же класса -> вероятно ПРОПУЩЕН объект
            for (c, cc, x, y, w, h) in preds:
                if cc < add_conf:
                    continue
                if not any(lc == c and _iou((x, y, w, h), (lx, ly, lw, lh)) >= iou_match
                           for (lc, lx, ly, lw, lh) in labels):
                    stat["missing"] += 1
                    additions.setdefault(lp, []).append(
                        [str(c), "%.6f" % x, "%.6f" % y, "%.6f" % w, "%.6f" % h])
                    if len(details) < 100:
                        nm = classes[c] if c < len(classes) else c
                        details.append("пропущен объект? %s (conf %.2f): %s" % (nm, cc, r.path))
            # 2) метка без поддержки модели / спорный класс
            for (lc, lx, ly, lw, lh) in labels:
                best_c, best_i = None, 0.0
                for (c, cc, x, y, w, h) in preds:
                    i = _iou((lx, ly, lw, lh), (x, y, w, h))
                    if i > best_i:
                        best_i, best_c = i, c
                if best_i < iou_match:
                    stat["unsupported"] += 1
                    if len(details) < 100:
                        nm = classes[lc] if lc < len(classes) else lc
                        details.append("метка без подтверждения моделью: %s: %s" % (nm, r.path))
                elif best_c is not None and best_c != lc:
                    stat["mismatch"] += 1
                    if len(details) < 100:
                        a = classes[lc] if lc < len(classes) else lc
                        b = classes[best_c] if best_c < len(classes) else best_c
                        details.append("спорный класс: размечено %s, модель думает %s: %s" % (a, b, r.path))
    return additions, stat, details


# ── авто-создание val-сплита (если его нет) ───────────────────────────────────────
def make_val_split(dataset, splits, frac):
    """Переносит долю train -> val (картинка + её метка), если val отсутствует. Возвращает
    число перенесённых картинок. Структура val зеркалит train."""
    import random
    train_dir = splits.get("train") or splits.get("all")
    if not train_dir:
        return 0
    imgs = _list_images(train_dir)
    if len(imgs) < 10:
        _log("  слишком мало картинок (%d) — val не выделяю." % len(imgs))
        return 0
    random.seed(1337)
    k = max(1, int(len(imgs) * frac))
    picked = set(random.sample(imgs, k))
    # путь val зеркалит train: .../images/train -> .../images/val и .../labels/train -> .../labels/val
    moved = 0
    for img in picked:
        lp = _label_path(img)
        v_img = img.replace(os.sep + "train", os.sep + "val")
        if v_img == img:
            return moved                            # нестандартная раскладка — не рискуем
        v_lp = _label_path(v_img)
        os.makedirs(os.path.dirname(v_img), exist_ok=True)
        os.makedirs(os.path.dirname(v_lp), exist_ok=True)
        try:
            shutil.move(img, v_img)
            if os.path.isfile(lp):
                shutil.move(lp, v_lp)
            moved += 1
        except Exception as e:
            _log("  не смог перенести %s: %s" % (img, e))
    return moved


# ── основная проверка ─────────────────────────────────────────────────────────────
def check(dataset, classes, do_fix, nn=False, nn_weights="yolov8x-worldv2.pt",
          nn_conf=0.25, nn_add_conf=0.40, nn_iou=0.45, nn_fix=False, make_val=0.0):
    ncls = len(classes) if classes else None
    splits = find_splits(dataset)
    if not splits:
        _log("ОШИБКА: не нашёл ни картинок, ни сплитов в %s" % dataset)
        _log("  ожидаю одну из раскладок: images/<train|val>, <train|val>/images или <split>/")
        return 2

    _log("=" * 64)
    _log("ПРОВЕРКА ДАТАСЕТА: %s" % dataset)
    _log("  классы (%s): %s" % (ncls if ncls is not None else "?",
                                ", ".join(classes) if classes else "не заданы — диапазон id не проверяю"))
    _log("  сплиты: %s" % ", ".join("%s=%s" % (k, os.path.relpath(v, dataset)) for k, v in splits.items()))
    _log("  режим: %s" % ("ИСПРАВЛЕНИЕ (с бэкапом)" if do_fix else "только анализ (ничего не меняю)"))
    _log("=" * 64)

    # сначала собираем, ЧТО надо менять — бэкап делаем только если реально есть починки
    plan = []          # (label_path, new_text) для перезаписи
    orphans = []       # метки-сироты на удаление
    totals = dict(images=0, boxes=0, no_label=0, empty_label=0, bad_images=0, undecodable=0)
    per_class = {}
    tag_counts = {}
    hard_errors = []   # то, что НЕ чинится автоматически
    no_decoder_warned = False

    def bump(tag, n=1):
        tag_counts[tag] = tag_counts.get(tag, 0) + n

    for split, img_dir in splits.items():
        imgs = _list_images(img_dir)
        label_dirs_seen = set()
        basenames = {}
        _log("\n[%s] картинок: %d  (%s)" % (split, len(imgs), img_dir))
        for img in imgs:
            totals["images"] += 1
            bn = os.path.basename(img)
            basenames.setdefault(bn.lower(), []).append(img)
            ok, _wh = _img_ok(img)
            if ok is None and not no_decoder_warned:
                _log("  (нет PIL/opencv — проверку читаемости картинок пропускаю; "
                     "структуру меток проверяю полностью)")
                no_decoder_warned = True
            elif ok is False:
                totals["bad_images"] += 1
                hard_errors.append("битая/нечитаемая картинка: %s" % img)
                continue
            lp = _label_path(img)
            label_dirs_seen.add(os.path.dirname(lp))
            if not os.path.isfile(lp):
                totals["no_label"] += 1
                continue
            try:
                lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()
            except Exception as e:
                hard_errors.append("не прочитать метку %s: %s" % (lp, e))
                continue
            new_lines, seen_box, changed, nonempty = [], set(), False, 0
            for ln in lines:
                if not ln.strip():
                    changed = True                  # пустые строки выкидываем
                    continue
                fixed, tag = fix_line(ln, ncls)
                if fixed is None:
                    bump(tag); changed = True
                    if tag == "not_normalized":
                        hard_errors.append("не нормированные (пиксельные?) координаты: %s | '%s'"
                                           % (lp, ln.strip()))
                    continue
                key = tuple(fixed)
                if key in seen_box:                 # дубликат коробки
                    bump("duplicate"); changed = True
                    continue
                seen_box.add(key)
                if tag != "ok":
                    bump(tag)
                    if tag in ("clamped", "poly", "trunc"):
                        changed = True
                new_lines.append(" ".join(fixed))
                nonempty += 1
                per_class[fixed[0]] = per_class.get(fixed[0], 0) + 1
            totals["boxes"] += nonempty
            if nonempty == 0:
                totals["empty_label"] += 1
            if changed:
                plan.append((lp, "\n".join(new_lines) + ("\n" if new_lines else "")))

        # метки-сироты (есть .txt, нет картинки)
        for ld in label_dirs_seen:
            if not os.path.isdir(ld):
                continue
            img_names = {os.path.splitext(os.path.basename(i))[0] for i in imgs}
            for txt in glob.glob(os.path.join(ld, "*.txt")):
                if os.path.splitext(os.path.basename(txt))[0] not in img_names:
                    orphans.append(txt)
        # конфликты имён (build_coco берёт basename -> одинаковые имена столкнутся)
        for bn, lst in basenames.items():
            if len(lst) > 1:
                hard_errors.append("дублирующиеся имена картинок в сплите '%s': %s (x%d)"
                                   % (split, bn, len(lst)))

    # ── отчёт ──
    _log("\n" + "=" * 64)
    _log("ИТОГИ")
    _log("  всего картинок: %d, объектов: %d" % (totals["images"], totals["boxes"]))
    _log("  картинок без метки (фон): %d" % totals["no_label"])
    _log("  пустых меток (0 объектов): %d" % totals["empty_label"])
    if totals["bad_images"]:
        _log("  битых картинок: %d" % totals["bad_images"])

    if per_class:
        _log("\n  объектов по классам:")
        for cid in sorted(per_class, key=lambda x: int(x)):
            name = classes[int(cid)] if classes and int(cid) < len(classes) else "id %s" % cid
            _log("    %-20s %d" % (name, per_class[cid]))
        if classes:
            empty = [classes[i] for i in range(len(classes)) if str(i) not in per_class]
            if empty:
                _log("  [!] классы без единого объекта: %s" % ", ".join(empty))
            vals = [per_class.get(str(i), 0) for i in range(len(classes))]
            if vals and min(vals) > 0 and max(vals) / min(vals) >= 20:
                _log("  [!] сильный перекос по классам (x%.0f) — модель будет хуже видеть редкие"
                     % (max(vals) / min(vals)))

    issue_names = {
        "malformed": "строки не из 5 чисел / мусор",
        "bad_class": "id класса вне диапазона",
        "degenerate": "вырожденные коробки (нулевой размер)",
        "clamped": "координаты за пределами кадра (подрезка)",
        "duplicate": "дубликаты коробок",
        "poly": "полигоны сегментации -> bbox",
        "trunc": "лишние числа в строке (обрезка до 4)",
        "not_normalized": "НЕ нормированные координаты (пиксели?)",
    }
    problems = sum(tag_counts.get(t, 0) for t in issue_names)
    if tag_counts:
        _log("\n  найдено в разметке:")
        for t, label in issue_names.items():
            if tag_counts.get(t):
                _log("    %-40s %d" % (label, tag_counts[t]))
    if orphans:
        _log("    %-40s %d" % ("метки-сироты (нет картинки)", len(orphans)))

    if "val" not in splits and "valid" not in splits:
        _log("\n  [!] нет val-сплита: тренер будет считать метрики по train (оптимистично). "
             "Желательно отделить ~10-20%% в val.")
    if classes and per_class:
        max_id = max(int(c) for c in per_class)
        if max_id >= len(classes):
            _log("  [!] в метках есть id=%d, а классов задано %d — проверь список классов!"
                 % (max_id, len(classes)))

    if hard_errors:
        _log("\n  ТРЕБУЕТ РУЧНОГО ВНИМАНИЯ (не чиню автоматически):")
        for e in hard_errors[:50]:
            _log("    - %s" % e)
        if len(hard_errors) > 50:
            _log("    … и ещё %d" % (len(hard_errors) - 50))

    fixable_struct = len(plan) + len(orphans)

    # единый бэкап: создаётся лениво перед ПЕРВЫМ изменением и только один раз
    state = {"backup": None}

    def ensure_backup():
        if state["backup"] is None:
            state["backup"] = make_backup(dataset)
        return state["backup"]

    # ── структурная починка (если просили) ──
    fixed_files = deleted = added_boxes = moved_val = 0
    if do_fix and fixable_struct:
        _log("\nИСПРАВЛЕНИЕ структуры: %d файлов меток, %d сирот." % (len(plan), len(orphans)))
        ensure_backup()
        for lp, text in plan:
            try:
                open(lp, "w", encoding="utf-8").write(text); fixed_files += 1
            except Exception as e:
                _log("  не смог переписать %s: %s" % (lp, e))
        for txt in orphans:
            try:
                os.remove(txt); deleted += 1
            except Exception as e:
                _log("  не смог удалить сироту %s: %s" % (txt, e))

    # ── нейро-проверка YOLO-World (на уже починенных метках) ──
    pending_nn = 0
    if nn:
        res = nn_audit(splits, classes, nn_weights, nn_conf, nn_add_conf, nn_iou)
        if res:
            additions, nstat, details = res
            pending_nn = sum(len(v) for v in additions.values())
            _log("\n  НЕЙРО-ПРОВЕРКА (модель: %s):" % os.path.basename(nn_weights))
            _log("    картинок прогнано: %d, объектов найдено моделью: %d" % (nstat["imgs"], nstat["preds"]))
            _log("    вероятно ПРОПУЩЕНО (нет метки): %d   меток без подтверждения: %d   спорный класс: %d"
                 % (nstat["missing"], nstat["unsupported"], nstat["mismatch"]))
            for d in details[:30]:
                _log("      - %s" % d)
            if len(details) > 30:
                _log("      … всего примеров расхождений: %d (полный список в отчёте)" % len(details))
            try:
                rep = os.path.join(dataset, "_dataset_check_nn_report.txt")
                open(rep, "w", encoding="utf-8").write("\n".join(details) + "\n")
                _log("    подробный отчёт: %s" % rep)
            except Exception:
                pass
            if do_fix and nn_fix and additions:
                _log("\n  ДОЗАПОЛНЯЮ пропущенные объекты: %d коробок в %d файлах "
                     "(человеческие метки не удаляю, только добавляю)." % (pending_nn, len(additions)))
                ensure_backup()
                for lp, adds in additions.items():
                    try:
                        os.makedirs(os.path.dirname(lp), exist_ok=True)
                        prev = ""
                        if os.path.isfile(lp):
                            prev = open(lp, encoding="utf-8", errors="replace").read()
                            if prev and not prev.endswith("\n"):
                                prev += "\n"
                        prev += "\n".join(" ".join(a) for a in adds) + "\n"
                        open(lp, "w", encoding="utf-8").write(prev)
                        added_boxes += len(adds)
                    except Exception as e:
                        _log("  не смог дополнить %s: %s" % (lp, e))
            elif additions:
                _log("    дозаполнить пропуски (+%d коробок) можно нейро-починкой (--fix --nn --nn-fix)."
                     % pending_nn)

    # ── авто-создание val-сплита ──
    if do_fix and make_val and "val" not in splits and "valid" not in splits:
        _log("\n  СОЗДАЮ val-сплит (%.0f%% из train)…" % (make_val * 100))
        ensure_backup()
        moved_val = make_val_split(dataset, splits, make_val)
        _log("    перенесено в val: %d картинок" % moved_val)

    # ── финал ──
    _log("\n" + "=" * 64)
    if not do_fix:
        if fixable_struct or problems or pending_nn:
            _log("ВЕРДИКТ: есть что улучшить.")
            if fixable_struct:
                _log("  структура: %d файлов меток + %d сирот — чинит --fix (сначала бэкап)."
                     % (len(plan), len(orphans)))
            if pending_nn:
                _log("  нейросеть нашла %d возможно пропущенных объектов — добавит --fix --nn --nn-fix."
                     % pending_nn)
        else:
            _log("ВЕРДИКТ: критичных проблем не найдено — датасет готов к обучению.")
        if hard_errors:
            _log("  (пункты «требует ручного внимания» авто-починка не трогает.)")
        return 1 if (fixable_struct or hard_errors or pending_nn) else 0

    if state["backup"] is None:
        _log("ИСПРАВЛЯТЬ НЕЧЕГО (авто-починка). Бэкап не делался.")
    else:
        _log("ГОТОВО. Меток переписано: %d, сирот удалено: %d, коробок добавлено: %d, в val перенесено: %d."
             % (fixed_files, deleted, added_boxes, moved_val))
        _log("  оригинал сохранён в бэкапе: %s" % state["backup"])
    if hard_errors:
        _log("  ВНИМАНИЕ: пункты «требует ручного внимания» НЕ исправлены — реши их вручную.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Чекер/ремонтник YOLO-датасета для NanoDet-тренера.")
    ap.add_argument("--dataset", default=os.environ.get("TRAIN_DATASET", ""),
                    help="корень датасета (или переменная TRAIN_DATASET)")
    ap.add_argument("--classes", default=os.environ.get("TRAIN_CLASSES", ""),
                    help="классы через запятую в порядке id (или TRAIN_CLASSES)")
    ap.add_argument("--fix", action="store_true",
                    help="исправить найденное (СНАЧАЛА делается бэкап внутри папки датасета)")
    ap.add_argument("--nn", action="store_true",
                    help="нейро-проверка разметки YOLO-World (как в генераторе датасетов)")
    ap.add_argument("--nn-fix", action="store_true",
                    help="дозаполнить пропущенные объекты находками модели (аддитивно; требует --fix --nn)")
    ap.add_argument("--nn-weights", default=os.environ.get("CHECK_NN_WEIGHTS", "yolov8x-worldv2.pt"),
                    help="веса YOLO-World (скачаются при первом запуске)")
    ap.add_argument("--nn-conf", type=float, default=0.25, help="порог уверенности предсказаний")
    ap.add_argument("--nn-add-conf", type=float, default=0.40,
                    help="порог для ДОЗАПОЛНЕНИЯ пропусков (выше — надёжнее)")
    ap.add_argument("--nn-iou", type=float, default=0.45, help="IoU сопоставления предсказание<->метка")
    ap.add_argument("--make-val", type=float, default=0.0,
                    help="если val-сплита нет — выделить эту долю из train (напр. 0.15); требует --fix")
    args = ap.parse_args()
    # дубли-переключатели через окружение (удобно для GUI)
    nn = args.nn or os.environ.get("CHECK_NN", "").lower() in ("1", "true", "yes", "on")
    nn_fix = args.nn_fix or os.environ.get("CHECK_NN_FIX", "").lower() in ("1", "true", "yes", "on")
    do_fix = args.fix or os.environ.get("CHECK_FIX", "").lower() in ("1", "true", "yes", "on")
    make_val = args.make_val or float(os.environ.get("CHECK_MAKE_VAL", "0") or 0)

    dataset = args.dataset.strip()
    if not dataset or not os.path.isdir(dataset):
        _log("ОШИБКА: папка датасета не найдена: %s" % (dataset or "(пусто)"))
        _log("  укажи --dataset <путь> или переменную TRAIN_DATASET")
        return 2
    classes = [c.strip() for c in args.classes.split(",") if c.strip()] if args.classes else []
    return check(os.path.abspath(dataset), classes, do_fix, nn=nn, nn_weights=args.nn_weights,
                 nn_conf=args.nn_conf, nn_add_conf=args.nn_add_conf, nn_iou=args.nn_iou,
                 nn_fix=nn_fix, make_val=make_val)


if __name__ == "__main__":
    sys.exit(main())
