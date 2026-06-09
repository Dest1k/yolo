#!/usr/bin/env python3
"""
Convert a YOLO detection dataset (the layout Ultralytics uses) to Pascal VOC for
MediaPipe Model Maker (`train_object_detector.py`).

Input  (YOLO):                         Output (Pascal VOC):
  images/train/*.jpg                     mediapipe_dataset/train/images/*.jpg
  labels/train/*.txt                     mediapipe_dataset/train/Annotations/*.xml
  images/val/*.jpg                       mediapipe_dataset/val/images/*.jpg
  labels/val/*.txt                       mediapipe_dataset/val/Annotations/*.xml

YOLO .txt lines are: `class_id x_center y_center w h` (all normalised 0..1).

IMPORTANT: CLASSES below must be in the SAME ORDER as `names:` in the data.yaml
you trained YOLO with — otherwise the class labels get silently swapped.
"""

import os
import shutil
import cv2
from xml.etree.ElementTree import Element, SubElement, ElementTree

# ── EDIT THIS: your classes, in the SAME ORDER as your YOLO data.yaml `names:` ──
CLASSES = ["Birds", "Drones", "Dron2"]   # id 0, id 1, id 2, …
# ──────────────────────────────────────────────────────────────────────────────

INPUT_DIR = "."
OUTPUT_DIR = "mediapipe_dataset"


def convert_split(split):
    img_dir = os.path.join(INPUT_DIR, "images", split)
    lbl_dir = os.path.join(INPUT_DIR, "labels", split)
    out_img_dir = os.path.join(OUTPUT_DIR, split, "images")
    out_xml_dir = os.path.join(OUTPUT_DIR, split, "Annotations")   # capital A — MediaPipe wants this
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_xml_dir, exist_ok=True)

    if not os.path.isdir(img_dir):
        print(f"  '{img_dir}' not found — skipping split '{split}'")
        return

    n_img = n_obj = n_bad_box = n_bad_cls = 0
    for file in os.listdir(img_dir):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        base = os.path.splitext(file)[0]
        img_path = os.path.join(img_dir, file)
        txt_path = os.path.join(lbl_dir, base + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: can't read image, skipping: {img_path}")
            continue
        h, w, c = img.shape

        shutil.copy(img_path, os.path.join(out_img_dir, file))

        root = Element("annotation")
        SubElement(root, "folder").text = "images"
        SubElement(root, "filename").text = file
        SubElement(root, "path").text = file
        size = SubElement(root, "size")
        SubElement(size, "width").text = str(w)
        SubElement(size, "height").text = str(h)
        SubElement(size, "depth").text = str(c)

        if os.path.exists(txt_path):
            with open(txt_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                    except ValueError:
                        continue

                    # Out-of-range class id → skip (don't invent a phantom class).
                    if class_id < 0 or class_id >= len(CLASSES):
                        n_bad_cls += 1
                        continue

                    xmin = int(round((xc - bw / 2) * w)); ymin = int(round((yc - bh / 2) * h))
                    xmax = int(round((xc + bw / 2) * w)); ymax = int(round((yc + bh / 2) * h))
                    # Clamp into the image.
                    xmin = max(0, min(xmin, w - 1)); ymin = max(0, min(ymin, h - 1))
                    xmax = max(0, min(xmax, w - 1)); ymax = max(0, min(ymax, h - 1))
                    # Drop degenerate boxes (zero/negative area) — they break training.
                    if xmax <= xmin or ymax <= ymin:
                        n_bad_box += 1
                        continue

                    obj = SubElement(root, "object")
                    SubElement(obj, "name").text = CLASSES[class_id]
                    SubElement(obj, "pose").text = "Unspecified"
                    SubElement(obj, "truncated").text = "0"
                    SubElement(obj, "difficult").text = "0"
                    bnd = SubElement(obj, "bndbox")
                    SubElement(bnd, "xmin").text = str(xmin)
                    SubElement(bnd, "ymin").text = str(ymin)
                    SubElement(bnd, "xmax").text = str(xmax)
                    SubElement(bnd, "ymax").text = str(ymax)
                    n_obj += 1

        xml_path = os.path.join(out_xml_dir, base + ".xml")
        with open(xml_path, "wb") as fh:
            ElementTree(root).write(fh, encoding="utf-8", xml_declaration=True)
        n_img += 1

    print(f"  {split}: {n_img} images, {n_obj} objects"
          + (f"  (skipped {n_bad_box} degenerate boxes)" if n_bad_box else "")
          + (f"  (skipped {n_bad_cls} out-of-range class ids)" if n_bad_cls else ""))


if __name__ == "__main__":
    print(f"Classes (id→name): {dict(enumerate(CLASSES))}")
    print("  ⚠️  this order MUST match `names:` in your YOLO data.yaml")
    for sp in ("train", "val"):
        convert_split(sp)
    print(f"Done → {OUTPUT_DIR}/  (train|val with images/ + Annotations/)")
