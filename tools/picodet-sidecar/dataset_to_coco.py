#!/usr/bin/env python3
"""
Convert a YOLO or Pascal VOC detection dataset to COCO JSON — the format
PaddleDetection (PicoDet training) expects.

Input, either:
  YOLO:   images/{train,val}/*.jpg   labels/{train,val}/*.txt   (class xc yc w h, normalised)
  VOC:    {train,val}/images/*.jpg   {train,val}/Annotations/*.xml

Output (PaddleDetection-friendly):
  picodet_dataset/
    train/*.jpg                 val/*.jpg
    annotations/instances_train.json   annotations/instances_val.json

Set CLASSES below to your class names. For YOLO the order MUST match your
data.yaml `names:`. For VOC the names come from the XML <name> tags, but CLASSES
fixes the category id order in the COCO file.
"""

import os
import json
import glob
import shutil
import xml.etree.ElementTree as ET
import cv2

# ── EDIT: your class names, in order (YOLO id 0,1,2… → these) ──────────────────
CLASSES = ["Birds", "Drones", "Dron2"]
# Input format: "yolo" or "voc"
INPUT_FORMAT = "yolo"
INPUT_DIR = "."
OUTPUT_DIR = "picodet_dataset"
# ──────────────────────────────────────────────────────────────────────────────

NAME_TO_ID = {n: i for i, n in enumerate(CLASSES)}   # COCO category ids start at 0 here


def _coco_skeleton():
    return {
        "images": [], "annotations": [],
        "categories": [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(CLASSES)],
    }


def _add(coco, img_id, ann_id, w, h, file_name, boxes):
    """boxes: list of (cls_id, x1, y1, x2, y2) in pixels."""
    coco["images"].append({"id": img_id, "file_name": file_name, "width": w, "height": h})
    for (cid, x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        coco["annotations"].append({
            "id": ann_id, "image_id": img_id, "category_id": cid,
            "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0,
        })
        ann_id += 1
    return ann_id


def _yolo_boxes(txt_path, w, h):
    boxes, bad = [], 0
    if not os.path.exists(txt_path):
        return boxes, bad
    for line in open(txt_path):
        p = line.split()
        if len(p) != 5:
            continue
        try:
            cid = int(p[0]); xc, yc, bw, bh = map(float, p[1:])
        except ValueError:
            continue
        if cid < 0 or cid >= len(CLASSES):
            bad += 1; continue
        x1 = (xc - bw / 2) * w; y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w; y2 = (yc + bh / 2) * h
        x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
        if x2 - x1 < 1 or y2 - y1 < 1:
            bad += 1; continue
        boxes.append((cid, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)))
    return boxes, bad


def _voc_boxes(xml_path):
    boxes, bad = [], 0
    if not os.path.exists(xml_path):
        return boxes, bad
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in NAME_TO_ID:
            bad += 1; continue
        b = obj.find("bndbox")
        x1 = float(b.findtext("xmin")); y1 = float(b.findtext("ymin"))
        x2 = float(b.findtext("xmax")); y2 = float(b.findtext("ymax"))
        if x2 - x1 < 1 or y2 - y1 < 1:
            bad += 1; continue
        boxes.append((NAME_TO_ID[name], x1, y1, x2, y2))
    return boxes, bad


def convert_split(split):
    if INPUT_FORMAT == "yolo":
        img_dir = os.path.join(INPUT_DIR, "images", split)
        lbl_dir = os.path.join(INPUT_DIR, "labels", split)
    else:  # voc
        img_dir = os.path.join(INPUT_DIR, split, "images")
        lbl_dir = os.path.join(INPUT_DIR, split, "Annotations")
    if not os.path.isdir(img_dir):
        print(f"  '{img_dir}' not found — skipping split '{split}'")
        return

    out_img = os.path.join(OUTPUT_DIR, split)
    out_ann = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_ann, exist_ok=True)

    coco = _coco_skeleton()
    img_id, ann_id, n_obj, n_bad = 1, 1, 0, 0
    for file in sorted(os.listdir(img_dir)):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        base = os.path.splitext(file)[0]
        img = cv2.imread(os.path.join(img_dir, file))
        if img is None:
            continue
        h, w = img.shape[:2]
        if INPUT_FORMAT == "yolo":
            boxes, bad = _yolo_boxes(os.path.join(lbl_dir, base + ".txt"), w, h)
        else:
            boxes, bad = _voc_boxes(os.path.join(lbl_dir, base + ".xml"))
        shutil.copy(os.path.join(img_dir, file), os.path.join(out_img, file))
        ann_id = _add(coco, img_id, ann_id, w, h, file, boxes)
        img_id += 1; n_obj += len(boxes); n_bad += bad

    out_json = os.path.join(out_ann, f"instances_{split}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"  {split}: {img_id - 1} images, {n_obj} objects → {out_json}"
          + (f"  (skipped {n_bad} bad/degenerate)" if n_bad else ""))


if __name__ == "__main__":
    print(f"Format: {INPUT_FORMAT}   classes (id→name): {dict(enumerate(CLASSES))}")
    for sp in ("train", "val"):
        convert_split(sp)
    print(f"Done → {OUTPUT_DIR}/  (train|val + annotations/instances_*.json)")
