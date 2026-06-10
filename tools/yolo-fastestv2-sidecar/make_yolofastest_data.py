#!/usr/bin/env python3
"""
Prepare a YOLO dataset for YOLO-FastestV2 training (builds the file lists +
.names + .data the upstream repo expects). YOLO-FastestV2 trains directly on
YOLO-format labels, so no box conversion — this just lists the files and writes
the config.

>>> Everything is configured in the CONFIG block below — just edit it and run:
        python make_yolofastest_data.py
    (No environment variables needed. Windows-friendly.)

Input layout (standard Ultralytics):
    <DATASET>/images/train/*.jpg   <DATASET>/labels/train/*.txt
    <DATASET>/images/val/*.jpg     <DATASET>/labels/val/*.txt
(the repo finds each label by swapping 'images'->'labels' and the ext -> '.txt')

Output (into OUT):  train.txt  val.txt  custom.names  custom.data
"""

import os
import glob

# ========================= CONFIG — EDIT ME ==================================
DATASET   = r"C:\Users\dest\Desktop\test\merged_dataset"   # YOLO dataset root
CLASSES   = ["Birds", "Drones", "Dron2"]                   # in YOLO id order (data.yaml `names:`)
OUT       = "yf_data"                                       # output folder
INPUT     = 352                                             # train/infer square size
EPOCHS    = 300                                             # training epochs
BATCH     = 96                                              # see note in train_yolofastest.py
LR        = 0.001                                           # learning rate
# repo COCO anchors — train_yolofastest.py recomputes these for your data (genanchors)
ANCHORS   = "12.64,19.39, 37.88,51.48, 55.71,138.31, 126.91,78.23, 131.57,214.55, 279.92,258.87"
# =============================================================================


def list_images(split):
    d = os.path.join(DATASET, "images", split)
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files += glob.glob(os.path.join(d, ext))
    return sorted(os.path.abspath(f) for f in files)


def build(out=OUT):
    if not os.path.isdir(DATASET):
        raise SystemExit(f"ERROR: DATASET not found: {DATASET}")
    os.makedirs(out, exist_ok=True)
    for split in ("train", "val"):
        imgs = list_images(split)
        if not imgs:
            print(f"  WARNING: no images in {DATASET}/images/{split}")
        with open(os.path.join(out, f"{split}.txt"), "w") as f:
            f.write("\n".join(imgs) + ("\n" if imgs else ""))
        print(f"  {split}: {len(imgs)} images")

    names_path = os.path.join(out, "custom.names")
    with open(names_path, "w") as f:
        f.write("\n".join(CLASSES) + "\n")

    data_path = os.path.join(out, "custom.data")
    with open(data_path, "w") as f:
        f.write(f"""[name]
model_name=custom

[train-configure]
epochs={EPOCHS}
steps={int(EPOCHS * 0.6)},{int(EPOCHS * 0.85)}
batch_size={BATCH}
subdivisions=1
learning_rate={LR}

[model-configure]
pre_weights=None
classes={len(CLASSES)}
width={INPUT}
height={INPUT}
anchor_num=3
anchors={ANCHORS}

[data-configure]
train={os.path.abspath(os.path.join(out, 'train.txt'))}
val={os.path.abspath(os.path.join(out, 'val.txt'))}
names={os.path.abspath(names_path)}
""")
    print(f"  classes={CLASSES}")
    print(f"Done -> {data_path}")
    return data_path


if __name__ == "__main__":
    build()
