#!/usr/bin/env python3
"""
Prepare a YOLO dataset for YOLO-FastestV2 training.

YOLO-FastestV2 trains directly on YOLO-format labels (no box conversion needed) —
it just needs file lists + a .names + a .data config. This generates them from a
standard Ultralytics-style layout:

  <dataset>/images/train/*.jpg   <dataset>/labels/train/*.txt
  <dataset>/images/val/*.jpg     <dataset>/labels/val/*.txt

(YOLO-FastestV2 finds each label by replacing 'images' with 'labels' and the image
extension with '.txt' — exactly this layout.)

Output (into OUT, default ./yf_data):
  train.txt  val.txt  custom.names  custom.data

Env:
  YF_DATASET   path to the YOLO dataset root                  [required]
  YF_CLASSES   comma-separated class names, in id order       [required]
  OUT          output dir                                     (default yf_data)
  YF_INPUT     train/infer input size                         (default 352)
  YF_EPOCHS    epochs                                         (default 200)
  YF_BATCH     batch size                                     (default 64)

Then compute anchors and train via train_yolofastest.py (it runs the repo's
genanchors + train.py).
"""

import os
import glob

def env(k, d=None):
    v = os.environ.get(k)
    return v if v not in (None, "") else d

DATASET = env("YF_DATASET")
CLASSES = env("YF_CLASSES")
OUT = env("OUT", "yf_data")
INPUT = int(env("YF_INPUT", "352"))
EPOCHS = int(env("YF_EPOCHS", "200"))
BATCH = int(env("YF_BATCH", "64"))
# repo COCO anchors — replaced by genanchors during training for your data
DEFAULT_ANCHORS = "12.64,19.39, 37.88,51.48, 55.71,138.31, 126.91,78.23, 131.57,214.55, 279.92,258.87"


def list_images(split):
    d = os.path.join(DATASET, "images", split)
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files += glob.glob(os.path.join(d, ext))
    return sorted(os.path.abspath(f) for f in files)


def main():
    if not DATASET or not CLASSES:
        raise SystemExit("ERROR: set YF_DATASET (YOLO dataset root) and YF_CLASSES (comma names)")
    classes = [c.strip() for c in CLASSES.split(",") if c.strip()]
    os.makedirs(OUT, exist_ok=True)

    for split in ("train", "val"):
        imgs = list_images(split)
        if not imgs:
            print(f"  WARNING: no images in {DATASET}/images/{split}")
        with open(os.path.join(OUT, f"{split}.txt"), "w") as f:
            f.write("\n".join(imgs) + ("\n" if imgs else ""))
        print(f"  {split}: {len(imgs)} images → {OUT}/{split}.txt")

    names_path = os.path.join(OUT, "custom.names")
    with open(names_path, "w") as f:
        f.write("\n".join(classes) + "\n")

    data_path = os.path.join(OUT, "custom.data")
    with open(data_path, "w") as f:
        f.write(f"""[name]
model_name=custom

[train-configure]
epochs={EPOCHS}
steps={int(EPOCHS*0.6)},{int(EPOCHS*0.85)}
batch_size={BATCH}
subdivisions=1
learning_rate=0.001

[model-configure]
pre_weights=None
classes={len(classes)}
width={INPUT}
height={INPUT}
anchor_num=3
anchors={DEFAULT_ANCHORS}

[data-configure]
train={os.path.abspath(os.path.join(OUT, 'train.txt'))}
val={os.path.abspath(os.path.join(OUT, 'val.txt'))}
names={os.path.abspath(names_path)}
""")
    print(f"  classes={classes}")
    print(f"Done → {data_path}  (anchors are placeholders; train_yolofastest.py runs genanchors)")


if __name__ == "__main__":
    main()
