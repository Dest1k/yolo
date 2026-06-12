#!/usr/bin/env python3
"""
Fetch a ready, COCO-pretrained EfficientDet-Lite0 .tflite for a first smoke-test of
the MediaPipe sidecar — before you train your own. One command, no GPU:

    python get_model.py
    # then it prints the exact line to run the sidecar.

Downloads Google's official EfficientDet-Lite0 object detector (COCO 80 classes).
The labels are baked into the .tflite, so nothing else is needed.
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = ("https://storage.googleapis.com/mediapipe-models/object_detector/"
       "efficientdet_lite0/float32/latest/efficientdet_lite0.tflite")


def main():
    dst = os.path.join(HERE, "efficientdet_lite0.tflite")
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  have {dst} ({os.path.getsize(dst)} B)")
    else:
        print("Fetching EfficientDet-Lite0 (COCO) .tflite…")
        try:
            with urllib.request.urlopen(URL, timeout=120) as r, open(dst, "wb") as f:
                f.write(r.read())
        except Exception as e:
            sys.exit(f"ERROR: failed to download {URL}\n  {e}")
        print(f"  → {dst} ({os.path.getsize(dst)} B)")
    print("\nDone. Smoke-test the sidecar:")
    print(f"  YOLO_MODEL={dst} YOLO_SOURCE=rpicam \\")
    print(f"    python3 yolo_mediapipe_sidecar.py")
    print("  (USB cam: YOLO_SOURCE=0)")


if __name__ == "__main__":
    main()
