import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
imgsz = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
model = YOLO(str(ROOT.parent / "models" / "ball_v1.pt"))
cap = cv2.VideoCapture(str(ROOT / "data" / "foochi_test.mp4"))
out = ROOT / "outputs" / f"ball_v1_look_{imgsz}_{conf}"
out.mkdir(parents=True, exist_ok=True)

idx = n = total = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    idx += 1
    if (idx - 1) % 3:
        continue
    n += 1
    r = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    total += len(r.boxes)
    if n % 15 == 0:
        cv2.imwrite(str(out / f"{idx:04d}.jpg"), r.plot())
cap.release()
print(f"imgsz={imgsz} conf={conf}: {total / n:.2f} detections per frame (true value is at most 1)")
