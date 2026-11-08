import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
video = ROOT / "data" / "foochi_test.mp4"
cap = cv2.VideoCapture(str(video))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not cap.isOpened() or total == 0:
    sys.exit(f"Cannot read video: {video}")
wanted = {int(i * total / 40) for i in range(40)}
out = ROOT / "data" / "to_label" / "test"
out.mkdir(parents=True, exist_ok=True)
idx = saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx in wanted:
        cv2.imwrite(str(out / f"foochi_test_f{idx:04d}.jpg"), frame)
        saved += 1
    idx += 1
cap.release()
print(f"saved {saved} test frames to {out}")
