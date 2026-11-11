import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
video, n = Path(sys.argv[1]), int(sys.argv[2])
crop_top = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0      # fraction of height to cut at top (UI bar)
crop_bottom = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0   # fraction to cut at bottom
if "foochi_test" in video.name:
    sys.exit("REFUSED: foochi_test is the held-out test clip - never use it for training.")


def crop(frame):
    h = frame.shape[0]
    return frame[int(h * crop_top): h - int(h * crop_bottom)]


cap = cv2.VideoCapture(str(video))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not video.exists() or not cap.isOpened() or total == 0:
    sys.exit(f"Cannot read video: {video}")
step = max(1, total // (n * 4))
valid, idx = [], 0
while True:                       # pass 1: find non-blank candidate frames
    ok, frame = cap.read()
    if not ok:
        break
    if idx % step == 0:
        g = cv2.cvtColor(cv2.resize(crop(frame), (160, 90)), cv2.COLOR_BGR2GRAY)
        if g.std() >= 8:
            valid.append(idx)
    idx += 1
cap.release()

chosen = {valid[int(i * len(valid) / n)] for i in range(min(n, len(valid)))}   # spread evenly
out = ROOT / "data" / "to_label" / "train"
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(video))
idx = saved = 0
while True:                       # pass 2: save them
    ok, frame = cap.read()
    if not ok:
        break
    if idx in chosen:
        cv2.imwrite(str(out / f"{video.stem}_f{idx:04d}.jpg"), crop(frame))
        saved += 1
    idx += 1
cap.release()
print(f"{video.name}: saved {saved} frames (of {idx}) to {out}")
