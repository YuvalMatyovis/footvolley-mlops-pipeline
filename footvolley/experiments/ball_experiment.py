import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent      # footvolley/
MODELS = ROOT.parent / "models"
VIDEO = ROOT / "data" / "foochi_test.mp4"
OUT = ROOT / "outputs" / "ball_experiment"

BALL = 32
STRIDE = 3        # use every 3rd frame
SAVE_EVERY = 15   # save an annotated image every N sampled frames

CONFIGS = [
    {"name": "current",   "kind": "yolo", "model": "yolov8n.pt", "imgsz": 960,  "conf": 0.15},
    {"name": "v8s-1280",  "kind": "yolo", "model": "yolov8s.pt", "imgsz": 1280, "conf": 0.10},
    {"name": "11n-1280",  "kind": "yolo", "model": "yolo11n.pt", "imgsz": 1280, "conf": 0.10},
    {"name": "11s-1280",  "kind": "yolo", "model": "yolo11s.pt", "imgsz": 1280, "conf": 0.10},
    {"name": "hsv-red",   "kind": "hsv"},
]


def detect_yolo(model, cfg, frame):
    r = model(frame, classes=[BALL], conf=cfg["conf"], imgsz=cfg["imgsz"], verbose=False)[0]
    return r.boxes.conf.tolist(), r.plot


def detect_hsv(frame):
    """Rough red-ball detector: red mask + circular blobs. Needs tuning per video."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not 30 <= area <= 3000:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per) if per else 0
        if circ >= 0.6:
            found.append((circ, cv2.boundingRect(c)))

    def render():
        out = frame.copy()
        for _, (x, y, w, h) in found:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 3)
        return out

    return [c for c, _ in found], render


def run(cfg):
    model = YOLO(str(MODELS / cfg["model"])) if cfg["kind"] == "yolo" else None
    detect = (lambda f: detect_yolo(model, cfg, f)) if model else detect_hsv

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {VIDEO}")

    ok, first = cap.read()          # warm-up, not timed
    detect(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_dir = OUT / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = n = hits = multi = 0
    confs, elapsed = [], 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % STRIDE:
            continue
        n += 1
        t0 = time.perf_counter()
        scores, render = detect(frame)
        elapsed += time.perf_counter() - t0
        if scores:
            hits += 1
            confs += scores
        if len(scores) > 1:
            multi += 1
        if n % SAVE_EVERY == 0:
            cv2.imwrite(str(out_dir / f"{idx:04d}.jpg"), render())
    cap.release()

    return {
        "Config": cfg["name"],
        "FPS": f"{n / elapsed:.2f}",
        "Ball det. rate": f"{hits / n * 100:.1f}% ({hits}/{n})",
        "Avg conf": f"{sum(confs) / len(confs):.2f}" if confs and cfg["kind"] == "yolo" else "-",
        "Frames >1 ball": multi,
    }


def main():
    rows = []
    for cfg in CONFIGS:
        print(f"Running {cfg['name']}...")
        rows.append(run(cfg))

    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]
    table = "\n".join(lines)
    print("\n" + table)
    (OUT / "results.md").write_text(table + "\n")


if __name__ == "__main__":
    main()