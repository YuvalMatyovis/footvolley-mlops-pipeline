import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent      # footvolley/
MODELS = ROOT.parent / "models"                    # shared weights
VIDEO = ROOT / "data" / "foochi_test.mp4"
OUT_DIR = ROOT / "outputs"

PERSON, BALL = 0, 32
TOUCH_MARGIN = 0.15   # touch distance as a fraction of player height
MAX_GAP_SEC = 2.0     # longer gap without a touch = chain broken
COOLDOWN_SEC = 0.25   # ignore repeated hits by the same player


def dist_to_box(pt, box):
    """Distance from a point to the nearest edge of a box (0 if inside)."""
    x, y = pt
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    player_model = YOLO(str(MODELS / "yolov8n.pt"))
    ball_model = YOLO(str(MODELS / "yolov8n.pt"))  # separate instance: keeps tracker state clean

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(OUT_DIR / "footvolley_analyzed.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    max_gap = int(fps * MAX_GAP_SEC)
    cooldown = int(fps * COOLDOWN_SEC)

    stats = {}                 # pid -> touches / passes_made / passes_received
    events, chains = [], []
    last_pid, last_frame, streak = None, None, 0
    frame_idx, ball_frames = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # players: tracked, with persistent IDs
        res = player_model.track(frame, persist=True, classes=[PERSON], conf=0.25,
                                 tracker="bytetrack.yaml", verbose=False)[0]
        players = []
        if res.boxes is not None and res.boxes.id is not None:
            for box, pid in zip(res.boxes.xyxy.tolist(), res.boxes.id.int().tolist()):
                players.append((pid, box))

        # ball: plain detection, highest confidence wins, no ID needed
        bres = ball_model(frame, classes=[BALL], conf=0.15, imgsz=960, verbose=False)[0]
        ball = None
        if len(bres.boxes):
            i = int(bres.boxes.conf.argmax())
            x1, y1, x2, y2 = bres.boxes.xyxy[i].tolist()
            ball = ((x1 + x2) / 2, (y1 + y2) / 2)
            ball_frames += 1

        # chain broken if too long without a touch
        if last_frame is not None and frame_idx - last_frame > max_gap:
            chains.append(streak)
            events.append({"frame": frame_idx, "type": "chain_broken", "length": streak})
            streak, last_pid, last_frame = 0, None, None

        # who touches the ball (closest player within margin)
        touch = None
        if ball:
            best = float("inf")
            for pid, box in players:
                d = dist_to_box(ball, box)
                if d <= TOUCH_MARGIN * (box[3] - box[1]) and d < best:
                    best, touch = d, pid

        if touch is not None and not (touch == last_pid and frame_idx - last_frame < cooldown):
            s = stats.setdefault(touch, {"touches": 0, "passes_made": 0, "passes_received": 0})
            s["touches"] += 1
            if last_pid is not None and last_pid != touch:
                streak += 1
                stats[last_pid]["passes_made"] += 1
                s["passes_received"] += 1
                events.append({"frame": frame_idx, "type": "pass", "from": last_pid, "to": touch})
                print(f"Frame {frame_idx:04d} | PASS #{last_pid} -> #{touch} (chain={streak})")
            last_pid, last_frame = touch, frame_idx

        # draw
        out = res.plot()
        if ball:
            cv2.circle(out, (int(ball[0]), int(ball[1])), 12, (0, 255, 255), 2)
        cv2.putText(out, f"Chain: {streak}  Best: {max(chains + [streak])}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y = 60
        for pid, s in sorted(stats.items()):
            cv2.putText(out, f"#{pid} touches={s['touches']} passes={s['passes_made']}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 24
        writer.write(out)

    cap.release()
    writer.release()
    chains.append(streak)

    summary = {"frames": frame_idx,
               "ball_detection_rate": round(ball_frames / max(frame_idx, 1), 3),
               "best_chain": max(chains), "chains": chains, "players": stats}
    (OUT_DIR / "events.json").write_text(json.dumps({"summary": summary, "events": events}, indent=2))

    print("\n=== FOOTVOLLEY REPORT ===")
    print(f"Frames: {frame_idx} | Ball detected in {summary['ball_detection_rate']:.1%} of frames")
    print(f"Best chain: {summary['best_chain']} passes | Chains: {chains}")
    for pid, s in sorted(stats.items()):
        print(f"Player #{pid}: {s}")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()