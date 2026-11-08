"""perception.py - per-frame player and ball detection with the winning setup.

Players: yolo11n (COCO), run at a high resolution so far-away players are found.
Ball:    ball_v2 (custom single-class detector).

Writes to footvolley/outputs/<model>_<date>_<time>/ :
    tracking_data.json   [{"frame", "people": [[x1,y1,x2,y2,conf]...], "ball": [x1,y1,x2,y2,conf] | null}, ...]
    meta.json            fps, size, and every setting used
    preview.mp4          H.264 video for viewing (orange boxes = small players the old filter dropped)

Usage (from the repo root):
    python footvolley/src/perception.py --clip footvolley/data/clips/rally_56.mp4
Quick test on the first 150 frames:
    python footvolley/src/perception.py --clip <clip> --max-frames 150
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser(description="Detect players and the ball in every frame")
    ap.add_argument("--clip", default="footvolley/data/clips/rally_56.mp4")
    ap.add_argument("--ball-model", default="footvolley/models/ball_v2.pt")
    ap.add_argument("--people-model", default="footvolley/models/yolo11n.pt")
    ap.add_argument("--out-root", default="footvolley/outputs")
    ap.add_argument("--ball-imgsz", type=int, default=1088)
    ap.add_argument("--conf-ball", type=float, default=0.25)
    ap.add_argument("--min-ball-area", type=float, default=15)
    ap.add_argument("--people-imgsz", type=int, default=1280)
    ap.add_argument("--conf-person", type=float, default=0.25)
    ap.add_argument("--min-person-area", type=float, default=1500)
    ap.add_argument("--old-area", type=float, default=3500, help="boxes below this area are drawn orange")
    ap.add_argument("--trail", type=int, default=15, help="ball trail length (frames) in the preview")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--progress-every", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.clip):
        raise SystemExit(f"clip not found: {args.clip}")

    ball_model = YOLO(args.ball_model)
    people_model = YOLO(args.people_model)
    ball_ids = [k for k, v in ball_model.names.items() if "ball" in v.lower()] or [0]
    print("ball model classes:", ball_model.names, "-> using", ball_ids)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    model_name = os.path.splitext(os.path.basename(args.ball_model))[0]
    out_dir = os.path.join(args.out_root, f"{model_name}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    raw = os.path.join(out_dir, "raw.mp4")
    final = os.path.join(out_dir, "preview.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    print(f"clip {args.clip}: {total} frames, {fps:.1f} fps, {W}x{H} -> {out_dir}")

    data, trail = [], []
    idx, t0 = 0, time.time()
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- players
        pr = people_model(frame, classes=[0], conf=args.conf_person, imgsz=args.people_imgsz, verbose=False)[0]
        people = []
        for x1, y1, x2, y2, c, _ in pr.boxes.data.tolist():
            area = (x2 - x1) * (y2 - y1)
            if area >= args.min_person_area:
                people.append([int(x1), int(y1), int(x2), int(y2), round(c, 2)])
                small = area < args.old_area
                color = (0, 165, 255) if small else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = ("small " if small else "") + f"{c:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---- ball
        br = ball_model(frame, classes=ball_ids, conf=args.conf_ball, imgsz=args.ball_imgsz, verbose=False)[0]
        ball = None
        for x1, y1, x2, y2, c, _ in br.boxes.data.tolist():
            if (x2 - x1) * (y2 - y1) >= args.min_ball_area and (ball is None or c > ball[4]):
                ball = [int(x1), int(y1), int(x2), int(y2), round(c, 2)]
        center = None
        if ball:
            center = ((ball[0] + ball[2]) // 2, (ball[1] + ball[3]) // 2)
            cv2.rectangle(frame, (ball[0], ball[1]), (ball[2], ball[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ball {ball[4]}", (ball[0], ball[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        trail.append(center)
        if len(trail) > args.trail:
            trail.pop(0)
        for i in range(1, len(trail)):
            if trail[i] is not None and trail[i - 1] is not None:
                cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), 2)

        cv2.putText(frame, f"frame {idx}  t={idx / fps:.1f}s  people={len(people)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
        data.append({"frame": idx, "people": people, "ball": ball})
        idx += 1

        if idx % args.progress_every == 0 or idx == total:
            el = time.time() - t0
            print(f"[{idx / total * 100:5.1f}%] frame {idx}/{total} | {el / idx:.2f}s/frame | "
                  f"elapsed {el / 60:.1f} min | ETA {el / idx * (total - idx) / 60:.1f} min", flush=True)

    cap.release()
    writer.release()
    json.dump(data, open(os.path.join(out_dir, "tracking_data.json"), "w"))
    meta = {k: v for k, v in vars(args).items()}
    meta.update({"fps": fps, "width": W, "height": H, "frames": idx})
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", final], check=True)
    os.remove(raw)

    balls = sum(1 for d in data if d["ball"])
    avg_people = sum(len(d["people"]) for d in data) / max(idx, 1)
    print("\n" + "=" * 60)
    print(f" frames {idx} | ball detected in {balls / max(idx, 1) * 100:.1f}% | avg people per frame {avg_people:.2f}")
    print(f" output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
