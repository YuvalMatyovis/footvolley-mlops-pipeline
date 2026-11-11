"""render_events.py - burn seat IDs, touches and pass counters into the video.

Use it to check count_passes by eye: watch the video, count passes yourself and
compare with the on-screen counter.

Usage:
    python footvolley/src/render_events.py --video <clip.mp4> --run <run folder>
Needs tracking_data.json, meta.json and pass_events.json in the run folder
(perception.py creates the first two, count_passes.py the third).
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np

import count_passes as cp

SEAT_COLORS = {1: (0, 0, 255), 2: (0, 200, 0), 3: (255, 120, 0), 4: (0, 165, 255), 5: (200, 0, 200), 6: (200, 200, 0)}
FLASH_FRAMES = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seats", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bg-filter", action="store_true", help="show the run made with count_passes --bg-filter")
    args = ap.parse_args()

    data, meta = cp.load_run(args.run)
    events = json.load(open(os.path.join(args.run, "pass_events_bgfilter.json" if args.bg_filter else "pass_events.json")))
    fps = float(meta["fps"])
    seats, n_seats = cp.assign_seats(data, args.seats)

    touches_by_frame = {t["frame"]: t for t in events["touches"]}
    passes_by_frame = {p["frame"]: p for p in events["passes"]}

    cap = cv2.VideoCapture(args.video)
    W, H = int(cap.get(3)), int(cap.get(4))
    out_path = args.out or os.path.join(args.run, "events_preview_bgfilter.mp4" if args.bg_filter else "events_preview.mp4")
    raw = out_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    ball_data = cp.remove_background(data, fps)[0] if args.bg_filter else data
    pos, _ = cp.clean_ball_track(ball_data)
    pos, _ = cp.despike(pos)
    pos, _ = cp.drop_static_runs(pos, fps)
    pos, _ = cp.fill_gaps(pos, fps)

    n_touch = n_pass = 0
    flash = []          # (until_frame, text, color, xy)
    last_pass_text = ""
    idx = 0
    while idx < len(data):
        ok, frame = cap.read()
        if not ok:
            break
        # seat boxes
        for si, box in seats[idx].items():
            col = SEAT_COLORS.get(si + 1, (255, 255, 255))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), col, 2)
            cv2.putText(frame, f"S{si + 1}", (box[0], box[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        # ball
        if not np.isnan(pos[idx][0]):
            cv2.circle(frame, (int(pos[idx][0]), int(pos[idx][1])), 9, (0, 255, 0), 2)
        # events
        if idx in touches_by_frame:
            t = touches_by_frame[idx]
            n_touch += 1
            flash.append((idx + FLASH_FRAMES, f"TOUCH S{t['seat']}", SEAT_COLORS.get(t["seat"], (255, 255, 255)), tuple(t["ball"])))
        if idx in passes_by_frame:
            p = passes_by_frame[idx]
            n_pass += 1
            last_pass_text = f"pass S{p['from_seat']} -> S{p['to_seat']}"
        flash = [f for f in flash if f[0] >= idx]
        for _, text, col, xy in flash:
            cv2.circle(frame, xy, 18, col, 3)
            cv2.putText(frame, text, (max(5, xy[0] - 40), max(30, xy[1] - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        # HUD
        cv2.rectangle(frame, (0, 0), (W, 78), (0, 0, 0), -1)
        cv2.putText(frame, f"t={idx / fps:5.1f}s  frame {idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"touches {n_touch}   passes {n_pass}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, last_pass_text, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], check=True)
    os.remove(raw)
    print("saved", out_path, "| touches", n_touch, "| passes", n_pass)


if __name__ == "__main__":
    main()
