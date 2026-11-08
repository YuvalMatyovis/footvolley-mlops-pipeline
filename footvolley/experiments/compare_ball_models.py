#!/usr/bin/env python3
"""
compare_ball_models.py  -  השוואת מודלי כדור V2 מול V3 על אותו קליפ
-------------------------------------------------------------------
A = ball_v2 (כדור) + yolo11n (אנשים)
B = ball_v3 (כדור + אנשים, מחלקות 0=ball, 1=person) בהרצה אחת

פלט (בתיקייה אחת עם תאריך ושעה):
  A_ball-v2__people-yolo11n.mp4      וידאו לצפייה (H.264)
  B_ball-v3__people-v3.mp4           וידאו לצפייה (H.264)
  *_tracking.json                    נתוני זיהוי לכל פריים
  summary.txt / summary.json         טבלת ההשוואה

דוגמאות:
  python footvolley/src/compare_ball_models.py --max-frames 100   # בדיקת זמנים
  python footvolley/src/compare_ball_models.py                    # הרצה מלאה
"""
import argparse, json, os, subprocess, time
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MIN_PERSON_AREA = 3500   # שטח מינימלי לשחקן (פיקסלים)
MIN_BALL_AREA = 15       # שטח מינימלי לכדור (פיקסלים)


def parse_args():
    ap = argparse.ArgumentParser(description="Compare ball_v2 vs ball_v3 on one clip")
    ap.add_argument("--clip", default="footvolley/data/clips/rally_56s.mp4")
    ap.add_argument("--models-dir", default="footvolley/models")
    ap.add_argument("--out-dir", default="footvolley/outputs")
    ap.add_argument("--conf-ball", type=float, default=0.25)
    ap.add_argument("--conf-person", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=1088)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--progress-every", type=int, default=50, help="print progress every X frames")
    ap.add_argument("--device", default="auto", help="auto / cpu / 0")
    return ap.parse_args()


def class_ids(model, word):
    """ מחזיר את ה-ID של מחלקה לפי שמה (לא מניחים שהכדור הוא 0) """
    return [k for k, v in model.names.items() if word in v.lower()]


def rows(result):
    return result.boxes.data.tolist()   # x1, y1, x2, y2, conf, cls


def pick_ball(rs):
    """ הכדור עם הביטחון הגבוה ביותר שעובר את סף השטח """
    best = None
    for x1, y1, x2, y2, c, _ in rs:
        if (x2 - x1) * (y2 - y1) > MIN_BALL_AREA and (best is None or c > best[4]):
            best = [int(x1), int(y1), int(x2), int(y2), round(c, 2)]
    return best


def pick_people(rs, conf_min):
    return [[int(x1), int(y1), int(x2), int(y2), round(c, 2)]
            for x1, y1, x2, y2, c, _ in rs
            if c >= conf_min and (x2 - x1) * (y2 - y1) > MIN_PERSON_AREA]


def draw_overlay(frame, people, ball, label, idx):
    for p in people:
        cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)
    if ball:
        cv2.rectangle(frame, (ball[0], ball[1]), (ball[2], ball[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Ball {ball[4]}", (ball[0], ball[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"{label} | frame {idx}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def ball_stats(b, n_frames):
    """ b = [(frame, cx, cy, conf), ...] """
    if not b:
        return dict(rate=0.0, conf=0.0, static=0.0, median_step=0.0, jumps=0)
    steps = [np.hypot(b[i][1] - b[i - 1][1], b[i][2] - b[i - 1][2])
             for i in range(1, len(b)) if b[i][0] == b[i - 1][0] + 1]   # רק פריימים עוקבים
    steps = np.array(steps) if steps else np.array([0.0])
    return dict(rate=len(b) / n_frames * 100,
                conf=float(np.mean([x[3] for x in b])),
                static=float((steps < 3).mean() * 100),
                median_step=float(np.median(steps)),
                jumps=int((steps > 200).sum()))


def people_stats(counts):
    c = np.array(counts)
    med = int(np.median(c))
    return dict(avg=float(c.mean()), median=med, at_median=float((c == med).mean() * 100))


def to_h264(raw, final):
    """ המרה ל-H.264 כדי שהדפדפן / VS Code יוכלו לנגן """
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", final], check=True)
        os.remove(raw)
        return final
    except Exception as e:
        print(f"WARNING: ffmpeg failed ({e}); raw video kept: {raw}")
        return raw


def main():
    args = parse_args()
    use_gpu = torch.cuda.is_available() if args.device == "auto" else args.device != "cpu"
    device = (0 if args.device == "auto" else args.device) if use_gpu else "cpu"
    half = use_gpu
    print(f"Device: {'GPU ' + torch.cuda.get_device_name(0) if use_gpu else 'CPU'} | half precision: {half}")

    # ---- טעינת מודלים
    people_yolo = YOLO(os.path.join(args.models_dir, "yolo11n.pt"))
    v2 = YOLO(os.path.join(args.models_dir, "ball_v2.pt"))
    v3 = YOLO(os.path.join(args.models_dir, "ball_v3.pt"))
    v2_ball, v3_ball, v3_person = class_ids(v2, "ball"), class_ids(v3, "ball"), class_ids(v3, "person")
    print(f"V2 classes {v2.names} | V3 classes {v3.names}")
    print(f"Class filter -> V2 ball {v2_ball} | V3 ball {v3_ball} | V3 person {v3_person}")
    if not (v2_ball and v3_ball and v3_person):
        raise SystemExit("ERROR: could not find a 'ball'/'person' class in a model's names.")

    # ---- וידאו וקבצי פלט
    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open clip {args.clip}")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = os.path.join(args.out_dir, f"compare_v2_vs_v3_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    names = {"A": "A_ball-v2__people-yolo11n", "B": "B_ball-v3__people-v3"}
    raw = {k: os.path.join(out_dir, f"{n}_raw.mp4") for k, n in names.items()}
    final = {k: os.path.join(out_dir, f"{n}.mp4") for k, n in names.items()}
    writers = {k: cv2.VideoWriter(raw[k], cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) for k in names}

    data = {k: [] for k in names}
    balls = {k: [] for k in names}
    counts = {k: [] for k in names}
    t_model = {"people_yolo11n": 0.0, "ball_v2": 0.0, "ball_v3": 0.0}
    print(f"Clip: {args.clip} | {total} frames | output: {out_dir}\n")

    # ---- לולאה ראשית
    idx, t0 = 0, time.time()
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break

        # A: כדור מ-V2, אנשים מ-yolo11n
        s = time.time()
        pr = people_yolo(frame, classes=[0], conf=args.conf_person, device=device, half=half, verbose=False)[0]
        t_model["people_yolo11n"] += time.time() - s
        ppl_a = pick_people(rows(pr), args.conf_person)

        s = time.time()
        r2 = v2(frame, conf=args.conf_ball, imgsz=args.imgsz, classes=v2_ball,
                device=device, half=half, verbose=False)[0]
        t_model["ball_v2"] += time.time() - s
        ball_a = pick_ball(rows(r2))

        # B: V3 בהרצה אחת, ואז פיצול לפי מחלקה
        s = time.time()
        r3 = rows(v3(frame, conf=min(args.conf_ball, args.conf_person), imgsz=args.imgsz,
                     device=device, half=half, verbose=False)[0])
        t_model["ball_v3"] += time.time() - s
        ball_b = pick_ball([r for r in r3 if int(r[5]) in v3_ball and r[4] >= args.conf_ball])
        ppl_b = pick_people([r for r in r3 if int(r[5]) in v3_person], args.conf_person)

        for key, ppl, ball in (("A", ppl_a, ball_a), ("B", ppl_b, ball_b)):
            data[key].append({"frame": idx, "people": ppl, "ball": ball})
            counts[key].append(len(ppl))
            if ball:
                balls[key].append((idx, (ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2, ball[4]))
            f = frame.copy()
            draw_overlay(f, ppl, ball, names[key], idx)
            writers[key].write(f)

        idx += 1
        # ---- הודעת התקדמות כל X פריימים
        if idx == 1 or idx % args.progress_every == 0 or idx == total:
            el = time.time() - t0
            eta = el / idx * (total - idx)
            print(f"[{idx / total * 100:5.1f}%] frame {idx}/{total} | {el / idx:.2f}s/frame | "
                  f"elapsed {el / 60:.1f} min | ETA {eta / 60:.1f} min | "
                  f"ball hit-rate so far A {len(balls['A']) / idx * 100:.0f}% / B {len(balls['B']) / idx * 100:.0f}%",
                  flush=True)

    cap.release()
    for w in writers.values():
        w.release()

    # ---- שמירה והמרה
    for k in names:
        json.dump(data[k], open(os.path.join(out_dir, f"{names[k]}_tracking.json"), "w"), indent=2)
        final[k] = to_h264(raw[k], final[k])

    # ---- טבלת סיכום
    sa, sb = ball_stats(balls["A"], idx), ball_stats(balls["B"], idx)
    pa, pb = people_stats(counts["A"]), people_stats(counts["B"])
    total_min = (time.time() - t0) / 60
    L = []
    L.append("=" * 76)
    L.append(f" COMPARISON | {os.path.basename(args.clip)} | {idx} frames | imgsz {args.imgsz}")
    L.append("=" * 76)
    L.append(f" {'metric':<40}{'A: v2 + yolo11n':>17}{'B: v3 only':>17}")
    L.append("-" * 76)
    L.append(f" {'BALL detection rate %':<40}{sa['rate']:>17.2f}{sb['rate']:>17.2f}")
    L.append(f" {'BALL avg confidence':<40}{sa['conf']:>17.2f}{sb['conf']:>17.2f}")
    L.append(f" {'BALL static <3px %  (high = suspicious)':<40}{sa['static']:>17.0f}{sb['static']:>17.0f}")
    L.append(f" {'BALL median step px':<40}{sa['median_step']:>17.1f}{sb['median_step']:>17.1f}")
    L.append(f" {'BALL jumps >200px  (high = false hits)':<40}{sa['jumps']:>17d}{sb['jumps']:>17d}")
    L.append(f" {'PEOPLE avg per frame':<40}{pa['avg']:>17.2f}{pb['avg']:>17.2f}")
    L.append(f" {'PEOPLE median count':<40}{pa['median']:>17d}{pb['median']:>17d}")
    L.append(f" {'PEOPLE frames at median count %':<40}{pa['at_median']:>17.0f}{pb['at_median']:>17.0f}")
    L.append("-" * 76)
    L.append(f" time: total {total_min:.1f} min | yolo11n {t_model['people_yolo11n'] / 60:.1f} | "
             f"ball_v2 {t_model['ball_v2'] / 60:.1f} | ball_v3 {t_model['ball_v3'] / 60:.1f}")
    L.append("=" * 76)
    L.append("Videos to watch:")
    L += [f"  {final[k]}" for k in names]
    print("\n" + "\n".join(L))
    open(os.path.join(out_dir, "summary.txt"), "w").write("\n".join(L))
    json.dump({"A": {"ball": sa, "people": pa}, "B": {"ball": sb, "people": pb},
               "frames": idx, "minutes": total_min},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
