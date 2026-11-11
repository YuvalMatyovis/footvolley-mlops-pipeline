"""perception.py - per-frame player and ball detection.

Players: a choice of detector (--people-backend):
    yolo11n       Ultralytics YOLO11n (COCO), the current baseline (AGPL-3.0)
    rfdetr-nano   RF-DETR Nano (Roboflow, Apache-2.0 per the vendor; verify before commercial use)
    rfdetr-small  RF-DETR Small
Ball:    ball_v2 (custom single-class detector, trained with Ultralytics).

Only the player detector changes between backends. Everything else (ball model,
thresholds, filters) is identical, so runs can be compared one change at a time.

Detections are made at a low --store-conf and the extra ones are saved separately,
so later experiments can re-filter offline without running the models again:
    people        players used by the pipeline (conf >= --conf-person, area >= --min-person-area)
    people_extra  the other player boxes (lower confidence or smaller), not used by count_passes
    ball          the ball used by the pipeline (best box with conf >= --conf-ball), unchanged logic
    ball_candidates  up to 3 best ball boxes per frame, including lower-confidence ones

Writes to <out-root>/<clip>_<backend>_<date>_<time>/ :
    tracking_data.json   [{"frame", "people", "people_extra", "ball", "ball_candidates"}, ...]
    meta.json            fps, size, every setting, library versions, git commit, timing
    preview.mp4          H.264 video for viewing (orange boxes = small players)

Usage (from the repo root):
    python footvolley/src/perception.py --clip footvolley/data/clips/rally_56s.mp4 --people-backend rfdetr-small
Quick test on the first 150 frames:
    python footvolley/src/perception.py --clip <clip> --people-backend yolo11n --max-frames 150
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import cv2

PEOPLE_BACKENDS = ("yolo11n", "rfdetr-nano", "rfdetr-small")
N_BALL_CANDIDATES = 3


def parse_args():
    ap = argparse.ArgumentParser(description="Detect players and the ball in every frame")
    ap.add_argument("--clip", default="footvolley/data/clips/rally_56s.mp4")
    ap.add_argument("--people-backend", choices=PEOPLE_BACKENDS, default="yolo11n")
    ap.add_argument("--people-model", default="footvolley/models/yolo11n.pt",
                    help="weights for the yolo11n backend (a bare 'yolo11n.pt' downloads the official file)")
    ap.add_argument("--ball-model", default="footvolley/models/ball_v2.pt")
    ap.add_argument("--out-root", default="footvolley/outputs")
    ap.add_argument("--ball-imgsz", type=int, default=1088)
    ap.add_argument("--conf-ball", type=float, default=0.25)
    ap.add_argument("--min-ball-area", type=float, default=15)
    ap.add_argument("--people-imgsz", type=int, default=1280, help="yolo11n only; RF-DETR uses its own resolution")
    ap.add_argument("--conf-person", type=float, default=0.25)
    ap.add_argument("--min-person-area", type=float, default=1500)
    ap.add_argument("--store-conf", type=float, default=0.10,
                    help="detections down to this confidence are saved (as *_extra / candidates), not used")
    ap.add_argument("--old-area", type=float, default=3500, help="boxes below this area are drawn orange")
    ap.add_argument("--trail", type=int, default=15, help="ball trail length (frames) in the preview")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--progress-every", type=int, default=200)
    return ap.parse_args()


# ---------------------------------------------------------------- player detectors
class YoloPeople:
    """Ultralytics YOLO, COCO class 0 = person."""

    def __init__(self, weights, imgsz):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz = imgsz

    def __call__(self, frame, conf):
        r = self.model(frame, classes=[0], conf=conf, imgsz=self.imgsz, verbose=False)[0]
        return [[x1, y1, x2, y2, c] for x1, y1, x2, y2, c, _ in r.boxes.data.tolist()]


class RfdetrPeople:
    """RF-DETR (pip package 'rfdetr'), COCO-pretrained; keeps only the person class."""

    CLASS_NAMES = {"rfdetr-nano": "RFDETRNano", "rfdetr-small": "RFDETRSmall"}

    def __init__(self, backend):
        import rfdetr
        name = self.CLASS_NAMES[backend]
        if not hasattr(rfdetr, name):
            raise SystemExit(f"this rfdetr version has no {name}; installed: {getattr(rfdetr, '__version__', '?')}")
        self.model = getattr(rfdetr, name)()
        self.person_id = self._person_id()
        self.class_ids_seen = set()

    @staticmethod
    def _person_id():
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES
            for k, v in COCO_CLASSES.items():
                if str(v).lower() == "person":
                    return int(k)
        except ImportError:
            pass
        return 1  # COCO category id of "person"

    def __call__(self, frame, conf):
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = self.model.predict(Image.fromarray(rgb), threshold=conf)
        out = []
        for box, c, k in zip(det.xyxy.tolist(), det.confidence.tolist(), det.class_id.tolist()):
            self.class_ids_seen.add(int(k))
            if int(k) == self.person_id:
                out.append([*box, c])
        return out


def make_people_detector(args):
    if args.people_backend == "yolo11n":
        return YoloPeople(args.people_model, args.people_imgsz)
    return RfdetrPeople(args.people_backend)


# ---------------------------------------------------------------- helpers
def versions():
    out = {}
    for mod in ("ultralytics", "rfdetr", "torch", "cv2"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def box_int(b):
    return [int(b[0]), int(b[1]), int(b[2]), int(b[3]), round(float(b[4]), 2)]


def main():
    args = parse_args()
    if not os.path.exists(args.clip):
        raise SystemExit(f"clip not found: {args.clip}")

    from ultralytics import YOLO
    ball_model = YOLO(args.ball_model)
    ball_ids = [k for k, v in ball_model.names.items() if "ball" in v.lower()] or [0]
    print("ball model classes:", ball_model.names, "-> using", ball_ids)
    people_det = make_people_detector(args)
    print("people backend:", args.people_backend)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    clip_name = os.path.splitext(os.path.basename(args.clip))[0]
    out_dir = os.path.join(args.out_root, f"{clip_name}_{args.people_backend}_{stamp}")
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
    t_people = t_ball = 0.0
    idx, t0 = 0, time.time()
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- players: detect at store_conf, split into used / extra
        s = time.time()
        found = people_det(frame, args.store_conf)
        t_people += time.time() - s
        people, people_extra = [], []
        for b in found:
            x1, y1, x2, y2, c = b
            area = (x2 - x1) * (y2 - y1)
            if c >= args.conf_person and area >= args.min_person_area:
                people.append(box_int(b))
                small = area < args.old_area
                color = (0, 165, 255) if small else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = ("small " if small else "") + f"{c:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                people_extra.append(box_int(b))

        # ---- ball: same selection as before; lower-confidence candidates are only saved
        s = time.time()
        br = ball_model(frame, classes=ball_ids, conf=args.store_conf, imgsz=args.ball_imgsz, verbose=False)[0]
        t_ball += time.time() - s
        cands = [b for b in br.boxes.data.tolist() if (b[2] - b[0]) * (b[3] - b[1]) >= args.min_ball_area]
        cands.sort(key=lambda b: -b[4])
        ball = box_int(cands[0]) if cands and cands[0][4] >= args.conf_ball else None
        ball_candidates = [box_int(b) for b in cands[:N_BALL_CANDIDATES]]

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

        cv2.putText(frame, f"frame {idx}  t={idx / fps:.1f}s  people={len(people)}  [{args.people_backend}]",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        data.append({"frame": idx, "people": people, "people_extra": people_extra,
                     "ball": ball, "ball_candidates": ball_candidates})
        idx += 1

        if idx % args.progress_every == 0 or idx == total:
            el = time.time() - t0
            print(f"[{idx / total * 100:5.1f}%] frame {idx}/{total} | {el / idx:.2f}s/frame | "
                  f"elapsed {el / 60:.1f} min | ETA {el / idx * (total - idx) / 60:.1f} min", flush=True)

    cap.release()
    writer.release()
    json.dump(data, open(os.path.join(out_dir, "tracking_data.json"), "w"))
    meta = dict(vars(args))
    meta.update({"fps": fps, "width": W, "height": H, "frames": idx,
                 "versions": versions(), "git_commit": git_commit(),
                 "sec_per_frame_people": round(t_people / max(idx, 1), 4),
                 "sec_per_frame_ball": round(t_ball / max(idx, 1), 4)})
    if isinstance(people_det, RfdetrPeople):
        meta["rfdetr_person_class_id"] = people_det.person_id
        meta["rfdetr_class_ids_seen"] = sorted(people_det.class_ids_seen)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", final], check=True)
    os.remove(raw)

    balls = sum(1 for d in data if d["ball"])
    avg_people = sum(len(d["people"]) for d in data) / max(idx, 1)
    print("\n" + "=" * 60)
    print(f" backend {args.people_backend} | frames {idx} | ball in {balls / max(idx, 1) * 100:.1f}% | "
          f"avg people {avg_people:.2f}")
    print(f" people model {meta['sec_per_frame_people']:.3f}s/frame | ball model {meta['sec_per_frame_ball']:.3f}s/frame")
    print(f" output: {out_dir}")
    print("=" * 60)
    if avg_people == 0:
        print("WARNING: no players found at all - check the person class id in meta.json")


if __name__ == "__main__":
    main()
