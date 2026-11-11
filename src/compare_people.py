"""compare_people.py - side-by-side table of player-detector runs on the same clip.

Each run folder needs tracking_data.json and meta.json (perception.py); if pass_eval.json
(evaluate_passes.py) is there, the pass metrics are added.

Player metrics (no ground truth needed):
    avg players per frame, % frames with >= 3 and with exactly 4 players,
    flicker = % of frames where the player count differs from the previous frame (lower = steadier),
    seat occupancy % per seat (how often each of the 4 seats has a player assigned),
    people-model seconds per frame (speed, for the cost model).

Usage:
    python src/compare_people.py outputs/<run A> outputs/<run B> ... \
        --out outputs/comparison_people.md
"""
import argparse
import json
import os

import numpy as np

import count_passes as cp


def stats(run):
    data, meta = cp.load_run(run)
    counts = np.array([len(f["people"]) for f in data])
    flicker = float((counts[1:] != counts[:-1]).mean() * 100) if len(counts) > 1 else 0.0
    seats, n_seats = cp.assign_seats(data, 4)
    occ = [float(np.mean([si in s for s in seats]) * 100) for si in range(n_seats)]
    row = {"run": os.path.basename(os.path.normpath(run)), "backend": meta.get("people_backend", "yolo11n"),
           "frames": len(data), "avg_people": float(counts.mean()), "pct_ge3": float((counts >= 3).mean() * 100),
           "pct_eq4": float((counts == 4).mean() * 100), "flicker": flicker, "seat_occ": occ,
           "sec_people": meta.get("sec_per_frame_people")}
    ev = os.path.join(run, "pass_eval.json")
    if os.path.exists(ev):
        row["passes"] = json.load(open(ev))["main"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [stats(r) for r in args.runs]

    L = ["# People detector comparison", "",
         "| metric | " + " | ".join(r["backend"] for r in rows) + " |",
         "| :--- | " + " | ".join("---:" for _ in rows) + " |"]

    def line(name, fn):
        L.append(f"| {name} | " + " | ".join(fn(r) for r in rows) + " |")

    line("frames", lambda r: str(r["frames"]))
    line("avg players per frame", lambda r: f"{r['avg_people']:.2f}")
    line("frames with >= 3 players %", lambda r: f"{r['pct_ge3']:.0f}")
    line("frames with exactly 4 players %", lambda r: f"{r['pct_eq4']:.0f}")
    line("count flicker % (lower = steadier)", lambda r: f"{r['flicker']:.1f}")
    line("seat occupancy % (S1 / S2 / S3 / S4)", lambda r: " / ".join(f"{o:.0f}" for o in r["seat_occ"]))
    line("people model s/frame", lambda r: f"{r['sec_people']:.3f}" if r["sec_people"] else "-")
    if all("passes" in r for r in rows):
        m0 = rows[0]["passes"]
        L.append(f"| **passes vs manual** (window {m0['window_s']}s, offset {m0['offset_s']}s) | " +
                 " | ".join("" for _ in rows) + " |")
        line("system passes / manual", lambda r: f"{r['passes']['system_passes']} / {r['passes']['truth_passes']}")
        line("matched", lambda r: str(r["passes"]["matched"]))
        line("missed", lambda r: str(r["passes"]["missed"]))
        line("invented", lambda r: str(r["passes"]["invented"]))
        line("receiver correct", lambda r: str(r["passes"]["receiver_correct"]))
        line("who-to-whom correct", lambda r: str(r["passes"]["who_to_whom_correct"]))
    L += ["", "Runs: " + ", ".join(r["run"] for r in rows)]
    text = "\n".join(L)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
        print("saved", args.out)


if __name__ == "__main__":
    main()
