"""diagnose_misses.py - why was each manual pass missed?

For every manual pass that evaluate_passes marked "missed", looks at what the system saw
around it (0.6 s before to 0.6 s after) and puts it in one group:
    touch_wrong_seat   a touch was detected near the right time, but at a seat other than the receiver
                       (often the same seat as the previous possession, so no pass was counted)
    touch_right_seat   a touch at the receiver's seat was detected near the right time, but no pass
                       was counted (merged with the previous touch, flight too short, ...)
    no_touch_ball_seen no touch nearby although the ball was detected in most frames
    no_ball            the ball was missing in most frames around the pass

Run it after every experiment and compare the group sizes.

Usage:
    python src/diagnose_misses.py --run outputs/<run> \
        --truth ground_truth/rally_56_manual_gavish_2026-11-10.json [--suffix _bgfilter]
"""
import argparse
import collections
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--suffix", default="", help="e.g. _bgfilter to read pass_events_bgfilter.json")
    args = ap.parse_args()

    data = json.load(open(os.path.join(args.run, "tracking_data.json")))
    fps = float(json.load(open(os.path.join(args.run, "meta.json")))["fps"])
    events = json.load(open(os.path.join(args.run, f"pass_events{args.suffix}.json")))
    ev = json.load(open(os.path.join(args.run, f"pass_eval{args.suffix}.json")))
    truth = json.load(open(args.truth))["passes"]
    offset = ev["main"]["offset_s"]
    touches = events["touches"]

    groups = collections.Counter()
    rows = []
    for p in ev["passes"]:
        if p["status"] != "missed":
            continue
        t = p["truth_time_s"]
        tr = min(truth, key=lambda x: abs(x["time_s"] - offset - t))
        a, b = max(0, int((t - 0.6) * fps)), min(len(data), int((t + 0.6) * fps))
        ball_share = sum(1 for f in data[a:b] if f["ball"]) / max(b - a, 1)
        near = [x for x in touches if abs(x["time_s"] - t) <= 0.6]
        if any(x["seat"] == tr["to_seat"] for x in near):
            g = "touch_right_seat"
        elif near:
            g = "touch_wrong_seat"
        elif ball_share >= 0.5:
            g = "no_touch_ball_seen"
        else:
            g = "no_ball"
        groups[g] += 1
        rows.append(f" {t:6.2f}s  S{tr['from_seat']}->S{tr['to_seat']}  {g:<19} touches near: "
                    f"{[('S%d' % x['seat'], x['time_s']) for x in near]}  ball in {ball_share * 100:.0f}% of frames")

    print("=" * 90)
    print(f" missed passes: {sum(groups.values())}  |  " + "  ".join(f"{k} {v}" for k, v in groups.most_common()))
    print("-" * 90)
    print("\n".join(rows))
    print("=" * 90)
    out = os.path.join(args.run, f"miss_diagnosis{args.suffix}.json")
    json.dump({"groups": dict(groups), "rows": rows}, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
