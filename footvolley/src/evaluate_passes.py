"""evaluate_passes.py - compare the system's passes with a manual count, pass by pass.

Inputs:
    --run    a run folder with pass_events.json (from count_passes.py)
    --truth  a manual count from the team's pass counter (footvolley/ground_truth/*.json)

Matching (one-to-one):
    1. The manual times are shifted earlier by a FIXED --offset (key-press delay, default 0.1 s),
       the same for every run, so different runs are compared fairly.
    2. All (manual, system) pairs are sorted by time difference; each pair is matched
       if both are still free and the difference is <= --window (default 0.5 s).

Reported separately (no netting out of errors):
    matched / missed (manual passes with no match) / invented (system passes with no match)
    receiver correct (same to_seat) / who-to-whom correct (same from_seat and to_seat)
For reference only, the same numbers at a 1.0 s window and with the per-run median offset
used on 10.11 (that offset is fitted to the run itself, so it is not the main metric).

Writes pass_eval.json into the run folder.

Usage:
    python footvolley/src/evaluate_passes.py --run footvolley/outputs/<run> \
        --truth footvolley/ground_truth/rally_56_manual_gavish_2026-11-10.json
"""
import argparse
import json
import os
import statistics


def match(truth, system, window, offset):
    """truth/system: lists of dicts with time_s, from_seat, to_seat. Returns metrics and pairs."""
    t_times = [t["time_s"] - offset for t in truth]
    pairs = sorted((abs(tt - s["time_s"]), i, j) for i, tt in enumerate(t_times) for j, s in enumerate(system))
    used_t, used_s, matched = set(), set(), []
    for d, i, j in pairs:
        if d > window:
            break
        if i in used_t or j in used_s:
            continue
        used_t.add(i)
        used_s.add(j)
        matched.append((i, j, round(system[j]["time_s"] - t_times[i], 3)))
    receiver = sum(1 for i, j, _ in matched if truth[i]["to_seat"] == system[j]["to_seat"])
    both = sum(1 for i, j, _ in matched if truth[i]["to_seat"] == system[j]["to_seat"]
               and truth[i]["from_seat"] == system[j]["from_seat"])
    metrics = {"truth_passes": len(truth), "system_passes": len(system), "matched": len(matched),
               "missed": len(truth) - len(matched), "invented": len(system) - len(matched),
               "receiver_correct": receiver, "who_to_whom_correct": both,
               "window_s": window, "offset_s": round(offset, 3)}
    return metrics, sorted(matched)


def median_offset(truth, system):
    """The 10.11 method: median of (truth time - nearest system time). Fitted per run."""
    if not system:
        return 0.0
    diffs = [t["time_s"] - min(system, key=lambda s: abs(s["time_s"] - t["time_s"]))["time_s"] for t in truth]
    return statistics.median(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--window", type=float, default=0.5)
    ap.add_argument("--offset", type=float, default=0.1, help="key-press delay subtracted from manual times")
    ap.add_argument("--events", default="pass_events.json", help="events file in the run folder")
    args = ap.parse_args()

    truth = json.load(open(args.truth))["passes"]
    system = json.load(open(os.path.join(args.run, args.events)))["passes"]

    main_m, pairs = match(truth, system, args.window, args.offset)
    wide_m, _ = match(truth, system, 1.0, args.offset)
    med = median_offset(truth, system)
    med_m, _ = match(truth, system, args.window, med)

    def row(name, m):
        return (f" {name:<34}{m['matched']:>8}{m['missed']:>8}{m['invented']:>9}"
                f"{m['receiver_correct']:>10}{m['who_to_whom_correct']:>12}")

    print("=" * 82)
    print(f" {os.path.basename(os.path.normpath(args.run))} [{args.events}] vs {os.path.basename(args.truth)}")
    print(f" manual passes {len(truth)} | system passes {len(system)}")
    print("-" * 82)
    print(f" {'':<34}{'matched':>8}{'missed':>8}{'invented':>9}{'receiver':>10}{'who->whom':>12}")
    print(row(f"MAIN: {args.window}s window, offset {args.offset}s", main_m))
    print(row(f"ref: 1.0s window, offset {args.offset}s", wide_m))
    print(row(f"ref: {args.window}s, run-fitted offset {med:+.2f}s", med_m))
    print("=" * 82)

    t_times = [t["time_s"] - args.offset for t in truth]
    detail = []
    matched_t = {i: (j, d) for i, j, d in pairs}
    matched_s = {j for _, j, _ in pairs}
    for i, t in enumerate(truth):
        if i in matched_t:
            j, d = matched_t[i]
            s = system[j]
            detail.append({"truth_time_s": round(t_times[i], 2), "system_time_s": s["time_s"], "diff_s": d,
                           "truth": [t["from_seat"], t["to_seat"]], "system": [s["from_seat"], s["to_seat"]],
                           "status": "matched"})
        else:
            detail.append({"truth_time_s": round(t_times[i], 2), "truth": [t["from_seat"], t["to_seat"]],
                           "status": "missed"})
    for j, s in enumerate(system):
        if j not in matched_s:
            detail.append({"system_time_s": s["time_s"], "system": [s["from_seat"], s["to_seat"]],
                           "status": "invented"})
    detail.sort(key=lambda x: x.get("truth_time_s", x.get("system_time_s")))

    out = {"truth_file": os.path.basename(args.truth), "main": main_m, "ref_window_1s": wide_m,
           "ref_run_fitted_offset": med_m, "passes": detail}
    path = os.path.join(args.run, args.events.replace("pass_events", "pass_eval"))
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
