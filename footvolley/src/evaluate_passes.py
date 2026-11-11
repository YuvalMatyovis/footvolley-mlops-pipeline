"""evaluate_passes.py - compare the system's passes with a manual count, pass by pass.

Inputs:
    --run    a run folder with pass_events.json (from count_passes.py)
    --truth  a manual count from the team's pass counter (footvolley/ground_truth/*.json)

MAIN metric (from 11.11): a system pass counts only if it is at the right time AND goes to the
same receiver (to_seat) as the manual pass. Time-only matching was found to be close to chance level
(passes come about every 1.25 s, so a 0.5 s window catches something almost always).

Matching (one-to-one):
    1. The manual times are shifted earlier by a FIXED --offset (key-press delay, 0.2 s measured
       on 11.11), the same for every run.
    2. Pairs with the same receiver are sorted by time difference; each pair is matched if both are
       still free and the difference is <= --window (0.5 s).

Reported: matched / chance (median matches when the system's times are shifted by several seconds,
i.e. pure coincidence) / missed / invented / who-to-whom correct (receiver and passer both right).
Only the part of "matched" above "chance" is real.

Writes pass_eval.json into the run folder.

Usage:
    python footvolley/src/evaluate_passes.py --run footvolley/outputs/<run> \
        --truth footvolley/ground_truth/rally_56_manual_gavish_2026-11-10.json
"""
import argparse
import json
import os
import statistics


def match(truth, system, window, offset, same_receiver=False):
    """truth/system: lists of dicts with time_s, from_seat, to_seat. Returns metrics and pairs.
    same_receiver=True: a pair can only match if the receiver (to_seat) is the same (the main metric)."""
    t_times = [t["time_s"] - offset for t in truth]
    pairs = sorted((abs(tt - s["time_s"]), i, j) for i, tt in enumerate(t_times) for j, s in enumerate(system)
                   if not same_receiver or truth[i]["to_seat"] == s["to_seat"])
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


def chance_level(truth, system, window, offset, same_receiver=True,
                 shifts=(3, 5, 7, 9, 11, 13, 17, 19, 23, 29)):
    """Matches expected by chance: the system's times are shifted (cyclically over the clip) so that any
    remaining match is a coincidence. Returns the median number of matches over the shifts."""
    if not system:
        return 0.0
    end = max(max(t["time_s"] for t in truth), max(s["time_s"] for s in system)) + 0.1
    counts = []
    for sh in shifts:
        moved = [dict(s, time_s=(s["time_s"] + sh) % end) for s in system]
        counts.append(match(truth, moved, window, offset, same_receiver)[0]["matched"])
    return statistics.median(counts)


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
    ap.add_argument("--offset", type=float, default=0.2,
                    help="key-press delay subtracted from manual times (0.2 s measured on 11.11, see experiments_log)")
    ap.add_argument("--events", default="pass_events.json", help="events file in the run folder")
    args = ap.parse_args()

    truth = json.load(open(args.truth))["passes"]
    system = json.load(open(os.path.join(args.run, args.events)))["passes"]

    # MAIN: a pass counts only if it is at the right time AND goes to the same receiver
    main_m, pairs = match(truth, system, args.window, args.offset, same_receiver=True)
    main_m["chance"] = chance_level(truth, system, args.window, args.offset, same_receiver=True)
    # reference only: time-only matching (the old metric, close to chance level)
    time_m, _ = match(truth, system, args.window, args.offset)
    time_m["chance"] = chance_level(truth, system, args.window, args.offset, same_receiver=False)

    print("=" * 86)
    print(f" {os.path.basename(os.path.normpath(args.run))} [{args.events}] vs {os.path.basename(args.truth)}")
    print(f" manual passes {len(truth)} | system passes {len(system)} | window {args.window}s, offset {args.offset}s")
    print("-" * 86)
    print(f" {'':<40}{'matched':>8}{'chance':>8}{'missed':>8}{'invented':>9}{'who->whom':>11}")
    print(f" {'MAIN: right time + right receiver':<40}{main_m['matched']:>8}{main_m['chance']:>8.1f}"
          f"{main_m['missed']:>8}{main_m['invented']:>9}{main_m['who_to_whom_correct']:>11}")
    print(f" {'ref: right time only (old metric)':<40}{time_m['matched']:>8}{time_m['chance']:>8.1f}"
          f"{time_m['missed']:>8}{time_m['invented']:>9}{'':>11}")
    print("=" * 86)

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

    out = {"truth_file": os.path.basename(args.truth), "metric": "right time + right receiver",
           "main": main_m, "ref_time_only": time_m, "passes": detail}
    path = os.path.join(args.run, args.events.replace("pass_events", "pass_eval"))
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
