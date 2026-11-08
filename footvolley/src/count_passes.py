"""count_passes.py - touches and passes from tracking_data.json.

Pipeline: clean ball track -> fill short gaps -> assign players to seats ->
detect touches (sharp velocity change while the ball is near a player) ->
count passes (ball goes from one seat to a different seat).

Usage:
    python footvolley/src/count_passes.py --run footvolley/outputs/<run folder>
(the folder must contain tracking_data.json and meta.json from perception)
"""
import argparse
import json
import math
import os

import numpy as np

# ---- tunables (all time values are in seconds, converted with the clip fps) ----
MAX_BALL_SPEED = 130.0     # px per frame; faster jumps are treated as false detections
MAX_GAP_FILL_S = 0.25      # fill ball gaps up to this long
SMOOTH_WIN = 2             # frames on each side for velocity estimation
TOUCH_NEAR_PX = 45.0       # ball must be this close to a player box (px)
MIN_DV = 3.5               # min change of velocity (px/frame) to count as a touch
MIN_TOUCH_GAP_S = 0.15     # min time between two touches
MERGE_SAME_SEAT_S = 0.4      # same-seat touches closer than this = one contact (noise)
MIN_FLIGHT_S = 0.2           # min time between touches of two seats to count a pass
SEAT_MAX_DIST = 150.0      # max distance (px) between a detection and its seat
SEAT_EMA = 0.15            # how fast seat anchors follow their player
SPIKE_PX = 45.0            # a detection this far from both neighbours (that are close to each other) is a spike
STATIC_PX = 8.0            # detections that stay within this radius ...
STATIC_S = 0.40            # ... for at least this long are a stuck false detection, not a flying ball
MIN_SPEED = 2.0            # a touch needs the ball to move at least this fast (px/frame) before or after


def load_run(run_dir):
    data = json.load(open(os.path.join(run_dir, "tracking_data.json")))
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    return data, meta


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def clean_ball_track(data):
    """Return an array [n, 2] of ball centers (nan = missing) after removing
    detections that jump too far from the last accepted one."""
    n = len(data)
    pos = np.full((n, 2), np.nan)
    last_i, last_p = None, None
    rejected = 0
    for i, fr in enumerate(data):
        b = fr["ball"]
        if not b:
            continue
        p = np.array(box_center(b))
        if last_p is not None:
            gap = i - last_i
            limit = MAX_BALL_SPEED * min(gap, 6)
            if np.hypot(*(p - last_p)) > limit:
                rejected += 1
                # allow recovery: after a long silence accept the new position
                if gap < 30:
                    continue
        pos[i] = p
        last_i, last_p = i, p
    return pos, rejected


def despike(pos):
    """Remove single-frame detections that jump away from the track and come back."""
    out = pos.copy()
    valid = np.where(~np.isnan(pos[:, 0]))[0]
    removed = 0
    for k in range(1, len(valid) - 1):
        a, b, c = valid[k - 1], valid[k], valid[k + 1]
        if c - a > 8:
            continue
        d_ab = np.hypot(*(pos[b] - pos[a]))
        d_bc = np.hypot(*(pos[c] - pos[b]))
        d_ac = np.hypot(*(pos[c] - pos[a]))
        if d_ab > SPIKE_PX and d_bc > SPIKE_PX and d_ac < SPIKE_PX:
            out[b] = np.nan
            removed += 1
    return out, removed


def drop_static_runs(pos, fps):
    """Remove long runs where the 'ball' does not move at all (stuck false detection)."""
    out = pos.copy()
    valid = np.where(~np.isnan(pos[:, 0]))[0]
    min_len = int(round(STATIC_S * fps))
    removed = 0
    i = 0
    while i < len(valid):
        anchor = pos[valid[i]]
        j = i
        while j + 1 < len(valid) and valid[j + 1] - valid[j] <= 6 and np.hypot(*(pos[valid[j + 1]] - anchor)) < STATIC_PX:
            j += 1
        if valid[j] - valid[i] >= min_len:
            for k in range(i, j + 1):
                out[valid[k]] = np.nan
                removed += 1
            i = j + 1
        else:
            i += 1
    return out, removed


def fill_gaps(pos, fps):
    max_gap = int(round(MAX_GAP_FILL_S * fps))
    filled = pos.copy()
    valid = np.where(~np.isnan(pos[:, 0]))[0]
    n_filled = 0
    for a, b in zip(valid[:-1], valid[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                filled[a + k] = pos[a] * (1 - t) + pos[b] * t
                n_filled += 1
    return filled, n_filled


def assign_seats(data, n_seats=4):
    """Seats are anchored where the players stand in the first frame that shows
    n_seats people (sorted left to right), then follow them slowly."""
    start = next((i for i, f in enumerate(data) if len(f["people"]) >= n_seats), None)
    if start is None:
        start = max(range(len(data)), key=lambda i: len(data[i]["people"]))
    ppl = sorted(data[start]["people"], key=lambda p: -p[4])[:n_seats]
    ppl = sorted(ppl, key=lambda p: (p[0] + p[2]) / 2.0)
    anchors = [np.array([(p[0] + p[2]) / 2.0, p[3]]) for p in ppl]
    n_seats = len(anchors)
    per_frame = []   # list of dict seat_index -> person box
    for fr in data:
        people = fr["people"]
        pairs = []
        for si, a in enumerate(anchors):
            for pi, p in enumerate(people):
                foot = np.array([(p[0] + p[2]) / 2.0, p[3]])
                pairs.append((float(np.hypot(*(foot - a))), si, pi))
        pairs.sort()
        used_s, used_p, assign = set(), set(), {}
        for dist, si, pi in pairs:
            if dist > SEAT_MAX_DIST:
                break
            if si in used_s or pi in used_p:
                continue
            used_s.add(si)
            used_p.add(pi)
            assign[si] = people[pi]
            p = people[pi]
            foot = np.array([(p[0] + p[2]) / 2.0, p[3]])
            anchors[si] = (1 - SEAT_EMA) * anchors[si] + SEAT_EMA * foot
        per_frame.append(assign)
    return per_frame, n_seats


def box_distance(pt, b):
    dx = max(b[0] - pt[0], 0, pt[0] - b[2])
    dy = max(b[1] - pt[1], 0, pt[1] - b[3])
    return math.hypot(dx, dy)


def detect_touches(pos, seats_per_frame, fps):
    n = len(pos)
    w = SMOOTH_WIN
    dv = np.full(n, np.nan)
    for t in range(w, n - w):
        a, c, b = pos[t - w], pos[t], pos[t + w]
        if np.isnan(a[0]) or np.isnan(c[0]) or np.isnan(b[0]):
            continue
        v_in = (c - a) / w
        v_out = (b - c) / w
        if max(np.hypot(*v_in), np.hypot(*v_out)) < MIN_SPEED:
            continue
        dv[t] = float(np.hypot(*(v_out - v_in)))
    min_gap = int(round(MIN_TOUCH_GAP_S * fps))
    touches = []
    t = w
    while t < n - w:
        if np.isnan(dv[t]) or dv[t] < MIN_DV:
            t += 1
            continue
        # local maximum within the next min_gap frames
        end = min(t + min_gap, n - w)
        best = t + int(np.nanargmax(dv[t:end + 1]))
        # nearest seat within a small window around the peak
        cand = None
        for tt in range(max(0, best - 3), min(n, best + 4)):
            if np.isnan(pos[tt][0]):
                continue
            for si, box in seats_per_frame[tt].items():
                d = box_distance(pos[tt], box)
                if cand is None or d < cand[0]:
                    cand = (d, si, tt)
        if cand is not None and cand[0] <= TOUCH_NEAR_PX:
            touches.append({"frame": int(best), "time_s": round(best / fps, 2),
                            "seat": int(cand[1]) + 1, "dist_px": round(cand[0], 1),
                            "dv": round(float(dv[best]), 1),
                            "ball": [int(pos[best][0]), int(pos[best][1])]})
            t = best + min_gap
        else:
            t += 1
    return touches


def merge_bursts(touches):
    """Noise suppression: a second 'touch' by the same seat within MERGE_SAME_SEAT_S
    of the previous kept one is treated as jitter of the same contact (a noisy ball
    track next to one player) and dropped."""
    kept = []
    for x in touches:
        if kept and kept[-1]["seat"] == x["seat"] and x["time_s"] - kept[-1]["time_s"] < MERGE_SAME_SEAT_S:
            continue
        kept.append(x)
    return kept


def count_passes(touches):
    """Group consecutive touches of the same seat into one possession. A pass is a
    change of possession whose flight (last touch of A -> first touch of B) lasts at
    least MIN_FLIGHT_S. Faster seat flips are noise, not a real pass."""
    if not touches:
        return [], 0
    poss = [[touches[0]]]
    for x in touches[1:]:
        if x["seat"] == poss[-1][-1]["seat"]:
            poss[-1].append(x)
        else:
            poss.append([x])
    passes, skipped = [], 0
    for a, b in zip(poss[:-1], poss[1:]):
        flight = b[0]["time_s"] - a[-1]["time_s"]
        if flight < MIN_FLIGHT_S:
            skipped += 1
            continue
        passes.append({"from_seat": a[0]["seat"], "to_seat": b[0]["seat"],
                       "frame": b[0]["frame"], "time_s": b[0]["time_s"],
                       "flight_s": round(flight, 2)})
    return passes, len(poss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with tracking_data.json + meta.json")
    ap.add_argument("--seats", type=int, default=4)
    args = ap.parse_args()

    data, meta = load_run(args.run)
    fps = float(meta["fps"])
    pos, rejected = clean_ball_track(data)
    pos, spikes = despike(pos)
    pos, static = drop_static_runs(pos, fps)
    raw_n = int(np.sum(~np.isnan(pos[:, 0])))
    pos, filled = fill_gaps(pos, fps)
    seats, n_seats = assign_seats(data, args.seats)
    raw_touches = detect_touches(pos, seats, fps)
    touches = merge_bursts(raw_touches)
    passes, longest = count_passes(touches)

    occ = [np.mean([si in s for s in seats]) * 100 for si in range(n_seats)]
    print("=" * 62)
    print(f" frames {len(data)} | {len(data) / fps:.1f}s | {fps:.1f} fps")
    print(f" ball: kept {raw_n} detections | removed: {rejected} jumps, {spikes} spikes, {static} static | filled {filled} frames")
    print(" seat occupancy %: " + "  ".join(f"S{i + 1} {o:.0f}" for i, o in enumerate(occ)))
    print(f" TOUCHES: {len(touches)}   PASSES: {len(passes)}   (possessions: {longest}; raw touches before noise filter: {len(raw_touches)})")
    per_seat = {}
    for t in touches:
        per_seat[t["seat"]] = per_seat.get(t["seat"], 0) + 1
    print(" touches per seat: " + "  ".join(f"S{s}: {c}" for s, c in sorted(per_seat.items())))
    print("=" * 62)

    out = {"fps": fps, "touches": touches, "passes": passes,
           "summary": {"touches": len(touches), "passes": len(passes),
                       "touches_per_seat": per_seat}}
    path = os.path.join(args.run, "pass_events.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
