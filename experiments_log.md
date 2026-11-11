# Experiments log

One entry per run. One change per experiment. Agree on the three lines (what / clip / success) **before** running.

Reference for evaluation: manual counts made by the team with the pass counter (`footvolley/ground_truth/`).
Main metric (from 11.11): a system pass counts only if it is at the right time (0.5 s window,
manual times shifted by a fixed 0.2 s key-press delay) AND goes to the same receiver. Always reported
next to its chance level (the system's times shifted by several seconds). Also: missed, invented,
who-to-whom. Time-only matching is no longer used: it is close to chance level.

## Template

### YYYY-MM-DD - short name
- **What:** the one thing that changes
- **Clip:** e.g. rally_56 (full)
- **Success means:** decided before the run
- **Branch / commit:**
- **Where it ran:** Codespaces / Colab (GPU type) / HPC
- **Settings:** models, imgsz, conf, thresholds
- **Results:**
- **Notes / decision:**

---

### 2026-11-11 - people detector: yolo11n vs RF-DETR Nano vs RF-DETR Small (ALT-38) - done, RF-DETR Small preferred
- **What:** only the player detector changes. Ball model, thresholds and pass logic are identical.
- **Clip:** rally_56 (full, 3,465 frames)
- **Success means:** an RF-DETR run is not worse than yolo11n on the player metrics (players per frame,
  frames with >= 3 / exactly 4, count flicker, seat occupancy) and on the pass metrics vs Gavish's count
  (missed, invented, receiver, who-to-whom; 0.5 s window, fixed 0.1 s offset). What counts as noise vs
  worse is decided by the team after seeing the numbers.
- **Branch / commit:** exp/people-detector-rfdetr
- **Where it ran:** Colab T4 (notebooks/exp_people_detector.ipynb)
- **Also saved for later experiments (not used in this one):** up to 3 ball candidates per frame,
  player boxes below the confidence/size thresholds.
- **Results** (files: `results/2026-11-11_people_detector/`):

| metric | yolo11n | RF-DETR Nano | RF-DETR Small |
| :--- | ---: | ---: | ---: |
| frames with > 4 players (false boxes; no people in the background) | 6.3% | 2.2% | 2.3% |
| player-count flicker | 13.0% | 6.9% | 6.4% |
| frames with exactly 4 players | 35% | 39% | 41% |
| passes matched / missed / invented | 27 / 19 / 20 | 25 / 21 / 19 | 26 / 20 / **9** |
| receiver / who-to-whom correct | 12 / 3 | 10 / 4 | 11 / 7 |
| people model s/frame (T4; RF-DETR not in optimized mode) | 0.016 | 0.022 | 0.031 |

- **Notes / decision:**
  - Run-to-run noise: this yolo11n run has the same settings as the 10.11 run; the ball is identical in
    97.5% of frames, players in 92%, and passes differ by 1 (27/19/20 vs 26/20/21). A difference of
    1-2 passes is noise.
  - 15 of the ~20 missed passes are missed by all three detectors: misses come from the ball/touch stage.
  - Seat swap: when the near player (S1) leaves the frame at ~40 s, yolo11n gives his seat to the left
    player (S2); RF-DETR leaves the seat empty. Confirmed by eye (40-44 s, 46-47 s, 54-58 s).
  - Who-to-whom is based on small numbers and is sensitive to the time offset; do not weigh it heavily.
  - Decision (Sinai, 11.11): prefer RF-DETR Small. Before commercial use: verify the license of the exact
    weights, dependencies and training data (Ido). Measure speed again in RF-DETR's optimized mode.
  - Seat names (agreed 11.11): S1 near the camera, S2 left, S3 far, S4 right.
  - Ground truth note: Gavish's count likely misses one last pass S2->S3 at the very end of the clip
    (~57.7 s, heel touch). Treated as noise; file not changed.

---

### 2026-11-11 - ball background filter v1 - done, FAILED (not merged into the pipeline)
- **What:** remove ball detections at fixed "background spots" (a spot where the ball is seen in many
  different seconds and does not move). Only change; people = RF-DETR Small run, all thresholds as before.
- **Clip:** rally_56, the RF-DETR Small run of 11.11 (local, no Colab)
- **Success means (agreed before):** matched >= 29 (beyond noise from 26), invented <= 9, S1 touches far below 40.
- **Branch:** exp/ball-background-filter (`count_passes.py --bg-filter`)
- **Settings (fixed before the run):** grid cell = median ball size (19 px); spot = seen in >= 12% of the
  clip's seconds (min 5; here 7) and median distance from the spot's center <= 0.35 ball size
- **Results** (files: `results/2026-11-11_ball_bg_filter_v1/`):

| | touches (S1/S2/S3/S4) | passes | matched | missed | invented | receiver | who-to-whom |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (RF-DETR Small) | 86 (40/21/6/19) | 35 | 26 | 20 | 9 | 11 | 7 |
| background filter v1 | 59 (14/14/11/20) | 41 | 27 | 19 | 14 | 13 | 7 |

  Missed-pass groups: baseline touch_right_seat 9, touch_wrong_seat 8, no_touch_ball_seen 3 ->
  filter touch_right_seat 4, touch_wrong_seat 6, no_touch_ball_seen 8, no_ball 1.
- **Notes / decision:** FAILED on matched and invented. The S1 overcount is gone (40 -> 14 touches) and
  the spots at the left edge (x 25-90, y 230-360) are the object seen by eye. But the filter also marked
  spots in the middle of the circle (x 230-330, y 150-270) where the real ball passes often: 1,071 of
  2,767 ball detections were removed, real ones included, so "no_touch_ball_seen" grew from 3 to 8.
  Cause: with a cell the size of one ball, any cell has a small spread, so the "does not move" test did
  not separate a still object from a moving ball. Next: v2 with a stillness test over time (separate entry).

---

### 2026-11-11 - measurement: main metric, Gavish's delay, oracle mask, Ido's labels - done
- **What:** no pipeline change. Fix the yardstick and find where the errors come from.
- **Clip:** rally_56, the three people-detector runs of 11.11 (local)
- **Files:** `results/2026-11-11_measurement/`, `footvolley/ground_truth/rally_56_ball_labels_ido_2026-11-10.json`

1. **Time-only matching is close to chance.** With the system's times shifted by 3-29 s (pure
   coincidence) time-only matching still gives ~22-25 "matches" vs 25-27 real. New main metric:
   right time + right receiver, with its chance level reported.
2. **Gavish's key-press delay, measured:** pairs where the system and Gavish agree on the receiver
   (58 pairs over the 3 runs): median delay 0.21 s (per run 0.19 / 0.21 / 0.25). Fixed: offset 0.2 s,
   window 0.5 s (below half the ~1.25 s between passes). Not to be changed per experiment.
3. **The three detectors under the new metric** (right receiver / chance / missed / invented / who-to-whom):
   yolo11n 15 / 8.5 / 31 / 32 / 5; RF-DETR Nano 15 / 8.5 / 31 / 29 / 5; RF-DETR Small 15 / 7.0 / 31 / 20 / 8.
   Real signal is about 7-8 passes above chance. RF-DETR Small still best (fewest invented).
4. **Oracle mask** (diagnosis only, never in the pipeline): ball detections at the left-edge object
   (x < 100, y 210-370) and near the camera (y >= 700) removed by hand, RF-DETR Small run:
   right receiver 15 -> 19 (chance 6), invented 20 -> 13, who-to-whom 8 -> 10. S1 touches 40 -> 13.
   Caveat: the mask also removed real balls near the left player (S2 touches 21 -> 6), so the true
   ceiling is somewhat higher. Remaining misses: touch at wrong seat 12, no ball 6, right seat but no
   pass 5, ball seen but no touch 4. Conclusion: the fake ball costs a few passes, but it is not the
   main problem; touch logic and missing detections matter as much.
5. **Ido's 57 labelled frames** (49 with a ball; frame k assumed to be second k-1, one image per
   second): the labelled ball is the chosen ball in 31 (63%), a lower-ranked candidate in 3 (6%), and
   not among the 3 saved candidates (conf >= 0.10) in 15 (31%). Even when searching +-30 frames around
   each assumed frame, 10 of 49 (20%) are never found: the detector misses the ball in at least ~20%
   of these frames. In 2 of 8 frames with no ball the system reported one.
- **Decision:** both code and training are needed. Code (selection by trajectory, touch-to-seat logic)
  can recover part; ~20% detector misses need a better ball model.

---

## Prior results (reported 10.11, code not saved, not re-verified)

Count_passes v1 vs Gavish's count (46 passes), 0.5 s window, one-to-one matching:

| Run | Touches | Passes | Matched | Missed | Invented | Receiver correct | Who-to-whom correct |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (count_passes v1) | 96 | 47 | 26 | 20 | 21 | 9 | 1 |
| With second-ball filter | 75 | 42 | 25 | 21 | 17 | 10 | 4 |

Use as a rough reference only; new runs produce their own baseline in the same setup.
