# Experiments log

One entry per run. One change per experiment. Agree on the three lines (what / clip / success) **before** running.

Reference for evaluation: manual counts made by the team with the pass counter (`footvolley/ground_truth/`).
Main metric: pass-by-pass match against the manual count, 0.5 s window, reported separately as
missed, invented (no netting out), receiver correct, and who-to-whom correct. Totals are secondary.

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

## Prior results (reported 10.11, code not saved, not re-verified)

Count_passes v1 vs Gavish's count (46 passes), 0.5 s window, one-to-one matching:

| Run | Touches | Passes | Matched | Missed | Invented | Receiver correct | Who-to-whom correct |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (count_passes v1) | 96 | 47 | 26 | 20 | 21 | 9 | 1 |
| With second-ball filter | 75 | 42 | 25 | 21 | 17 | 10 | 4 |

Use as a rough reference only; new runs produce their own baseline in the same setup.
