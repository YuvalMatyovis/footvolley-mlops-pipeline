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

### 2026-11-11 - people detector: yolo11n vs RF-DETR Nano vs RF-DETR Small (ALT-38) - planned
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
- **Results:**
- **Notes / decision:**

---

## Prior results (reported 10.11, code not saved, not re-verified)

Count_passes v1 vs Gavish's count (46 passes), 0.5 s window, one-to-one matching:

| Run | Touches | Passes | Matched | Missed | Invented | Receiver correct | Who-to-whom correct |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (count_passes v1) | 96 | 47 | 26 | 20 | 21 | 9 | 1 |
| With second-ball filter | 75 | 42 | 25 | 21 | 17 | 10 | 4 |

Use as a rough reference only; new runs produce their own baseline in the same setup.
