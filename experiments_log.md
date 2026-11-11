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

## Prior results (reported 10.11, code not saved, not re-verified)

Count_passes v1 vs Gavish's count (46 passes), 0.5 s window, one-to-one matching:

| Run | Touches | Passes | Matched | Missed | Invented | Receiver correct | Who-to-whom correct |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (count_passes v1) | 96 | 47 | 26 | 20 | 21 | 9 | 1 |
| With second-ball filter | 75 | 42 | 25 | 21 | 17 | 10 | 4 |

Use as a rough reference only; new runs produce their own baseline in the same setup.
