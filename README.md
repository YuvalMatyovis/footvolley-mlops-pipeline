# Footvolley Analytics Pipeline

Computer-vision pipeline for footvolley and ball-keeping circles: detect the players and the ball in every frame, assign players to seats around the circle, and count touches and passes.

Status: **proof of concept**. Everything below was measured on one 58-second clip (`rally_56`, 464x832, 60 fps, 4 players) plus a 40-frame hand-labeled test set. Treat the numbers as directional, not final.

## How it works

```
video -> perception.py -> tracking_data.json -> count_passes.py -> pass_events.json -> render_events.py -> events_preview.mp4
          players: yolo11n (COCO) or RF-DETR (--people-backend rfdetr-small)
          ball:    ball_v2 (custom single-class detector)
```

1. **`perception.py`** runs the two detectors on every frame and writes `tracking_data.json`, `meta.json` and a preview video (orange boxes are small, far-away players that an older size filter used to drop).
2. **`count_passes.py`** cleans the ball track (jump, spike and stuck-detection filters), fills short gaps (up to 0.25 s), assigns players to seats by position, detects touches (sharp velocity change while the ball is near a player) and merges same-seat touch bursts (ball-track noise next to one player) into one contact, groups touches into possessions, and counts a pass when the possession changes seat with a flight of at least 0.2 s.
3. **`render_events.py`** burns seat IDs, touch flashes and live counters into the video so the result can be checked by eye against a manual count.

## Results so far

| Experiment | Result |
| :--- | :--- |
| `ball_v2` on 40 hand-labeled test frames (`foochi_test`) | precision 0.89, recall 0.73, F1 0.80 (preliminary; the 95% interval on recall is roughly 56-85%) |
| V2 vs V3 on the first 2,690 frames of `rally_56` (V3 filtered to its `ball` class) | ball found in 65.5% vs 12.3% of frames, average confidence 0.71 vs 0.35, average players per frame 2.02 vs 0.90 |
| Final run, V2 + yolo11n (people at imgsz 1280, conf 0.25, min box area 1500) on the full clip | ball in 80.1% of frames (avg conf 0.68, longest gap 0.5 s), 3.23 players per frame on average, at least 3 players in 82% of frames |
| `count_passes` on that output | raw detector: 132 touches / 59 passes. After noise filtering (burst merge 0.4 s, min flight 0.2 s): 96 touches / 47 passes. Manual count: 47 passes (about 52 touches expected). Pass total matches, but the thresholds were tuned on this one clip and the result is sensitive to min flight (0.15 s gives 53, 0.25 s gives 41), so the match is partly luck; touches are still overcounted |

Notes:
- Detection rate is not accuracy. There is no ground truth for `rally_56` yet; the detections were checked visually in the preview video.
- `ball_v3` (extra frames labeled by a collaborator, classes `ball` and `person`) looked excellent at first (99.5% detection) only because the `person` class was being counted as a ball. With a proper class filter it finds the ball much less often than `ball_v2`. Its validation numbers (P 0.91 / R 0.84) are probably averaged over both classes and are not comparable. Why it is weaker has not been diagnosed (labels, class setup and resolution are the suspects).
- The people-detection settings (imgsz 1280, conf 0.25, min area 1500) were chosen to catch the player farthest from the camera and confirmed by eye, not by a systematic search.

### Update 11.11: pass-by-pass evaluation

Matching only the total pass count turned out to be misleading, so passes are now compared one by one with the manual count (`evaluate_passes.py`). A system pass counts only if it is at the right time (0.5 s window, manual times shifted 0.2 s for the key-press delay) **and** goes to the right receiver. Each result is shown next to a "chance" number (the same matching with the system times shifted by a few seconds), because with a pass every ~1.25 s some matches happen by luck.

| Run on `rally_56` (47 manual passes) | right receiver | chance | invented |
| :--- | ---: | ---: | ---: |
| yolo11n players | 15 | 8.5 | 32 |
| RF-DETR Small players | 15 | 7 | 20 |

So the pass detection is still weak: about 15 out of 47, not much above chance. RF-DETR Small gives the same hits with fewer invented passes and steadier seats, so it is the preferred player detector for now (license still to be checked). Three quick ideas to clean the ball track (background filter, touch = break in the arc, picking the ball by trajectory) did not help. Checking the false touches by eye showed the main problem is the ball detector: fixed objects on the sand are often picked as the ball, and on the 57 frames Ido labeled the ball is missed in at least 20% of them. Details and all numbers: [`experiments_log.md`](experiments_log.md) and [`results/`](results/).

## Quickstart

```bash
pip install -r requirements.txt
# put the clip in footvolley/data/clips/ and the weights in footvolley/models/ (weights are not in git)

python footvolley/src/perception.py --clip footvolley/data/clips/rally_56.mp4
python footvolley/src/count_passes.py --run footvolley/outputs/<run folder>
python footvolley/src/render_events.py --video footvolley/data/clips/rally_56.mp4 --run footvolley/outputs/<run folder>

# compare with the manual count, and see why passes were missed
python footvolley/src/evaluate_passes.py --run footvolley/outputs/<run folder> --truth footvolley/ground_truth/rally_56_manual_gavish_2026-11-10.json
python footvolley/src/diagnose_misses.py --run footvolley/outputs/<run folder> --truth footvolley/ground_truth/rally_56_manual_gavish_2026-11-10.json
```

The notebook `notebooks/exp_people_detector.ipynb` runs the same steps on Colab (clip and weights from Google Drive).

Perception is much faster on a GPU (for example a free Google Colab T4) than on CPU, where the 58 s clip takes a long time.

## Repository layout

```
footvolley/
  src/          active pipeline: perception.py, count_passes.py, render_events.py,
                evaluation: evaluate_passes.py, diagnose_misses.py, compare_people.py, evaluate_detectors.py
  tools/        data preparation: frame extraction, label export -> YOLO dataset
  ground_truth/ manual counts from the team's pass counter (the reference for evaluation)
  training/     model cards and training logs (ball_v1, ball_v2, ball_v3)
  models/       weights (git-ignored)   data/  clips, labels (git-ignored)   outputs/  runs (git-ignored)
notebooks/      Colab notebook for GPU runs
results/        small result files from each experiment (one folder per experiment)
docs/           status reports and early experiment notes
experiments_log.md  one entry per experiment: what was tested, result, decision
```

## Known limitations

- Seats are anchored where the players stand in the first frame with four visible players. Players who swap places or leave the frame for long periods can inherit the wrong seat.
- Touches are inferred from the ball trajectory only. False ball detections near a player create false touches, so the touch count is probably too high until the thresholds are calibrated against a manual count.
- Drops (ball hitting the floor) are not detected yet.
- One clip, one camera setup; no claim about other scenes, colorways or moving cameras.

## Next steps

- [x] Compare passes one by one with the manual count (done 11.11, see above).
- [ ] Try a multi-frame ball tracker (for example VballNet or WASB, licenses to check) instead of a single-frame ball detector.
- [ ] Filter fixed objects that are detected as the ball (they don't move over time).
- [ ] Retrain the ball detector with more labeled frames; get a second clip to validate the thresholds.
- [ ] Drop detection, per-player statistics, longest chain without a drop.
- [ ] Audit the labels used for `ball_v3` and retrain.
- [ ] Later: experiment tracking (MLflow), data versioning (DVC), CI, Docker, a simple dashboard.

## Licensing notes (to verify before any commercial use)

- The detectors are built with Ultralytics YOLO, which is released under AGPL-3.0; a closed-source commercial product needs an Ultralytics commercial license. Check their current terms.
- The public FOTEVOLLY dataset used for `ball_v1` has an unverified license; credit the author before publishing.

Early ball-detector experiments are kept in [`docs/early-experiments.md`](docs/early-experiments.md).

## Credits

- Ido labeled the additional ~400 frames used to train `ball_v3`.
- Ball detector `ball_v1` was fine-tuned from the public FOTEVOLLY dataset (license to be verified).
