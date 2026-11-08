# Footvolley Analytics Pipeline

Computer-vision pipeline for footvolley and ball-keeping circles: detect the players and the ball in every frame, assign players to seats around the circle, and count touches and passes.

Status: **proof of concept**. Everything below was measured on one 58-second clip (`rally_56`, 464x832, 60 fps, 4 players) plus a 40-frame hand-labeled test set. Treat the numbers as directional, not final.

## How it works

```
video -> perception.py -> tracking_data.json -> count_passes.py -> pass_events.json -> render_events.py -> events_preview.mp4
          players: yolo11n (COCO), high resolution
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

## Quickstart

```bash
pip install -r requirements.txt
# put the clip in footvolley/data/clips/ and the weights in footvolley/models/ (weights are not in git)

python footvolley/src/perception.py --clip footvolley/data/clips/rally_56.mp4
python footvolley/src/count_passes.py --run footvolley/outputs/<run folder>
python footvolley/src/render_events.py --video footvolley/data/clips/rally_56.mp4 --run footvolley/outputs/<run folder>
```

Perception is much faster on a GPU (for example a free Google Colab T4) than on CPU, where the 58 s clip takes a long time.

## Repository layout

```
footvolley/
  src/          active pipeline: perception.py, count_passes.py, render_events.py, evaluate_detectors.py
  tools/        data preparation: frame extraction, label export -> YOLO dataset
  experiments/  earlier or one-off scripts (kept for reference, may not run as is)
  training/     model cards and training logs (ball_v1, ball_v2, ball_v3)
  models/       weights (git-ignored)   data/  clips, labels (git-ignored)   outputs/  runs (git-ignored)
docs/           status reports and early experiment notes
```

## Known limitations

- Seats are anchored where the players stand in the first frame with four visible players. Players who swap places or leave the frame for long periods can inherit the wrong seat.
- Touches are inferred from the ball trajectory only. False ball detections near a player create false touches, so the touch count is probably too high until the thresholds are calibrated against a manual count.
- Drops (ball hitting the floor) are not detected yet.
- One clip, one camera setup; no claim about other scenes, colorways or moving cameras.

## Next steps

- [ ] Check the pass list in `render_events.py` video against the manual count pass by pass (not just the total); get a second clip to validate the thresholds.
- [ ] Label about 30 frames of `rally_56` to get real precision/recall for the ball and for far players.
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
