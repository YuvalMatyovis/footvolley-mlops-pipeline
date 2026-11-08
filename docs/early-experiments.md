# Footvolley / ball-keeping circles

Goal: with 4 players in a circle, count touches, passes between players and
the longest chain without the ball dropping.

## Run
```bash
python footvolley/src/footvolley.py        # tracking + touches/passes + annotated video
python footvolley/src/ball_experiment.py   # ball detector comparison
```
Put the clip in `footvolley/data/foochi_test.mp4`.

## Findings (one 18s clip, no ground truth yet)
- Players: detected and tracked reliably (~90%+ by visual check, stable IDs).
- Ball: the bottleneck. `footvolley.py` detected it in 9.6% of frames, so no
  pass chain was ever formed (best chain = 0).

Ball detector comparison (every 3rd frame, 178 frames, CPU):

| Config | FPS | Ball det. rate | Avg conf | Frames >1 ball |
|---|---|---|---|---|
| yolov8n, 960, conf 0.15 | 4.06 | 10.1% | 0.34 | 0 |
| yolov8s, 1280, conf 0.10 | 0.94 | 8.4% | 0.25 | 0 |
| yolo11n, 1280, conf 0.10 | 2.37 | 16.3% | 0.37 | 0 |
| yolo11s, 1280, conf 0.10 | 0.94 | 29.2% | 0.33 | 0 |
| HSV red color (OpenCV) | 441.73 | 43.8% | - | 25 |

Notes: detection rate is not accuracy. HSV results are not yet visually
verified (25 frames had multiple candidates, so some are false positives).
Off-the-shelf COCO models struggle with the small, blurred ball. Results from
one clip do not generalize (in basketball a bigger model/resolution helped).

## Next steps
- [ ] Verify HSV / yolo11s detections visually
- [ ] Ball tracker: HSV candidates + temporal filter + proximity to players
- [ ] Manual pass count as ground truth
- [ ] Pseudo-labels from HSV, verify ~50, fine-tune YOLO in Colab, before/after table
- [ ] More clips; touch via velocity change; drop detection; progress printing
- [ ] Slim requirements.txt (direct deps, CPU torch)
- [ ] MLflow, DVC, GitHub Actions, Docker

> **Reading this table:** 40 hand-labeled frames from ONE clip (33 contain the ball). One frame is ~3 points and the 95% interval for ball_v2 recall is roughly 56-85%, so this is directional evidence only. conf=0.25 was fixed, not tuned on test. The "insight" column holds hypotheses, not verified causes.
