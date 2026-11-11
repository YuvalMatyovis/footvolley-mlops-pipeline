# Footvolley Analytics Pipeline

A computer-vision project for footvolley / ball-keeping circles. It finds the players and the ball in every frame, gives each player a seat around the circle, and counts touches and passes.

**Status: proof of concept.** Everything here was tested on one 58-second clip (`rally_56`, 4 players, phone on the sand, 60 fps). The numbers are a direction, not a final result.

## How it works

```
video -> perception.py -> count_passes.py -> render_events.py -> preview video
```

1. **`perception.py`** - finds the players (yolo11n, or RF-DETR with `--people-backend rfdetr-small`) and the ball (my own model, `ball_v2`) in every frame.
2. **`count_passes.py`** - cleans the ball track, gives each player a seat (S1-S4), finds touches (the ball changes direction near a player) and counts a pass when the ball moves to another seat.
3. **`render_events.py`** - draws seats, touches and counters on the video, so I can check it by eye.

To check the result against a manual count:
- **`evaluate_passes.py`** - compares the passes one by one with the manual count.
- **`diagnose_misses.py`** - says why each missed pass was missed.
- **`compare_people.py`** - a table to compare player detectors.

## Results so far

**Ball model:** `ball_v2` finds the ball in about 80% of the frames of the clip. On 40 hand-labeled test frames: precision 0.89, recall 0.73 (small test set, so not very exact). `ball_v3` was weaker: it found the ball in 12% of the frames vs 65% for `ball_v2` (details in [`docs/status_2026-11-08.md`](docs/status_2026-11-08.md)).

**Passes:** at first the total looked perfect (47 passes, same as the manual count), but that was partly luck. When I compared pass by pass (right time **and** right receiver), only about 15 of the 47 were right:

| Players detector | right passes (of 47) | by chance | invented passes |
| :--- | ---: | ---: | ---: |
| yolo11n | 15 | 8.5 | 32 |
| RF-DETR Small | 15 | 7 | 20 |

"By chance" = how many would match by luck (passes come every ~1.25 s). Only what is above it is real.

What I learned:
- RF-DETR Small gives fewer invented passes and steadier seats, so it is the preferred player detector for now.
- The main problem is the ball detector: objects on the sand are picked as the ball, and it misses the ball in at least 20% of the frames Ido labeled.
- Three quick fixes to the ball track did not help.

All experiments, one by one: [`experiments_log.md`](experiments_log.md). Result files: [`results/`](results/).

## Quickstart

```bash
pip install -r requirements.txt
# put the clip in data/clips/ and the weights in models/ (not in git)

python src/perception.py --clip data/clips/rally_56.mp4
python src/count_passes.py --run outputs/<run folder>
python src/render_events.py --video data/clips/rally_56.mp4 --run outputs/<run folder>

# compare with the manual count
python src/evaluate_passes.py --run outputs/<run folder> --truth ground_truth/rally_56_manual_gavish_2026-11-10.json
python src/diagnose_misses.py --run outputs/<run folder> --truth ground_truth/rally_56_manual_gavish_2026-11-10.json
```

On CPU it is slow. I run it on a free Colab GPU with `notebooks/exp_people_detector.ipynb` (clip and weights from Google Drive).

## Repository layout

```
src/            the pipeline and the evaluation scripts
tools/          preparing frames and labels for training
training/       model cards for ball_v1, ball_v2, ball_v3
ground_truth/   manual pass counts and Ido's ball labels
notebooks/      Colab notebook for GPU runs
results/        small result files, one folder per experiment
docs/           status reports and early notes
data/ models/ outputs/   clips, weights and runs (local only, not in git)
experiments_log.md   what each experiment tested, the result and the decision
```

## Known limitations

- Only one clip and one camera setup.
- Seats are set from the first frame; if players swap places, the seats can get mixed up.
- Wrong ball detections near a player create false touches.
- Drops (ball on the sand) are not detected yet.

## Next steps

- [ ] Try a ball tracker that uses several frames (for example VballNet or WASB).
- [ ] Filter objects that are detected as the ball but never move.
- [ ] Retrain the ball model with more labeled frames, and test on a second clip.
- [ ] Later: drop detection, per-player stats, MLflow / DVC / CI.

## Licenses (to check before any commercial use)

- The ball models are trained with Ultralytics YOLO (AGPL-3.0). A closed commercial product would need their commercial license.
- RF-DETR license still to be checked.
- The public FOTEVOLLY dataset used for `ball_v1` - license not verified yet.

## Credits

- Ido labeled ~400 frames for `ball_v3` and the ball frames used for checking.
- Gavish did the manual pass count used as the reference.
- `ball_v1` was fine-tuned from the public FOTEVOLLY dataset.
