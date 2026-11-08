# ball_v3 (ball + person) - NOT used; kept for reference

- Base: fine-tuned from `ball_v2.pt` in Google Colab, 50 epochs, training imgsz 1088 (checkpoint metadata).
- Classes: `{0: ball, 1: person}`. Trained on a larger set of frames (about 400) labeled by a collaborator in addition to our own.
- Reported validation numbers: precision 0.91, recall 0.84. These are probably averaged over both classes (the person class is easy), so they say little about the ball. Not verified.
- First use in the pipeline was misleading: without a class filter the `person` class was counted as a ball, giving a 99.5% "ball detection rate" with most boxes on a distant, static person.
- With the ball class selected (`classes=[0]`) on the first 2,690 frames of `rally_56`: ball found in 12.3% of frames (avg confidence 0.35) vs 65.5% for `ball_v2` under the same settings; its person detections were also weaker than yolo11n (0.90 vs 2.02 per frame at the older settings).
- Not diagnosed yet. Suspects: label quality or inconsistent ball boxes in the added data, class setup, resolution mismatch. A lower confidence threshold (0.10-0.15) was not tried.
- Decision: keep `ball_v2` for the ball and `yolo11n` for people until the added labels are audited and the model retrained.
- Weights: `footvolley/models/ball_v3.pt` (not in git).
