# ball_v2 (single-class ball detector) - current ball model

- Base: yolo11n, fine-tuned in Google Colab (T4), 50 epochs, training imgsz 640 (checkpoint metadata). In the pipeline it is run at imgsz 1088.
- Classes: `{0: ball}`
- Data: our own hand-labeled frames from three iPhone clips (static and POV, three ball colorways), exported from Ultralytics Platform and converted with `tools/prepare_own_dataset.py`: 80 train frames (72 with a ball), 20 val frames (18 with a ball). Test set: 40 frames from `foochi_test` (33 with a ball), never used for training.
- Test results (40 frames, conf 0.25 fixed): precision 0.89, recall 0.73, F1 0.80. One frame is about 3 points; the 95% interval on recall is roughly 56-85%. Directional evidence only.
- Full-clip check on `rally_56` (58 s): ball found in 80.1% of frames at conf 0.25, average confidence 0.68, longest gap 0.5 s. Checked visually; no ground truth for that clip.
- Weights: `models/ball_v2.pt` (not in git).
