import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES_DIR = ROOT / "data" / "foochi_yolo" / "images" / "test"
TEST_LABELS_DIR = ROOT / "data" / "foochi_yolo" / "labels" / "test"

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    b1_x1, b1_x2 = x1 - w1/2, x1 + w1/2
    b1_y1, b1_y2 = y1 - h1/2, y1 + h1/2
    b2_x1, b2_x2 = x2 - w2/2, x2 + w2/2
    b2_y1, b2_y2 = y2 - h2/2, y2 + h2/2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = w1 * h1
    b2_area = w2 * h2
    
    return inter_area / float(b1_area + b2_area - inter_area) if (b1_area + b2_area) > 0 else 0

def detect_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask = cv2.dilate(cv2.erode(mask1 | mask2, None, iterations=2), None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return [x + w/2, y + h/2, w, h]
    return None

def evaluate():
    models = {}
    
    models["YOLO11n (COCO)"] = (YOLO("yolo11n.pt"), 32)
    
    v1_path = ROOT / "models" / "ball_v1.pt"
    if v1_path.exists():
        models["ball_v1"] = (YOLO(v1_path), 0)
        
    v2_path = ROOT / "models" / "ball_v2.pt"
    if v2_path.exists():
        models["ball_v2"] = (YOLO(v2_path), 0)
    
    results = {name: {"TP": 0, "FP": 0, "FN": 0} for name in models}
    results["HSV Color"] = {"TP": 0, "FP": 0, "FN": 0}
    
    for img_path in TEST_IMAGES_DIR.glob("*.jpg"):
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        
        lbl_path = TEST_LABELS_DIR / f"{img_path.stem}.txt"
        gt_box = None
        if lbl_path.exists() and lbl_path.read_text().strip():
            parts = lbl_path.read_text().strip().split()
            gt_box = [float(p) * dim for p, dim in zip(parts[1:], [w, h, w, h])]
            
        for name, (model, cls_id) in models.items():
            preds = model(frame, verbose=False)[0].boxes
            pred_box = None
            if preds is not None:
                for box in preds:
                    if int(box.cls.item()) == cls_id:
                        pred_box = box.xywh[0].cpu().numpy()
                        break
            _update_metrics(results[name], gt_box, pred_box, w, h)
            
        hsv_box = detect_hsv(frame)
        _update_metrics(results["HSV Color"], gt_box, hsv_box, w, h)

    print("\n=== EVALUATION RESULTS (Test Set: 40 frames) ===")
    print(f"{'Model':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 50)
    for name, m in results.items():
        precision = m['TP'] / (m['TP'] + m['FP']) if (m['TP'] + m['FP']) > 0 else 0
        recall = m['TP'] / (m['TP'] + m['FN']) if (m['TP'] + m['FN']) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{name:<15} | {precision:.2f}      | {recall:.2f}      | {f1:.2f}")

def _update_metrics(metrics, gt_box, pred_box, w, h, iou_thresh=0.3):
    # Fix: explicit None check so NumPy arrays don't crash the logic
    if gt_box is not None and pred_box is not None:
        if iou(gt_box, pred_box) >= iou_thresh:
            metrics["TP"] += 1
        else:
            metrics["FP"] += 1
            metrics["FN"] += 1
    elif gt_box is not None and pred_box is None:
        metrics["FN"] += 1
    elif gt_box is None and pred_box is not None:
        metrics["FP"] += 1

if __name__ == "__main__":
    evaluate()
