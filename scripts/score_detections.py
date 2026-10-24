"""Measure operating-point precision/recall with unique same-class IoU matching.
This measures object detections, not the correctness of site safety incidents.
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def iou(box, boxes):
    if len(boxes) == 0:
        return np.array([])
    boxes = np.asarray(boxes)
    intersection = np.maximum(
        0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])
    ).prod(1)
    union = (
        (box[2] - box[0]) * (box[3] - box[1])
        + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        - intersection
    )
    return intersection / np.maximum(union, 1e-9)


def match_counts(predictions, ground_truth, threshold, names):
    counts = {int(c): dict(tp=0, fp=0, fn=0) for c in names}
    matched = set()
    for detection in sorted(predictions, key=lambda d: -d["confidence"]):
        if detection["confidence"] < threshold:
            continue
        cid = detection["class_id"]
        if cid not in counts:
            continue
        possible = [
            j
            for j, g in enumerate(ground_truth)
            if g["class_id"] == cid and j not in matched
        ]
        overlaps = iou(
            np.asarray(detection["box"]), [ground_truth[j]["box"] for j in possible]
        )
        if len(overlaps) and overlaps.max() >= 0.5:
            matched.add(possible[int(overlaps.argmax())])
            counts[cid]["tp"] += 1
        else:
            counts[cid]["fp"] += 1
    for j, gt in enumerate(ground_truth):
        if j not in matched:
            counts[gt["class_id"]]["fn"] += 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data", default="configs/construction-ppe.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg["path"])
    names = cfg["names"]
    val = Path(cfg["val"])
    val = val if val.is_absolute() else root / val
    images = (
        [Path(x) for x in val.read_text().splitlines()]
        if val.is_file()
        else sorted(val.glob("*"))
    )
    model = YOLO(args.model)
    thresholds = [0.2, 0.35, 0.5, 0.65, 0.8]
    all_counts = {
        t: {int(c): dict(tp=0, fp=0, fn=0) for c in names} for t in thresholds
    }
    for path in images:
        im = cv2.imread(str(path))
        h, w = im.shape[:2]
        labels = (
            path.parent.parent.parent
            / "labels"
            / path.parent.name
            / (path.stem + ".txt")
        )
        gt = []
        for line in labels.read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            gt.append(
                {
                    "class_id": int(c),
                    "box": [
                        (x - bw / 2) * w,
                        (y - bh / 2) * h,
                        (x + bw / 2) * w,
                        (y + bh / 2) * h,
                    ],
                }
            )
        result = model.predict(
            im,
            conf=0.2,
            iou=0.45,
            imgsz=args.imgsz,
            device=0,
            rect=False,
            verbose=False,
        )[0]
        predictions = [
            {
                "class_id": int(b.cls.item()),
                "confidence": float(b.conf.item()),
                "box": b.xyxy[0].tolist(),
            }
            for b in result.boxes
        ]
        for threshold in thresholds:
            counts = match_counts(predictions, gt, threshold, names)
            for cid, values in counts.items():
                for key, value in values.items():
                    all_counts[threshold][cid][key] += value
    rows = []
    for threshold, classes in all_counts.items():
        for cid, values in classes.items():
            tp, fp, fn = values["tp"], values["fp"], values["fn"]
            rows.append(
                {
                    "threshold": threshold,
                    "class_id": cid,
                    "name": names[cid],
                    **values,
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / (tp + fn) if tp + fn else None,
                }
            )
    out = {
        "model": args.model,
        "data": args.data,
        "imgsz": args.imgsz,
        "images": len(images),
        "iou": 0.5,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
