"""Qualitative comparison on explicitly inspected workplace images.
Does not populate the live dashboard or claim a video/field evaluation.
"""

import json
from pathlib import Path
import cv2
import numpy as np
from backend.app.config import Settings
from backend.app.services.yolo_service import Detector


def main():
    paths = [
        Path("datasets/construction-ppe/images/val/image524.jpeg"),
        Path("datasets/construction-ppe/images/val/image141.jpeg"),
    ]
    models = [
        ("Previous: 5 epochs / 320 px", "models/ppe/weights/best.pt", 320),
        (
            "Updated: 60 more epochs / 640 px",
            "models/ppe_extended/weights/best.pt",
            640,
        ),
    ]
    results = {}
    panels = []
    for title, model, size in models:
        detector = Detector(
            Settings(model_path=model, image_size=size, enable_vehicles=False)
        )
        column = []
        for path in paths:
            im = cv2.imread(str(path))
            predictions = detector.predict(im)
            visible = [d for d in predictions if d["confidence"] >= 0.5]
            results.setdefault(str(path), {})[title] = visible
            canvas = im.copy()
            for d in visible:
                box = list(map(int, d["box"]))
                color = {
                    "helmet": (80, 220, 90),
                    "vest": (220, 180, 40),
                    "Person": (240, 120, 80),
                }.get(d["label"], (100, 200, 230))
                cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), color, 2)
                cv2.putText(
                    canvas,
                    f"{d['label']} {d['confidence']:.2f}",
                    (box[0], max(16, box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                )
            h, w = canvas.shape[:2]
            canvas = cv2.resize(canvas, (640, round(h / w * 640)))
            pane = np.full((700, 660, 3), 245, np.uint8)
            canvas = canvas[:640, :640]
            pane[48 : 48 + canvas.shape[0], 10:650] = canvas
            cv2.putText(
                pane, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1
            )
            cv2.putText(
                pane,
                f"{path.name} | confidence >= 0.50 | predictions, not ground truth",
                (12, 39),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (60, 60, 60),
                1,
            )
            column.append(pane)
        panels.append(np.vstack(column))
    out = Path("results/improvement")
    cv2.imwrite(str(out / "prediction-comparison.jpg"), np.hstack(panels))
    (out / "qualitative-predictions.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
