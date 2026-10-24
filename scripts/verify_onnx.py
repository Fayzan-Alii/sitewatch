import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from backend.app.config import Settings, settings
from backend.app.services.yolo_service import Detector


def iou(a, b):
    left = np.maximum(a[:2], b[:2])
    right = np.minimum(a[2:], b[2:])
    inter = np.prod(np.maximum(0, right - left))
    return inter / max(
        1e-8,
        np.prod(np.array(a[2:]) - a[:2]) + np.prod(np.array(b[2:]) - b[:2]) - inter,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(settings.model_path).with_suffix(".pt")))
    p.add_argument("--image", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()
    image = cv2.imread(a.image)
    pt = Detector(
        Settings(model_path=a.model, image_size=a.imgsz, enable_vehicles=False)
    )
    ort = Detector(
        Settings(
            model_path=str(Path(a.model).with_suffix(".onnx")),
            runtime="onnx",
            image_size=a.imgsz,
            enable_vehicles=False,
        )
    )
    # Square input gives both runtimes identical letterboxing (Ultralytics uses rect by default).
    h, w = image.shape[:2]
    side = max(h, w)
    square = np.full((side, side, 3), 114, np.uint8)
    square[
        (side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w
    ] = image
    x = pt.predict(square)
    y = ort.predict(square)
    matches = []
    used = set()
    for d in x:
        candidates = [
            (j, e)
            for j, e in enumerate(y)
            if e["class_id"] == d["class_id"] and j not in used
        ]
        if candidates:
            j, e = max(candidates, key=lambda pair: iou(d["box"], pair[1]["box"]))
            used.add(j)
            matches.append(
                {
                    "iou": float(iou(d["box"], e["box"])),
                    "confidence_delta": abs(d["confidence"] - e["confidence"]),
                }
            )
    passed = (
        bool(x)
        and len(x) == len(y) == len(matches)
        and all(m["iou"] > 0.95 and m["confidence_delta"] < 0.02 for m in matches)
    )
    out = {
        "passed": passed,
        "pytorch_count": len(x),
        "onnx_count": len(y),
        "matches": matches,
        "provider": ort.provider,
        "image": a.image,
        "model": a.model,
    }
    Path("results/onnx-verification.json").write_text(json.dumps(out, indent=2))
    print(out)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
