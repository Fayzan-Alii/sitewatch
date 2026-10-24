import argparse
import json
from pathlib import Path
from backend.app.config import settings
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(settings.model_path).with_suffix(".pt")))
    p.add_argument("--data", default="configs/construction-ppe.yaml")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--output", default="results/metrics.json")
    a = p.parse_args()
    r = YOLO(a.model).val(
        data=a.data,
        device=a.device,
        imgsz=a.imgsz,
        batch=2,
        workers=0,
        plots=False,
        split=a.split,
    )
    out = {
        "model": a.model,
        "dataset": a.data,
        "split": a.split,
        "imgsz": a.imgsz,
        "precision": float(r.box.mp),
        "recall": float(r.box.mr),
        "mAP50": float(r.box.map50),
        "mAP50_95": float(r.box.map),
        "per_class": [
            {
                "class_id": int(cid),
                "name": r.names[int(cid)],
                "precision": float(r.box.p[i]),
                "recall": float(r.box.r[i]),
                "mAP50": float(r.box.ap50[i]),
                "mAP50_95": float(r.box.ap[i]),
            }
            for i, cid in enumerate(r.box.ap_class_index)
        ],
    }
    Path("results").mkdir(exist_ok=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
