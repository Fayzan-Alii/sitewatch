"""Combine reviewed crops with original training set; retain untouched validation."""

import argparse
import json
from pathlib import Path
import yaml
from backend.app.config import settings
from backend.app.db import Store
from backend.app.services.annotations import export_reviews


def prepare():
    base = yaml.safe_load(Path(settings.dataset_path).read_text())
    names = base["names"]
    result = export_reviews(Store(settings.data_dir), names)
    if not result["count"]:
        raise ValueError("No accepted/corrected reviews available")
    root = Path(base["path"]).resolve()
    train = base["train"] if isinstance(base["train"], list) else [base["train"]]
    base["train"] = [str(root / x) for x in train] + [
        str(Path(result["path"]) / "images/train")
    ]
    base["val"] = str(root / base["val"])
    base["path"] = str(root)
    out = Path("configs/retraining.yaml")
    out.write_text(yaml.safe_dump(base))
    return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()
    from ultralytics import YOLO

    data = prepare()
    model = YOLO(settings.model_path)
    result = model.train(
        data=data,
        epochs=a.epochs,
        batch=a.batch,
        imgsz=a.imgsz,
        device=settings.resolved_device,
        workers=0,
        project=str(Path("models").resolve()),
        name="reviewed",
        seed=42,
        amp=True,
        plots=False,
    )
    Path("results/retraining.json").write_text(
        json.dumps(
            {
                "checkpoint": str(model.trainer.best),
                "metrics": result.results_dict,
                "settings": vars(a),
                "data": data,
            },
            indent=2,
        )
    )
    print(result.results_dict)


if __name__ == "__main__":
    main()
