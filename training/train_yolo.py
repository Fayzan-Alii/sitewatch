import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/construction-ppe.yaml")
    p.add_argument("--model", default="models/yolo11s.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--resume", default=None)
    p.add_argument("--name", default="ppe")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--cos-lr", action="store_true")
    args = p.parse_args()
    model = YOLO(args.resume or args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        fraction=args.fraction,
        resume=bool(args.resume),
        project=str(Path("models").resolve()),
        name=args.name,
        exist_ok=bool(args.resume),
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        close_mosaic=10,
        save_period=10,
        amp=True,
        seed=42,
        plots=False,
        cache=False,
    )
    Path("results").mkdir(exist_ok=True)
    Path(f"results/training-{args.name}.json").write_text(
        json.dumps(
            {
                "settings": vars(args),
                "metrics": results.results_dict,
                "checkpoint": str(model.trainer.best),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
