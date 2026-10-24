import argparse
import json
from pathlib import Path
from backend.app.config import settings
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(settings.model_path).with_suffix(".pt")))
    p.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    a = p.parse_args()
    out = {"format": a.format, "model": a.model, "imgsz": a.imgsz}
    try:
        if a.format == "engine":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
        path = YOLO(a.model).export(
            format=a.format,
            imgsz=a.imgsz,
            device=a.device,
            half=a.format == "engine",
            dynamic=False,
            nms=False,
            **({"workspace": 2} if a.format == "engine" else {"opset": 17}),
        )
        out.update(status="completed", artifact=str(path))
    except Exception as e:
        out.update(status="unavailable", error=f"{type(e).__name__}: {e}")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/export-{a.format}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if out["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
