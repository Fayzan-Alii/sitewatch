import argparse
import csv
import gc
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from backend.app.config import Settings, settings
from backend.app.services.yolo_service import Detector
from backend.app.services.sam_service import SamService


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(detector, image, iterations, warmup, sam=None):
    def run():
        detections = detector.predict(image)
        if sam and detections:
            sam.segment(image, [detections[0]["box"]])
        sync()

    for _ in range(warmup):
        run()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    samples = []
    stages = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        run()
        samples.append((time.perf_counter() - start) * 1000)
        stages.append(detector.timings.copy())
    return dict(
        average_latency_ms=float(np.mean(samples)),
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        fps=1000 / float(np.mean(samples)),
        throughput_images_s=1000 / float(np.mean(samples)),
        gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else None,
        gpu_memory_scope="PyTorch allocator only; excludes ORT/TensorRT external allocations",
        model_size_mb=Path(detector.path).stat().st_size / 1024**2,
        stages={k: float(np.mean([s[k] for s in stages])) for k in stages[0]},
        provider=detector.provider,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(settings.model_path).with_suffix(".pt")))
    p.add_argument("--image", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--sam", action="store_true")
    a = p.parse_args()
    image = cv2.imread(a.image)
    if image is None:
        raise ValueError("Image not found")
    output = dict(
        available=True,
        model=a.model,
        image=a.image,
        imgsz=a.imgsz,
        iterations=a.iterations,
        warmup=a.warmup,
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        results=[],
    )
    for name, runtime, half, path in [
        ("PyTorch FP32", "pytorch", False, a.model),
        ("PyTorch FP16", "pytorch", True, a.model),
        ("ONNX Runtime", "onnx", False, str(Path(a.model).with_suffix(".onnx"))),
        ("TensorRT FP16", "tensorrt", True, str(Path(a.model).with_suffix(".engine"))),
    ]:
        row = {"name": name, "runtime": runtime}
        detector = None
        try:
            if half and not torch.cuda.is_available():
                raise RuntimeError("FP16 requires CUDA")
            if not Path(path).exists():
                report = Path("results/export-engine.json")
                if runtime == "tensorrt" and report.exists():
                    issue = json.loads(report.read_text())
                    raise RuntimeError(
                        issue.get(
                            "installation_blocker",
                            issue.get("error", f"Missing engine: {path}"),
                        )
                    )
                raise FileNotFoundError(path)
            detector = Detector(
                Settings(
                    model_path=path,
                    runtime=runtime,
                    image_size=a.imgsz,
                    enable_vehicles=False,
                ),
                half=half,
            )
            row.update(
                status="completed", **measure(detector, image, a.iterations, a.warmup)
            )
        except Exception as e:
            row.update(status="unavailable", error=f"{type(e).__name__}: {e}")
        output["results"].append(row)
        print(row, flush=True)
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if a.sam:
        detector = Detector(
            Settings(model_path=a.model, image_size=a.imgsz, enable_vehicles=False)
        )
        row = {"name": "PyTorch + SAM2 (one box each frame)", "runtime": "pytorch_sam2"}
        try:
            sam = SamService("models/sam2.1_t.pt", detector.cfg.resolved_device)
            row.update(
                status="completed",
                **measure(detector, image, a.iterations, a.warmup, sam),
            )
        except Exception as e:
            row.update(status="unavailable", error=f"{type(e).__name__}: {e}")
        output["results"].append(row)
    Path("results/benchmarks.json").write_text(json.dumps(output, indent=2))
    keys = [
        "name",
        "status",
        "average_latency_ms",
        "p50_ms",
        "p95_ms",
        "fps",
        "throughput_images_s",
        "gpu_memory_mb",
        "model_size_mb",
        "provider",
        "error",
    ]
    with Path("results/benchmarks.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output["results"])


if __name__ == "__main__":
    main()
