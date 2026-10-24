"""Build a clearly labelled smoke MP4 from real images and execute the pipeline."""

import argparse
import json
from pathlib import Path
import cv2
from backend.app.config import Settings
from backend.app.db import Store
from backend.app.services.video_service import VideoService


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="results/test-image.jpg")
    p.add_argument("--model", default="models/yolo11s.pt")
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--frames", type=int, default=30)
    a = p.parse_args()
    image = cv2.imread(a.source)
    if image is None:
        raise ValueError(a.source)
    h, w = image.shape[:2]
    image = cv2.resize(image, (640, round(h / w * 640) // 2 * 2))
    h, w = image.shape[:2]
    video = Path("results/smoke-input.mp4")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(a.frames):
        writer.write(image)
    writer.release()
    cfg = Settings(
        model_path=a.model,
        yolo_model="models/yolo11s.pt",
        image_size=320,
        enable_vehicles=False,
        enable_sam2=not a.no_sam,
        sam_interval=10,
        temporal_window=0.5,
        uncertainty_threshold=0.15,
        active_learning_batch_size=3,
    )
    store = Store(cfg.data_dir)
    job = store.put(
        "job",
        dict(
            kind="video",
            name="Smoke test · repeated real image",
            status="queued",
            source=str(video),
        ),
    )
    VideoService(cfg, store).run(job["id"], str(video))
    result = store.get(job["id"])
    print(json.dumps(result, indent=2))
    Path("results/smoke-video.json").write_text(json.dumps(result, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
