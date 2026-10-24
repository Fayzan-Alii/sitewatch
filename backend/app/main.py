import json
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .config import settings
from .db import Store
from .schemas import ReviewAction, TrainRequest, VideoRequest
from .services.video_service import VideoService
from .services.yolo_service import Detector
from .services.annotations import export_reviews

store = Store(settings.data_dir)
(settings.data_dir / "media").mkdir(exist_ok=True)
# Serialize GPU tasks for 6GB VRAM. HTTP reads never wait on this executor.
worker = ThreadPoolExecutor(max_workers=1)
app = FastAPI(title="SiteWatch AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=settings.data_dir / "media"), name="media")

# A process interruption is not a running job after restart.
for job in store.list("job"):
    if job.get("status") in ("queued", "running"):
        store.put(
            "job",
            {
                **job,
                "status": "failed",
                "error": "Server restarted before job completed",
            },
            job["id"],
        )


def classes():
    path = Path(settings.dataset_path)
    return yaml.safe_load(path.read_text())["names"] if path.exists() else {}


@app.get("/api/health")
def health():
    import torch

    return dict(
        status="ok",
        cuda=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        runtime=settings.runtime,
        model=settings.model_path,
        checkpoint_exists=Path(settings.model_path).exists(),
        active_model=settings.model_path
        if Path(settings.model_path).exists()
        else settings.yolo_model,
        model_status="PPE checkpoint configured"
        if Path(settings.model_path).exists()
        else "COCO fallback: PPE detection unavailable",
        sam2={
            "enabled": settings.enable_sam2,
            "checkpoint_exists": Path(settings.sam_model).exists(),
        },
        dino={"enabled": settings.enable_dino, "model": "facebook/dinov2-small"},
        display_confidence=settings.display_confidence,
        event_confidence=settings.event_confidence,
        classes=classes(),
    )


@app.get("/api/jobs")
def jobs():
    return store.list("job")


@app.get("/api/jobs/{id}")
def job_status(id: str):
    item = store.get(id)
    if not item or "status" not in item:
        raise HTTPException(404, "Job not found")
    return item


@app.post("/api/videos", status_code=202)
async def upload_video(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv"):
        raise HTTPException(400, "Upload an MP4, MOV, AVI or MKV")
    id = uuid.uuid4().hex
    directory = settings.data_dir / "media" / id
    directory.mkdir(parents=True)
    source = directory / ("input" + Path(file.filename).suffix.lower())
    size = 0
    with source.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 2 * 1024**3:
                f.close()
                source.unlink()
                raise HTTPException(413, "Maximum upload size is 2GB")
            f.write(chunk)
    item = store.put(
        "job",
        dict(kind="video", name=file.filename, status="queued", source=str(source)),
        id,
    )
    worker.submit(VideoService(settings, store).run, id, str(source))
    return item


@app.post("/api/videos/local", status_code=202)
def local_video(request: VideoRequest):
    # This API is intended for localhost; reject remote URLs and unlimited webcams.
    if "://" in request.source:
        raise HTTPException(400, "Use a local file or webcam index")
    if request.source.isdigit() and request.max_frames == 0:
        raise HTTPException(400, "Webcam capture requires max_frames")
    if not request.source.isdigit() and not Path(request.source).is_file():
        raise HTTPException(404, "Local video not found")
    item = store.put(
        "job",
        dict(
            kind="video",
            name=Path(request.source).name,
            status="queued",
            source=request.source,
        ),
    )
    worker.submit(
        VideoService(settings, store).run,
        item["id"],
        request.source,
        request.max_frames,
    )
    return item


@app.post("/api/infer", status_code=202)
async def infer(file: UploadFile = File(...)):
    data = await file.read(25 * 1024 * 1024 + 1)
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "Maximum image size is 25MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid image")
    item = store.put("job", dict(kind="image", name=file.filename, status="queued"))

    def run():
        try:
            detector = Detector(settings)
            predictions = detector.predict(image, vehicles=True)
            store.put(
                "job",
                {
                    **item,
                    "status": "completed",
                    "detections": predictions,
                    "timings": detector.timings,
                },
                item["id"],
            )
        except Exception as e:
            store.put("job", {**item, "status": "failed", "error": str(e)}, item["id"])

    worker.submit(run)
    return item


@app.get("/api/events")
def events():
    return store.list("event")


@app.get("/api/reviews")
def reviews():
    return store.list("review")


@app.patch("/api/reviews/{id}")
def review(id: str, action: ReviewAction):
    item = store.get(id)
    if not item or "detection" not in item:
        raise HTTPException(404, "Review not found")
    if action.class_id is not None and action.class_id not in classes():
        raise HTTPException(400, "Unknown dataset class")
    if action.box:
        image = cv2.imread(str(store.root / item["image"]))
        if image is None:
            raise HTTPException(409, "Review image missing")
        if action.box[2] > image.shape[1] or action.box[3] > image.shape[0]:
            raise HTTPException(400, "Box outside image")
    if action.action == "accept" and "label" in item["detection"]:
        from .services.safety_engine import canonical

        matches = [
            cid
            for cid, name in classes().items()
            if canonical(name) == canonical(item["detection"]["label"])
        ]
        if not matches:
            raise HTTPException(
                409,
                "Prediction class is outside the dataset ontology; correct its class or reject it",
            )
        if item["detection"]["class_id"] != matches[0]:
            item["corrected_class"] = matches[0]
    item["status"] = action.action
    if action.class_id is not None:
        item["corrected_class"] = action.class_id
    if action.box:
        item["corrected_box"] = action.box
    return store.put("review", item, id)


@app.post("/api/reviews/export")
def export():
    return export_reviews(store, classes())


@app.get("/api/metrics")
def metrics():
    p = settings.data_dir / "metrics.json"
    return (
        json.loads(p.read_text())
        if p.exists()
        else {"available": False, "reason": "Validation has not run"}
    )


@app.get("/api/benchmarks")
def benchmarks():
    p = settings.data_dir / "benchmarks.json"
    return (
        json.loads(p.read_text())
        if p.exists()
        else {"available": False, "reason": "Benchmarks have not run", "results": []}
    )


@app.post("/api/retrain", status_code=202)
def retrain(request: TrainRequest):
    if not any(
        r.get("status") in ("accept", "correct") for r in store.list("review", 100000)
    ):
        raise HTTPException(409, "Accept or correct reviews first")
    item = store.put(
        "job",
        dict(
            kind="training",
            name="Review retraining",
            status="queued",
            settings=request.model_dump(),
        ),
    )

    def run():
        store.put("job", {**item, "status": "running"}, item["id"])
        log = settings.data_dir / f"train-{item['id']}.log"
        with log.open("w") as output:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.retrain",
                    "--epochs",
                    str(request.epochs),
                    "--batch",
                    str(request.batch),
                    "--imgsz",
                    str(request.imgsz),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        store.put(
            "job",
            {
                **item,
                "status": "completed" if result.returncode == 0 else "failed",
                "log": str(log),
                "error": log.read_text()[-2000:] if result.returncode else None,
            },
            item["id"],
        )

    worker.submit(run)
    return item


if Path("frontend/dist").exists():
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="dashboard")
