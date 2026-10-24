import subprocess
import time
import uuid
import cv2
from .yolo_service import Detector, Tracker
from .safety_engine import SafetyEngine
from .sam_service import SamService
from .dino_service import DinoService
from .active_learning import CandidatePool


class VideoService:
    def __init__(self, settings, store):
        self.cfg = settings
        self.store = store

    def run(self, job_id, source, max_frames=0):
        cfg = self.cfg
        folder = self.store.root / "media" / job_id
        folder.mkdir(parents=True, exist_ok=True)
        job = self.store.get(job_id)
        job.update(status="running", progress=0, warnings=[])
        self.store.put("job", job, job_id)
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
        writer = None
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video source: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            if not 1 <= fps <= 240:
                fps = 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            detector = Detector(cfg)
            tracker = Tracker(max(1, round(fps / cfg.detection_interval)))
            safety = SafetyEngine(cfg.event_cooldown, cfg.event_confidence)
            sam = SamService(cfg.sam_model, cfg.resolved_device)
            pool = CandidatePool(cfg, DinoService(cfg.dino_device))
            frames = 0
            detections = []
            masks = {}
            all_latency = []
            start = time.perf_counter()
            use_sam = cfg.enable_sam2
            use_dino = cfg.enable_dino

            def flush():
                for item in pool.select():
                    rid = uuid.uuid4().hex
                    image = f"media/{job_id}/review-{rid}.jpg"
                    cv2.imwrite(str(self.store.root / image), item["image"])
                    maskpath = None
                    if item["mask"] is not None:
                        maskpath = f"media/{job_id}/mask-{rid}.png"
                        cv2.imwrite(
                            str(self.store.root / maskpath),
                            item["mask"].astype("uint8") * 255,
                        )
                    self.store.put(
                        "review",
                        dict(
                            job_id=job_id,
                            image=image,
                            mask=maskpath,
                            detection=item["detection"],
                            timestamp=item["timestamp"],
                            frame=item["frame"],
                            uncertainty=item["uncertainty"],
                            status="pending",
                            class_names=detector.names,
                            reason=f"Confidence uncertainty ≥ {cfg.uncertainty_threshold}; temporal gap ≥ {cfg.temporal_window}s; DINOv2 cosine core-set selection (uncertainty weight {cfg.uncertainty_weight}, diversity weight {cfg.diversity_weight})",
                        ),
                        rid,
                    )
                pool.items.clear()

            while True:
                ok, frame = cap.read()
                if not ok or (max_frames and frames >= max_frames):
                    break
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(folder / "processed-raw.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV MP4 encoder unavailable")
                tick = time.perf_counter()
                timestamp = frames / fps
                events = []
                if frames % cfg.detection_interval == 0:
                    detections = tracker.update(
                        detector.predict(frame, vehicles=True), frame.shape[:2]
                    )
                    events = safety.evaluate(detections, frames, timestamp)
                # Masks only rendered on the frame where computed: never stale masks.
                masks = {}
                if use_sam and frames % cfg.sam_interval == 0:
                    selected = [
                        d
                        for d in detections
                        if d["label"].lower() == "person"
                        and d["confidence"] >= cfg.display_confidence
                    ][: cfg.sam_max_objects]
                    if not selected:
                        selected = [
                            d
                            for d in detections
                            if d["confidence"] >= cfg.display_confidence
                        ][:1]
                    try:
                        for d, m in zip(
                            selected, sam.segment(frame, [d["box"] for d in selected])
                        ):
                            masks[
                                d.get("track_id")
                                if d.get("track_id") is not None
                                else tuple(d["box"])
                            ] = m
                    except Exception as e:
                        job["warnings"].append(
                            f"SAM2 disabled for this job: {type(e).__name__}: {e}"
                        )
                        use_sam = False
                if use_dino and frames % cfg.detection_interval == 0:
                    try:
                        # Vehicle COCO class IDs belong to a separate model, not the PPE export ontology.
                        pool.collect(
                            frame,
                            [d for d in detections if d["class_id"] < 1000],
                            timestamp,
                            frames,
                            masks,
                        )
                        if len(pool.items) >= max(
                            32, cfg.active_learning_batch_size * 4
                        ):
                            flush()
                    except Exception as e:
                        job["warnings"].append(
                            f"DINOv2 collection disabled for this job: {type(e).__name__}: {e}"
                        )
                        use_dino = False
                rendered = SamService.overlay(frame, list(masks.values()))
                for d in detections:
                    if d["confidence"] < cfg.display_confidence:
                        continue
                    x1, y1, x2, y2 = map(int, d["box"])
                    color = (60, 220, 130)
                    cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        rendered,
                        f"{d['label']} {d['confidence']:.2f}"
                        + (
                            f" #{d['track_id']}"
                            if d.get("track_id") is not None
                            else ""
                        ),
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )
                for i, event in enumerate(events):
                    cv2.putText(
                        rendered,
                        event["event_type"],
                        (15, 60 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (50, 80, 255),
                        2,
                    )
                    screenshot = f"media/{job_id}/event-{frames}-{i}.jpg"
                    cv2.imwrite(str(self.store.root / screenshot), rendered)
                    self.store.put(
                        "event", {**event, "job_id": job_id, "image": screenshot}
                    )
                latency = (time.perf_counter() - tick) * 1000
                all_latency.append(latency)
                cv2.putText(
                    rendered,
                    f"{1000 / max(latency, 0.001):.1f} processing FPS | {latency:.1f} ms | frame {frames}",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (240, 240, 240),
                    1,
                )
                writer.write(rendered)
                if frames % 10 == 0:
                    cv2.imwrite(str(folder / "latest.jpg"), rendered)
                    job.update(
                        frames=frames + 1,
                        progress=min(
                            1,
                            (frames + 1)
                            / (
                                min(total, max_frames)
                                if max_frames and total
                                else total
                            ),
                        )
                        if total
                        else None,
                        latest=f"media/{job_id}/latest.jpg",
                        latency_ms=latency,
                        processing_fps=(frames + 1) / (time.perf_counter() - start),
                        detections=detections,
                    )
                    self.store.put("job", job, job_id)
                frames += 1
            if not frames:
                raise ValueError("No decodable frames in video")
            flush()
            writer.release()
            writer = None
            raw = folder / "processed-raw.mp4"
            output = folder / "processed.mp4"
            try:
                import imageio_ffmpeg

                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(raw),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                output = raw
                job["warnings"].append(
                    f"H.264 transcode unavailable; raw MPEG4 may need external playback: {e}"
                )
            job.update(
                status="completed",
                progress=1,
                frames=frames,
                video=str(output.relative_to(self.store.root)),
                processing_fps=frames / (time.perf_counter() - start),
                average_latency_ms=sum(all_latency) / len(all_latency),
                model=detector.path,
                runtime=cfg.runtime,
                classes=detector.names,
            )
            self.store.put("job", job, job_id)
        except Exception as e:
            job.update(status="failed", error=f"{type(e).__name__}: {e}")
            self.store.put("job", job, job_id)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
