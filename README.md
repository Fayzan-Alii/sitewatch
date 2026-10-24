# SiteWatch AI

Local video safety monitoring and active learning for a 6 GB NVIDIA GPU. Python/FastAPI, React/Vite, SQLite, YOLO11s, ByteTrack, SAM2.1 tiny, DINOv2 ViT-S/14, direct ONNX Runtime, and optional TensorRT FP16.

## Current trained model

The active checkpoint has completed 60 additional training epochs at 640 px. Fresh validation mAP50 improved from 51.16% to 60.07%, and held-out test mAP50 from 49.46% to 55.44%. Missing-helmet recall remains weak. See [the improvement report](IMPROVEMENT.md) for the full comparison, thresholds and limitations.

## Run

From the project root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r requirements.txt
npm ci --prefix frontend
cp .env.example .env
.venv/bin/python -m scripts.download_dataset
.venv/bin/python -m scripts.setup_models
npm run build --prefix frontend
./scripts/run.sh
```

Open **http://127.0.0.1:8000**. API documentation: **http://127.0.0.1:8000/docs**. During UI development, run `npm run dev --prefix frontend` separately; Vite proxies the API and media to port 8000. Bind to localhost: this is a trusted local operator tool with filesystem/video access, without authentication.

If the PPE checkpoint does not exist, inference explicitly falls back to the COCO pretrained YOLO11s. That model can detect people/vehicles, but cannot detect PPE. Train the PPE checkpoint before expecting helmet alerts. CUDA is selected automatically when available. DINOv2 runs on CPU by default to preserve GPU memory. SAM2 is loaded lazily. Failed optional model components are recorded in each video job's `warnings`; they are never replaced by synthetic outputs.

## Dataset and training

The downloader uses the official [Ultralytics Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/), validates the archive, and records its SHA-256 in `results/dataset.json`. The upstream YAML is retained in `configs/construction-ppe-upstream.yaml`; the local path YAML is generated on setup. The dataset has 11 classes: helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots. It **does not include no_vest, vehicles or machinery**. A separate COCO YOLO11s supplies car/bus/truck detections during PPE video inference. Machinery and explicit no-vest need additional labelled training data. The rule engine supports those names when a suitable model is supplied.

```bash
# Standard fine tuning; use --device cpu if necessary
.venv/bin/python -m training.train_yolo --model models/yolo11s.pt --epochs 30 --batch 2 --imgsz 640 --device 0
# Fast integration smoke run (not deployment training)
.venv/bin/python -m training.train_yolo --model models/yolo11s.pt --epochs 1 --batch 2 --imgsz 320 --fraction 0.1 --name smoke
# Resume interrupted training, retaining optimizer/scheduler state
.venv/bin/python -m training.train_yolo --resume models/ppe/weights/last.pt
.venv/bin/python -m training.evaluate --model models/ppe/weights/best.pt --split val
```

Checkpoints and Ultralytics run logs are saved under `models/<name>/`; validation measurements go to `results/metrics.json`. mAP, precision and recall are measured on the named dataset split. No metrics imply field readiness, especially after a short smoke training run. Use the test split only for final evaluation after tuning on validation.

## Video and safety

The dashboard has no preloaded demo footage. Earlier integration-test footage was a repeated dataset still, including an off-site restaurant image; those records have been archived. The active checkpoint is a development model. Its boxes are predictions requiring review, and missing PPE outside a PPE-required area is not a safety incident. This system has no automatic construction-scene or restricted-zone classifier.


Upload MP4/MOV/AVI/MKV in Monitor, or submit a local video:

```bash
curl -X POST http://127.0.0.1:8000/api/videos/local \
  -H 'Content-Type: application/json' \
  -d '{"source":"path/to/video.mp4","max_frames":300}'
# Webcam: source "0", with a mandatory finite max_frames.
```

Jobs run on a single background worker, serializing GPU usage to fit 6 GB VRAM. Poll `/api/jobs/{id}` for progress, latest frame, errors and warnings. The frontend polls every 2.5 seconds and shows the latest annotated frame while processing, then an H.264 MP4. Processing FPS is measured compute throughput, not the encoded source playback rate. Boxes are reused between detection intervals; SAM masks are rendered only on frames where computed. ByteTrack sees detection frames, with its track buffer scaled to the inference rate.

Safety rules associate explicit no-helmet/no-vest evidence with the smallest containing person's appropriate head/body region. Missing detections alone do not establish a PPE violation. Person–vehicle proximity is a 2D box-gap heuristic, not calibrated physical distance. Events are deduplicated per video, tracked person and rule for a configurable cooldown. Unknown IDs use a quantized-position fallback. Do not infer real-world distances from these alerts.

SAM2 uses real box-prompted masks on selected people (at most two by default) every `SAM_INTERVAL` frames. It does not propagate masks through video: recomputing selected prompts avoids stale geometry and keeps memory predictable. See [Ultralytics SAM2](https://docs.ultralytics.com/models/sam-2/). DINO uses the pretrained [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) model (ViT-S/14), with normalized CLS-token embeddings.

## Active learning and review

Uncertainty is `1 - detection confidence`, a ranking heuristic rather than calibrated probability. Candidates must exceed the uncertainty threshold and a temporal gap. The lowest-confidence qualifying object is embedded from each sampled frame. A bounded candidate pool uses uncertainty-weighted farthest-first cosine selection. Pools flush at `max(32, batch_size * 4)` candidates and at end-of-video, so long videos may produce multiple selection batches. Selection reasons and optional masks are persisted alongside the frame, box, confidence, frame number and timestamp.

Accept, reject, or correct class in Human Review. The API also accepts a corrected `box: [x1,y1,x2,y2]`. SQLite stores actions across restarts. `/api/reviews/export` rebuilds a YOLO dataset snapshot from accepted/corrected object crops; rejected records are excluded. Crops reduce incomplete full-frame annotations but are not exhaustively labelled scenes. For best training quality, review additional objects inside a crop before using it as a complete scene.

```bash
.venv/bin/python -m training.retrain --epochs 10 --batch 2 --imgsz 640
```

Retraining combines reviewed crops with the original training set and retains the **original validation split**, avoiding review/validation leakage. A new checkpoint is written under `models/reviewed*`; it is not silently promoted. Change `MODEL_PATH` after evaluating it, then restart the server. The UI retraining button starts the same workflow as a background job. Failed job logs are shown in Human Review.

## ONNX, TensorRT and performance

```bash
.venv/bin/python -m training.export_model --format onnx --imgsz 640
.venv/bin/python -m scripts.verify_onnx --image path/to/image.jpg --imgsz 640
# Optional TensorRT dependencies, matched to CUDA 12:
uv pip install --python .venv/bin/python tensorrt-cu12
.venv/bin/python -m training.export_model --format engine --imgsz 640
.venv/bin/python -m benchmarks.benchmark --image path/to/image.jpg --imgsz 640 --warmup 5 --iterations 30 --sam
```

Set `RUNTIME=onnx` and `MODEL_PATH=models/ppe/weights/best.onnx`, or `RUNTIME=tensorrt` and the `.engine` path. Export and runtime image sizes must agree. ONNX uses `InferenceSession.run`, real letterboxing, class-aware NMS and coordinate restoration; it never calls PyTorch detection. CUDA EP is requested when CUDA is available, with CPU fallback reported by provider name. The parity script uses a square-padded image so both runtimes receive identical preprocessing, then performs unique same-class matches with IoU >0.95 and confidence delta <0.02.

TensorRT uses the genuine Ultralytics engine backend. Export blockers are saved in `results/export-engine.json`; absent or failing runtimes are `unavailable`, never represented by fake numbers. Engines are hardware/runtime-specific and should be rebuilt on the target GPU. The export workspace is limited to 2 GB.

Benchmarks warm up each runtime, synchronize CUDA, and report mean/p50/p95 wall latency, sequential batch-one throughput, model size and per-stage timings. GPU memory is **PyTorch allocator peak only**, excluding external ONNX/TensorRT allocations; it is not total device usage. SAM comparison prompts one actual detection every benchmark iteration, a deliberate stress comparison; the normal video pipeline runs SAM less frequently. Metrics and benchmarks are consumed from JSON by the frontend; CSV is also saved. Missing measurements display as unavailable.

## Configuration and persistence

All settings live in `backend/app/config.py` with `.env` overrides and validation. See `.env.example`. Run commands from the project root. Defaults: detector 640 px, detection interval 2, SAM interval 15, 2 SAM objects, active learning batch 12, temporal window 2 seconds. Inference confidence and IoU are configurable. `DEVICE=auto` chooses CUDA or CPU. SQLite, videos, review images, logs and measurements are stored in `results/`; datasets/models remain local and are git-ignored. Only `results/media` is served publicly, not the database or logs.

The tested dependency snapshot is in `requirements-tested.txt`. `scripts/setup.sh` installs compatible CUDA 12 wheels and the normal dependencies. The development `.env` uses `models/ppe_extended/weights/best.pt` at 640 px. `.env.example` retains the initial training-output path for fresh installations.

## Tests

```bash
.venv/bin/python -m pytest -q
npm run build --prefix frontend
# Real model smoke test; writes a clearly labelled repeated-image MP4
.venv/bin/python -m scripts.smoke_video --model models/ppe/weights/best.pt --source path/to/image.jpg
```

Unit/API tests cover conservative safety rules, association and deduplication, temporal filtering, cosine diversity, tracking ID continuity, annotation bounds, export lifecycle, persistence, config validation, missing model handling and endpoint errors. Actual model execution results and environmental limitations are recorded in `results/` and `VERIFICATION.md`.

## Licensing

Ultralytics software/checkpoints and dataset have their own terms (AGPL-3.0 or commercial licensing as applicable); consult [Ultralytics licensing](https://www.ultralytics.com/license). DINOv2 and SAM2 checkpoints use their upstream licenses. This project does not relicense third-party assets.
