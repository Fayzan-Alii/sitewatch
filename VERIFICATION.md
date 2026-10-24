# Initial execution verification — 2026-10-05

**Historical report:** the active model and current measurements have been superseded by the 60-epoch training run documented in [IMPROVEMENT.md](IMPROVEMENT.md). Details below describe the original integration checks.

SiteWatch AI runs at http://127.0.0.1:8000. Restart with `./scripts/run.sh` from the project root. FastAPI serves the production React build.

## Verified with real execution

- **GPU:** NVIDIA RTX 2060 Max-Q, 6 GB VRAM, CUDA available. Python 3.12.3, PyTorch 2.14.0+cu126, Ultralytics 8.4.161. Dependency snapshot: `requirements-tested.txt`.
- **Dataset:** Official Construction-PPE ZIP downloaded, extracted, hashed; 1,132 training and 143 validation images. Provenance and archive SHA-256: `results/dataset.json`.
- **Training:** YOLO11s fine-tuned for five epochs on the full training split, batch 4, image size 320, seed 42, AMP enabled. Active checkpoint: `models/ppe/weights/best.pt`. Run: `models/ppe/results.csv`; settings: `results/training-ppe.json`.
- **Validation:** Precision **0.736349**, recall **0.472549**, mAP50 **0.511589**, mAP50–95 **0.249092**, on the original validation split at 320 px. Source: `results/metrics.json`. These are short-run development measurements, not deployment readiness.
- **Detection/tracking/video:** Real YOLO detections and ByteTrack IDs; H.264 MP4 output; a model-predicted `PERSON_WITHOUT_HELMET` event persisted with evidence image on an off-site restaurant image (not a valid construction safety incident). `results/smoke-video.json`, `results/ppe-video.log`, SQLite event records. Smoke videos repeat real images to test the pipeline; they are not a field-video evaluation.
- **SAM2:** Meta SAM2.1 Hiera tiny weights, used by the Ultralytics SAM2 implementation. A real YOLO box prompt produced a 1080×1920 mask with 29,340 foreground pixels. `results/sam-verification.json`, `results/sam-mask.png`, `results/sam-overlay.jpg`. Masks also rendered in the repeated-image test MP4. No video propagation is claimed.
- **DINOv2:** Pretrained ViT-S/14 returned a real 384-dimensional unit-normalized embedding. `results/download-dino.log`. Embeddings actually drove cosine diversity selection into persisted review records.
- **Active learning:** Uncertainty filtering, temporal deduplication and weighted farthest-first selection ran during videos. Reasons, confidence, frame and timestamps were persisted; test records are now archived outside the live dashboard.
- **Review/retraining:** An automated correction from official training ground truth persisted through the API and SQLite, exported one YOLO-labelled crop, and fed one full retraining epoch alongside the original training images. `results/review-verification.json`, `results/retraining.json`, `models/reviewed/weights/best.pt`. This verifies mechanics; it is not a claimed human decision or learning improvement. Retraining validation was worse (mAP50 0.401937), so this checkpoint was not promoted.
- **FastAPI:** HTTP MP4 upload processed 40 frames without component warnings. HTTP image inference returned four detections. Health responded in 2.64 ms while video processing ran; video byte-range playback returned HTTP 206. `results/api-verification.json`.
- **ONNX:** Trained PPE model exported to `models/ppe/weights/best.onnx`. Direct ONNX Runtime 1.22.0 executed on CUDAExecutionProvider. All four detections matched PyTorch with IoU >0.999998 and confidence differences <0.000001. `results/onnx-verification.json`.
- **Frontend:** Production build passed. Chromium exercised Monitor, Events, Human Review and Performance with real API data. No JavaScript errors; no horizontal overflow at 390 px width. `results/browser-verification.json`; screenshots: `results/dashboard-*.png`.
- **Tests:** 16 passed. `results/tests.log`. Covers rules, association, deduplication, diversity, tracking, export, persistence, config, model failure and API validation/ontology handling.

## Measured benchmark

Trained PPE checkpoint, 320 px, one off-site restaurant image from the dataset, sequential batch 1, five warm-ups and 20 measured iterations. Wall latency includes preprocessing/inference/postprocessing and CUDA synchronization. Results apply to this image and workload, not full-pipeline video FPS.

| Runtime | Mean latency | Throughput |
|---|---:|---:|
| PyTorch FP32 / CUDA | 14.92 ms | 67.00 images/s |
| PyTorch FP16 / CUDA | 14.33 ms | 69.78 images/s |
| ONNX Runtime / CUDA | 8.38 ms | 119.37 images/s |
| PyTorch + SAM2, one box each frame | 160.05 ms | 6.25 images/s |
| TensorRT FP16 | Unavailable | Unavailable |

Machine-readable p50/p95, stages, memory scope and model sizes: `results/benchmarks.json` and `results/benchmarks.csv`. GPU memory reports the PyTorch allocator only; ONNX/TensorRT external allocations are excluded. Video uses less frequent SAM2 prompts than this stress comparison.

## Remaining blocker and limitations

**TensorRT was attempted but is not verified.** The `tensorrt-cu12==11.3.0.99` installation required a 4,387,712,625-byte library wheel from NVIDIA's package server. The transfer stalled around 171 MiB for several minutes and was terminated after over eight minutes. The subsequent export attempt failed with `ModuleNotFoundError: No module named 'tensorrt'`. No engine or TensorRT benchmark is presented as working. Exact record: `results/export-engine.json`. Export and genuine engine inference paths remain implemented; retry the documented installation/export when the dependency can be downloaded.

The initially selected ONNX Runtime 1.30 GPU wheel required CUDA 13 libraries (`libcublasLt.so.13` missing). Pinning to 1.22.0 resolved this; CUDA execution was verified afterward.

Construction-PPE does not label missing vests or machinery. Rules support those labels, but this checkpoint cannot produce them. A secondary COCO model supplies car/bus/truck detections. PPE association/proximity uses 2D heuristics; masks are selectively recomputed rather than propagated. The five-epoch model's missing-PPE recall remains limited; measure on representative held-out site footage before operational use.

## Correction after inspecting the demo image

The selected `image1117.jpg` shows a man eating a sandwich in a restaurant. It was selected automatically because model predictions triggered a missing-helmet rule, without visually verifying the scene. Earlier descriptions of it as a construction example were incorrect. The video repeated this still image; it was not real construction footage. The predictions verify execution only, not safety accuracy or scene relevance. Detector outputs alone cannot establish that a person is in a helmet-required area.

All assistant-created test jobs, events and review samples have been archived from the live dashboard into `results/archived-verification-records.json` (also retained in SQLite under verification record kinds). The dashboard now starts empty for genuine user footage. The model needs representative site-video evaluation, better training and site/zone context before safety alerts are meaningful. Benchmark numbers remain measured compute timings on that particular off-site image.
