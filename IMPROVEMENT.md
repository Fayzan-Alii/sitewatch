# Training improvement report

The 60-epoch run completed. The best validation checkpoint was epoch 49 of the additional run and is now active at `models/ppe_extended/weights/best.pt`, with 640-pixel inference. The original five-epoch checkpoint remains at `models/ppe/weights/best.pt` for rollback. Training used all 1,132 training images, batch 8, AdamW, cosine learning-rate decay and mixed precision. It took approximately 22.7 minutes for the training epochs, plus initialization/final validation.

## Measured comparison

These are fresh evaluations using the same evaluation code and batch size. The old deployment used 320 px; the new deployment uses 640 px. Training's automatic final evaluation used a different batch/rectangular preprocessing grouping and reported 62.34% mAP50; the dashboard uses the fresh 60.07% result below consistently.

| Metric | Previous: 5 epochs, 320 px | Updated: 60 additional epochs, 640 px |
|---|---:|---:|
| Validation precision | 73.63% | 57.89% |
| Validation recall | 47.25% | 61.87% |
| Validation mAP50 | 51.16% | 60.07% |
| Validation mAP50–95 | 24.91% | 29.72% |
| Test mAP50 | 49.46% | 55.44% |
| Test mAP50–95 | 24.38% | 27.92% |

Precision and recall in this table use the validator's F1-selected operating point. They are not the same as the app's fixed confidence threshold. Overall precision declined at that point, while recall and mAP improved. This is an improvement in aggregate detection, not a uniform gain in every class or a demonstration of reliable safety enforcement.

## Matched-resolution and similarity checks

At the same 640 px resolution, the old/new validation mAP50 scores are **41.10% / 60.07%**, and mAP50–95 scores are **17.43% / 29.72%**. A perceptual-hash audit flagged 16 validation and 15 test images as similar to training examples. On the remaining 127 validation images, old/new mAP50 at 640 px is **37.64% / 57.57%**. The candidate scored **54.52% mAP50** on the remaining 126 test images. These filtered subsets reduce some obvious similarity; they are not proven source-independent splits.

## Actual operating point: confidence 0.50, IoU matching 0.50

These are object-detection counts over all 143 validation images, with unique same-class matches. They do not assess whether PPE is required in the scene or whether a safety incident occurred.

| Class | Previous TP / FP / FN (320 px) | Updated TP / FP / FN (640 px) | Updated precision | Updated recall |
|---|---:|---:|---:|---:|
| helmet | 148 / 16 / 53 | 152 / 25 / 49 | 85.9% | 75.6% |
| vest | 123 / 16 / 48 | 130 / 14 / 41 | 90.3% | 76.0% |
| Person | 195 / 18 / 44 | 207 / 24 / 32 | 89.6% | 86.6% |
| no_helmet | 0 / 0 / 45 | 4 / 0 / 41 | 100.0% | 8.9% |

**Missing-helmet detection remains poor:** only 4 of 45 annotated objects are found at 0.50 confidence. No false positives among four detections is a small-sample observation, not proof of perfect precision. Lowering to 0.35 finds 14 with 3 false positives; to 0.20 finds 24 with 14 false positives. The application retains a conservative 0.50 display/alert threshold, and still collects uncertain detections from 0.20 for human review. More representative missing-PPE examples and validation on actual site footage are needed.

## Application changes and checks

- Active model switched to the better checkpoint, image size increased to 640.
- Live boxes and safety alerts require 0.50 confidence; lower-confidence predictions remain in the active-learning path.
- SAM2 prompts follow the display confidence filter. Untracked boxes no longer display `#None`.
- Performance now shows per-class metrics and explicit limitations.
- ONNX export rebuilt at 640 px; all 20 detections on an inspected workplace image matched PyTorch on CUDA (IoU above 0.99999).
- Runtime benchmarks rerun for the new checkpoint and resolution; see `results/benchmarks.json`. TensorRT remains unavailable for the previously recorded dependency-download blocker.
- Software video regression uses an isolated results directory and a clearly labelled repeated construction still. It is not a field-video evaluation and does not preload the live dashboard.
- Side-by-side images are in `results/improvement/prediction-comparison.jpg`. They were chosen by visually inspecting workplace content before candidate inference. They illustrate predictions rather than prove general accuracy.

## Reproduce and rollback

Training command:

```bash
.venv/bin/python -m training.train_yolo --model models/ppe/weights/best.pt --epochs 60 --batch 8 --imgsz 640 --device 0 --name ppe_extended --patience 15 --workers 2 --lr0 0.001 --cos-lr
```

Completed-run configuration: `models/ppe_extended/args.yaml`. Epoch measurements: `models/ppe_extended/results.csv`. Fresh evaluation files, operating-point counts and comparison JSON: `results/improvement/`.

To roll back, restore `MODEL_PATH=models/ppe/weights/best.pt` and `IMAGE_SIZE=320` in `.env`, then restart `./scripts/run.sh`. Restore `results/improvement/previous-metrics.json` as `results/metrics.json` if also rolling back the dashboard's accuracy report. Existing videos retain their original rendered predictions; upload/process footage again to use the improved checkpoint.
