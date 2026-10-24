import ast
import time
from pathlib import Path
import cv2
import numpy as np


class Detector:
    """Direct ORT execution; Ultralytics for PyTorch and actual TensorRT engines."""

    def __init__(self, settings, half=False):
        self.cfg = settings
        self.half = half
        self.model = None
        self.session = None
        self.vehicle_model = None
        self.names = {}
        self.timings = {}
        self.provider = None
        self.path = (
            settings.model_path
            if Path(settings.model_path).exists()
            else settings.yolo_model
        )
        if settings.runtime != "pytorch" and not Path(settings.model_path).exists():
            raise FileNotFoundError(
                f"{settings.runtime} artifact missing: {settings.model_path}. Run training/export_model.py."
            )

    def load(self):
        if self.model is not None or self.session is not None:
            return
        if self.cfg.runtime == "onnx":
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.cfg.resolved_device != "cpu"
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(self.path, providers=providers)
            metadata = self.session.get_modelmeta().custom_metadata_map
            self.names = ast.literal_eval(metadata["names"])
            self.names = {int(k): v for k, v in self.names.items()}
            self.provider = self.session.get_providers()[0]
        else:
            from ultralytics import YOLO

            self.model = YOLO(self.path, task="detect")
            self.names = self.model.names
            self.provider = self.cfg.resolved_device

    def predict(self, image, vehicles=False):
        self.load()
        start = time.perf_counter()
        if self.session is not None:
            result = self._onnx(image)
        else:
            r = self.model.predict(
                image,
                conf=self.cfg.yolo_confidence,
                iou=self.cfg.yolo_iou,
                imgsz=self.cfg.image_size,
                device=self.cfg.resolved_device,
                half=self.half,
                rect=False,
                verbose=False,
            )[0]
            result = [
                dict(
                    box=b.xyxy[0].tolist(),
                    class_id=int(b.cls.item()),
                    label=self.names[int(b.cls.item())],
                    confidence=float(b.conf.item()),
                    track_id=None,
                )
                for b in r.boxes
            ]
            self.timings = {k + "_ms": float(v) for k, v in r.speed.items()}
        if (
            vehicles
            and self.cfg.enable_vehicles
            and "helmet" in {str(v).lower() for v in self.names.values()}
        ):
            if self.vehicle_model is None:
                from ultralytics import YOLO

                self.vehicle_model = YOLO(self.cfg.yolo_model)
            r = self.vehicle_model.predict(
                image,
                classes=[2, 5, 7],
                conf=self.cfg.yolo_confidence,
                device=self.cfg.resolved_device,
                imgsz=self.cfg.image_size,
                verbose=False,
            )[0]
            result += [
                dict(
                    box=b.xyxy[0].tolist(),
                    class_id=1000 + int(b.cls.item()),
                    label=r.names[int(b.cls.item())],
                    confidence=float(b.conf.item()),
                    track_id=None,
                )
                for b in r.boxes
            ]
        self.timings["end_to_end_ms"] = (time.perf_counter() - start) * 1000
        return result

    def _onnx(self, image):
        t = time.perf_counter()
        h, w = image.shape[:2]
        shape = self.session.get_inputs()[0].shape
        size = int(shape[2]) if isinstance(shape[2], int) else self.cfg.image_size
        ratio = min(size / w, size / h)
        nw, nh = round(w * ratio), round(h * ratio)
        left = (size - nw) // 2
        top = (size - nh) // 2
        padded = np.full((size, size, 3), 114, np.uint8)
        padded[top : top + nh, left : left + nw] = cv2.resize(image, (nw, nh))
        dtype = (
            np.float16
            if self.session.get_inputs()[0].type == "tensor(float16)"
            else np.float32
        )
        x = (
            np.ascontiguousarray(
                padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=dtype
            )
            / 255
        )
        a = time.perf_counter()
        output = self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0]
        b = time.perf_counter()
        if output.shape[0] == 4 + len(self.names):
            output = output.T
        if output.shape[1] != 4 + len(self.names):
            raise ValueError(
                "Expected YOLO11 raw output; export with nms=False, end2end=False"
            )
        cls = output[:, 4:].argmax(1)
        scores = output[np.arange(len(output)), cls + 4]
        keep = scores >= self.cfg.yolo_confidence
        output = output[keep]
        cls = cls[keep]
        scores = scores[keep]
        boxes = output[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        ids = []
        for c in np.unique(cls):
            ix = np.flatnonzero(cls == c)
            selected = cv2.dnn.NMSBoxes(
                boxes[ix].tolist(),
                scores[ix].tolist(),
                self.cfg.yolo_confidence,
                self.cfg.yolo_iou,
            )
            ids.extend(ix[np.asarray(selected, dtype=int).reshape(-1)].tolist())
        result = []
        for i in sorted(ids, key=lambda i: -scores[i]):
            x, y, bw, bh = boxes[i]
            xyxy = [
                max(0, (x - left) / ratio),
                max(0, (y - top) / ratio),
                min(w, (x + bw - left) / ratio),
                min(h, (y + bh - top) / ratio),
            ]
            result.append(
                dict(
                    box=list(map(float, xyxy)),
                    class_id=int(cls[i]),
                    label=self.names[int(cls[i])],
                    confidence=float(scores[i]),
                    track_id=None,
                )
            )
        self.timings = {
            "preprocess_ms": (a - t) * 1000,
            "inference_ms": (b - a) * 1000,
            "postprocess_ms": (time.perf_counter() - b) * 1000,
        }
        return result


class Tracker:
    def __init__(self, fps=30):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from types import SimpleNamespace

        self.tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                track_buffer=max(1, round(fps)),
                match_thresh=0.8,
                fuse_score=True,
            )
        )

    def update(self, detections, shape):
        from ultralytics.engine.results import Boxes

        entities = [
            i
            for i, d in enumerate(detections)
            if d["label"].lower()
            in (
                "person",
                "car",
                "bus",
                "truck",
                "vehicle",
                "machinery",
                "forklift",
                "excavator",
            )
        ]
        rows = np.array(
            [
                detections[i]["box"]
                + [detections[i]["confidence"], detections[i]["class_id"]]
                for i in entities
            ],
            dtype=np.float32,
        ).reshape(-1, 6)
        tracks = self.tracker.update(Boxes(rows, shape))
        for row in tracks:
            i = int(row[-1])
            if 0 <= i < len(entities):
                detections[entities[i]]["track_id"] = int(row[4])
        return detections
