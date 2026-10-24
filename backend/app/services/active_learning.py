import numpy as np


def uncertainty(confidence):
    return 1 - float(confidence)


def select_diverse(
    embeddings, uncertainties, count, uncertainty_weight=0.55, diversity_weight=0.45
):
    """Greedy uncertainty-weighted cosine core-set; deterministic tie breaking."""
    if len(embeddings) == 0:
        return []
    x = np.asarray(embeddings, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    u = np.asarray(uncertainties)
    chosen = []
    distances = np.ones(len(x), dtype=np.float32)
    for _ in range(min(count, len(x))):
        score = uncertainty_weight * u + diversity_weight * distances
        score[chosen] = -np.inf
        i = int(np.argmax(score))
        chosen.append(i)
        distances = np.minimum(distances, np.clip(1 - x @ x[i], 0, 2) / 2)
    return chosen


class CandidatePool:
    def __init__(self, settings, dino):
        self.settings = settings
        self.dino = dino
        self.items = []
        self.last_time = -1e9

    def collect(self, frame, detections, timestamp, frame_number, masks=None):
        uncertain = [
            d
            for d in detections
            if uncertainty(d["confidence"]) >= self.settings.uncertainty_threshold
        ]
        if not uncertain or timestamp - self.last_time < self.settings.temporal_window:
            return
        d = min(uncertain, key=lambda d: d["confidence"])
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, d["box"])
        crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size == 0:
            return
        embedding = self.dino.embed(crop)
        self.items.append(
            dict(
                image=frame.copy(),
                detection=d,
                embedding=embedding,
                timestamp=timestamp,
                frame=frame_number,
                uncertainty=uncertainty(d["confidence"]),
                mask=(masks or {}).get(
                    d.get("track_id")
                    if d.get("track_id") is not None
                    else tuple(d["box"])
                ),
            )
        )
        self.last_time = timestamp

    def select(self):
        indices = select_diverse(
            [x["embedding"] for x in self.items],
            [x["uncertainty"] for x in self.items],
            self.settings.active_learning_batch_size,
            self.settings.uncertainty_weight,
            self.settings.diversity_weight,
        )
        return [self.items[i] for i in indices]
