"""Conservative PPE rules: explicit missing-PPE evidence, never absence alone."""

import math


def canonical(label):
    return (
        label.lower()
        .replace("_", "-")
        .replace(" ", "-")
        .replace("hardhat", "helmet")
        .replace("safety-vest", "vest")
    )


def inside(center, box):
    x, y = center
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


class SafetyEngine:
    def __init__(self, cooldown=10, min_confidence=0.5):
        self.cooldown = cooldown
        self.min_confidence = min_confidence
        self.last = {}

    def evaluate(self, detections, frame, timestamp):
        people = [d for d in detections if canonical(d["label"]) == "person"]
        missing = [
            d for d in detections if canonical(d["label"]) in ("no-helmet", "no-vest")
        ]
        vehicles = [
            d
            for d in detections
            if canonical(d["label"])
            in ("car", "truck", "bus", "vehicle", "machinery", "excavator", "forklift")
        ]
        evidence = []
        for p in people:
            x1, y1, x2, y2 = p["box"]
            height = max(1, y2 - y1)
            for gear in missing:
                b = gear["box"]
                center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                # Assign evidence exclusively to the smallest containing person.
                owners = [q for q in people if inside(center, q["box"])]
                owner = min(
                    owners,
                    key=lambda q: (
                        (q["box"][2] - q["box"][0]) * (q["box"][3] - q["box"][1])
                    ),
                    default=None,
                )
                if owner is not p:
                    continue
                rel = (center[1] - y1) / height
                label = canonical(gear["label"])
                if (label == "no-helmet" and rel <= 0.45) or (
                    label == "no-vest" and 0.15 <= rel <= 0.85
                ):
                    evidence.append(
                        (
                            p,
                            "PERSON_WITHOUT_HELMET"
                            if label == "no-helmet"
                            else "PERSON_WITHOUT_VEST",
                            min(p["confidence"], gear["confidence"]),
                        )
                    )
            for v in vehicles:
                a = p["box"]
                b = v["box"]
                dx = max(b[0] - a[2], a[0] - b[2], 0)
                dy = max(b[1] - a[3], a[1] - b[3], 0)
                if math.hypot(dx, dy) < height * 0.35:
                    evidence.append(
                        (
                            p,
                            "PERSON_NEAR_VEHICLE",
                            min(p["confidence"], v["confidence"]),
                        )
                    )
                    break
        events = []
        for p, kind, confidence in evidence:
            if confidence < self.min_confidence:
                continue
            # Quantized position is a short-lived fallback when ByteTrack has no ID yet.
            identity = (
                p.get("track_id")
                if p.get("track_id") is not None
                else tuple(round(v / 50) for v in p["box"])
            )
            key = (kind, identity)
            if timestamp - self.last.get(key, -1e9) < self.cooldown:
                continue
            self.last[key] = timestamp
            events.append(
                dict(
                    event_type=kind,
                    confidence=confidence,
                    frame=frame,
                    timestamp=timestamp,
                    track_id=p.get("track_id"),
                    box=p["box"],
                )
            )
        self.last = {
            k: t
            for k, t in self.last.items()
            if timestamp - t < max(60, self.cooldown * 2)
        }
        return events
