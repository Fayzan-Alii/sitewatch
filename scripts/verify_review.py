"""Exercise a persisted class/box correction using official training ground truth.
This is an automated integration check, not a claimed human review.
"""

import json
from pathlib import Path
import cv2
import requests
from backend.app.db import Store
from backend.app.services.annotations import export_reviews
import yaml


def main():
    cfg = yaml.safe_load(Path("configs/construction-ppe.yaml").read_text())
    root = Path(cfg["path"])
    store = Store("results")
    candidates = [
        r
        for r in store.list("review")
        if r.get("job_id")
        == json.loads(Path("results/smoke-video.json").read_text())["id"]
    ]
    if not candidates:
        raise RuntimeError("No active learning candidates from the latest smoke video")
    source = Path(
        json.loads(Path("results/selected-ppe-image.json").read_text())["image"]
    )
    label = root / "labels/train" / f"{source.stem}.txt"
    rows = [list(map(float, line.split())) for line in label.read_text().splitlines()]
    r = candidates[0]
    im = cv2.imread(str(store.root / r["image"]))
    h, w = im.shape[:2]

    # Use the nearest true box to this candidate rather than claiming its predicted label was verified by a person.
    def box(row):
        c, x, y, bw, bh = row
        return [
            max(0, (x - bw / 2) * w),
            max(0, (y - bh / 2) * h),
            min(w, (x + bw / 2) * w),
            min(h, (y + bh / 2) * h),
        ]

    truth = min(
        rows,
        key=lambda row: sum(
            (a - b) ** 2 for a, b in zip(box(row), r["detection"]["box"])
        ),
    )
    action = {"action": "correct", "class_id": int(truth[0]), "box": box(truth)}
    response = requests.patch(
        "http://127.0.0.1:8000/api/reviews/" + r["id"], json=action, timeout=10
    )
    response.raise_for_status()
    persisted = Store("results").get(r["id"])
    assert persisted["status"] == "correct"
    persisted["verification_note"] = (
        "Automated integration correction from official training ground truth; not a human decision."
    )
    store.put("review", persisted, r["id"])
    export = export_reviews(store, cfg["names"])
    assert export["count"] > 0
    Path("results/review-verification.json").write_text(
        json.dumps(
            {
                "review_id": r["id"],
                "action": action,
                "persisted": True,
                "export": export,
                "source": str(source),
                "note": persisted["verification_note"],
            },
            indent=2,
        )
    )
    print(export)


if __name__ == "__main__":
    main()
