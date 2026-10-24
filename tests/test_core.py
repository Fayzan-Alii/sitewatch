import numpy as np
import pytest
from pydantic import ValidationError
from backend.app.config import Settings
from backend.app.services.safety_engine import SafetyEngine
from backend.app.services.active_learning import (
    uncertainty,
    select_diverse,
    CandidatePool,
)
from backend.app.services.annotations import yolo_line, export_reviews
from backend.app.services.yolo_service import Detector, Tracker
from backend.app.db import Store


def det(label, box, confidence=0.9, id=1):
    return dict(label=label, box=box, confidence=confidence, track_id=id, class_id=0)


def test_ppe_association_and_cooldown():
    e = SafetyEngine(10)
    person = det("Person", [0, 0, 100, 200])
    helmet = det("no_helmet", [20, 10, 60, 50])
    assert (
        e.evaluate([person, helmet], 0, 0)[0]["event_type"] == "PERSON_WITHOUT_HELMET"
    )
    assert e.evaluate([person, helmet], 1, 1) == []
    assert len(e.evaluate([person, helmet], 300, 10)) == 1
    assert (
        SafetyEngine().evaluate([person, det("no_helmet", [120, 10, 160, 50])], 0, 0)
        == []
    )
    assert SafetyEngine().evaluate([person], 0, 0) == []


def test_vest_and_vehicle():
    person = det("person", [0, 0, 100, 200])
    missing = det("no-safety-vest", [20, 60, 80, 130])
    vehicle = det("truck", [130, 100, 300, 230], id=2)
    assert {
        e["event_type"]
        for e in SafetyEngine().evaluate([person, missing, vehicle], 0, 0)
    } == {"PERSON_WITHOUT_VEST", "PERSON_NEAR_VEHICLE"}


def test_nested_people_unique_association():
    people = [
        det("person", [0, 0, 200, 300], id=1),
        det("person", [10, 0, 80, 200], id=2),
    ]
    events = SafetyEngine().evaluate(
        people + [det("no_helmet", [20, 10, 40, 30])], 0, 0
    )
    assert len(events) == 1 and events[0]["track_id"] == 2


def test_uncertainty_and_diversity():
    assert uncertainty(0.2) == pytest.approx(0.8)
    assert select_diverse([[1, 0], [1, 0], [0, 1]], [0.8, 0.8, 0.7], 2) == [0, 2]
    assert select_diverse([], [], 10) == []
    assert len(select_diverse([[1, 0], [0, 1]], [0.5, 0.5], 12)) == 2


def test_temporal_dedup():
    class Dino:
        def embed(self, image):
            return np.array([1, 0])

    pool = CandidatePool(Settings(temporal_window=2), Dino())
    im = np.zeros((100, 100, 3), np.uint8)
    d = det("Person", [0, 0, 30, 40], 0.3)
    for t in [0, 0.1, 1, 2]:
        pool.collect(im, [d], t, int(t * 30))
    assert len(pool.items) == 2


def test_annotation_conversion():
    assert (
        yolo_line(2, [0, 0, 100, 100], 200, 200)
        == "2 0.250000 0.250000 0.500000 0.500000"
    )
    with pytest.raises(ValueError):
        yolo_line(0, [-1, 0, 3, 3], 10, 10)


def test_configuration():
    with pytest.raises(ValidationError):
        Settings(detection_interval=0)
    with pytest.raises(ValidationError):
        Settings(runtime="fake")
    assert Settings(device="cpu").resolved_device == "cpu"


def test_missing_model_error():
    with pytest.raises(FileNotFoundError):
        Detector(Settings(runtime="onnx", model_path="missing.onnx"))


def test_store_persistence(tmp_path):
    s = Store(tmp_path)
    r = s.put("review", {"status": "pending"})
    s.put("review", {"status": "accept"}, r["id"])
    assert Store(tmp_path).get(r["id"])["status"] == "accept"
    assert len(s.list("review")) == 1


def test_tracking_ids():
    t = Tracker()
    a = t.update([det("Person", [10, 10, 50, 90])], (100, 100))[0]
    b = t.update([det("Person", [12, 10, 52, 90])], (100, 100))[0]
    assert a["track_id"] is not None and a["track_id"] == b["track_id"]


def test_review_export_excludes_rejected(tmp_path):
    import cv2

    s = Store(tmp_path / "results")
    cv2.imwrite(str(s.root / "im.jpg"), np.zeros((100, 100, 3), np.uint8))
    r = s.put(
        "review",
        {
            "status": "accept",
            "image": "im.jpg",
            "detection": {"class_id": 0, "box": [10, 10, 30, 40]},
        },
    )
    s.put(
        "review",
        {
            "status": "reject",
            "image": "im.jpg",
            "detection": {"class_id": 0, "box": [50, 50, 80, 90]},
        },
    )
    out = export_reviews(s, {0: "Person"}, tmp_path / "dataset")
    assert out["count"] == 1
    s.put("review", {**r, "status": "reject"}, r["id"])
    assert export_reviews(s, {0: "Person"}, tmp_path / "dataset")["count"] == 0
    assert not list((tmp_path / "dataset/labels/train").glob("*.txt"))


def test_export_rejects_wrong_model_ontology(tmp_path):
    import cv2

    s = Store(tmp_path / "results")
    cv2.imwrite(str(s.root / "im.jpg"), np.zeros((40, 40, 3), np.uint8))
    row = s.put(
        "review",
        {
            "status": "accept",
            "image": "im.jpg",
            "detection": {"class_id": 0, "label": "person", "box": [0, 0, 20, 20]},
        },
    )
    assert export_reviews(s, {0: "helmet"}, tmp_path / "dataset")["count"] == 0
    s.put("review", {**row, "status": "correct", "corrected_class": 0}, row["id"])
    assert export_reviews(s, {0: "helmet"}, tmp_path / "dataset")["count"] == 1


def test_invalid_video_records_failure(tmp_path):
    from backend.app.services.video_service import VideoService

    s = Store(tmp_path)
    job = s.put("job", {"status": "queued", "kind": "video"})
    VideoService(Settings(data_dir=tmp_path), s).run(
        job["id"], str(tmp_path / "missing.mp4")
    )
    assert s.get(job["id"])["status"] == "failed"
    assert "Cannot open video" in s.get(job["id"])["error"]


def test_ppe_boxes_do_not_steal_person_tracks():
    t = Tracker()
    person = det("Person", [10, 10, 50, 90])
    gear = det("helmet", [12, 12, 48, 35], id=None)
    a = t.update([gear, person], (100, 100))
    assert a[0]["track_id"] is None and a[1]["track_id"] is not None


def test_uncertain_ppe_does_not_generate_alert_or_consume_cooldown():
    engine = SafetyEngine(min_confidence=0.5)
    person = det("Person", [0, 0, 100, 200])
    weak = det("no_helmet", [20, 10, 60, 50], confidence=0.27)
    assert engine.evaluate([person, weak], 0, 0) == []
    stronger = {**weak, "confidence": 0.8}
    assert len(engine.evaluate([person, stronger], 1, 0.1)) == 1


def test_detection_scoring_counts_duplicate_predictions_as_false_positives():
    from scripts.score_detections import match_counts

    truth = [{"class_id": 0, "box": [0, 0, 10, 10]}]
    predictions = [
        {"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9},
        {"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.8},
    ]
    assert match_counts(predictions, truth, 0.5, {0: "helmet"})[0] == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
    }
    assert match_counts(predictions, truth, 0.95, {0: "helmet"})[0] == {
        "tp": 0,
        "fp": 0,
        "fn": 1,
    }
