from fastapi.testclient import TestClient
from backend.app.db import Store
from backend.app import main


def test_api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", Store(tmp_path))
    with TestClient(main.app) as client:
        assert client.get("/api/health").status_code == 200
        for endpoint in ["events", "reviews", "jobs", "metrics", "benchmarks"]:
            assert client.get("/api/" + endpoint).status_code == 200
        assert client.get("/api/jobs/nonexistent").status_code == 404
        assert (
            client.post(
                "/api/infer", files={"file": ("bad.jpg", b"bad", "image/jpeg")}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/videos", files={"file": ("bad.txt", b"bad", "text/plain")}
            ).status_code
            == 400
        )
        assert client.post("/api/retrain", json={}).status_code == 409
        row = main.store.put(
            "review",
            {"status": "pending", "detection": {"class_id": 0, "box": [0, 0, 10, 10]}},
        )
        assert (
            client.patch(
                "/api/reviews/" + row["id"], json={"action": "accept"}
            ).status_code
            == 200
        )
        assert main.store.get(row["id"])["status"] == "accept"
        assert (
            client.patch(
                "/api/reviews/" + row["id"], json={"action": "correct"}
            ).status_code
            == 422
        )
        assert (
            client.patch(
                "/api/reviews/" + row["id"],
                json={"action": "correct", "class_id": 9999},
            ).status_code
            == 400
        )


def test_accept_maps_foreign_class_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setattr(main, "classes", lambda: {0: "helmet", 6: "Person"})
    row = main.store.put(
        "review",
        {
            "status": "pending",
            "detection": {"class_id": 0, "label": "person", "box": [0, 0, 10, 10]},
        },
    )
    with TestClient(main.app) as client:
        r = client.patch("/api/reviews/" + row["id"], json={"action": "accept"})
        assert r.status_code == 200 and r.json()["corrected_class"] == 6
        row = main.store.put(
            "review",
            {
                "status": "pending",
                "detection": {
                    "class_id": 0,
                    "label": "traffic light",
                    "box": [0, 0, 10, 10],
                },
            },
        )
        assert (
            client.patch(
                "/api/reviews/" + row["id"], json={"action": "accept"}
            ).status_code
            == 409
        )
