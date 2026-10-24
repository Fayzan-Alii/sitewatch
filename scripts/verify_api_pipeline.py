import json
import time
from pathlib import Path
import requests

BASE = "http://127.0.0.1:8000/api"


def wait(job):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        r = requests.get(BASE + "/jobs/" + job["id"], timeout=10)
        r.raise_for_status()
        result = r.json()
        if result["status"] in ("completed", "failed"):
            assert result["status"] == "completed", result
            return result
        time.sleep(0.5)
    raise TimeoutError("API job did not finish in 300 seconds")


with Path("results/smoke-input.mp4").open("rb") as f:
    r = requests.post(
        BASE + "/videos",
        files={"file": ("PPE integration verification.mp4", f, "video/mp4")},
        timeout=30,
    )
    r.raise_for_status()
    video = r.json()
# Health must remain responsive while video processing is queued/running.
start = time.perf_counter()
health = requests.get(BASE + "/health", timeout=10)
health.raise_for_status()
health_ms = (time.perf_counter() - start) * 1000
video = wait(video)
source = json.loads(Path("results/selected-ppe-image.json").read_text())["image"]
with Path(source).open("rb") as f:
    r = requests.post(
        BASE + "/infer",
        files={"file": (Path(source).name, f, "image/jpeg")},
        timeout=30,
    )
    r.raise_for_status()
    image = wait(r.json())
assert image["detections"]
media = requests.get(
    "http://127.0.0.1:8000/" + video["video"],
    headers={"Range": "bytes=0-1023"},
    timeout=10,
)
assert media.status_code == 206
out = {
    "video_job": video["id"],
    "video_status": video["status"],
    "video_frames": video["frames"],
    "warnings": video["warnings"],
    "image_job": image["id"],
    "image_detections": len(image["detections"]),
    "health_while_processing_ms": health_ms,
    "video_range_status": media.status_code,
}
Path("results/api-verification.json").write_text(json.dumps(out, indent=2))
print(out)
