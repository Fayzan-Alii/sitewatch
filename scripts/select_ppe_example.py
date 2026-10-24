import json
from pathlib import Path
import cv2
from backend.app.config import Settings
from backend.app.services.yolo_service import Detector, Tracker
from backend.app.services.safety_engine import SafetyEngine

model = Detector(
    Settings(
        model_path="models/ppe/weights/best.pt", image_size=320, enable_vehicles=False
    )
)
root = Path("datasets/construction-ppe")
chosen = None
for label in sorted((root / "labels/train").glob("*.txt")):
    ids = {
        int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()
    }
    if not {6, 7}.issubset(ids):
        continue
    paths = list((root / "images/train").glob(label.stem + ".*"))
    if not paths:
        continue
    image = cv2.imread(str(paths[0]))
    detections = Tracker().update(model.predict(image), image.shape[:2])
    events = SafetyEngine().evaluate(detections, 0, 0)
    if events:
        chosen = {"image": str(paths[0]), "detections": detections, "events": events}
        break
if chosen is None:
    raise RuntimeError(
        "No training example generated a PPE event at the configured threshold"
    )
Path("results/selected-ppe-image.json").write_text(json.dumps(chosen, indent=2))
print(chosen["image"], len(chosen["events"]))
