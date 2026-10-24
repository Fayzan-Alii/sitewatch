from pathlib import Path
import yaml


def yolo_line(class_id, box, width, height):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box outside image")
    return f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def export_reviews(store, names, output="datasets/reviewed"):
    """Export object crops, not incompletely labelled full frames as negatives."""
    import cv2

    root = Path(output).resolve()
    (root / "images/train").mkdir(parents=True, exist_ok=True)
    (root / "labels/train").mkdir(parents=True, exist_ok=True)
    # Rebuild snapshot: later rejected/corrected records cannot leave stale labels.
    for folder in ("images/train", "labels/train"):
        for old in (root / folder).glob("*"):
            old.unlink()
    count = 0
    for r in store.list("review", 100000):
        if r.get("status") not in ("accept", "correct"):
            continue
        cid = r.get("corrected_class", r["detection"]["class_id"])
        if cid not in names:
            continue
        if "corrected_class" not in r:
            from .safety_engine import canonical

            if canonical(r["detection"].get("label", names[cid])) != canonical(
                names[cid]
            ):
                continue
        image = cv2.imread(str(store.root / r["image"]))
        if image is None:
            continue
        h, w = image.shape[:2]
        box = r.get("corrected_box", r["detection"]["box"])
        yolo_line(cid, box, w, h)
        x1, y1, x2, y2 = box
        # Tight crop avoids silently marking unreviewed people/PPE elsewhere negative.
        left, top, right, bottom = (
            int(x1),
            int(y1),
            min(w, int(x2) + 1),
            min(h, int(y2) + 1),
        )
        crop = image[top:bottom, left:right]
        cv2.imwrite(str(root / f"images/train/{r['id']}.jpg"), crop)
        line = yolo_line(
            cid, [x1 - left, y1 - top, x2 - left, y2 - top], right - left, bottom - top
        )
        (root / f"labels/train/{r['id']}.txt").write_text(line + "\n")
        count += 1
    (root / "classes.yaml").write_text(yaml.safe_dump({"names": names}))
    return {"count": count, "path": str(root)}
