"""Flag perceptually similar images across splits; preserve original annotations."""

import json
from pathlib import Path
import cv2
import numpy as np
import yaml


def phash(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Cannot decode {path}")
    dct = cv2.dct(cv2.resize(im, (32, 32)).astype(np.float32))[:8, :8].reshape(-1)[1:]
    return dct > np.median(dct)


def main():
    cfg = yaml.safe_load(Path("configs/construction-ppe.yaml").read_text())
    root = Path(cfg["path"])
    out = Path("results/improvement")
    paths = {
        split: sorted((root / "images" / split).glob("*"))
        for split in ("train", "val", "test")
    }
    hashes = {split: np.array([phash(p) for p in pp]) for split, pp in paths.items()}
    report = {
        "method": "63-bit DCT perceptual hash; Hamming distance <= 6 flags a suspected near duplicate, not proof of shared provenance",
        "splits": {},
    }
    for split in ("val", "test"):
        distances = np.count_nonzero(
            hashes[split][:, None] != hashes["train"][None, :], axis=2
        )
        nearest = distances.argmin(axis=1)
        pairs = [
            {
                "image": str(paths[split][i]),
                "nearest_train": str(paths["train"][j]),
                "distance": int(distances[i, j]),
            }
            for i, j in enumerate(nearest)
        ]
        flagged = [p for p in pairs if p["distance"] <= 6]
        clean = [p for p in pairs if p["distance"] > 6]
        report["splits"][split] = {
            "total": len(pairs),
            "flagged": flagged,
            "remaining": len(clean),
        }
        listing = out / f"{split}-less-similar.txt"
        listing.write_text("\n".join(p["image"] for p in clean) + "\n")
    Path("results/improvement/near-duplicates.json").write_text(
        json.dumps(report, indent=2)
    )
    filtered = dict(cfg)
    filtered["val"] = str((out / "val-less-similar.txt").resolve())
    filtered["test"] = str((out / "test-less-similar.txt").resolve())
    Path("configs/construction-ppe-audited.yaml").write_text(yaml.safe_dump(filtered))
    pairs = report["splits"]["val"]["flagged"][:12]
    canvas = np.full((max(1, len(pairs)) * 130, 440, 3), 245, np.uint8)
    for i, pair in enumerate(pairs):
        for j, key in enumerate(("image", "nearest_train")):
            im = cv2.imread(pair[key])
            h, w = im.shape[:2]
            ratio = min(215 / w, 105 / h)
            im = cv2.resize(im, (round(w * ratio), round(h * ratio)))
            canvas[i * 130 : i * 130 + im.shape[0], j * 220 : j * 220 + im.shape[1]] = (
                im
            )
            cv2.putText(
                canvas,
                f"{Path(pair[key]).name} | d={pair['distance']}",
                (j * 220 + 2, i * 130 + 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
            )
    cv2.imwrite(str(out / "near-duplicate-pairs.jpg"), canvas)
    print(
        {
            s: {"flagged": len(v["flagged"]), "remaining": v["remaining"]}
            for s, v in report["splits"].items()
        }
    )


if __name__ == "__main__":
    main()
