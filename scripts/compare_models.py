"""Evaluate a fixed candidate against the previous deployment checkpoint.
Run after training ends. This script reports evidence; promotion is explicit.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def evaluate(model, data, split, imgsz, output):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.evaluate",
            "--model",
            model,
            "--data",
            data,
            "--split",
            split,
            "--imgsz",
            str(imgsz),
            "--output",
            str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="models/ppe_extended/weights/best.pt")
    p.add_argument("--baseline", default="models/ppe/weights/best.pt")
    a = p.parse_args()
    root = Path("results/improvement")
    root.mkdir(exist_ok=True)
    results = {}
    for label, model, data, split, size in [
        (
            "baseline-deployed-val",
            a.baseline,
            "configs/construction-ppe.yaml",
            "val",
            320,
        ),
        ("candidate-val", a.candidate, "configs/construction-ppe.yaml", "val", 640),
        (
            "candidate-audited-val",
            a.candidate,
            "configs/construction-ppe-audited.yaml",
            "val",
            640,
        ),
        ("baseline-test", a.baseline, "configs/construction-ppe.yaml", "test", 320),
        ("candidate-test", a.candidate, "configs/construction-ppe.yaml", "test", 640),
        (
            "candidate-audited-test",
            a.candidate,
            "configs/construction-ppe-audited.yaml",
            "test",
            640,
        ),
    ]:
        results[label] = evaluate(model, data, split, size, root / (label + ".json"))
    for label in ("baseline-val", "baseline-audited-val"):
        results[label] = json.loads((root / (label + ".json")).read_text())
    old = results["baseline-deployed-val"]
    new = results["candidate-val"]
    changes = {
        key: new[key] - old[key] for key in ("precision", "recall", "mAP50", "mAP50_95")
    }
    report = {
        "baseline": a.baseline,
        "candidate": a.candidate,
        "comparison": results,
        "changes_vs_previous_deployment": changes,
        "promoted": False,
        "note": "Validation selects the candidate. Held-out test is a final assessment. Perceptual-hash filtering flags some near duplicates; it does not establish independent source provenance.",
    }
    (root / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(changes, indent=2))


if __name__ == "__main__":
    main()
