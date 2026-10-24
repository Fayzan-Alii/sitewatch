"""Download the official archive and generate a local, portable dataset YAML."""

from pathlib import Path
import hashlib
import json
import zipfile
import yaml


def main():
    root = Path("datasets").resolve()
    root.mkdir(exist_ok=True)
    upstream = yaml.safe_load(
        Path("configs/construction-ppe-upstream.yaml").read_text()
    )
    archive = root / "construction-ppe.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        from scripts.download_parallel import download

        download(upstream["download"], archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError("Unsafe archive entry")
        destination = (
            root
            if any(n.startswith("construction-ppe/") for n in z.namelist())
            else root / "construction-ppe"
        )
        z.extractall(destination)
    cfg = {k: v for k, v in upstream.items() if k != "download"}
    cfg["path"] = str(root / "construction-ppe")
    Path("configs/construction-ppe.yaml").write_text(yaml.safe_dump(cfg))
    Path("results/dataset.json").write_text(
        json.dumps(
            {
                "source": upstream["download"],
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "classes": cfg["names"],
            },
            indent=2,
        )
    )
    print("Dataset ready: configs/construction-ppe.yaml")


if __name__ == "__main__":
    main()
