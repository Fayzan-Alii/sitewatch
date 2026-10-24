"""Resumable HTTP range download with validated per-part lengths."""

import argparse
import fcntl
import concurrent.futures
from pathlib import Path
import requests


def download(url, destination, workers=8):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    head = requests.get(url, stream=True, timeout=30)
    head.raise_for_status()
    final = head.url.replace(
        "release-assets.githubusercontent.com",
        "releaseassetproduction.blob.core.windows.net",
    )
    size = int(head.headers["Content-Length"])
    head.close()
    chunk = (size + workers - 1) // workers

    def part(i):
        start = i * chunk
        end = min(size - 1, start + chunk - 1)
        expected = end - start + 1
        p = path.with_suffix(path.suffix + f".part{workers}.{i}")
        if p.exists() and p.stat().st_size == expected:
            return p
        offset = p.stat().st_size if p.exists() else 0
        if offset > expected:
            p.unlink()
            offset = 0
        r = requests.get(
            final,
            headers={"Range": f"bytes={start + offset}-{end}"},
            stream=True,
            timeout=(30, 60),
        )
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("Server did not honor Range request")
        with p.open("ab") as f:
            for block in r.iter_content(1024 * 128):
                f.write(block)
        if p.stat().st_size != expected:
            raise RuntimeError("Partial download length mismatch")
        print(f"Part {i + 1}/{workers} complete", flush=True)
        return p

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(part, range(workers)))
    temporary = path.with_suffix(path.suffix + ".assembling")
    with temporary.open("wb") as f:
        for p in parts:
            with p.open("rb") as source:
                while block := source.read(1024 * 1024):
                    f.write(block)
    temporary.replace(path)
    for p in parts:
        p.unlink()
    print(f"Downloaded {path}: {size} bytes", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("destination")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    download(a.url, a.destination, a.workers)
