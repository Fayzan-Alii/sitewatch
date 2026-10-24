#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.12 .venv
# CUDA 12 wheels keep PyTorch and ONNX Runtime on compatible CUDA libraries.
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r requirements.txt
npm ci --prefix frontend
.venv/bin/python -m scripts.download_dataset
.venv/bin/python -m scripts.setup_models
npm run build --prefix frontend
