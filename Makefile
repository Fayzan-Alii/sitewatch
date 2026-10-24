.PHONY: setup dataset models train evaluate api frontend test build benchmark
setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu126
	uv pip install --python .venv/bin/python -r requirements.txt
	npm ci --prefix frontend
dataset:
	.venv/bin/python -m scripts.download_dataset
models:
	.venv/bin/python -m scripts.setup_models
train:
	.venv/bin/python -m training.train_yolo --model models/yolo11s.pt
evaluate:
	.venv/bin/python -m training.evaluate
api:
	./scripts/run.sh
frontend:
	npm run dev --prefix frontend
test:
	.venv/bin/python -m pytest -q
build:
	npm run build --prefix frontend
benchmark:
	.venv/bin/python -m benchmarks.benchmark --image $(IMAGE) --sam
