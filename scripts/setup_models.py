from pathlib import Path
from ultralytics import YOLO, SAM
from backend.app.services.dino_service import DinoService


def main():
    Path("models").mkdir(exist_ok=True)
    YOLO("models/yolo11s.pt")
    if not Path("models/sam2.1_t.pt").exists():
        import urllib.request

        urllib.request.urlretrieve(
            "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt",
            "models/sam2.1_t.pt",
        )
    SAM("models/sam2.1_t.pt")
    DinoService().load()
    print("YOLO11s, SAM2.1 tiny and DINOv2 ViT-S/14 ready.")


if __name__ == "__main__":
    main()
