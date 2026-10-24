from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    yolo_model: str = "models/yolo11s.pt"
    model_path: str = "models/ppe/weights/best.pt"
    yolo_confidence: float = Field(0.20, ge=0.01, le=1)
    display_confidence: float = Field(0.50, ge=0.01, le=1)
    event_confidence: float = Field(0.50, ge=0.01, le=1)
    yolo_iou: float = Field(0.45, gt=0, le=1)
    device: str = "auto"
    image_size: int = Field(640, ge=32, multiple_of=32)
    enable_sam2: bool = True
    sam_model: str = "models/sam2.1_t.pt"
    sam_interval: int = Field(15, ge=1)
    sam_max_objects: int = Field(2, ge=1)
    enable_dino: bool = True
    dino_device: str = "cpu"
    uncertainty_threshold: float = Field(0.45, ge=0, le=1)
    active_learning_batch_size: int = Field(12, ge=1)
    uncertainty_weight: float = Field(0.55, ge=0)
    diversity_weight: float = Field(0.45, ge=0)
    temporal_window: float = Field(2, ge=0)
    detection_interval: int = Field(2, ge=1)
    runtime: str = Field("pytorch", pattern="^(pytorch|onnx|tensorrt)$")
    dataset_path: str = "configs/construction-ppe.yaml"
    data_dir: Path = Path("results")
    enable_vehicles: bool = True
    event_cooldown: float = Field(10, ge=0)

    @property
    def resolved_device(self):
        import torch

        return (
            ("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.device == "auto"
            else self.device
        )


settings = Settings()
