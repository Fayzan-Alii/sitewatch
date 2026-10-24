from typing import Literal
from pydantic import BaseModel, Field, model_validator


class Detection(BaseModel):
    box: list[float] = Field(min_length=4, max_length=4)
    class_id: int
    label: str
    confidence: float
    track_id: int | None = None


class ReviewAction(BaseModel):
    action: Literal["accept", "reject", "correct"]
    class_id: int | None = Field(None, ge=0)
    box: list[float] | None = Field(None, min_length=4, max_length=4)

    @model_validator(mode="after")
    def check(self):
        if self.action == "correct" and self.class_id is None:
            raise ValueError("A corrected class_id is required")
        if self.box and (
            self.box[2] <= self.box[0]
            or self.box[3] <= self.box[1]
            or min(self.box) < 0
        ):
            raise ValueError("Invalid xyxy box")
        return self


class TrainRequest(BaseModel):
    epochs: int = Field(10, ge=1, le=300)
    batch: int = Field(2, ge=1, le=32)
    imgsz: int = Field(640, ge=32, le=1280, multiple_of=32)


class VideoRequest(BaseModel):
    source: str
    max_frames: int = Field(0, ge=0)
