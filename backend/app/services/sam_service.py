import cv2
import numpy as np


class SamService:
    def __init__(self, checkpoint, device="cpu"):
        self.checkpoint = checkpoint
        self.device = device
        self.model = None

    def segment(self, image, boxes):
        if not boxes:
            return []
        if self.model is None:
            from ultralytics import SAM

            self.model = SAM(self.checkpoint)
        results = self.model.predict(
            image, bboxes=boxes, device=self.device, verbose=False
        )
        if results[0].masks is None:
            return []
        h, w = image.shape[:2]
        return [
            cv2.resize(m.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST).astype(
                bool
            )
            for m in results[0].masks.data
        ]

    @staticmethod
    def overlay(image, masks):
        out = image.copy()
        for mask in masks:
            out[mask] = (out[mask] * 0.60 + np.array([180, 200, 35]) * 0.40).astype(
                np.uint8
            )
        return out
