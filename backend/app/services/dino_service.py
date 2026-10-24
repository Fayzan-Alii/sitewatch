import cv2
import torch
import torch.nn.functional as F


class DinoService:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None

    def load(self):
        if self.model is None:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.model = (
                AutoModel.from_pretrained("facebook/dinov2-small")
                .eval()
                .to(self.device)
            )

    @torch.inference_mode()
    def embed(self, image):
        self.load()
        inputs = self.processor(
            images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_tensors="pt"
        ).to(self.device)
        vector = self.model(**inputs).last_hidden_state[:, 0]
        return F.normalize(vector.float(), dim=-1)[0].cpu().numpy()
