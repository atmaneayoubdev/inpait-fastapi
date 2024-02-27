import os
import cv2
import numpy as np
import torch

from helper import (
    norm_img,
)
from schema import InpaintRequest
from base import InpaintModel


class LaMa(InpaintModel):
    name = "lama"
    pad_mod = 8
    is_erase_model = True

    def init_model(self, device, **kwargs):
        # Update this with the actual path to your model file
        model_path = "checkpoints/big-lama.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model = torch.jit.load(model_path, map_location=device).eval()

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists("path/to/big-lama.pt")

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res
