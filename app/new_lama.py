import os
import cv2
import torch
import numpy as np
from loguru import logger

from app.helper import (
    boxes_from_mask,
    pad_img_to_modulo,
    norm_img
)
from app.schema import InpaintRequest, HDStrategy


class NewLaMa:
    name = "lama"
    pad_mod = 8
    pad_to_square = False
    is_erase_model = True
    min_size = None

    def __init__(self, device, **kwargs):
        self.device = device
        self.init_model(device, **kwargs)

    def init_model(self, device, **kwargs):
        model_path = "app/checkpoints/big-lama.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = torch.jit.load(model_path, map_location=device).eval()

    def forward(self, image, mask, config: InpaintRequest):
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

    def _pad_forward(self, image, mask, config: InpaintRequest):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )
        pad_mask = pad_img_to_modulo(
            mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )
        image, mask = self.forward_pre_process(image, mask, config)
        result = self.forward(pad_image, pad_mask, config)
        result = result[0:origin_height, 0:origin_width, :]
        result, image, mask = self.forward_post_process(
            result, image, mask, config)
        if config.sd_keep_unmasked_area:
            mask = mask[:, :, np.newaxis]
            result = result * (mask / 255) + \
                image[:, :, ::-1] * (1 - (mask / 255))
        return result

    def forward_pre_process(self, image, mask, config):
        return image, mask

    def forward_post_process(self, result, image, mask, config):
        return result, image, mask

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        inpaint_result = None
        if config.hd_strategy == HDStrategy.CROP:
            if max(image.shape) > config.hd_strategy_crop_trigger_size:
                logger.info(f"Run crop strategy")
                boxes = boxes_from_mask(mask)
                crop_result = []
                for box in boxes:
                    crop_image, crop_box = self._run_box(
                        image, mask, box, config)
                    crop_result.append((crop_image, crop_box))
                inpaint_result = image[:, :, ::-1]
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    inpaint_result[y1:y2, x1:x2, :] = crop_image
        if inpaint_result is None:
            inpaint_result = self._pad_forward(image, mask, config)
        return inpaint_result

    def _crop_box(self, image, mask, box, config: InpaintRequest):
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]
        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2
        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2
        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h
        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)
        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]
        return crop_img, crop_mask, [l, t, r, b]

    def _run_box(self, image, mask, box, config: InpaintRequest):
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(
            image, mask, box, config)
        return self._pad_forward(crop_img, crop_mask, config), [l, t, r, b]

# Example usage:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = NewLaMa(device)
# inpainted_image = model(image, mask, config)


