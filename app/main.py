# Import necessary modules
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from typing import Optional, Dict
import cv2
from loguru import logger
import numpy as np
import torch
from PIL import Image
import base64
import io
import time
from fastapi import APIRouter
from utils import torch_gc
# Import utility functions
from utils import decode_base64_to_image, concat_alpha_channel, pil_to_bytes
from schema import InpaintRequest
# Define API endpoints
import time
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


# Create FastAPI app instance
app = FastAPI()
# Get the base directory of the project
base_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to the parent directory of the app folder
parent_dir = os.path.dirname(base_dir)
# Construct the path to the model file
model_path = os.path.join(parent_dir, 'models', 'big-lama.pt')
map_location = torch.device('cpu')  # Load the model on CPU
model = torch.load(model_path, map_location=map_location)
model.eval()


router = APIRouter(prefix="/api/v1")


@router.post("/inpaint")
async def inpaint_images(req: InpaintRequest):
    print("===========Inpainting==========")
    # Decode image and mask
    image_np, alpha_channel, infos = decode_base64_to_image(req.image)
    mask_np, _, _ = decode_base64_to_image(req.mask, gray=True)

    # Threshold the mask
    mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)[1]

    # Check if image and mask sizes match
    if image_np.shape[:2] != mask_np.shape[:2]:
        raise HTTPException(
            400,
            detail=f"Image size({image_np.shape[:2]}) and mask size({mask_np.shape[:2]}) not match.",
        )

    # Perform inpainting
    start = time.time()
    with torch.no_grad():
        image_tensor = torch.from_numpy(
            image_np.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(
            0).unsqueeze(0).float() / 255.0
        inpainted_image_tensor = model(image_tensor, mask_tensor)
    logging.info(f"process time: {(time.time() - start) * 1000:.2f}ms")

    inpainted_image = (inpainted_image_tensor.squeeze(0).permute(
        1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)

    alpha_channel = None

    inpainted_image = concat_alpha_channel(inpainted_image, alpha_channel)

    inpainted_pil_image = Image.fromarray(inpainted_image)

    res_img_bytes = pil_to_bytes(inpainted_pil_image, ext="png", quality=95)

    # Return response
    return Response(
        content=res_img_bytes,
        media_type="image/png",
    )


@router.post("/retouch")
def api_inpaint(self, req: InpaintRequest):
    image, alpha_channel, infos = decode_base64_to_image(req.image)
    mask, _, _ = decode_base64_to_image(req.mask, gray=True)

    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    if image.shape[:2] != mask.shape[:2]:
        raise HTTPException(
            400,
            detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
        )

    if req.paint_by_example_example_image:
        paint_by_example_image, _, _ = decode_base64_to_image(
            req.paint_by_example_example_image
        )

    start = time.time()
    rgb_np_img = self.model_manager(image, mask, req)
    logger.info(f"process time: {(time.time() - start) * 1000:.2f}ms")
    torch_gc()

    rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

    ext = "png"
    res_img_bytes = pil_to_bytes(
        Image.fromarray(rgb_res),
        ext=ext,
        quality=self.config.quality,
        infos=infos,
    )

    return Response(
        content=res_img_bytes,
        media_type=f"image/{ext}",
        headers={"X-Seed": str(req.sd_seed)},
    )


app.include_router(router)
