import time
import cv2
import numpy as np
import torch
try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass
from PIL import Image
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.helper import (
    decode_base64_to_image,
    pil_to_bytes,
    concat_alpha_channel,
    torch_gc
)
from app.schema import InpaintRequest
from app.lama import LaMa

app = FastAPI()

cors_options = {
    "allow_methods": ["*"],
    "allow_headers": ["*"],
    "allow_origins": ["*"],
    "allow_credentials": True,
}

app.add_middleware(CORSMiddleware, **cors_options)

lama_model = LaMa(device='cuda' if torch.cuda.is_available() else 'cpu')


@app.get("/")
def home():
    return {"Health Check": "OK"}


@app.post("/api/v1/inpaint")
async def api_inpaint(req: InpaintRequest):
    image, alpha_channel, infos = decode_base64_to_image(req.image)
    mask, _, _ = decode_base64_to_image(req.mask, gray=True)

    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    if image.shape[:2] != mask.shape[:2]:
        raise HTTPException(
            400,
            detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
        )

    start = time.time()
    # Use the LaMa model for inpainting
    rgb_np_img = lama_model(image, mask, req)
    logger.info(f"process time: {(time.time() - start) * 1000:.2f}ms")
    torch_gc()

    rgb_np_img = cv2.cvtColor(
        rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

    ext = "png"
    res_img_bytes = pil_to_bytes(
        Image.fromarray(rgb_res),
        ext=ext,
        quality=95,  # Assuming config is defined somewhere else
        infos=infos,
    )

    return Response(
        content=res_img_bytes,
        media_type=f"image/{ext}",
    )
