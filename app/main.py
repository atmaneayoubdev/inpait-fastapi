import base64
import io
import json
import traceback

import requests
from app.helper import (
    check_image_exists_in_gcs,
    decode_base64_to_image,
    download_image_from_gcs,
    pil_to_bytes,
    concat_alpha_channel,
    torch_gc
)
from loguru import logger
from fastapi import FastAPI, HTTPException, Response, status
from PIL import Image
import torch
import numpy as np
import cv2
import time
from app.lama import LaMa
from app.schema import AgencyInpaintRequest, InpaintRequest
from app.helper import check_image_exists_in_gcs
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Response, status
import os

# Set the path to the service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "app/valuator-381307-0ceea1748d71.json"

# Import FastAPI and other necessary modules
# Other imports and code...

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

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


@app.post("/check-agency-watermark")
async def check_agency_watermark(request_data: dict):
    try:
        # Get the image name from the request data
        image_name = request_data.get('name')

        if not image_name:
            return Response(content=json.dumps({'error': 'Image name is required'}).encode('utf-8'), status_code=status.HTTP_400_BAD_REQUEST)

        # Hardcoded GCS details for testing
        project_id = 'valuator-381307'
        bucket_name = 'water_mark_remover_masks'

        # Check if the image exists in the GCS bucket
        image_exists = check_image_exists_in_gcs(
            project_id, bucket_name, image_name)

        # Return a response with the result
        return Response(content=json.dumps({'exists': image_exists}).encode('utf-8'), status_code=status.HTTP_200_OK)

    except Exception as e:
        return Response(content=json.dumps({'error': str(e)}).encode('utf-8'), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/agency-inpaint")
async def agency_inpaint(req: AgencyInpaintRequest):
    try:

        # Download image from imageUrl
        image_response = requests.get(req.image_url)
        image_base64 = base64.b64encode(image_response.content).decode('utf-8')

        # Download mask from GCS using req.mask_name
        mask_base64 = download_image_from_gcs(req.mask_name)
        if mask_base64 is None:
            raise HTTPException(
                404,
                detail=f"Mask image '{req.mask_name}' not found in GCS."
            )

        # print("Image base64:", image_base64)
        # print("Mask base64:", mask_base64)

        image, alpha_channel, infos = decode_base64_to_image(image_base64)
        mask, _, _ = decode_base64_to_image(mask_base64, gray=True)

        # Resize mask to match the size of image if they don't match
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Convert mask to binary
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

    except Exception as e:
        # Print the full traceback in case of an error
        traceback.print_exc()
        # Raise an HTTP exception with the error details
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
