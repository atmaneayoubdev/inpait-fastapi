import logging
import uvicorn
from app.helper import (
    compress_image,
    decode_base64_to_image,
    pil_to_bytes,
    concat_alpha_channel,
    get_bounding_box_coordinates,
    download_and_create_mask,
    upload_img_to_gcp,
    validate_api_key,
    make_post_request,
    create_mask_from_base64
)
from loguru import logger
from fastapi import FastAPI, HTTPException, Response, status
from PIL import Image
import torch
import numpy as np
import cv2
import time
from app.lama import LaMa
from app.schema import InpaintRequest, WatermarkImgRemoverRequest, WatermarkRemoverRequest
from fastapi.middleware.cors import CORSMiddleware
import os

# Set the path to the service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "app/valuator-381307-0ceea1748d71.json"
logging.basicConfig(level=logging.DEBUG)


# Import FastAPI and other necessary modules
# Other imports and code...

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    # torch._C._jit_set_nvfuser_enabled(False)
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


@app.post("/api/v1/magic-eraser")
async def api_inpaint(req: InpaintRequest):
    try:
        start_time = time.time()

        req.image = req.image.split(",")[1] if "," in req.image else req.image
        req.mask = req.mask.split(",")[1] if "," in req.mask else req.mask

        if await validate_api_key(req.api_key) != 200:
            raise HTTPException(status_code=401, detail="Invalid API key")

        image, alpha_channel, infos = decode_base64_to_image(req.image)
        mask, _, _ = decode_base64_to_image(req.mask, gray=True)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        if image.shape[:2] != mask.shape[:2]:
            raise HTTPException(
                status_code=400, detail=f"Image size ({image.shape[:2]}) and mask size ({mask.shape[:2]}) do not match."
            )

        process_start_time = time.time()
        inpainted_image = lama_model(image, mask, req)
        logger.info(
            f"Model inference time: {(time.time() - process_start_time) * 1000:.2f} ms")

        inpainted_image = cv2.cvtColor(
            inpainted_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        result_image = concat_alpha_channel(inpainted_image, alpha_channel)
        result_image_bytes = pil_to_bytes(Image.fromarray(
            result_image), ext="png", quality=95, infos=infos)

        time_taken = int((time.time() - start_time) * 1000)
        proceed_request_data = {
            "apiKey": req.api_key,
            "creditsToDeduct": 1,
            "endpointName": "Magic Eraser",
            "origin": req.origin,
            "timeTaken": time_taken
        }
        await make_post_request(proceed_request_data)
        logger.info(proceed_request_data)
        logger.info("API request processed successfully")

        return Response(content=result_image_bytes, media_type="image/png")

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/watermark-remover-url")
async def watermark_remover_with_url(req: WatermarkRemoverRequest):
    try:
        start = time.time()

        if await validate_api_key(req.api_key) != 200:
            raise HTTPException(status_code=401, detail="Invalid API key")

        top_left_x, top_left_y, bottom_right_x, bottom_right_y = await get_bounding_box_coordinates(req.image_url)

        base64_img, base64_mask = await download_and_create_mask(req.image_url, top_left_x, top_left_y,
                                                                 bottom_right_x, bottom_right_y)
        image, alpha_channel, infos = decode_base64_to_image(base64_img)
        mask, _, _ = decode_base64_to_image(base64_mask, gray=True)

        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        if image.shape[:2] != mask.shape[:2]:
            raise HTTPException(
                status_code=400,
                detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
            )

        # Use the LaMa model for inpainting
        process_start_time = time.time()
        rgb_np_img = lama_model(image, mask, req)
        logger.info(
            f"Model inference time: {(time.time() - process_start_time) * 1000:.2f} ms")
        # torch_gc()

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

        time_taken = int((time.time() - start) * 1000)

        proceed_request_data = {
            "apiKey": req.api_key,
            "creditsToDeduct": 1,
            "endpointName": "Watermark Remover",
            "origin": req.origin,
            "timeTaken": time_taken
        }

        # Make the proceed request
        await make_post_request(proceed_request_data)

        # Log the proceed request data
        logger.info(proceed_request_data)

        return Response(
            content=res_img_bytes,
            media_type=f"image/{ext}",
        )

    except HTTPException as http_exception:
        # Handle HTTPException raised by get_bounding_box_coordinates
        raise http_exception

    except Exception as e:
        logger.info(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/watermark-remover-img")
async def watermark_remover_with_img(req: WatermarkImgRemoverRequest):

    try:
        start = time.time()

        if "," in req.image:
            req.image = req.image.split(",")[1]

        if await validate_api_key(req.api_key) != 200:
            raise HTTPException(status_code=401, detail="Invalid API key")

        image, alpha_channel, infos = decode_base64_to_image(req.image)
        image_url = upload_img_to_gcp(req.image)
        logger.info(image_url)

        top_left_x, top_left_y, bottom_right_x, bottom_right_y = await get_bounding_box_coordinates(
            image_url)

        base64_img, base64_mask = await create_mask_from_base64(req.image, top_left_x, top_left_y,
                                                                bottom_right_x, bottom_right_y)

        image, alpha_channel, infos = decode_base64_to_image(base64_img)
        mask, _, _ = decode_base64_to_image(base64_mask, gray=True)

        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        if image.shape[:2] != mask.shape[:2]:
            raise HTTPException(
                status_code=400,
                detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
            )

        # Use the LaMa model for inpainting
        process_start_time = time.time()
        rgb_np_img = lama_model(image, mask, req)
        logger.info(
            f"Model inference time: {(time.time() - process_start_time) * 1000:.2f} ms")

        # torch_gc()

        rgb_np_img = cv2.cvtColor(
            rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

        ext = "png"
        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext=ext,
            quality=95,
            infos=infos,
        )

        time_taken = int((time.time() - start) * 1000)

        proceed_request_data = {
            "apiKey": req.api_key,
            "creditsToDeduct": 1,
            "endpointName": "Watermark Remover",
            "origin": req.origin,
            "timeTaken": time_taken
        }

        # Make the proceed request
        await make_post_request(proceed_request_data)

        # Log the proceed request data
        logger.info(proceed_request_data)

        return Response(
            content=res_img_bytes,
            media_type=f"image/{ext}",
        )

    except HTTPException as http_exception:
        # Handle HTTPException raised by get_bounding_box_coordinates
        raise http_exception

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=e)


@app.post("/api/v1/watermark-remover-dp")
async def watermark_remover_dp(req: WatermarkRemoverRequest):

    try:
        start = time.time()

        # if await validate_api_key(req.api_key) != 200:
        #     raise HTTPException(status_code=401, detail="Invalid API key")

        top_left_x, top_left_y, bottom_right_x, bottom_right_y = await get_bounding_box_coordinates(req.image_url)

        base64_img, base64_mask = await download_and_create_mask(req.image_url, top_left_x, top_left_y,
                                                                 bottom_right_x, bottom_right_y)
        image, alpha_channel, infos = decode_base64_to_image(base64_img)
        mask, _, _ = decode_base64_to_image(base64_mask, gray=True)

        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        if image.shape[:2] != mask.shape[:2]:
            raise HTTPException(
                status_code=400,
                detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
            )

        # Compress the image
        compressed_image = compress_image(image, target_size_mb=1)

        start = time.time()
        rgb_np_img = lama_model(compressed_image, mask, req)
        logger.info(f"Process time: {(time.time() - start) * 1000:.2f}ms")

        # Convert data type to uint8
        rgb_np_img = np.uint8(rgb_np_img)
        pil_image = Image.fromarray(rgb_np_img)

        # # Add watermark
        # watermark_path = os.path.join("app", "media", "DpLogo.png")
        # pil_image = resize_and_position_watermark(pil_image, watermark_path)

        # Convert PIL image to bytes
        ext = "jpeg"
        res_img_bytes = pil_to_bytes(
            pil_image,
            ext=ext,
            quality=95,  # Assuming config is defined somewhere else
            infos=infos,
        )

        time_taken = int((time.time() - start) * 1000)

        proceed_request_data = {
            "apiKey": req.api_key,
            "creditsToDeduct": 1,
            "endpointName": "Watermark Remover",
            "origin": req.origin,
            "timeTaken": time_taken
        }

        # Make the proceed request
        await make_post_request(proceed_request_data)

        # Log the proceed request data
        logger.info(proceed_request_data)

        return Response(
            content=res_img_bytes,
            media_type=f"image/{ext}",
        )

    except HTTPException as http_exception:
        # Handle HTTPException raised by get_bounding_box_coordinates
        raise http_exception

    except Exception as e:
        logger.info(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


################################## Free Version############################


if __name__ == "__main__":
    # num_workers = multiprocessing.cpu_count()
    # logger.success(f"Num Workers: {num_workers}")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
