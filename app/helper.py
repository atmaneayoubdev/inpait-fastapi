import random
from PIL import Image
import base64
import datetime
import gc
import imghdr
import io
import os
import sys
from typing import Any, List, Optional, Dict, Tuple
import cv2
from PIL import Image, ImageOps, PngImagePlugin
from fastapi import HTTPException
from loguru import logger
import numpy as np
import requests
import torch
from google.cloud import storage
from app.schema import ProceedRequest
import logging
import httpx
logging.basicConfig(level=logging.INFO)


def save_image_bytes(image_bytes, output_dir, filename):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file path
    output_path = os.path.join(output_dir, filename)

    # Save the image bytes to the output file
    with open(output_path, 'wb') as f:
        f.write(image_bytes)

    print(f"Image saved to: {output_path}")


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def pil_to_bytes(pil_img, ext: str, quality: int = 95, infos={}) -> bytes:
    with io.BytesIO() as output:
        kwargs = {k: v for k, v in infos.items() if v is not None}
        if ext == "jpg":
            ext = "jpeg"
        if "png" == ext.lower() and "parameters" in kwargs:
            pnginfo_data = PngImagePlugin.PngInfo()
            pnginfo_data.add_text("parameters", kwargs["parameters"])
            kwargs["pnginfo"] = pnginfo_data

        pil_img.save(output, format=ext, quality=quality, **kwargs)
        image_bytes = output.getvalue()
    return image_bytes


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def numpy_to_bytes(image_numpy: np.ndarray, ext: str) -> bytes:
    data = cv2.imencode(
        f".{ext}",
        image_numpy,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(
            cv2.IMWRITE_PNG_COMPRESSION), 0],
    )[1]
    image_bytes = data.tobytes()
    return image_bytes


def load_img(img_bytes, gray: bool = False, return_info: bool = False):
    alpha_channel = None
    image = Image.open(io.BytesIO(img_bytes))

    if return_info:
        infos = image.info

    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    if return_info:
        return np_img, alpha_channel, infos
    return np_img, alpha_channel


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def resize_max_size(
    np_img, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img


def pad_img_to_modulo(
    img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None
):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w, 1)  0~255

    Returns:

    """
    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h]).astype(int)

        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes


def only_keep_largest_contour(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w)  0~255

    Returns:

    """
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_index = -1
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_index = i

    if max_index != -1:
        new_mask = np.zeros_like(mask)
        return cv2.drawContours(new_mask, contours, max_index, 255, -1)
    else:
        return mask


def is_mac():
    return sys.platform == "darwin"


def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w


def encode_pil_to_base64(image: Image, quality: int, infos: Dict) -> bytes:
    img_bytes = pil_to_bytes(
        image,
        "png",
        quality=quality,
        infos=infos,
    )
    return base64.b64encode(img_bytes)


def adjust_mask(mask: np.ndarray, kernel_size: int, operate):
    # fronted brush color "ffcc00bb"
    # kernel_size = kernel_size*2+1
    mask[mask >= 127] = 255
    mask[mask < 127] = 0

    if operate == "reverse":
        mask = 255 - mask
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1)
        )
        if operate == "expand":
            mask = cv2.dilate(
                mask,
                kernel,
                iterations=1,
            )
        else:
            mask = cv2.erode(
                mask,
                kernel,
                iterations=1,
            )
    res_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    res_mask[mask > 128] = [255, 203, 0, int(255 * 0.73)]
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_BGRA2RGBA)
    return res_mask


def gen_frontend_mask(bgr_or_gray_mask):
    if len(bgr_or_gray_mask.shape) == 3 and bgr_or_gray_mask.shape[2] != 1:
        bgr_or_gray_mask = cv2.cvtColor(bgr_or_gray_mask, cv2.COLOR_BGR2GRAY)

    # fronted brush color "ffcc00bb"
    # TODO: how to set kernel size?
    kernel_size = 9
    bgr_or_gray_mask = cv2.dilate(
        bgr_or_gray_mask,
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1,
    )
    res_mask = np.zeros(
        (bgr_or_gray_mask.shape[0], bgr_or_gray_mask.shape[1], 4), dtype=np.uint8
    )
    res_mask[bgr_or_gray_mask > 128] = [255, 203, 0, int(255 * 0.73)]
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_BGRA2RGBA)
    return res_mask


def check_image_exists_in_gcs(project_id, bucket_name, image_name):
    """Check if an image with a given name exists in a specific GCS bucket."""
    client = storage.Client(project=project_id)

    bucket = client.get_bucket(bucket_name)

    try:
        # Check if the image exists in the bucket
        blob = bucket.blob(image_name)
        exists = blob.exists()

        if exists:
            print(
                f"The image '{image_name}' exists in bucket '{bucket_name}'.")
            return True
        else:
            print(
                f"The image '{image_name}' does not exist in bucket '{bucket_name}'.")
            return False

    except Exception as e:
        print(f"Error checking image '{image_name}': {str(e)}")
        return False


def download_image_from_gcs(image_name):
    project_id = 'valuator-381307'
    bucket_name = 'water_mark_remover_masks'

    client = storage.Client(project=project_id)
    bucket = client.get_bucket(bucket_name)

    try:
        # Download the image as bytes
        blob = bucket.blob(image_name)
        image_bytes = blob.download_as_string()

        # Convert image bytes to base64 string
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        print(f"The image '{image_name}' has been downloaded as base64.")
        return image_base64
    except Exception as e:
        print(f"Error downloading image '{image_name}': {str(e)}")
        return None


def upload_img_to_gcp(image_base64):
    project_id = "valuator-381307"
    bucket_name = "watermark-removing-temp"

    try:
        # Initialize the GCS client
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)

        # Convert base64 image data to bytes
        image_bytes = base64.b64decode(image_base64)

        # Upload image to GCS with correct content type
        blob = bucket.blob("temp_image.png")  # Use a temporary file name
        blob.upload_from_string(image_bytes, content_type="image/png")

        # Set expiration time for the URL
        expiration = datetime.timedelta(hours=1)  # URL expires in 1 hour

        # Generate signed URL for the uploaded image
        url = blob.generate_signed_url(expiration=expiration)

        print(
            f"Image uploaded to GCS bucket '{bucket_name}' and URL generated.")
        return url

    except Exception as e:
        print(f"Error uploading image to GCS: {str(e)}")
        return None


def download_image(url: str) -> Optional[np.array]:
    response = requests.get(url)
    if response.status_code == 200:
        # Convert the downloaded image bytes to NumPy array
        image = np.array(Image.open(io.BytesIO(response.content)))
        return image
    else:
        return None


def resize_image(image: np.array, target_shape: Tuple[int, int]) -> np.array:
    # Resize the image to the target shape
    resized_image = cv2.resize(image, target_shape[::-1])
    return resized_image


async def get_bounding_box_coordinates(image_url):
    url = "https://api-us.restb.ai/vision/v2/multipredict"
    headers = {
        "X-Property-ID": "1"
    }
    params = {
        "model_id": "re_logo",
        "client_key": "8c57199bd6239b90d070bd146d25fc24c1a63b2c2a99cbaa8941f52ce2c2f01b",
        "image_url": image_url
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes

            # Parse the JSON response
            json_data = response.json()
            error = json_data.get("error")

            # Check if the response indicates an error
            if error == "true":
                message = json_data.get("message")
                if message == "API rate limit exceeded":
                    raise HTTPException(status_code=429, detail=message)
                error_id = json_data.get("error_id")
                time = json_data.get("time")
                correlation_id = json_data.get("correlation_id")
                version = json_data.get("version")

                # Log the error details
                logging.error(
                    f"Error {error_id} occurred at {time}: {message}, Correlation ID: {correlation_id}, Version: {version}")

                # Raise an HTTPException with status code 400 and custom message
                raise HTTPException(status_code=400, detail=message)

            # Extract bounding box coordinates
            detections = json_data.get("response", {}).get(
                "solutions", {}).get("re_logo", {}).get("detections", [])
            if not detections:
                logging.info("Watermark not detected")
                raise HTTPException(
                    status_code=400, detail="Watermark not detected in the provided image.")

            bounding_box = detections[0]["bounding_box"]
            top_left_x = bounding_box["top_left_x"]
            top_left_y = bounding_box["top_left_y"]
            bottom_right_x = bounding_box["bottom_right_x"]
            bottom_right_y = bounding_box["bottom_right_y"]

            return top_left_x, top_left_y, bottom_right_x, bottom_right_y

    except httpx.RequestError as e:
        # Log the error and raise an HTTPException with status code 400
        logging.error(f"Error occurred while processing the image: {str(e)}")
        raise HTTPException(
            status_code=400, detail="Error occurred while processing the image.")


def draw_bounding_box(image_url, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    # Download the image from the URL
    response = requests.get(image_url)
    if response.status_code == 200:
        # Create a stream-like object from the image content
        image_content = io.BytesIO(response.content)
        # Decode the image content with OpenCV
        image = cv2.imdecode(np.frombuffer(
            image_content.read(), np.uint8), cv2.IMREAD_COLOR)

        # Calculate bounding box coordinates in pixel values
        height, width, _ = image.shape
        x1 = int(top_left_x * width)
        y1 = int(top_left_y * height)
        x2 = int(bottom_right_x * width)
        y2 = int(bottom_right_y * height)

        # Draw bounding box on the image
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Save the mask image
        # create_mask(image.shape, top_left_x, top_left_y,
        #             bottom_right_x, bottom_right_y, "Images/restb_mask.jpg")

        # Display the image with bounding box
        cv2.imshow("Image with Bounding Box", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to download image from the URL")


async def download_and_create_mask(image_url, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        if response.status_code == 200:
            # Create a stream-like object from the image content
            image_content = io.BytesIO(response.content)
            # Decode the image content with OpenCV
            image = cv2.imdecode(np.frombuffer(
                image_content.read(), np.uint8), cv2.IMREAD_COLOR)

            # Calculate image dimensions
            height, width, _ = image.shape

            # Calculate bounding box coordinates in pixel values
            x1 = int(top_left_x * width)
            y1 = int(top_left_y * height)
            x2 = int(bottom_right_x * width)
            y2 = int(bottom_right_y * height)

            # Create a black image mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # Draw the bounding box on the mask
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

            # Encode the image and mask as base64
            _, image_buffer = cv2.imencode(".png", image)
            _, mask_buffer = cv2.imencode(".png", mask)

            image_base64 = base64.b64encode(image_buffer).decode()
            mask_base64 = base64.b64encode(mask_buffer).decode()

            return image_base64, mask_base64

        else:
            print("Failed to download image from the URL")
            return None, None


async def create_mask_from_base64(image_base64, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    try:
        # Log the first 50 characters of the base64 string
        logging.info("Received image_base64: %s", image_base64[:50])

        # Decode base64 image data
        image_bytes = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Calculate image dimensions
        height, width, _ = image.shape

        # Calculate bounding box coordinates in pixel values
        x1 = int(top_left_x * width)
        y1 = int(top_left_y * height)
        x2 = int(bottom_right_x * width)
        y2 = int(bottom_right_y * height)

        # Create a black image mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw the bounding box on the mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Encode the mask as base64
        _, mask_buffer = cv2.imencode(".png", mask)
        mask_base64 = base64.b64encode(mask_buffer).decode()

        # Encode the image as base64
        _, image_buffer = cv2.imencode(".png", image)
        image_base64 = base64.b64encode(image_buffer).decode()

        return image_base64, mask_base64

    except Exception as e:
        logging.error(f"Error creating mask from base64 image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def compress_image(image_np, target_size_mb=1):
    # Convert image to PIL Image
    image_pil = Image.fromarray(image_np)

    # Define compression quality
    quality = 95

    # Initialize compression loop parameters
    min_quality = 1
    max_quality = 100
    step = 5

    # Initialize compressed image size
    compressed_size = target_size_mb * 1024 * 1024

    # Perform binary search for the optimal quality
    while True:
        # Save image to memory buffer with the current compression quality
        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG", quality=quality)

        # Calculate the size of the compressed image
        compressed_image_size = buffer.tell()

        # Adjust quality based on the comparison with the target size
        if compressed_image_size > compressed_size:
            max_quality = quality
            quality -= step
        else:
            min_quality = quality
            quality += step

        # If the difference between min and max quality is less than the step,
        # or if the quality is already at the minimum or maximum, break the loop
        if max_quality - min_quality < step or quality <= min_quality or quality >= max_quality:
            break

    # Convert the PIL Image back to NumPy array
    compressed_image_np = cv2.imdecode(np.frombuffer(
        buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

    return compressed_image_np


def compress_image_png(image_np, target_size_mb=1):
    # Convert image to PIL Image
    image_pil = Image.fromarray(image_np)

    # Define compression level
    compression_level = 6  # Adjust the compression level as needed

    # Initialize compression loop parameters
    min_compression = 0  # Minimum compression level
    max_compression = 9  # Maximum compression level

    # Initialize compressed image size
    compressed_size = target_size_mb * 1024 * 1024

    # Perform binary search for the optimal compression level
    while True:
        # Save image to memory buffer with the current compression level
        buffer = io.BytesIO()
        image_pil.save(buffer, format="PNG", compress_level=compression_level)

        # Calculate the size of the compressed image
        compressed_image_size = buffer.tell()

        # Adjust compression level based on the comparison with the target size
        if compressed_image_size > compressed_size:
            max_compression = compression_level
            compression_level = (min_compression + compression_level) // 2
        else:
            min_compression = compression_level
            compression_level = (max_compression + compression_level) // 2

        # If the difference between min and max compression is less than 1,
        # or if the compression level is already at the minimum or maximum, break the loop
        if max_compression - min_compression < 1 or compression_level <= min_compression or compression_level >= max_compression:
            break

    # Convert the PIL Image back to NumPy array
    compressed_image_np = cv2.imdecode(np.frombuffer(
        buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

    return compressed_image_np


# def validate_api_key(api_key: str) -> int:
#     VALIDATION_URL = f"https://sefarai.vercel.app/api/api-keys/{api_key}/validate"
#     url = VALIDATION_URL.format(key=api_key)
#     try:
#         response = requests.post(url)
#         return response.status_code
#     except requests.RequestException as e:
#         logger.error(f"Validation request failed: {e}")
#         return 500  # Internal Server Error


# def make_post_request(payload):
#     url = "https://sefarai.vercel.app/api/request/proceed"
#     headers = {
#         "Content-Type": "application/json"
#     }

#     response = requests.post(url, json=payload, headers=headers)

#     if response.status_code == 200:
#         return response.json()
#     else:
#         raise HTTPException(
#             status_code=response.status_code, detail=response.text)


# def decode_base64_to_image(
#     encoding: str, gray=False
# ) -> Tuple[np.array, Optional[np.array], Dict]:
#     if encoding.startswith("data:image/") or encoding.startswith(
#         "data:application/octet-stream;base64,"
#     ):

#         encoding = encoding.split(";")[1].split(",")[1]
#     image = Image.open(io.BytesIO(base64.b64decode(encoding)))

#     alpha_channel = None
#     try:
#         image = ImageOps.exif_transpose(image)
#     except:
#         pass
#     # exif_transpose will remove exif rotate info，we must call image.info after exif_transpose
#     infos = image.info

#     if gray:
#         image = image.convert("L")
#         np_img = np.array(image)
#     else:
#         if image.mode == "RGBA":
#             np_img = np.array(image)
#             alpha_channel = np_img[:, :, -1]
#             np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
#         else:
#             image = image.convert("RGB")
#             np_img = np.array(image)

#     return np_img, alpha_channel, infos


# def concat_alpha_channel(rgb_np_img, alpha_channel) -> np.ndarray:
#     if alpha_channel is not None:
#         if alpha_channel.shape[:2] != rgb_np_img.shape[:2]:
#             alpha_channel = cv2.resize(
#                 alpha_channel, dsize=(rgb_np_img.shape[1], rgb_np_img.shape[0])
#             )
#         rgb_np_img = np.concatenate(
#             (rgb_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
#         )
#     return rgb_np_img


###################################################################
async def validate_api_key(api_key: str) -> int:
    VALIDATION_URL = f"https://sefarai.vercel.app/api/api-keys/{api_key}/validate"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(VALIDATION_URL)
            return response.status_code
        except httpx.RequestError as e:
            logger.error(f"Validation request failed: {e}")
            return 500  # Internal Server Error


async def make_post_request(payload: Dict[str, Any]) -> Any:
    url = "https://sefarai.vercel.app/api/request/proceed"
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code, detail=response.text)


def decode_base64_to_image(encoding: str, gray: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    if encoding.startswith("data:image/") or encoding.startswith("data:application/octet-stream;base64,"):
        encoding = encoding.split(";")[1].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoding)))

    alpha_channel = None
    try:
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        logger.warning(f"Exif transpose failed: {e}")

    infos = image.info
    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    return np_img, alpha_channel, infos


def concat_alpha_channel(rgb_np_img: np.ndarray, alpha_channel: Optional[np.ndarray]) -> np.ndarray:
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != rgb_np_img.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, dsize=(
                rgb_np_img.shape[1], rgb_np_img.shape[0]))
        rgb_np_img = np.concatenate(
            (rgb_np_img, alpha_channel[:, :, np.newaxis]), axis=-1)
    return rgb_np_img


def pil_to_bytes(pil_img: Image.Image, ext: str, quality: int = 95, infos: Dict = {}) -> bytes:
    with io.BytesIO() as output:
        kwargs = {k: v for k, v in infos.items() if v is not None}
        if ext == "jpg":
            ext = "jpeg"
        if "png" == ext.lower() and "parameters" in kwargs:
            pnginfo_data = PngImagePlugin.PngInfo()
            pnginfo_data.add_text("parameters", kwargs["parameters"])
            kwargs["pnginfo"] = pnginfo_data

        pil_img.save(output, format=ext, quality=quality, **kwargs)
        image_bytes = output.getvalue()
    return image_bytes


def resize_and_position_watermark(image: Image.Image, watermark_path: str) -> Image.Image:
    # Load the watermark image
    watermark = Image.open(watermark_path).convert("RGBA")

    # Resize the watermark to 10% of the image width while maintaining aspect ratio
    image_width, image_height = image.size
    watermark_width = int(image_width * 0.3)
    aspect_ratio = watermark.height / watermark.width
    watermark_height = int(watermark_width * aspect_ratio)
    watermark = watermark.resize(
        (watermark_width, watermark_height), Image.LANCZOS)

    # Define margins and possible positions
    margin = 10
    positions = [
        (margin, margin),  # Top-left corner
        (image_width - watermark_width - margin, margin),  # Top-right corner
        (margin, image_height - watermark_height - margin),  # Bottom-left corner
        (image_width - watermark_width - margin, image_height - \
         watermark_height - margin)  # Bottom-right corner
    ]

    # Choose a random position for the watermark
    position = random.choice(positions)

    # Paste the watermark onto the image
    image.paste(watermark, position, watermark)
    return image


def apply_watermark(image: np.ndarray, margin: int = 20) -> np.ndarray:
    # Open the original image and watermark image
    original_image = Image.fromarray(image)
    watermark = Image.open("app/media/sefar_ai_logo.png").convert("RGBA")

    # Calculate the new size for the watermark (20% of the original image width)
    original_width, original_height = original_image.size
    new_watermark_width = int(original_width * 0.3)
    watermark_aspect_ratio = watermark.size[1] / watermark.size[0]
    new_watermark_height = int(new_watermark_width * watermark_aspect_ratio)

    # Resize the watermark
    watermark = watermark.resize(
        (new_watermark_width, new_watermark_height), Image.Resampling.LANCZOS)

    # Print the resized watermark size for verification
    print(f"Resized watermark size: {watermark.size}")

    # Calculate the position to place the watermark (bottom-right corner with margin)
    position = (original_width - new_watermark_width - margin,
                original_height - new_watermark_height - margin)

    # Add watermark to the original image
    transparent = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    transparent.paste(original_image, (0, 0))
    transparent.paste(watermark, position, mask=watermark)
    # Remove alpha for saving in jpg format
    watermarked_image = transparent.convert('RGB')

    return np.array(watermarked_image)


def resize_and_reduce_quality(image: np.ndarray, max_side: int = 500, quality: int = 85) -> np.ndarray:
    # Convert the image to PIL format
    pil_image = Image.fromarray(image)

    # Calculate the new size while maintaining the aspect ratio
    original_width, original_height = pil_image.size
    if original_width > original_height:
        new_width = max_side
        new_height = int((max_side / original_width) * original_height)
    else:
        new_height = max_side
        new_width = int((max_side / original_height) * original_width)

    # Resize the image
    resized_image = pil_image.resize(
        (new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized image to a BytesIO object with reduced quality
    output = io.BytesIO()
    resized_image.save(output, format='JPEG', quality=quality)
    output.seek(0)

    # Read the image back from the BytesIO object
    reduced_quality_image = Image.open(output)

    return np.array(reduced_quality_image)
# Example usage
# resized_image = resize_and_reduce_quality(image)
