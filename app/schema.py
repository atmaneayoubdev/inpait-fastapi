from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class HDStrategy(str, Enum):
    # Use original image size
    ORIGINAL = "Original"
    # then do inpainting on the resized image. Finally, resize the inpainting result to the original size.
    # The area outside the mask will not lose quality.
    RESIZE = "Resize"
    # Crop masking area(with a margin controlled by hd_strategy_crop_margin) from the original image to do inpainting
    CROP = "Crop"


class InpaintRequest(BaseModel):
    image: str = Field(None, description="base64 encoded image")
    mask: str = Field(None, description="base64 encoded mask")
    origin: str = Field(..., description="the origin source of the api call")
    api_key: str = Field(..., description="API key for validation")

    hd_strategy: str = Field(
        HDStrategy.CROP,
        description="Different way to preprocess image, only used by erase models(e.g. lama/mat)",
    )
    hd_strategy_crop_trigger_size: int = Field(
        800,
        description="Crop trigger size for hd_strategy=CROP, if the longer side of the image is larger than this value, use crop strategy",
    )
    hd_strategy_crop_margin: int = Field(
        128, description="Crop margin for hd_strategy=CROP"
    )
    hd_strategy_resize_limit: int = Field(
        1280, description="Resize limit for hd_strategy=RESIZE"
    )
    sd_keep_unmasked_area: bool = Field(
        True, description="Keep unmasked area unchanged"
    )


class WatermarkRemoverRequest(BaseModel):
    api_key: str = Field(..., description="API key for validation")
    image_url: str = Field(..., description="Image URL")
    origin: str = Field(..., description="the origin source of the api call")

    hd_strategy: str = Field(
        HDStrategy.CROP,
        description="Different way to preprocess image, only used by erase models(e.g. lama/mat)",
    )
    hd_strategy_crop_trigger_size: int = Field(
        800,
        description="Crop trigger size for hd_strategy=CROP, if the longer side of the image is larger than this value, use crop strategy",
    )
    hd_strategy_crop_margin: int = Field(
        128, description="Crop margin for hd_strategy=CROP"
    )
    hd_strategy_resize_limit: int = Field(
        1280, description="Resize limit for hd_strategy=RESIZE"
    )
    sd_keep_unmasked_area: bool = Field(
        True, description="Keep unmasked area unchanged"
    )


class WatermarkImgRemoverRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image")
    origin: str = Field(..., description="the origin source of the api call")
    api_key: str = Field(..., description="API key for validation")

    hd_strategy: str = Field(
        HDStrategy.CROP,
        description="Different way to preprocess image, only used by erase models(e.g. lama/mat)",
    )
    hd_strategy_crop_trigger_size: int = Field(
        800,
        description="Crop trigger size for hd_strategy=CROP, if the longer side of the image is larger than this value, use crop strategy",
    )
    hd_strategy_crop_margin: int = Field(
        128, description="Crop margin for hd_strategy=CROP"
    )
    hd_strategy_resize_limit: int = Field(
        1280, description="Resize limit for hd_strategy=RESIZE"
    )
    sd_keep_unmasked_area: bool = Field(
        True, description="Keep unmasked area unchanged"
    )


class ProceedRequest(BaseModel):
    apiKey: str
    creditsToDeduct: int
    endpointName: str
    origin: str
    timeTaken: int
