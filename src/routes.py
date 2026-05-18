import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

import service
from config import settings
from validator import ImageValidator

router = APIRouter()
templates = Jinja2Templates(directory="src/templates/")
img_validator = ImageValidator()


def get_model() -> YOLO:
    """
    Load and return a YOLO model in evaluation mode.

    Intended to be used as a FastAPI dependency so route handlers receive a ready-to-use
    model instance without managing its lifecycle directly.

    Returns:
        YOLO: YOLO model loaded from ``settings.model_path`` and set to evaluation mode.
    """
    model = YOLO(model=settings.model_path)
    model.eval()
    return model


@router.get("/")
async def home_page(request: Request) -> HTMLResponse:
    """
    Render the application's home page.

    Args:
        request (Request): Incoming HTTP request, required by Jinja2 to render the template.

    Returns:
        HTMLResponse: Rendered ``home.html`` template.
    """
    return templates.TemplateResponse(request=request, name="home.html")


@router.post("/dices/stats")
async def stats(
    request: Request,
    files: list[UploadFile],
    model: Annotated[YOLO, Depends(get_model)],
) -> HTMLResponse:
    """
    Detect dice across uploaded images and render aggregate statistics.

    Each file is validated, then dice are detected concurrently using the YOLO model.
    The total image count and total dice count are passed to the ``stats.html`` template.

    Args:
        request (Request): Incoming HTTP request, required by Jinja2 to render the template.
        files (list[UploadFile]): Image files to analyze.
        model (YOLO): YOLO model injected via dependency.

    Raises:
        HTTPException: 400 if any uploaded file fails image validation.

    Returns:
        HTMLResponse: Rendered ``stats.html`` template with total image and dice counts.
    """
    for file in files:
        result = await img_validator.validate_image(file)
        if not result["valid"]:
            raise HTTPException(status_code=400, detail=f"Image is not valid: {result['errors']}")

    detections = await asyncio.gather(*(service.detect_dices(file, model) for file in files))
    total_dices = service.total_dices(detections)
    return templates.TemplateResponse(
        request=request, name="stats.html", context={"total_images": len(files), "total_dices": total_dices}
    )


@router.post("/dices/distribution")
async def normal_distribution(
    request: Request, files: list[UploadFile], model: Annotated[YOLO, Depends(get_model)]
) -> HTMLResponse:
    """
    Detect dice across uploaded images and render per-value distribution statistics.

    Each file is validated, then dice are detected concurrently. Per-value count, variance,
    and standard deviation are computed and rendered in the ``distribution.html`` template.

    Args:
        request (Request): Incoming HTTP request, required by Jinja2 to render the template.
        files (list[UploadFile]): Image files to analyze.
        model (YOLO): YOLO model injected via dependency.

    Raises:
        HTTPException: 400 if any uploaded file fails image validation.

    Returns:
        HTMLResponse: Rendered ``distribution.html`` template with serialized distribution entries.
    """
    for file in files:
        result = await img_validator.validate_image(file)
        if not result["valid"]:
            raise HTTPException(status_code=400, detail=f"Image is not valid: {result['errors']}")

    detections = await asyncio.gather(*(service.detect_dices(file, model) for file in files))
    distributions = service.compute_dice_statistics(detections)
    return templates.TemplateResponse(
        request, "distribution.html", context={"distributions": [d.model_dump() for d in distributions]}
    )


@router.post("/dices/check")
async def check(request: Request, file: UploadFile, model: Annotated[YOLO, Depends(get_model)]) -> HTMLResponse:
    """
    Detect dice on a single uploaded image and return an annotated preview.

    The image is validated, dice are detected with the YOLO model, bounding boxes are
    drawn over the original image, and the annotated result is rendered as a base64-encoded
    image in the ``check.html`` template.

    Args:
        request (Request): Incoming HTTP request, required by Jinja2 to render the template.
        file (UploadFile): Image file to analyze.
        model (YOLO): YOLO model injected via dependency.

    Raises:
        HTTPException: 400 if the uploaded file fails image validation.

    Returns:
        HTMLResponse: Rendered ``check.html`` template containing the annotated image as base64.
    """
    validation_result = await img_validator.validate_image(file)
    if not validation_result["valid"]:
        raise HTTPException(status_code=400, detail=f"Image is not valid: {validation_result['errors']}")

    detection = await service.detect_dices(file, model)
    image = await service.draw_boxes(detection, file)
    image_base64 = await service.image_to_base64(image)
    return templates.TemplateResponse(request=request, name="check.html", context={"image_base64": image_base64})
