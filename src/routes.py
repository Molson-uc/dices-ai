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
templates = Jinja2Templates(directory="./templates/")
img_validator = ImageValidator()


def get_model() -> YOLO:
    model = YOLO(model=settings.model_path)
    model.eval()
    return model


def get_mock_model() -> YOLO:
    return YOLO()


@router.get("/")
async def home_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="home.html")


@router.post("/dices/stats")
async def stats(
    request: Request,
    files: list[UploadFile],
    model: Annotated[YOLO, Depends(get_model)],
) -> HTMLResponse:
    for file in files:
        result = await img_validator.validate_image(file)
        if not result["valid"]:
            raise HTTPException(status_code=400, detail=f"Image is not valid: {result['errors']}")

    detections = await asyncio.gather(*(service.detect_dices(file, model) for file in files))
    total_dices = service.count_dices(detections)
    return templates.TemplateResponse(
        request=request, name="stats.html", context={"total_images": len(files), "total_dices": total_dices}
    )


@router.post("/dices/check")
async def check(request: Request, file: UploadFile, model: Annotated[YOLO, Depends(get_model)]) -> HTMLResponse:
    validation_result = await img_validator.validate_image(file)
    if not validation_result["valid"]:
        raise HTTPException(status_code=400, detail=f"Image is not valid: {validation_result['errors']}")

    detection = await service.detect_dices(file, model)
    image = await service.draw_boxes(detection, file)
    image_base64 = await service.image_to_base64(image)
    return templates.TemplateResponse(request=request, name="check.html", context={"image_base64": image_base64})
