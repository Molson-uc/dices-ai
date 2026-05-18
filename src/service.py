import base64
import io
import math
from collections import defaultdict

from fastapi import UploadFile
from PIL import Image, ImageFont
from PIL.ImageDraw import ImageDraw
from PIL.ImageFile import ImageFile
from ultralytics import YOLO

from schemas import BoundingBox, DiceDetection, NormalDistribution


async def convert_image(file: UploadFile) -> ImageFile:
    await file.seek(0)
    img_bytes = await file.read()
    return Image.open(io.BytesIO(img_bytes))


async def detect_dices(file: UploadFile, model: YOLO) -> list[DiceDetection]:
    image = await convert_image(file)
    results = model(image)

    file_detections: list[DiceDetection] = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            file_detections.append(
                DiceDetection(
                    name=result.names[cls_id],
                    class_id=cls_id,
                    confidence=float(box.conf[0]),
                    box=BoundingBox(
                        x1=float(box.xyxy[0][0]),
                        y1=float(box.xyxy[0][1]),
                        x2=float(box.xyxy[0][2]),
                        y2=float(box.xyxy[0][3]),
                    ),
                )
            )
    return file_detections


def count_mean(dices_nr_sum: dict) -> float:
    total = sum(dices_nr_sum.values())
    if total > 0:
        return sum(int(k) * v for k, v in dices_nr_sum.items()) / total
    return 0.0


def count_dices(detections: list[list[DiceDetection]]) -> int:
    return sum(len(d) for d in detections)


def count_same_dices(detections: list[list[DiceDetection]]) -> dict[str, int]:
    out = defaultdict(int)
    for results in detections:
        for detection in results:
            out[detection.name] += 1
    return out


def normal_distribution(
    detections: list[list[DiceDetection]],
) -> list[NormalDistribution]:
    distributions = []
    dices_count = count_dices(detections)
    dice_number_count = count_same_dices(detections)
    dices_count_mean = count_mean(dice_number_count)
    for dice_number, count in dice_number_count.items():
        variance = (count - dices_count_mean) ** 2 / dices_count
        std = math.sqrt(variance)
        distributions.append(NormalDistribution(dice_name=dice_number, std=std, count=count))
    return distributions


async def draw_boxes(detections: list[DiceDetection], file: UploadFile) -> Image.Image:
    image = await convert_image(file)
    draw = ImageDraw(image)

    for detection in detections:
        box = detection.box
        coordinates = [box.x1, box.y1, box.x2, box.y2]
        font = ImageFont.load_default(size=20.0)
        draw.rectangle(coordinates, outline="red", width=4)
        text = f"{detection.name} - {detection.confidence:.2f}"
        text_bbox = draw.textbbox((box.x1, box.y1), text, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        text_w = text_bbox[2] - text_bbox[0]
        draw.rectangle((box.x1 + 4, box.y1 + 4, box.x1 + text_w, box.y1 + text_h + 7), fill="green")
        draw.text((box.x1, box.y1), text, fill="white", font=font)
    return image


async def image_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
