import base64
import io
import math
from collections import defaultdict

from fastapi import UploadFile
from PIL import Image, ImageFont
from PIL.ImageDraw import ImageDraw
from PIL.ImageFile import ImageFile
from ultralytics import YOLO

from src.schemas import BoundingBox, DiceDetection, DiceStatistics


async def convert_image(file: UploadFile) -> ImageFile:
    await file.seek(0)
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img.filename = file.filename
    return img


async def detect_dices(file: UploadFile, model: YOLO) -> list[DiceDetection]:
    """
    Detect dices on uploaded file using YOLO model.

    The file is converted to compability format. Detections are saved to list of ``DiceDetection``
    objects containing the detected dice class, name, bounding box and confidence score.

    Args:
        file (UploadFile): Uploaded image file.
        model (YOLO): Initialized YOLO model

    Returns:
        list[DiceDetection]: List of detected dices represented as ``DiceDetection``
    """
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


def _compute_mean(dices_nr_sum: dict) -> float:
    total = sum(dices_nr_sum.values())
    if total > 0:
        return sum(int(k) * v for k, v in dices_nr_sum.items()) / total
    return 0.0


def total_dices(detections: list[list[DiceDetection]]) -> int:
    return sum(len(d) for d in detections)


def _count_dices(detections: list[list[DiceDetection]]) -> dict[str, int]:
    """
    Count occurrences of detected dice grouped by dice name.

    Args:
        detections (list[list[DiceDetection]]): A nested list containing ``DiceDetection`` objects,
            typically grouped per processed image.

    Returns:
        dict[str, int]:  A dictionary where:
            - key: dice name,
            - value: number of occurrences of that dice.
    """
    out = defaultdict(int)
    for results in detections:
        for detection in results:
            out[detection.name] += 1
    return out


def compute_dice_statistics(
    detections: list[list[DiceDetection]],
) -> list[DiceStatistics]:
    """
    Compute variance, standard distribution and total dices for dice class.

    Args:
        detections (list[list[DiceDetection]]): A nested list containing ``DiceDetection`` objects,
            typically grouped per processed image.

    Returns:
        list[DiceStatistics]: List of ``DiceStatistics``
    """
    distributions = []
    dices_count = total_dices(detections)
    dice_number_count = _count_dices(detections)
    dices_count_mean = _compute_mean(dice_number_count)
    for dice_number, count in dice_number_count.items():
        variance = (count - dices_count_mean) ** 2 / dices_count
        std = math.sqrt(variance)
        distributions.append(DiceStatistics(dice_name=dice_number, variance=variance, std=std, count=count))
    return distributions


async def draw_boxes(detections: list[DiceDetection], file: UploadFile) -> Image.Image:
    """
    Draw bounding boxes and detection labels on uploaded image.

    The uploaded image is converted into an editable PIL image and annotated
    with bounding boxes, dice names, and confidence scores for each detected
    dice.

    Args:
        detections (list[DiceDetection]): List containing ``DiceDetection`` objects.
        file (UploadFile): Uploaded image file to annotate.

    Returns:
        Image.Image: PIL image instance with rendered bounding boxes and labels.
    """
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


async def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL image to to Base64 JPEG string.

    The image is converted to RGB mode and serialized as a JPEG image
    in memory before being encoded into a Base64 UTF-8 string.

    Args:
        image (Image.Image): PIL image instance to encode.

    Returns:
        str: Base64-encoded string representation of the JPEG image.
    """
    buffer = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
