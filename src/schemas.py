from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DiceDetection(BaseModel):
    name: str
    confidence: float
    box: BoundingBox
    class_id: int


class NormalDistribution(BaseModel):
    dice_name: str
    std: float
    count: int
