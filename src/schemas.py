from pydantic import BaseModel


class BoundingBox(BaseModel):
    """
    Bounding boxes coordinates describing a location of detected object.

    Attributes:
        x1(float): Left coordinate of the box.
        y1(float): Top coordinate of the box.
        x2(float): Right coordinate of the box.
        y2(float): Bottom coordinate of the box.
    """

    x1: float
    y1: float
    x2: float
    y2: float


class DiceDetection(BaseModel):
    """
    Representation of a single detected object.

    Attributes:
        name(str): Predicted dice class name.
        confidence(float): Detection confidence score returned by the model.
        box(BoundingBox): Bounding box coordinates of the detected dice.
        class_id(int): Numeric class identifier assigned by the model.
    """

    name: str
    confidence: float
    box: BoundingBox
    class_id: int


class DiceStatistics(BaseModel):
    """
    Representation of a dice class statistics.

    Attributes:
        dice_name(str): Name of analyzed dice class.
        variance(float): Variance of the dice occurrence distribution.
        std(float): Standard deviation of the dice occurrence distribution.
        count: Total number of detected dices for this class dices.
    """

    dice_name: str
    variance: float
    std: float
    count: int
