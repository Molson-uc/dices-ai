from collections.abc import Iterator
from http import HTTPStatus
from pathlib import Path
from typing import IO

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from ultralytics import YOLO

from src import routes
from src.main import app
from src.routes import get_model
from src.schemas import BoundingBox, DiceDetection

FileUpload = list[tuple[str, tuple[str, IO[bytes], str]]]

app.dependency_overrides[get_model] = lambda: None
test_client = TestClient(app)


@pytest.fixture
def mock_detections() -> list[DiceDetection]:
    return [
        DiceDetection(
            name="6",
            class_id=5,
            confidence=0.92594,
            box=BoundingBox(x1=382.27, y1=191.61, x2=490.80, y2=301.74),
        ),
        DiceDetection(
            name="5",
            class_id=4,
            confidence=0.93468,
            box=BoundingBox(x1=238.62, y1=192.78, x2=344.88, y2=299.85),
        ),
        DiceDetection(
            name="3",
            class_id=2,
            confidence=0.90327,
            box=BoundingBox(x1=383.42, y1=49.84, x2=489.84, y2=157.16),
        ),
        DiceDetection(
            name="1",
            class_id=0,
            confidence=0.90162,
            box=BoundingBox(x1=93.41, y1=49.05, x2=199.07, y2=156.33),
        ),
        DiceDetection(
            name="4",
            class_id=3,
            confidence=0.89507,
            box=BoundingBox(x1=94.44, y1=194.70, x2=198.86, y2=298.65),
        ),
        DiceDetection(
            name="2",
            class_id=1,
            confidence=0.89485,
            box=BoundingBox(x1=237.92, y1=49.57, x2=344.96, y2=156.39),
        ),
    ]


@pytest.fixture
def image_file() -> Iterator[FileUpload]:
    path = Path("./tests/dices_photos/dices_all.jpg")
    with path.open("rb") as f:
        yield [("file", ("dices_all.jpg", f, "image/jpeg"))]


@pytest.fixture
def image_files() -> Iterator[FileUpload]:
    path = Path("./tests/dices_photos/dices_all.jpg")
    with path.open("rb") as f1:
        yield [("files", ("dices_all.jpg", f1, "image/jpeg"))]


def test_check_dices_returns_rendered_image(
    monkeypatch: pytest.MonkeyPatch,
    image_file: FileUpload,
    mock_detections: list[DiceDetection],
) -> None:
    async def fake_detect(_file: UploadFile, _model: YOLO) -> list[DiceDetection]:
        return mock_detections

    monkeypatch.setattr(routes.service, "detect_dices", fake_detect)

    response = test_client.post("/dices/check", files=image_file)

    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"].startswith("text/html")
    assert "data:image/jpeg;base64," in response.text


def test_check_dices_rejects_invalid_extension() -> None:
    response = test_client.post(
        "/dices/check",
        files=[("file", ("bad.txt", b"not an image", "text/plain"))],
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "not valid" in response.text.lower()


def test_normal_distribution_returns_rendered_distribution(
    monkeypatch: pytest.MonkeyPatch, image_files: FileUpload, mock_detections: list[DiceDetection]
):
    async def fake_detect(_file: UploadFile, _model: YOLO) -> list[DiceDetection]:
        return mock_detections

    monkeypatch.setattr(routes.service, "detect_dices", fake_detect)

    response = test_client.post("/dices/distribution", files=image_files)

    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"].startswith("text/html")
    for name in ("1", "2", "3", "4", "5", "6"):
        assert name in response.text


def test_normal_distribution_rejects_invalid_extension() -> None:
    response = test_client.post(
        "/dices/distribution",
        files=[("files", ("bad.txt", b"not an image", "text/plain"))],
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "not valid" in response.text.lower()
