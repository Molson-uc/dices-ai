from uuid import uuid4

from PIL.ImageFile import ImageFile

from src.config import settings


def upload_file(image: ImageFile):
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    name = image.filename or f"{uuid4().hex}"
    image.save(settings.uploads_path / f"{name}")
