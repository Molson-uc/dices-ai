from pathlib import Path

from fastapi import UploadFile


class ImageValidator:
    def __init__(self, max_size: int = 10 * 1024 * 1024) -> None:
        self.max_size = max_size
        self.allowed_extensions = {".jpeg", ".jpg", ".png"}

    async def validate_image(self, image: UploadFile) -> dict:
        """Check if image is valid."""
        result = {"valid": True, "errors": []}

        if not image.filename or image.filename.strip() == "":
            result["valid"] = False
            result["errors"].append("No image selected")
            return result

        img_extension = Path(image.filename).suffix.lower()

        if img_extension not in self.allowed_extensions:
            result["valid"] = False
            result["errors"].append("No valid extension")
            return result

        return result
