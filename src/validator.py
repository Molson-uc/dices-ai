from pathlib import Path

from fastapi import UploadFile


class ImageValidator:
    """
    Validate uploaded image files against basic constraints.

    Checks performed by ``validate_image`` include presence of a filename and an allowed
    file extension. Configuration (max file size, allowed extensions) is set at construction
    time so the same validator can be reused across requests.

    Attributes:
        max_size (int): Maximum allowed file size in bytes.
        allowed_extensions (set[str]): File extensions accepted by the validator,
            stored in lowercase and including the leading dot (e.g. ``".png"``).
    """

    def __init__(self, max_size: int = 10 * 1024 * 1024) -> None:
        """
        Initialize the validator with size and extension constraints.

        Args:
            max_size (int): Maximum allowed file size in bytes. Defaults to 10 MiB.
        """
        self.max_size = max_size
        self.allowed_extensions = {".jpeg", ".jpg", ".png"}

    async def validate_image(self, image: UploadFile) -> dict:
        """
        Verify that an uploaded image has a filename and an allowed extension.

        The check is short-circuited: as soon as a problem is found, the result is
        returned without further checks.

        Args:
            image (UploadFile): File to validate.

        Returns:
            dict: A dictionary with two keys:
                - ``valid`` (bool): ``True`` if the image passes all checks, ``False`` otherwise.
                - ``errors`` (list[str]): Human-readable error messages; empty when ``valid`` is ``True``.
        """
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
