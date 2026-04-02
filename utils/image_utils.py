"""
utils/image_utils.py — Image loading, validation, resizing, and base64 encoding.
"""

import base64
import io
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MAX_FILE_SIZE_MB = 15


def validate_image_path(path: str) -> tuple[bool, str]:
    """
    Check that the path points to a valid, supported image file.
    Returns (is_valid, error_message). error_message is empty string on success.
    """
    p = Path(path)

    if not p.exists():
        return False, f"File not found: {path}"

    if not p.is_file():
        return False, f"Path is not a file: {path}"

    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"Unsupported format '{p.suffix}'. Supported: {supported}"

    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File is {size_mb:.1f} MB, maximum allowed is {MAX_FILE_SIZE_MB} MB"

    # Try opening the file header to confirm it is a real image
    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as e:
        return False, f"Cannot read image file: {e}"

    return True, ""


def load_and_encode_image(path: str, max_dim: int = 896) -> tuple[str, str]:
    """
    Load an image, resize if necessary, and return:
      - base64-encoded JPEG string
      - full data URI: "data:image/jpeg;base64,{b64}"

    Args:
        path:    Path to the image file.
        max_dim: Maximum width or height in pixels. Image is resized proportionally
                 if either dimension exceeds this value.
    """
    with Image.open(path) as img:
        # Normalize to RGB (handles RGBA, palette, grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if the image is too large (preserves aspect ratio)
        w, h = img.size
        if max(w, h) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        # Encode to JPEG in memory
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
        buffer.seek(0)
        raw_bytes = buffer.read()

    b64_string = base64.b64encode(raw_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_string}"
    return b64_string, data_uri


def get_image_dimensions(path: str) -> tuple[int, int]:
    """Return (width, height) of an image without loading the full file."""
    with Image.open(path) as img:
        return img.size
