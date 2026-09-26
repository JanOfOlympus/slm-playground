"""
Tesseract OCR helper — locates the binary/language data and runs OCR on an
image from a path or a file-like object (e.g. a Streamlit UploadedFile).

    from ocr_utils import ocr_image
    text = ocr_image(uploaded_file)
"""

import os
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

TESSDATA_DIR = Path(__file__).parent / "tessdata"

_TESS_CANDIDATES = [
    os.environ.get("TESSERACT_CMD"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]
for _cand in _TESS_CANDIDATES:
    if _cand and os.path.isfile(_cand):
        pytesseract.pytesseract.tesseract_cmd = _cand
        break

if TESSDATA_DIR.is_dir():
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


def thai_available() -> bool:
    """False if tessdata/ (and its tha.traineddata) is missing — Tesseract then
    silently falls back to eng-only and approximates Thai glyphs as Latin
    gibberish instead of erroring, which is easy to miss."""
    try:
        return "tha" in pytesseract.get_languages(config="")
    except pytesseract.TesseractNotFoundError:
        return False


def ocr_image(image_source, lang: str = "tha+eng") -> str:
    """image_source: a path (str/Path) or a file-like object (bytes, seekable)."""
    img = Image.open(image_source)
    img = ImageOps.exif_transpose(img)
    img = ImageOps.grayscale(img)
    config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    try:
        return pytesseract.image_to_string(img, lang=lang, config=config)
    except pytesseract.TesseractNotFoundError:
        return ""
