import os
from pathlib import Path

import pytesseract
import streamlit as st
from PIL import Image, ImageOps

# --- Config ---
IMAGE_DIR = Path(__file__).parent / "images"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# Project-local language data (holds tha/eng traineddata so no admin install is
# needed). Picked up via TESSDATA_PREFIX below when the folder exists.
TESSDATA_DIR = Path(__file__).parent / "tessdata"

# Locate the Tesseract binary: $TESSERACT_CMD wins, otherwise fall back to PATH
# and the standard Windows install locations so no env var is needed.
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

# Use the project-local language data if it's there. Done via TESSDATA_PREFIX
# rather than --tessdata-dir: pytesseract can't pass a quoted path on Windows,
# so a --tessdata-dir with spaces in it would break.
if TESSDATA_DIR.is_dir():
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


st.title("Image OCR Playground")


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def ocr_image(path: Path, lang: str) -> str:
    img = Image.open(path)
    # Respect EXIF orientation and drop alpha so Tesseract gets a clean grayscale.
    img = ImageOps.exif_transpose(img)
    img = ImageOps.grayscale(img)
    # Thai text has no inter-word spaces; keep the ones Tesseract does find.
    config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(img, lang=lang, config=config).strip()


IMAGE_DIR.mkdir(exist_ok=True)
images = list_images(IMAGE_DIR)

st.write(f"Reading images from `{IMAGE_DIR}`")

if not images:
    st.warning(
        f"No images found in `{IMAGE_DIR}`. "
        f"Drop some files in there ({', '.join(sorted(IMAGE_EXTS))}) and rerun."
    )
    st.stop()

try:
    installed = pytesseract.get_languages(config="")
except pytesseract.TesseractNotFoundError:
    st.error(
        "Tesseract binary not found. Install it and either add it to PATH or set "
        "the TESSERACT_CMD environment variable to its full path."
    )
    st.stop()

default_langs = [l for l in ("tha", "eng") if l in installed] or installed[:1]
selected = st.multiselect(
    "Tesseract language(s)",
    options=sorted(installed),
    default=default_langs,
    help="Pick 'tha' for Thai. Combine with 'eng' for mixed Thai/English documents.",
)
lang = "+".join(selected) if selected else "eng"

if "tha" not in installed:
    st.warning(
        f"Thai language data ('tha') not found. Put `tha.traineddata` in `{TESSDATA_DIR}` "
        "(alongside `eng.traineddata`). Download: "
        "https://github.com/tesseract-ocr/tessdata/raw/main/tha.traineddata"
    )

if st.button("Run OCR on all images"):
    for path in images:
        st.subheader(path.name)
        col_img, col_txt = st.columns(2)
        with col_img:
            st.image(str(path), use_container_width=True)
        with col_txt:
            with st.spinner(f"OCR: {path.name}"):
                try:
                    text = ocr_image(path, lang)
                except pytesseract.TesseractNotFoundError:
                    st.error(
                        "Tesseract binary not found. Install it and either add it to "
                        "PATH or set the TESSERACT_CMD environment variable to its full path."
                    )
                    st.stop()
            if text:
                st.code(text, height=300)
                st.download_button(
                    "Download .txt",
                    data=text,
                    file_name=f"{path.stem}.txt",
                    key=f"dl_{path.name}",
                )
            else:
                st.info("No text detected.")
