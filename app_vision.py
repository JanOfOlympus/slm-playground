"""
Slip field extractor — local vision model.

Reads slip images from ./images, sends each to a local Ollama vision model to
pull structured fields (who paid, who received, amount, fee, date, ref), and
overrides the numeric fields with a deterministic regex pass over Tesseract OCR
text so amounts are never at the model's mercy.

Nothing leaves the machine.

    streamlit run app_vision.py

Needs: `ollama serve` running + `ollama pull qwen2.5vl:7b`
"""

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytesseract
import requests
import streamlit as st
from PIL import Image, ImageOps

# --- Config ---
OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = "qwen2.5vl:7b"  # good Thai; alt: "minicpm-v". NOT llama3.2-vision.

IMAGE_DIR = Path(__file__).parent / "images"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TESSDATA_DIR = Path(__file__).parent / "tessdata"

# --- Locate Tesseract (only used for the deterministic amount/fee/date pass) ---
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


PROMPT = """This image is a Thai bank e-slip (a money transfer, a bill payment, \
or a QR/PromptPay payment).

Read it and return the transaction details. Layout rules:
- The PAYER (sender) is the account holder shown near the top of the slip.
- The PAYEE (recipient) is the shop, biller, or person shown BELOW the payer's
  masked account number.
- Ignore garbled single characters and icon glyphs.

Reply with ONLY this JSON object and nothing else:
{
  "slip_type": "transfer" | "bill_payment" | "qr_payment" | "unknown",
  "datetime": string or null,
  "from": {"name": string or null, "bank": string or null, "account": string or null},
  "to":   {"name": string or null, "bank": string or null, "account": string or null},
  "amount": string or null,
  "fee": string or null,
  "transaction_ref": string or null
}
Keep Thai text in Thai. Use null when a value is not visible. Do not guess."""


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def ocr_text(path: Path) -> str:
    """Tesseract text for the regex cross-check. Empty string if Tesseract is absent."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = ImageOps.grayscale(img)
        config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
        return pytesseract.image_to_string(img, lang="tha+eng", config=config)
    except pytesseract.TesseractNotFoundError:
        return ""


_AMOUNT_RE = re.compile(r"([\d,]+\.\d{2})")

_THAI_MONTHS = {
    "มกราคม": 1, "ม.ค.": 1, "ม.ค": 1,
    "กุมภาพันธ์": 2, "ก.พ.": 2, "ก.พ": 2,
    "มีนาคม": 3, "มี.ค.": 3, "มี.ค": 3,
    "เมษายน": 4, "เม.ย.": 4, "เม.ย": 4,
    "พฤษภาคม": 5, "พ.ค.": 5, "พ.ค": 5,
    "มิถุนายน": 6, "มิ.ย.": 6, "มิ.ย": 6,
    "กรกฎาคม": 7, "ก.ค.": 7, "ก.ค": 7,
    "สิงหาคม": 8, "ส.ค.": 8, "ส.ค": 8,
    "กันยายน": 9, "ก.ย.": 9, "ก.ย": 9,
    "ตุลาคม": 10, "ต.ค.": 10, "ต.ค": 10,
    "พฤศจิกายน": 11, "พ.ย.": 11, "พ.ย": 11,
    "ธันวาคม": 12, "ธ.ค.": 12, "ธ.ค": 12,
}
_THAI_DATETIME_RE = re.compile(
    r"(\d{1,2})\s+([ก-๙.]{2,10})\s+(\d{2,4})\s*[-–]?\s*(\d{1,2})[:.](\d{2})"
)


def normalize_thai_datetime(raw: str | None) -> str | None:
    """
    Parse a Buddhist-era Thai datetime like "8 ก.ย. 69 20:15 น." into ISO
    "YYYY-MM-DD HH:MM" (Gregorian year). Returns None (caller keeps the
    original) if it doesn't match — never guesses.
    """
    if not raw:
        return None
    m = _THAI_DATETIME_RE.search(raw)
    if not m:
        return None
    day, month_th, year_be, hour, minute = m.groups()
    month = _THAI_MONTHS.get(month_th)
    if not month:
        return None
    year_be = int(year_be)
    if year_be < 100:
        year_be += 2500  # 2-digit BE year, e.g. "69" -> 2569
    try:
        dt = datetime(year_be - 543, month, int(day), int(hour), int(minute))
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d %H:%M")


def _amount_near(lines: list[str], label_terms: tuple[str, ...]) -> str | None:
    for i, line in enumerate(lines):
        if any(term in line for term in label_terms):
            for probe in lines[i:i + 3]:
                m = _AMOUNT_RE.search(probe)
                if m:
                    return m.group(1).replace(",", "")
    return None


def quick_fields(text: str) -> dict:
    """
    Deterministic, Thai-label-anchored extraction of amount/fee only.
    datetime is NOT extracted here: Tesseract reliably mangles the Thai month
    abbreviation (e.g. "ก.ย." misread as a stray digit), which would silently
    turn "September" into "February" if trusted. The vision model reads the
    month from the image directly, so its answer is kept and just normalized.
    """
    if not text.strip():
        return {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: dict = {}

    amount = _amount_near(lines, ("จำนวน", "จํานวน"))
    if amount:
        out["amount"] = amount

    fee = _amount_near(lines, ("ค่าธรรมเนียม",))
    if fee:
        out["fee"] = fee

    return out


def extract_vision(path: Path, model: str) -> dict:
    """Ask the local vision model for the full field set. Returns {'_error': ...} on model trouble."""
    b64 = base64.b64encode(path.read_bytes()).decode()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": PROMPT,
                "images": [b64],
                "stream": False,
                "format": "json",
            },
            timeout=3600,
        )
    except requests.ConnectionError:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_URL}. Start it with `ollama serve`."
        )
    except requests.Timeout:
        return {"_error": "Model timed out (180s). Try a smaller model or a smaller image."}

    if resp.status_code == 404:
        return {"_error": f"Model '{model}' not found. Run: ollama pull {model}"}
    if not resp.ok:
        return {"_error": f"Ollama error {resp.status_code}: {resp.text[:200]}"}

    raw = resp.json().get("response", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_error": f"Model did not return JSON: {raw[:200]}"}


def extract_fields(path: Path, model: str) -> dict:
    """Vision result, with amount/fee overridden by the deterministic OCR pass
    (label-anchored, reliable) and datetime normalized to ISO from the model's
    own reading (Tesseract's is not trustworthy for the Thai month — see
    quick_fields)."""
    result = extract_vision(path, model)
    text = ocr_text(path)
    if "_error" not in result:
        for key, value in quick_fields(text).items():
            result[key] = value
        if result.get("datetime"):
            result["datetime"] = normalize_thai_datetime(result["datetime"]) or result["datetime"]
    result["_ocr"] = text
    return result


# --- UI ---
st.title("Slip Field Extractor — local vision")
st.caption(
    f"Reads `{IMAGE_DIR}`. Needs Ollama running (`ollama serve`) and the model "
    "pulled (`ollama pull qwen2.5vl:7b`). Everything runs locally."
)

model = st.text_input("Ollama vision model", value=VISION_MODEL)

images = list_images(IMAGE_DIR)
if not images:
    st.warning(
        f"No images in `{IMAGE_DIR}` ({', '.join(sorted(IMAGE_EXTS))}). "
        "Add some and rerun."
    )
    st.stop()

if st.button("Extract from all slips"):
    for path in images:
        st.subheader(path.name)
        col_img, col_out = st.columns(2)
        with col_img:
            st.image(str(path), use_container_width=True)
        with col_out:
            with st.spinner(f"Reading {path.name} with {model}…"):
                try:
                    result = extract_fields(path, model)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

            if "_error" in result:
                st.error(result["_error"])
                continue

            clean = {k: v for k, v in result.items() if not k.startswith("_")}
            st.json(clean)
            st.download_button(
                "Download .json",
                data=json.dumps(clean, ensure_ascii=False, indent=2),
                file_name=f"{path.stem}.json",
                key=f"js_{path.name}",
            )
            with st.expander("OCR text (debug)"):
                st.code(result.get("_ocr", "") or "(Tesseract not available)")
