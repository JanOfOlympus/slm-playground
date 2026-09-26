"""
Slip field extractor — pre-extracted text.

Reads clean OCR text (e.g. from Apple Live Text) saved as .txt files in
./transactions, and parses each into structured JSON: who paid, who received,
amount, fee, date, transaction ref.

A local text LLM (qwen2.5:3b via Ollama) does the structural parse — the OCR
lines are often out of reading order (labels grouped together, values following
as a separate block), which a model handles from context far better than line
pairing. A deterministic regex pass then overrides amount / fee / date / type.

Nothing leaves the machine.

    streamlit run app_text.py

Needs: `ollama serve` running + `ollama pull qwen2.5:3b`
"""

import json
import re
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

# --- Config ---
OLLAMA_URL = "http://localhost:11434"
PARSE_MODEL = "qwen2.5:3b"

TXT_DIR = Path(__file__).parent / "transactions"
JSON_DIR = Path(__file__).parent / "transactions_json"


PROMPT = """You are given OCR text from a Thai bank e-slip (a transfer, a bill \
payment, or a QR/PromptPay payment). The lines are often NOT in reading order: \
labels that end with ":" are frequently grouped together, with their values \
following as a separate block in the same order.

Rules:
- The PAYER (from) is the person/account named near the TOP of the slip.
- The PAYEE (to) is the shop, biller, government agency, or person named AFTER
  the payer's masked account number.
- Label -> field mapping (Thai):
  - "เลขที่รายการ" / "หมายเลขอ้างอิง" / "รหัสอ้างอิง" -> transaction_ref
  - "จำนวน" / "จำนวนเงิน" / "จานวนเงิน" -> amount
  - "ค่าธรรมเนียม" -> fee
  - "วันที่ทำรายการ" -> datetime
- amount and fee are the values written with "บาท"; fee is usually "0.00".
- Keep Thai text in Thai. Do not guess a value that is not present.
- "category": category of the PAYEE ("to"). ONLY assign a specific value when the
  payee is unmistakable from its name (a nationally known chain, bank, utility, or
  government agency). For a small/local shop, an unfamiliar name, a holding
  company, or any doubt -> use "other". It is better to answer "other" than to
  guess. Values: food, groceries, shopping, transport, utilities, health,
  entertainment, services, education, government, transfer, other.

Reply with ONLY this JSON object and nothing else:
{
  "slip_type": "transfer" | "bill_payment" | "qr_payment" | "unknown",
  "datetime": string or null,
  "from": {"name": string or null, "bank": string or null, "account": string or null},
  "to":   {"name": string or null, "bank": string or null, "account": string or null},
  "category": "food" | "groceries" | "shopping" | "transport" | "utilities" | "health" | "entertainment" | "services" | "education" | "government" | "transfer" | "other",
  "amount": string or null,
  "fee": string or null,
  "transaction_ref": string or null
}

OCR text:
<<<
{text}
>>>"""


def list_txt(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


_BAHT_RE = re.compile(r"([\d,]+\.\d{2})\s*บาท")
_DATETIME_RE = re.compile(
    r"\d{1,2}\s+[ก-๙.]{2,6}\s+\d{2,4}\s*[-–]?\s*\d{1,2}[:.]\d{2}(?:\s*น\.?)?"
)
_TITLE_RE = re.compile(r"^(?:น\.ส\.?|นาย|นาง|ด\.ช\.|ด\.ญ\.)\s*\S")
_ACCT_RE = re.compile(r"[xX]{2,4}-[xX0-9]-[xX0-9]{2,}-[xX0-9]")
_LATIN_SUFFIX_RE = re.compile(r"\(?[A-Za-z.\-/ ]+\)?")

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
    Parse a Buddhist-era Thai datetime like "8 ก.ย. 69 20:15 น." or
    "07 ก.ย. 2569 - 18:21" into ISO "YYYY-MM-DD HH:MM" (Gregorian year).
    Returns None (caller keeps the original string) if it doesn't match —
    never guesses.
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

CATEGORIES = (
    "food", "groceries", "shopping", "transport", "utilities", "health",
    "entertainment", "services", "education", "government", "transfer", "other",
)

# Only the offline-certain billers stay hardcoded — no point paying an API to
# recognise the electricity authority every month. Everything else goes:
# local SLM first, then (optionally) a third-party AI for whatever it can't place.
_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("utilities", ("การไฟฟ้า", "การประปา", "ประปา", "กปน", "กปภ", "กฟน", "กฟภ",
                   "ค่าไฟ", "ค่าน้ำ", "ค่าโทรศัพท์")),
    ("government", ("กรมสรรพากร", "สรรพากร", "กรมการปกครอง", "เทศบาล",
                    "กรมที่ดิน", "ขนส่งทางบก")),
]


def _category_from_rules(payee: str | None, text: str) -> str | None:
    hay = f"{payee or ''}\n{text}".lower()
    for category, needles in _CATEGORY_RULES:
        if any(n.lower() in hay for n in needles):
            return category
    return None


def _layout_names(text: str) -> tuple[str | None, str | None]:
    """
    On these slips the two parties are positional, not labelled:
    - payer  = the first line with a Thai personal title (น.ส./นาย/นาง/...)
    - payee  = the first Thai line after the payer's masked account number
               (skipping "K+" and label/number noise)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    from_name = next((ln for ln in lines if _TITLE_RE.match(ln)), None)

    to_name = None
    acct_idx = next((i for i, ln in enumerate(lines) if _ACCT_RE.search(ln)), None)
    if acct_idx is not None:
        rest = lines[acct_idx + 1:]
        for j, ln in enumerate(rest):
            if ln == "K+" or ln.endswith(":") or ln[0].isdigit():
                continue
            if not re.search(r"[ก-๙]", ln):
                continue
            to_name = ln
            nxt = rest[j + 1] if j + 1 < len(rest) else ""
            if nxt and _LATIN_SUFFIX_RE.fullmatch(nxt):
                to_name = f"{ln} {nxt}"
            break

    return from_name, to_name


def quick_fields(text: str) -> dict:
    """Deterministic extraction of the fields that don't need a model."""
    out: dict = {}

    if "ชำระเงินสำเร็จ" in text:
        out["slip_type"] = "qr_payment"
    elif "จ่ายบิลสำเร็จ" in text:
        out["slip_type"] = "bill_payment"
    elif "โอนเงินสำเร็จ" in text:
        out["slip_type"] = "transfer"

    baht = [m.replace(",", "") for m in _BAHT_RE.findall(text)]
    if len(baht) >= 2:
        out["amount"], out["fee"] = baht[0], baht[1]
    elif len(baht) == 1:
        out["amount"] = baht[0]

    dt = _DATETIME_RE.search(text)
    if dt:
        out["datetime"] = dt.group(0).strip()

    from_name, to_name = _layout_names(text)
    if from_name:
        out["_from_name"] = from_name
    if to_name:
        out["_to_name"] = to_name

    if out.get("slip_type") == "transfer" and from_name and to_name:
        out["_category"] = "transfer"  # person-to-person, no merchant
    rule_cat = _category_from_rules(to_name, text)
    if rule_cat:
        out["_category"] = rule_cat

    return out


def parse_llm(text: str, model: str) -> dict:
    """Ask the local text model to structure the slip. {'_error': ...} on model trouble."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": PROMPT.replace("{text}", text),
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
    except requests.ConnectionError:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_URL}. Start it with `ollama serve`."
        )
    except requests.Timeout:
        return {"_error": "Model timed out (120s)."}

    if resp.status_code == 404:
        return {"_error": f"Model '{model}' not found. Run: ollama pull {model}"}
    if not resp.ok:
        return {"_error": f"Ollama error {resp.status_code}: {resp.text[:200]}"}

    raw = resp.json().get("response", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_error": f"Model did not return JSON: {raw[:200]}"}


def extract_fields(text: str, model: str) -> dict:
    """LLM parse, with the deterministic fields (type, amount, fee, date, the two
    party names) overridden by the regex/layout pass. `category` is the local
    model's guess corrected by the rule table; anything it can't place stays
    "other" for step 2 (resolve_categories.py) to handle."""
    result = parse_llm(text, model)
    if "_error" in result:
        return result

    qf = quick_fields(text)
    from_name = qf.pop("_from_name", None)
    to_name = qf.pop("_to_name", None)
    rule_category = qf.pop("_category", None)
    for key, value in qf.items():
        result[key] = value
    if rule_category:
        result["category"] = rule_category
    result.setdefault("category", "other")

    for side, name in (("from", from_name), ("to", to_name)):
        if name:
            if not isinstance(result.get(side), dict):
                result[side] = {"name": None, "bank": None, "account": None}
            result[side]["name"] = name

    if result.get("datetime"):
        result["datetime"] = normalize_thai_datetime(result["datetime"]) or result["datetime"]

    return result


# --- UI ---
st.title("Slip Field Extractor — pre-extracted text")
st.caption(
    f"Reads `.txt` files from `{TXT_DIR}` (clean OCR, e.g. Apple Live Text). "
    "Needs Ollama running and `ollama pull qwen2.5:3b`. Everything runs locally."
)

model = st.text_input("Ollama text model", value=PARSE_MODEL)

files = list_txt(TXT_DIR)
if not files:
    st.warning(f"No .txt files in `{TXT_DIR}`. Add some slip texts and rerun.")
    st.stop()

st.caption(f"Parsed JSON is also written to `{JSON_DIR}`.")

if st.button("Parse all transactions"):
    JSON_DIR.mkdir(exist_ok=True)
    for path in files:
        st.subheader(path.name)
        text = path.read_text(encoding="utf-8")
        col_txt, col_out = st.columns(2)
        with col_txt:
            st.code(text, language=None)
        with col_out:
            with st.spinner(f"Parsing {path.name} with {model}…"):
                try:
                    result = extract_fields(text, model)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

            if "_error" in result:
                st.error(result["_error"])
                continue

            payload = json.dumps(result, ensure_ascii=False, indent=2)
            out_path = JSON_DIR / f"{path.stem}.json"
            out_path.write_text(payload, encoding="utf-8")

            st.json(result)
            st.caption(f"Saved to `{out_path}`")
            st.download_button(
                "Download .json",
                data=payload,
                file_name=f"{path.stem}.json",
                key=f"js_{path.name}",
            )
