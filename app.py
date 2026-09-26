"""
Spending Dashboard + Upload Slip

Two pages:
  - Dashboard: reads every JSON file in ./transactions_json (written by
    app_text.py / app_vision.py, categories filled in by resolve_categories.py)
    and lets you drill into spend by date range, time, and category/payee.
  - Upload Slip: upload a slip image, OCR it (Tesseract, via ocr_utils.py),
    edit the text if needed, parse it with the same local-LLM pipeline as
    app_text.py (slip_parser.py), and save into the same folders the
    dashboard reads from.

Nothing leaves the machine except the Ollama call for parsing.

    streamlit run app.py
"""

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ocr_utils import ocr_image, thai_available
from slip_parser import PARSE_MODEL, extract_fields

IMAGE_DIR = Path(__file__).parent / "images"
TXT_DIR = Path(__file__).parent / "transactions"
JSON_DIR = Path(__file__).parent / "transactions_json"
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="Spending Dashboard", layout="wide")


# ============================== Dashboard ================================

def load_transactions(folder: Path) -> pd.DataFrame:
    """One row per slip JSON. datetime/amount that don't parse become NaT/NaN."""
    rows = []
    for path in sorted(folder.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        dt = pd.to_datetime(d.get("datetime"), format="%Y-%m-%d %H:%M", errors="coerce")
        rows.append({
            "file": path.name,
            "datetime": dt,
            "date": dt.date() if pd.notna(dt) else None,
            "hour": dt.hour if pd.notna(dt) else None,
            "weekday": dt.strftime("%a") if pd.notna(dt) else None,
            "amount": pd.to_numeric(d.get("amount"), errors="coerce"),
            "fee": pd.to_numeric(d.get("fee"), errors="coerce"),
            "category": d.get("category") or "other",
            "slip_type": d.get("slip_type") or "unknown",
            "payee": (d.get("to") or {}).get("name") or "(unknown)",
            "payer": (d.get("from") or {}).get("name") or "(unknown)",
            "transaction_ref": d.get("transaction_ref"),
        })
    return pd.DataFrame(rows)


def render_dashboard():
    st.title("💸 Spending Dashboard")
    st.caption(f"Reads every slip JSON from `{JSON_DIR}`.")

    df = load_transactions(JSON_DIR)

    if df.empty:
        st.warning(f"No JSON files in `{JSON_DIR}`. Use the Upload Slip page, or run app_text.py / app_vision.py.")
        return

    no_amount = int(df["amount"].isna().sum())
    no_date = int(df["date"].isna().sum())
    df = df.dropna(subset=["date", "amount"]).copy()

    if df.empty:
        st.warning("No transactions have both a parseable date and amount yet.")
        return

    # --- Filters ---
    min_date, max_date = df["date"].min(), df["date"].max()
    with st.sidebar:
        st.header("Filters")
        date_range = st.date_input(
            "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )

        categories = sorted(df["category"].unique())
        selected_categories = st.multiselect("Category", categories, default=categories)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    mask = (
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
        & (df["category"].isin(selected_categories))
    )
    filtered = df[mask].sort_values("datetime", ascending=False)

    skip_notes = []
    if no_date:
        skip_notes.append(f"{no_date} slip(s) skipped — no parseable date")
    if no_amount:
        skip_notes.append(f"{no_amount} slip(s) skipped — no parseable amount")
    if skip_notes:
        st.caption("⚠️ " + "; ".join(skip_notes))

    if filtered.empty:
        st.info("No transactions match the current filters.")
        return

    # --- KPIs ---
    c1, c2 = st.columns(2)
    c1.metric("Total spend", f"{filtered['amount'].sum():,.2f} บาท")
    c2.metric("Transactions", f"{len(filtered)}")

    st.divider()

    # --- Time drill-down ---
    st.subheader("Spend over time")
    tab_day, tab_hour, tab_weekday = st.tabs(["By day", "By hour", "By weekday"])
    with tab_day:
        st.bar_chart(filtered.groupby("date")["amount"].sum().sort_index())
    with tab_hour:
        hourly = filtered.groupby("hour")["amount"].sum().reindex(range(24), fill_value=0)
        st.bar_chart(hourly)
    with tab_weekday:
        weekday = filtered.groupby("weekday")["amount"].sum().reindex(WEEKDAY_ORDER, fill_value=0)
        st.bar_chart(weekday)

    st.divider()

    # --- What it was spent on ---
    st.subheader("Spend by category")
    col_chart, col_table = st.columns([2, 1])
    by_cat = (
        filtered.groupby("category")["amount"]
        .agg(total="sum", count="count")
        .sort_values("total", ascending=False)
    )
    with col_chart:
        st.bar_chart(by_cat["total"])
    with col_table:
        st.dataframe(by_cat.round(2), width="stretch")

    st.subheader("Top payees")
    by_payee = (
        filtered.groupby("payee")["amount"]
        .agg(total="sum", count="count")
        .sort_values("total", ascending=False)
        .head(15)
    )
    st.bar_chart(by_payee["total"])

    st.divider()

    # --- Detail table (sortable by clicking a column header) ---
    st.subheader(f"Transactions ({len(filtered)})")
    st.dataframe(
        filtered[["datetime", "category", "payee", "payer", "amount", "fee", "slip_type", "transaction_ref"]],
        width="stretch",
        hide_index=True,
    )


# ============================== Upload Slip ===============================

@st.cache_data(show_spinner=False)
def _ocr_cached(file_bytes: bytes) -> str:
    return ocr_image(io.BytesIO(file_bytes))


def render_upload():
    st.title("📤 Upload Slip")
    st.caption(
        "Image → Tesseract OCR → edit if needed → parse with the local LLM → save. "
        "Same OCR as app_vision.py (ocr_utils.py) and same parser as app_text.py "
        "(slip_parser.py). Needs Ollama running + `ollama pull qwen2.5:3b`."
    )

    if not thai_available():
        st.warning(
            "Thai language data not found — OCR will silently degrade to English-only "
            "and garble Thai text instead of erroring. Recreate `tessdata/` (gitignored, "
            "so it's gone after a clean/fresh checkout): see README 'language data'."
        )

    model = st.text_input("Ollama text model", value=PARSE_MODEL, key="upload_model")

    uploaded = st.file_uploader(
        "Upload one or more slip images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
    )
    if not uploaded:
        st.info("Upload a slip image to begin.")
        return

    for i, file in enumerate(uploaded):
        key = f"{i}_{file.name}"
        file_bytes = file.getvalue()

        st.divider()
        st.subheader(file.name)

        col_img, col_txt = st.columns(2)
        with col_img:
            st.image(file_bytes, width="stretch")
        with col_txt:
            text = st.text_area(
                "OCR text (edit if needed before parsing)",
                value=_ocr_cached(file_bytes),
                height=240,
                key=f"text_{key}",
            )

        c1, c2 = st.columns(2)
        if c1.button("Parse to JSON", key=f"parse_{key}"):
            with st.spinner(f"Parsing with {model}…"):
                try:
                    st.session_state[f"json_{key}"] = extract_fields(text, model)
                except RuntimeError as e:
                    st.error(str(e))
                    st.session_state.pop(f"json_{key}", None)

        result = st.session_state.get(f"json_{key}")
        if result is None:
            continue

        if "_error" in result:
            st.error(result["_error"])
            continue

        st.json(result)
        if c2.button("Save (image + text + json)", key=f"save_{key}"):
            IMAGE_DIR.mkdir(exist_ok=True)
            TXT_DIR.mkdir(exist_ok=True)
            JSON_DIR.mkdir(exist_ok=True)

            stem = Path(file.name).stem
            ext = Path(file.name).suffix or ".jpg"

            (IMAGE_DIR / f"{stem}{ext}").write_bytes(file_bytes)
            (TXT_DIR / f"{stem}.txt").write_text(text, encoding="utf-8")
            (JSON_DIR / f"{stem}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            st.success(
                f"Saved `images/{stem}{ext}`, `transactions/{stem}.txt`, "
                f"`transactions_json/{stem}.json` — it'll show up on the Dashboard now."
            )


# ============================== Navigation ================================

pg = st.navigation([
    st.Page(render_dashboard, title="Dashboard", icon="💸", default=True),
    st.Page(render_upload, title="Upload Slip", icon="📤"),
])
pg.run()
