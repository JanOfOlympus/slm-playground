"""
Spending Dashboard

Reads every JSON file in ./transactions_json (written by app_text.py /
app_vision.py, categories filled in by resolve_categories.py) and lets you
drill into spend by date range, time, amount, and category/payee.

Nothing leaves the machine — this only reads local files.

    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

JSON_DIR = Path(__file__).parent / "transactions_json"
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="Spending Dashboard", layout="wide")


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


st.title("💸 Spending Dashboard")
st.caption(f"Reads every slip JSON from `{JSON_DIR}`.")

df = load_transactions(JSON_DIR)

if df.empty:
    st.warning(f"No JSON files in `{JSON_DIR}`. Run app_text.py or app_vision.py first.")
    st.stop()

no_amount = int(df["amount"].isna().sum())
no_date = int(df["date"].isna().sum())
df = df.dropna(subset=["date", "amount"]).copy()

if df.empty:
    st.warning("No transactions have both a parseable date and amount yet.")
    st.stop()

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
    st.stop()

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

# --- Detail table (sortable by clicking a column header — dig into amount/time) ---
st.subheader(f"Transactions ({len(filtered)})")
st.dataframe(
    filtered[["datetime", "category", "payee", "payer", "amount", "fee", "slip_type", "transaction_ref"]],
    width="stretch",
    hide_index=True,
)
