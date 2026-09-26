"""
Slip parser — step 1 of the slip pipeline.

Reads clean OCR text (e.g. from Apple Live Text) saved as .txt files in
./transactions, sends each to a local text LLM (qwen2.5:3b via Ollama), and
writes the structured result to ./transactions_json/<slip>.json.

The parsing itself lives in slip_parser.py (shared with app.py's "Upload Slip"
page). `category` left as "other" here is a job for resolve_categories.py.
Nothing leaves the machine.

    streamlit run app_text.py

Needs: `ollama serve` running + `ollama pull qwen2.5:3b`
"""

import json
from pathlib import Path

import streamlit as st

from slip_parser import PARSE_MODEL, extract_fields

TXT_DIR = Path(__file__).parent / "transactions"
JSON_DIR = Path(__file__).parent / "transactions_json"


def list_txt(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


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
