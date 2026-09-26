# slm-playground

## .venv
- py -m venv .venv
- .venv\Scripts\Activate.ps1
- deactivate

## pip
- py -m pip install streamlit requests numpy pandas chromadb pytesseract pillow

## tesseract (OCR engine — used by app_vision.py)
- install the binary: `winget install UB-Mannheim.TesseractOCR` (or the UB-Mannheim installer)
- app_vision.py auto-detects the binary at `C:\Program Files\Tesseract-OCR\tesseract.exe`;
  override with `$env:TESSERACT_CMD` if yours is elsewhere

## language data (project-local, no admin needed)
- app_vision.py loads traineddata from `./tessdata/` (via `TESSDATA_PREFIX`) when that folder exists
- create it once:
  - `mkdir tessdata`
  - `curl -L -o tessdata/tha.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/tha.traineddata`
  - `copy "C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata" tessdata\`
  - `copy "C:\Program Files\Tesseract-OCR\tessdata\osd.traineddata" tessdata\`
- `tessdata/` is gitignored; recreate it with the steps above after a fresh clone

## streamlit
- `streamlit run app.py` — **dashboard**: spend by date range, time, amount, category (`transactions_json/`)
- `streamlit run app_text.py` — step 1 of the slip pipeline (clean text -> JSON)
- `streamlit run app_vision.py` — structured slip fields straight from images (`images/`)
- `python resolve_categories.py` — step 2 (fill in `other` categories via a third-party AI)

## app.py — two pages (`st.navigation`)

### Dashboard
- reads every `transactions_json/*.json` (skips any with no parseable date/amount)
- sidebar filters: date range, category (multiselect)
- KPIs: total spend, transaction count (filtered)
- drill-downs: spend by day / hour / weekday (tabs), spend by category
  (chart + table), top payees (chart), full sortable transaction table
- purely local, no network calls

### Upload Slip
- upload an image -> Tesseract OCR (`ocr_utils.py`, same engine as app_vision.py)
  -> edit the text if the OCR is messy -> **Parse to JSON**
  (`slip_parser.py`, same pipeline as app_text.py) -> **Save**, which writes
  `images/<name>`, `transactions/<name>.txt`, `transactions_json/<name>.json`
  so it shows up on the Dashboard immediately
- needs: `ollama serve` + `ollama pull qwen2.5:3b`
- Tesseract's Thai OCR on a raw photo is noticeably worse than Apple Live Text
  (see app_text.py) — that's why the OCR text box is editable before parsing

`.claude/launch.json` has a `dashboard` config for previewing app.py in-editor.

## slip_parser.py / ocr_utils.py
- shared, Streamlit-free logic modules: `slip_parser.py` is the OCR-text ->
  JSON pipeline (used by app_text.py and app.py's Upload Slip page);
  `ocr_utils.py` is the Tesseract setup + image -> text helper (used by
  app.py's Upload Slip page)

## slip pipeline (two steps)

### step 1 — app_text.py (clean text -> JSON, fully local)
- put clean OCR text (e.g. Apple Live Text "Copy All Text") as `transactions/slip_*.txt`
- `qwen2.5:3b` structures each slip; a deterministic pass then pins slip_type,
  amount, fee, datetime, and the two party names (payer = first Thai-title line,
  payee = first Thai line after the masked account no.)
- `category` = food / groceries / shopping / transport / utilities / health /
  entertainment / services / education / government / transfer / other.
  Only rule-certain billers (electricity/water/gov) and P2P transfers are set
  here; everything else is left as `other` for step 2. The local model is told
  to answer `other` rather than guess.
- writes `transactions_json/<slip>.json`
- needs: `ollama serve` + `ollama pull qwen2.5:3b`; ~12-15 s per slip; nothing leaves the machine

### step 2 — resolve_categories.py (category lookup via Anthropic API)
- reads `transactions_json/*.json`; for each still `other`, sends **only** the
  payee name to Claude and writes back `category` + `category_source: api:anthropic`
- no amount / account / payer / date is ever sent
- `python resolve_categories.py` (add `--force` to redo all, `--dry-run` to preview)
- configure: `$env:ANTHROPIC_API_KEY = "sk-ant-..."`
  (optional `$env:CATEGORY_MODEL`, default `claude-haiku-4-5-20251001`)

## app_vision.py (image -> JSON) — fallback for messy OCR
- a local Ollama **vision** model reads each slip image directly
- amount / fee / date overridden by a regex pass over Tesseract text
- needs: `ollama serve` + `ollama pull qwen2.5vl:7b` (~6 GB)
- model editable in the UI (try `minicpm-v`; avoid `llama3.2-vision` for Thai)
- slow on CPU (~2-3 min per image)

## model useds
- command: ollama pull <models>
- decoding model: qwen2.5:3b
- vision model (app_vision.py): qwen2.5vl:7b
- embedding model: nomic-embed-text

## vector store
- chroma