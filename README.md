# slm-playground

## .venv
- py -m venv .venv
- .venv\Scripts\Activate.ps1
- deactivate

## pip
- py -m pip install streamlit requests numpy chromadb pytesseract pillow

## tesseract (OCR engine)
- install the binary: `winget install UB-Mannheim.TesseractOCR` (or the UB-Mannheim installer)
- app.py auto-detects the binary at `C:\Program Files\Tesseract-OCR\tesseract.exe`;
  override with `$env:TESSERACT_CMD` if yours is elsewhere

## language data (project-local, no admin needed)
- app.py loads traineddata from `./tessdata/` (via `TESSDATA_PREFIX`) when that folder exists
- create it once:
  - `mkdir tessdata`
  - `curl -L -o tessdata/tha.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/tha.traineddata`
  - `copy "C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata" tessdata\`
  - `copy "C:\Program Files\Tesseract-OCR\tessdata\osd.traineddata" tessdata\`
- `tessdata/` is gitignored; recreate it with the steps above after a fresh clone

## streamlit
- `streamlit run app.py` — OCR: image (`images/`) -> raw text (Tesseract)
- `streamlit run app_text.py` — structured slip fields from clean text (`transactions/`) **[recommended]**
- `streamlit run app_vision.py` — structured slip fields straight from images (`images/`)

## app_text.py (clean text -> JSON) — recommended
- put clean OCR text (e.g. Apple Live Text "Copy All Text") as `transactions/slip_*.txt`
- `qwen2.5:3b` structures each slip; a deterministic pass then pins slip_type,
  amount, fee, datetime, and the two party names (payer = first Thai-title line,
  payee = first Thai line after the masked account no.)
- needs: `ollama serve` + `ollama pull qwen2.5:3b`
- ~12-15 s per slip on CPU; nothing leaves the machine

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