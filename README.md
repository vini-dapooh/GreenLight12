# Artwork Readiness Check (Esko AIForge - Problem 2)

Upload packaging artwork (PDF or image), the approved content sheet (.xlsx) and optionally a company rulebook (.md). Get a ranked list of **definite problems**, **items needing human review**, and **things that look correct**, each with evidence on the artwork.

## Run
```bash
pip install -r requirements.txt        # also needs the tesseract binary
uvicorn app.server:app --port 8077     # open http://localhost:8077
pytest -q tests                        # regression tests on the sample
```
Optional AI layer: `export OPENAI_API_KEY=...` (model via `AIFORGE_MODEL`). Everything works without it.

## Layout
`app/extract.py` layers, OCR, panels, barcode, hidden text · `app/parse.py` structure · `app/checks.py` check primitives · `app/rulebook.py` + `app/rules/*.yaml` rules · `app/engine.py` orchestration · `app/llm.py` optional LLM · `web/index.html` UI · `docs/` architecture, write-up, demo script.

See `docs/ARCHITECTURE.md` and `docs/SUBMISSION.md`.
