# GreenLight: Is this packaging artwork ready to launch?

*GreenLight turns "is the artwork ready to print?" into a clear red / amber / green answer: definite problems, items for a human, and what is verified correct. Built for Esko AIForge, Problem 2.*

Upload packaging artwork (PDF or image), the approved content sheet (.xlsx) and optionally a company rulebook (.md). Get a ranked list of **definite problems**, **items needing human review**, and **things that look correct**, each with evidence on the artwork.

## Run
```bash
pip install -r requirements.txt        # also needs the Tesseract OCR binary installed (https://github.com/tesseract-ocr/tesseract)
uvicorn app.server:app --port 8077     # open http://localhost:8077
pytest -q tests                        # regression tests on the sample
```
Optional AI layer: put `OPENAI_API_KEY=...` in a `.env` file next to this README (gitignored), or export it (model via `GREENLIGHT_MODEL`). Everything works without it.

## Why "GreenLight"
The question a packaging team asks before print is "can we give this the green light?". Every finding lands in one of three colours: **red** (proven problem), **amber** (a human must decide), **green** (checked and correct). Nothing is flagged red unless it is proven by arithmetic, set comparison, regex or a decoded barcode.

## Layout
`app/extract.py` layers, OCR, panels, barcode, hidden text · `app/parse.py` structure · `app/checks.py` check primitives · `app/rulebook.py` + `app/rules/*.yaml` rules · `app/engine.py` orchestration · `app/llm.py` optional LLM · `web/index.html` UI · `docs/` architecture, write-up, demo script.

See `docs/ARCHITECTURE.md` and `docs/SUBMISSION.md`.
