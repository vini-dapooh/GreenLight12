# Is this packaging artwork ready to launch? - Submission write-up

**Problem 2, Esko AIForge.** Packaging artwork must match the approved product data and the rules (regulatory, retailer, company) before it is printed. Today a person reads the artwork against a spreadsheet and a rulebook. It is slow, and the errors that slip through are expensive: a wrong allergen, a wrong claim, a Nutrition Facts panel that does not add up.

## Approach
1. **Read the artwork like a prepress tool, not like a picture.** Dielines keep guide text on separate PDF layers. We read the text layer with positions, separate dieline guide text from printed text, find panels from the crease vectors, then switch the guide layers off to get a clean render.
2. **Cross-check what is encoded against what is visible.** OCR on the clean render is an independent witness. If a text block exists in the file but OCR cannot see it, it is covered by something else. On the sample this exposes a full UK nutrition table hidden under the US panel.
3. **Parse into structure**: nutrition rows, ingredients, allergens, claims, net quantity, barcode, marks. Each item keeps its bounding box.
4. **Run rules as code.** Regulation rules (YAML), content-sheet comparison, prepress rules, and a plain-English company rulebook that is split into clauses and routed to checks. Clauses nothing can check are listed for manual review.
5. **Report with evidence.** Each finding: found vs expected, the arithmetic, a pin on the artwork, a sheet cell reference, the rule clause, a suggested fix, a confidence.

## Where the AI is (and is not)
- **Deterministic core, no model required:** layout analysis, OCR (Tesseract), barcode decode (zxing-cpp), %DV and calorie arithmetic, ingredient set comparison, rule routing by keyword.
- **Optional LLM layer (OpenAI, needs `OPENAI_API_KEY`):** vision pass for graphic marks (USDA, vegan, recycling), rule splitting/routing against a closed catalogue of checks, judging clauses that are not machine-checkable, and answering follow-up questions about a finding.
- **Guardrails:** LLM-only verdicts are capped at "needs review"; cited quotes are verified against the PDF text; routing can only choose from existing check names; the app works fully offline.
- **Honest limitation:** the LLM layer is unit-tested with a fake client but has not been run against the live API in development.

## Result on the sample carton (15 s)
36 findings: **15 definite problems** (6 critical), **13 need a human**, 8 look correct. Highlights: %DV wrong in 9 of 11 rows; 166 kcal declared vs about 535 implied by the macros; "No Added Sugar" with honey, coconut sugar and cane sugar listed; tree nuts present but only "May Contain"; ingredients differ from the sheet (walnuts, flax, strawberries, pumpkin seeds, honey 1% vs 9%); vegan claim with honey; net weight only on the back and in metric only; no manufacturer address; hidden UK nutrition table.

## Limits
- Tuned on one carton; the structure is generic but the nutrition parser targets US-style panels.
- Image-only artwork is read by OCR and is deliberately conservative.
- Graphic marks need the vision model or a human.

## Tech
Python, FastAPI, PyMuPDF, Tesseract, zxing-cpp, openpyxl, RapidFuzz, YAML rules, vanilla JS/SVG viewer, optional OpenAI API.
