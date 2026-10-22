# GreenLight architecture

```mermaid
flowchart LR
  A[Artwork PDF / PNG] --> X
  S[Approved content sheet .xlsx] --> SH
  R1[Regulation rules YAML] --> RE
  R2[Company rulebook .md] --> RT[Rule router<br/>keywords, or LLM into a closed catalogue]
  RT --> RE

  subgraph X[1 Extract]
    X1[Text-layer spans<br/>PyMuPDF get_texttrace] --> X2[Dedupe + merge glyph runs]
    X2 --> X3[Classify guide text by layer / vocabulary]
    X3 --> X4[Panels from dieline crease vectors]
    X4 --> X5[Switch dieline layers OFF, render clean]
    X5 --> X6[OCR clean render - tesseract]
    X6 --> X7[Hidden-text detection:<br/>text layer vs what OCR can see]
    X5 --> X8[Barcode decode - zxing-cpp]
  end
  SH[Content sheet reader<br/>cell refs + 'unverified' flags]
  X --> P[2 Parse<br/>nutrition rows, ingredients, allergens,<br/>claims, net qty, marks - with bounding boxes]
  P --> RE[3 Rule engine<br/>about 30 deterministic check primitives]
  SH --> RE
  RE --> V{{Verdict per finding}}
  V -->|proven| F[FAIL definite problem]
  V -->|uncertain| W[REVIEW needs a human]
  V -->|verified| OK[PASS]
  LLM[Optional LLM layer<br/>vision marks, rule routing, judge, Q&A] -.->|never the sole source of a FAIL| RE
  F & W & OK --> UI[Web UI: evidence on artwork,<br/>confirm / false alarm, export JSON/CSV/HTML]
```

## Design rules that handle "uncertainty and wrong AI output"

| Principle | How it shows up |
|---|---|
| Deterministic first | Every FAIL comes from arithmetic, set comparison, regex or a decoded barcode - not from a model's opinion. |
| Three outcomes, not two | FAIL = proven problem. REVIEW = the system can't be sure. PASS = checked and OK. |
| The LLM can't convict | LLM-only findings are capped at REVIEW. Quotes it cites must exist in the PDF text layer, otherwise confidence is halved. |
| Evidence on every finding | Artwork box (pin-pointed to words), content-sheet cell reference, rule clause. |
| Source reliability | Content-sheet fields marked "verify / confirm / assumed" tag findings `approved-data-unverified`. |
| Degrades honestly | No text layer (PNG/scan)? Everything is OCR-read and any FAIL is capped to REVIEW at lower confidence, with the reason stated. |
| Nothing is silently skipped | Clauses with no automatic check (visual / subjective) appear as `manual` REVIEW items. |
| Human in the loop | Confirm / False-alarm per finding; exports carry the decision. |
