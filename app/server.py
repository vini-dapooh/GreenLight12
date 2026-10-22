"""FastAPI server: upload -> analyze -> review UI + exports."""
from __future__ import annotations
import base64, csv, io, json, os, tempfile, threading, html
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, HTMLResponse
from PIL import Image, ImageDraw, ImageFont

from .engine import analyze
from .llm import LLM, offline_answer

ROOT = os.path.dirname(os.path.dirname(__file__))
SAMPLES = os.path.join(ROOT, "samples")
app = FastAPI(title="AIForge Artwork Preflight")
JOBS: dict = {}
_sample_lock = threading.Lock()
LLM_CLIENT = LLM()


def _save(up: UploadFile, d: str) -> str:
    p = os.path.join(d, os.path.basename(up.filename or "upload"))
    with open(p, "wb") as f:
        f.write(up.file.read())
    return p


def _run(art_path, sheet_path, rb, art_name, sheet_name, ocr=True):
    report, art, P = analyze(art_path, sheet_path, rb, artwork_name=art_name, sheet_name=sheet_name, llm=LLM_CLIENT, ocr=ocr)
    JOBS[report["id"]] = dict(report=report, art=art, P=P, reviews={})
    return report


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/status")
def status():
    return dict(llm=LLM_CLIENT.available, model=LLM_CLIENT.model if LLM_CLIENT.available else None)


@app.get("/api/sample")
def sample():
    with _sample_lock:
        if "sample" not in JOBS:
            rb = [("company_rulebook.md", open(os.path.join(SAMPLES, "company_rulebook.md")).read())]
            rep = _run(os.path.join(SAMPLES, "carton.pdf"), os.path.join(SAMPLES, "content_sheet.xlsx"), rb,
                       "DancingFish_Granola_Berry_US_Carton.pdf", "DancingFish_Granola_BerryCrunch_US_Content_Sheet.xlsx")
            JOBS["sample"] = JOBS[rep["id"]]
        return JOBS["sample"]["report"]


@app.post("/api/analyze")
def analyze_upload(artwork: UploadFile = File(...), sheet: Optional[UploadFile] = File(None), rulebook: Optional[UploadFile] = File(None), ocr: bool = Form(True)):
    with tempfile.TemporaryDirectory() as d:
        ap = _save(artwork, d)
        sp = _save(sheet, d) if sheet and sheet.filename else None
        rb = []
        if rulebook and rulebook.filename:
            raw = rulebook.file.read()
            try:
                rb.append((rulebook.filename, raw.decode("utf-8", "ignore")))
            except Exception:
                pass
        try:
            return _run(ap, sp, rb, artwork.filename, sheet.filename if sheet and sheet.filename else None, ocr)
        except Exception as e:
            raise HTTPException(400, f"Could not analyse the artwork: {type(e).__name__}: {e}")


def _job(jid):
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[jid]


@app.get("/api/jobs/{jid}/page.jpg")
def page(jid: str, clean: int = 1):
    j = _job(jid)
    pg = j["art"].pages[0]
    return Response(pg.view_clean if clean else pg.view_full, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/jobs/{jid}/review")
def review(jid: str, body: dict):
    j = _job(jid)
    j["reviews"][body["finding_id"]] = dict(state=body.get("state", "open"), note=body.get("note", ""))
    return j["reviews"]


@app.get("/api/jobs/{jid}/reviews")
def reviews(jid: str):
    return _job(jid)["reviews"]


@app.post("/api/jobs/{jid}/chat")
def chat(jid: str, body: dict):
    j = _job(jid)
    f = next((x for x in j["report"]["findings"] if x["id"] == body.get("finding_id")), None)
    if not f:
        raise HTTPException(404, "unknown finding")
    q = body.get("question", "")
    if LLM_CLIENT.available:
        try:
            return dict(answer=LLM_CLIENT.answer(f, q, body.get("history")), source="llm")
        except Exception as e:
            pass
    return dict(answer=offline_answer(f, q), source="offline")


# ----------------------------------------------------------------------------- exports
def annotated_png(job, max_w=2000):
    art = job["art"]; rep = job["report"]
    img = Image.open(io.BytesIO(art.pages[0].clean_png)).convert("RGB")
    s = img.width / rep["page"]["width"]
    d = ImageDraw.Draw(img, "RGBA")
    col = {"fail": (217, 45, 32), "review": (220, 104, 3)}
    n = 0
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for f in rep["findings"]:
        if f["verdict"] == "pass":
            continue
        n += 1
        for e in f["evidence"]:
            if e["kind"] == "artwork" and e.get("bbox") and e["label"] != "Principal display panel" and not e["label"].startswith("Panel region"):
                x0, y0, x1, y1 = [v * s for v in e["bbox"]]
                c = col[f["verdict"]]
                d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=c + (255,), width=3, fill=c + (40,))
                d.rectangle([x0 - 2, y0 - 22, x0 + 26, y0], fill=c + (255,))
                d.text((x0, y0 - 22), str(n), fill=(255, 255, 255, 255), font=font)
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    b = io.BytesIO(); img.save(b, "JPEG", quality=82)
    return b.getvalue()


@app.get("/api/jobs/{jid}/report.json")
def report_json(jid: str):
    j = _job(jid)
    out = dict(j["report"]); out["reviews"] = j["reviews"]
    return JSONResponse(out, headers={"Content-Disposition": f'attachment; filename="preflight-{jid}.json"'})


@app.get("/api/jobs/{jid}/report.csv")
def report_csv(jid: str):
    j = _job(jid)
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["#", "verdict", "severity", "confidence", "rule", "title", "summary", "found", "expected", "authority", "evidence", "reviewer_state", "reviewer_note"])
    for i, f in enumerate(j["report"]["findings"], 1):
        ev = " | ".join(f"{e['label']} [{e.get('panel') or e.get('ref') or ''}]" for e in f["evidence"])
        r = j["reviews"].get(f["id"], {})
        w.writerow([i, f["verdict"], f["severity"], f["confidence"], f["rule_id"], f["title"], f["summary"], f["found"], f["expected"], f["authority"], ev, r.get("state", ""), r.get("note", "")])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="preflight-{jid}.csv"'})


@app.get("/api/jobs/{jid}/report.html")
def report_html(jid: str):
    j = _job(jid); rep = j["report"]
    img = base64.b64encode(annotated_png(j)).decode()
    e = html.escape
    rows = []
    for i, f in enumerate(rep["findings"], 1):
        r = j["reviews"].get(f["id"], {})
        badge = {"fail": "Definite problem", "review": "Needs a human", "pass": "Looks correct"}[f["verdict"]]
        det = ""
        if f["details"]:
            det = "<table class=d>" + "".join(f"<tr><td>{'✓' if d.get('ok') else '✗'}</td><td>{e(str(d['label']))}</td><td>{e(str(d['found']))}</td><td>{e(str(d['expected']))}</td></tr>" for d in f["details"]) + "</table>"
        rows.append(f"<section class='f {f['verdict']}'><h3><span class=n>{i}</span> {e(f['title'])} <em>{badge}</em></h3><p><b>{e(f['summary'])}</b></p><p>{e(f['explanation'])}</p>"
                    + (f"<p class=m>Found: {e(f['found'])}<br>Expected: {e(f['expected'])}</p>" if f["found"] or f["expected"] else "") + det
                    + f"<p class=m>Rule: {e(f['authority'])} · confidence {f['confidence']:.0%} · {e(f['method'])}"
                    + (f" · reviewer: <b>{e(r['state'])}</b> {e(r.get('note', ''))}" if r else "") + "</p></section>")
    s = rep["summary"]
    page = f"""<!doctype html><meta charset=utf-8><title>Preflight report</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1d2433}}h1{{margin:.2rem 0}}img{{max-width:100%;border:1px solid #ddd;border-radius:8px}}
.f{{border-left:6px solid #999;padding:.2rem 1rem;margin:1rem 0;background:#fafafa;border-radius:6px}}.f.fail{{border-color:#d92d20}}.f.review{{border-color:#dc6803}}.f.pass{{border-color:#12b76a}}
h3 em{{font-size:.7em;font-style:normal;color:#666;margin-left:.5rem}}.n{{display:inline-block;background:#1d2433;color:#fff;border-radius:4px;padding:0 .4rem;margin-right:.3rem}}.m{{color:#555;font-size:.9em}}
table.d{{border-collapse:collapse;font-size:.9em}}table.d td{{border:1px solid #e3e3e3;padding:2px 8px}}</style>
<h1>Artwork preflight: {e(rep['headline'] if 'headline' in rep else s['headline'])}</h1>
<p>{e(rep['artwork'])} vs {e(rep['sheet'] or 'no content sheet')} · {s['fail']} definite problems · {s['review']} need a human · {s['passed']} look correct</p>
<img src="data:image/jpeg;base64,{img}" alt="annotated artwork">{''.join(rows)}"""
    return HTMLResponse(page, headers={"Content-Disposition": f'attachment; filename="preflight-{jid}.html"'})
