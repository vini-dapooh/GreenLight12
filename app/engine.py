"""Orchestrates: artwork + sheet + rulebooks  ->  findings report."""
from __future__ import annotations
import base64, os, time, uuid
from typing import Optional

from . import checks as C
from .checks import Ctx, CHECKS
from .extract import load_artwork, Artwork
from .llm import LLM
from .models import Evidence, Finding, FAIL, REVIEW, PASS
from .parse import parse_artwork, Item
from .rulebook import Rule, load_yaml_rules, rules_from_text
from .sheet import load_sheet

HERE = os.path.dirname(__file__)
DEFAULT_RULES = os.path.join(HERE, "rules", "us_fda.yaml")
SEV_ORDER = {"critical": 0, "major": 1, "minor": 2}
VERD_ORDER = {FAIL: 0, REVIEW: 1, PASS: 2}


def catalogue():
    return {k: (v.__doc__ or ", ".join(C.KEYWORDS.get(k, ()))).strip().split("\n")[0] for k, v in CHECKS.items()}


def _result_to_finding(rule: Rule, res: dict, idx: int, extra_auth: list) -> Finding:
    ev = res.get("evidence", [])
    auth = res.get("authority") or rule.authority
    if extra_auth:
        auth = (auth + " · " if auth else "") + " · ".join(extra_auth)
    return Finding(
        id=f"{rule.id}.{idx}", rule_id=rule.id, title=res.get("title") or rule.title, verdict=res["verdict"],
        severity=res.get("severity") or rule.severity, confidence=res.get("confidence", 0.8), summary=res["summary"],
        explanation=res.get("explanation", ""), found=res.get("found", ""), expected=res.get("expected", ""),
        authority=auth, rule_source=res.get("rule_source") or rule.source, fix=res.get("fix", ""),
        evidence=[e for e in ev if e is not None], details=res.get("details", []), method=res.get("method", "deterministic"),
        tags=list(res.get("tags", [])))


def _cap_for_ocr(res: dict) -> dict:
    """Artwork with no text layer was read by OCR. OCR can drop or garble words, so 'not found' / 'differs' must never be
    presented as a definite problem: cap at 'needs review', lower the confidence and say why."""
    res = dict(res)
    res["tags"] = list(res.get("tags", [])) + ["ocr-only"]
    res["confidence"] = round(res.get("confidence", 0.8) * 0.65, 2)
    if res["verdict"] == FAIL:
        res["verdict"] = REVIEW
        res["explanation"] = ("Capped from 'definite problem' to 'needs review' because this artwork has no text layer and was read by OCR, which can miss or garble words. "
                              + res.get("explanation", ""))
    return res


def _manual(rule: Rule, ctx: Ctx, llm: Optional[LLM]):
    """Clause with no deterministic check."""
    art_text = "\n".join(b.text for b in ctx.art.print_blocks())
    ev_rule = Evidence("rule", rule.authority or rule.id, rule.clause or rule.title, ref=rule.id)
    if llm and llm.available:
        try:
            png = ctx.art.pages[0].clean_png
            j = llm.judge(rule.clause or rule.title, art_text, png)
            quotes = j.get("quotes", [])
            ok = llm.verify_quotes(quotes, art_text)
            ev = [ev_rule] + [Evidence("artwork", "AI-quoted text", q) for q in quotes[:3]]
            conf = float(j.get("confidence", 0.5)) * (1.0 if ok else 0.5)
            if j.get("verdict") == "compliant" and ok and conf >= 0.7:
                return [dict(verdict=PASS, summary=f"AI review: looks compliant. {j.get('reason','')}", evidence=ev, confidence=min(conf, 0.75), method="llm",
                             title=rule.title, tags=["ai-judged"])]
            return [dict(verdict=REVIEW, summary=f"AI review: {j.get('verdict','unclear')}. {j.get('reason','')}",
                         explanation="This clause has no deterministic checker, so an AI model judged it. AI-only verdicts are capped at 'needs review' (and quotes " +
                                     ("were verified against the PDF text." if ok else "could NOT be verified against the PDF text.") + ")",
                         evidence=ev, confidence=min(conf, 0.6), method="llm", tags=["ai-judged"])]
        except Exception as e:  # network / parsing -> fall through to manual
            pass
    return [dict(verdict=REVIEW, summary="No automatic check exists for this clause; a person must verify it.",
                 explanation="This rule is not machine-checkable offline (it is visual or subjective). It is listed rather than silently skipped. "
                             "With an API key the clause is evaluated by a vision-language model and the result is still capped at 'needs review'.",
                 evidence=[ev_rule], confidence=0.0, method="manual", tags=["manual"])]


def analyze(artwork_path: str, sheet_path: Optional[str] = None, rulebooks: Optional[list] = None, *, artwork_name=None, sheet_name=None,
            llm: Optional[LLM] = None, ocr=True, yaml_rules=DEFAULT_RULES) -> dict:
    t0 = time.time()
    trace = []
    art = load_artwork(artwork_path, artwork_name, ocr=ocr)
    trace += art.trace
    sheet = load_sheet(sheet_path, sheet_name) if sheet_path else None
    if sheet:
        unc = [f.label for f in sheet.fields.values() if f.uncertain]
        trace.append(f"Approved content sheet: {len(sheet.fields)} fields, {len(sheet.nutrition)} nutrient rows; {len(unc)} fields carry an 'unverified' note")
    P = parse_artwork(art)
    trace.append(f"Parsed: {len(P.nf_rows)} nutrition rows, {len(P.ingredients.text.split(',')) if P.ingredients else 0} ingredients, "
                 f"{len(P.net_qty)} net-quantity statement(s), {sum(len(v) for v in P.claims.values())} claim mentions")
    ctx = Ctx(art, P, sheet, llm)

    # optional vision pass for graphic marks
    if llm and llm.available and sheet:
        try:
            want = ["USDA Organic seal", "Certified Vegan mark (vegan.org)", "How2Recycle / recyclable paperboard mark"]
            pg = art.pages[0]
            res = llm.detect_marks(pg.clean_png, pg.width, pg.height, want)
            keymap = {"USDA Organic seal": "usda_organic", "Certified Vegan mark (vegan.org)": "certified_vegan", "How2Recycle / recyclable paperboard mark": "recycle"}
            for name, r in res.items():
                k = keymap.get(name)
                if k and r["present"] and r["bbox"] and k not in P.marks:
                    P.marks[k] = Item(f"{name} (vision model)", tuple(r["bbox"]), None, meta=dict(source="llm", confidence=0.6))
            trace.append(f"Vision model looked for {len(want)} graphic marks; {sum(1 for r in res.values() if r['present'])} seen")
        except Exception as e:
            trace.append(f"Vision pass skipped: {type(e).__name__}")

    # rules
    rules = load_yaml_rules(yaml_rules)
    trace.append(f"Loaded {len(rules)} regulatory rules from {os.path.basename(yaml_rules)}")
    for name, text in (rulebooks or []):
        rb = None
        if llm and llm.available:
            try:
                rb = [Rule(id=f"DF-{r['id']}" if not str(r['id']).startswith("DF") else r['id'], title=r["title"], check=("manual" if r["check"] == "judge" else r["check"]),
                           severity=r.get("severity", "major"), source="internal", clause=r.get("clause", ""), routed_by="llm",
                           authority=f"Company standard · {name}") for r in llm.route_rulebook(text, catalogue())]
                trace.append(f"Rulebook “{name}”: AI split it into {len(rb)} atomic rules (check names validated against a closed catalogue)")
            except Exception as e:
                rb = None
        if rb is None:
            rb = rules_from_text(text)
            nm = sum(1 for r in rb if r.check == "manual")
            trace.append(f"Rulebook “{name}”: {len(rb)} atomic rules from keyword routing ({nm} not machine-checkable → listed for manual review)")
        rules += rb

    # dedupe rules that map to the same check -> one finding, several authorities
    findings, seen = [], {}
    n_run = 0
    for rule in rules:
        key = (rule.check, tuple(sorted(rule.params.items())))
        if rule.check != "manual" and key in seen:
            seen[key]["extra"].append(f"{rule.id} ({rule.source})")
            continue
        if rule.check == "manual":
            results = _manual(rule, ctx, llm)
        else:
            fn = CHECKS.get(rule.check)
            if not fn:
                continue
            try:
                results = fn(ctx, rule.params)
            except Exception as e:  # a failing check must never take the report down
                results = [dict(verdict=REVIEW, summary=f"Check '{rule.check}' could not run ({type(e).__name__}).", explanation=str(e)[:200], confidence=0.0, evidence=[])]
            n_run += 1
        entry = dict(rule=rule, results=results, extra=[])
        seen[key] = entry
        findings.append(entry)
    ocr_only = bool(art.blocks) and all(b.source == "ocr" for b in art.blocks)
    out = []
    for entry in findings:
        for i, res in enumerate(entry["results"]):
            if ocr_only:
                res = _cap_for_ocr(res)
            out.append(_result_to_finding(entry["rule"], res, i + 1, entry["extra"]))
    out.sort(key=lambda f: (VERD_ORDER[f.verdict], SEV_ORDER.get(f.severity, 3)))
    counts = {FAIL: 0, REVIEW: 0, PASS: 0}
    for f in out:
        counts[f.verdict] += 1
    if counts[FAIL]:
        headline = "Not ready to launch"
    elif counts[REVIEW]:
        headline = "Needs human review"
    else:
        headline = "Ready to launch"
    trace.append(f"Ran {n_run} checks → {len(out)} findings ({counts[FAIL]} definite problems, {counts[REVIEW]} need a human, {counts[PASS]} look correct) in {time.time() - t0:.1f}s")

    pg = art.pages[0]
    report = dict(
        id=uuid.uuid4().hex[:10], artwork=art.filename, sheet=sheet.filename if sheet else None, rulebooks=[n for n, _ in (rulebooks or [])],
        llm=bool(llm and llm.available), model=llm.model if llm and llm.available else None,
        page=dict(width=pg.width, height=pg.height, dpi=pg.dpi),
        summary=dict(fail=counts[FAIL], review=counts[REVIEW], passed=counts[PASS], total=len(out), headline=headline,
                     critical=sum(1 for f in out if f.verdict == FAIL and f.severity == "critical")),
        findings=[f.to_dict() for f in out], trace=trace, stats=art.stats,
        panels={k: [round(v, 1) for v in bb] for k, bb in art.panels.items()},
        blocks=[b.to_dict() for b in art.blocks if b.kind in ("guide", "hidden")],
        seconds=round(time.time() - t0, 1),
    )
    return report, art, P
