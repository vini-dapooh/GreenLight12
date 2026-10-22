"""Regression tests: the planted issues in the Esko sample must be found, and the right things must PASS."""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from app.engine import analyze
from app.llm import LLM

S = os.path.join(os.path.dirname(__file__), "..", "samples")

@pytest.fixture(scope="module")
def rep():
    rb = open(os.path.join(S, "company_rulebook.md")).read()
    r, art, P = analyze(os.path.join(S, "carton.pdf"), os.path.join(S, "content_sheet.xlsx"), [("company_rulebook.md", rb)])
    return r

def by_title(rep, needle):
    return [f for f in rep["findings"] if needle.lower() in f["title"].lower()]

def verdict(rep, needle):
    fs = by_title(rep, needle)
    assert fs, f"no finding matching {needle!r}"
    return {f["verdict"] for f in fs}

def test_headline(rep):
    assert rep["summary"]["headline"] == "Not ready to launch"
    assert rep["summary"]["fail"] >= 10

@pytest.mark.parametrize("needle", ["added sugar", "vegan", "ingredient list differs", "place of business", "serving size"])
def test_definite_problems(rep, needle):
    assert "fail" in verdict(rep, needle)

@pytest.mark.skipif(not shutil.which("tesseract"), reason="hidden-text check needs the Tesseract binary")
def test_hidden_panel_is_review_not_fail(rep):
    assert verdict(rep, "hidden") == {"review"}

def test_good_things_pass(rep):
    assert "pass" in verdict(rep, "statement of identity")
    assert "pass" in verdict(rep, "net weight matches")

def test_manual_clauses_listed(rep):
    assert any("manual" in f["tags"] for f in rep["findings"])

def test_every_fail_has_evidence(rep):
    for f in rep["findings"]:
        if f["verdict"] == "fail":
            assert f["evidence"], f["id"]

def test_ocr_only_never_fails():
    from PIL import Image
    p = "/tmp/aif_raster.png"
    if not os.path.exists(p):
        pytest.skip("raster fixture missing")
    r, _, _ = analyze(p, os.path.join(S, "content_sheet.xlsx"), [], artwork_name="carton.png")
    assert r["summary"]["fail"] == 0

def test_llm_off_by_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not LLM(api_key=None).available

def test_llm_judge_is_capped_and_quotes_verified():
    from app.engine import _manual
    from app.rulebook import Rule
    from app.checks import Ctx
    class Fake(LLM):
        available = True
        def judge(self, *a, **k): return dict(verdict="compliant", confidence=0.99, reason="ok", quotes=["THIS TEXT IS NOT ON THE PACK"])
    r, art, P = analyze(os.path.join(S, "carton.pdf"), None, [])
    out = _manual(Rule(id="X", title="t", check="manual", clause="c"), Ctx(art, P, None, None), Fake(api_key=None))
    assert out[0]["verdict"] in ("review",)   # unverifiable quote can never produce PASS
