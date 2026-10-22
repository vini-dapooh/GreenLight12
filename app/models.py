"""Shared data structures. Everything that reaches the UI is built from these."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

BBox = tuple  # (x0, y0, x1, y1) in PDF points, origin top-left


@dataclass
class Block:
    """One run of text found on the artwork, with its location."""
    id: str
    text: str
    bbox: BBox
    page: int = 0
    size: float = 0.0
    color: str = "000000"
    layer: str = ""
    rot: int = 0                 # 0 / 90 / 180 / 270 / -1 (curved)
    source: str = "text-layer"   # text-layer | ocr
    kind: str = "print"          # print | guide | hidden
    panel: Optional[str] = None
    ocr_support: Optional[float] = None   # 0..100 agreement with what is visibly rendered

    @property
    def norm(self) -> str:
        return "".join(ch for ch in self.text.lower() if ch.isalnum())

    def to_dict(self):
        d = asdict(self)
        d["bbox"] = [round(v, 1) for v in self.bbox]
        return d


@dataclass
class Evidence:
    kind: str                     # artwork | sheet | rule
    label: str = ""
    text: str = ""
    bbox: Optional[list] = None   # artwork only
    page: int = 0
    panel: Optional[str] = None
    ref: Optional[str] = None     # sheet cell / rule clause

    def to_dict(self):
        d = asdict(self)
        if d["bbox"]:
            d["bbox"] = [round(v, 1) for v in d["bbox"]]
        return d


FAIL, REVIEW, PASS = "fail", "review", "pass"


@dataclass
class Finding:
    id: str
    rule_id: str
    title: str
    verdict: str                  # fail | review | pass
    severity: str                 # critical | major | minor
    confidence: float
    summary: str                  # one-line, plain language
    explanation: str              # why it matters + how we know
    found: str = ""
    expected: str = ""
    authority: str = ""
    rule_source: str = "regulation"   # regulation | internal | content-sheet
    fix: str = ""
    evidence: list = field(default_factory=list)
    details: list = field(default_factory=list)   # optional table rows [{label, found, expected, ok}]
    method: str = "deterministic"     # deterministic | llm | hybrid
    tags: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["evidence"] = [e if isinstance(e, dict) else e.to_dict() for e in self.evidence]
        d["confidence"] = round(self.confidence, 2)
        return d
