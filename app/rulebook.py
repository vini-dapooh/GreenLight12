"""Rule sources -> atomic rules.

* YAML rulebooks (regulations) are loaded directly.
* A free-text rulebook (Markdown/plain text, e.g. a company standard) is split into clauses and routed to a
  deterministic check. Offline we route by keyword; with an LLM key, llm.rulebook_to_rules() does the routing
  and may also create `llm_judge` rules. A clause nobody can check automatically becomes a *manual* rule that
  surfaces as "needs human review", never silently skipped.
"""
from __future__ import annotations
import re, yaml, os
from dataclasses import dataclass, field

from .checks import KEYWORDS


@dataclass
class Rule:
    id: str
    title: str
    check: str
    authority: str = ""
    severity: str = "major"
    source: str = "regulation"       # regulation | internal | industry
    params: dict = field(default_factory=dict)
    clause: str = ""                  # original wording (internal rules)
    routed_by: str = ""               # yaml | keyword | llm | manual


def load_yaml_rules(path: str):
    out = []
    for r in yaml.safe_load(open(path)) or []:
        out.append(Rule(id=r["id"], title=r["title"], check=r["check"], authority=r.get("authority", ""), severity=r.get("severity", "major"),
                        source=r.get("rule_source", "regulation"), params=r.get("params", {}), routed_by="yaml"))
    return out


def split_clauses(text: str):
    clauses = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:(\d+)[.)]|[-*•])\s+(.*\S)\s*$", line)
        if m:
            clauses.append((m.group(1) or str(len(clauses) + 1), m.group(2)))
    return clauses


def route(clause: str):
    """keyword router: deterministic checks that fit a clause (empty list if nothing fits)"""
    c = clause.lower()
    scored = []
    for name, kws in KEYWORDS.items():
        s = 0
        for k in kws:
            k = k.lower()
            if (len(k) <= 3 and re.search(r"\b" + re.escape(k) + r"\b", c)) or (len(k) > 3 and k in c):
                s += 1
        if s:
            scored.append((s, name))
    if not scored:
        return []
    scored.sort(reverse=True)
    best = scored[0][0]
    return [n for s, n in scored if s >= max(1, 0.6 * best)][:3]


def rules_from_text(text: str, prefix="DF", source="internal"):
    rules = []
    for num, clause in split_clauses(text):
        checks = route(clause)
        title = clause if len(clause) < 110 else clause[:107] + "…"
        if not checks:
            rules.append(Rule(id=f"{prefix}-{num}", title=title, check="manual", source=source, severity="minor", clause=clause, routed_by="manual",
                              authority=f"Company standard {prefix} clause {num}"))
            continue
        for k, ck in enumerate(checks):
            rules.append(Rule(id=f"{prefix}-{num}" + (f"{chr(97 + k)}" if len(checks) > 1 else ""), title=title, check=ck, source=source,
                              severity="major", clause=clause, routed_by="keyword", authority=f"Company standard {prefix} clause {num}"))
    return rules
