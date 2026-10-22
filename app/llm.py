"""Optional LLM layer (OpenAI). Everything here degrades gracefully when no API key is set.

Where the LLM is used - and where it is deliberately NOT trusted:
  * detect_marks   : multimodal look at the rendered artwork for graphic marks that have no text layer.
                     Result is only ever "present / not seen" with a low-confidence box -> never a FAIL on its own.
  * route_rulebook : free-text rulebook clauses -> atomic rules in our schema (check name from a closed catalogue,
                     or a free-text `judge` rule). Unknown check names are rejected.
  * judge          : evaluates a clause that has no deterministic check. Must return verbatim quotes from the artwork;
                     quotes are verified against the real text layer. Unverifiable -> downgraded to "needs review".
                     An LLM-only verdict is capped at REVIEW (it can PASS only with verified quotes).
  * answer         : explains a finding / answers follow-ups, grounded on that finding's evidence only.
"""
from __future__ import annotations
import base64, io, json, os, re
from typing import Optional

from rapidfuzz import fuzz

DEFAULT_MODEL = os.environ.get("GREENLIGHT_MODEL", "gpt-4o")


class LLM:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, client=None):
        self.model = model or DEFAULT_MODEL
        self.client = client
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if self.client is None and key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=key)
            except Exception:
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ plumbing
    def _chat(self, system: str, messages: list, max_tokens: int) -> str:
        r = self.client.chat.completions.create(model=self.model, max_completion_tokens=max_tokens,
                                                messages=[{"role": "system", "content": system}] + messages)
        return r.choices[0].message.content or ""

    def _call(self, system: str, content, max_tokens=1500) -> str:
        return self._chat(system, [{"role": "user", "content": content}], max_tokens)

    def _json(self, system, content, max_tokens=1500):
        txt = self._call(system + "\nReturn ONLY a JSON object. No prose, no code fences.", content, max_tokens)
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise ValueError("model did not return JSON")
        return json.loads(m.group(0))

    @staticmethod
    def _img(png: bytes, max_edge=1800):
        from PIL import Image
        im = Image.open(io.BytesIO(png)).convert("RGB")
        s = max_edge / max(im.size)
        if s < 1:
            im = im.resize((int(im.width * s), int(im.height * s)))
        b = io.BytesIO(); im.save(b, "JPEG", quality=88)
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}, im.size

    # ------------------------------------------------------------------ graphic marks
    def detect_marks(self, png: bytes, page_w: float, page_h: float, expected: list):
        """returns {name: {present: bool, bbox: [x0,y0,x1,y1] in pt | None, note: str}}"""
        img, (w, h) = self._img(png)
        sys_ = ("You inspect packaging artwork. For each requested mark/logo say whether it is visibly present on the artwork. "
                "Give a tight bounding box as fractions of image width/height [x0,y0,x1,y1]. Do not guess: if unsure, present=false.")
        user = [img, {"type": "text", "text": "Marks to look for: " + json.dumps(expected) +
                      '\nJSON: {"marks":[{"name":str,"present":bool,"bbox":[x0,y0,x1,y1]|null,"note":str}]}'}]
        data = self._json(sys_, user)
        out = {}
        for m in data.get("marks", []):
            bb = m.get("bbox")
            if m.get("present") and bb and len(bb) == 4 and all(0 <= v <= 1 for v in bb):
                bb = [bb[0] * page_w, bb[1] * page_h, bb[2] * page_w, bb[3] * page_h]
            else:
                bb = None
            out[m.get("name", "")] = dict(present=bool(m.get("present")), bbox=bb, note=m.get("note", ""))
        return out

    # ------------------------------------------------------------------ rulebook
    def route_rulebook(self, text: str, catalogue: dict):
        """catalogue: {check_name: one-line description}. Returns list of rule dicts."""
        sys_ = ("You convert a packaging rulebook into atomic, individually checkable rules. For each rule choose `check` from the catalogue "
                "if one deterministic check fully covers it; otherwise use `judge` and keep the clause wording. One obligation per rule. "
                "Never invent check names.")
        user = ("Catalogue:\n" + "\n".join(f"- {k}: {v}" for k, v in catalogue.items()) + "\n\nRulebook:\n" + text +
                '\n\nJSON: {"rules":[{"id":str,"title":str,"check":str,"clause":str,"severity":"critical|major|minor"}]}')
        data = self._json(sys_, user, 3000)
        rules = []
        for r in data.get("rules", []):
            chk = r.get("check")
            if chk != "judge" and chk not in catalogue:
                chk = "judge"
            rules.append(dict(id=str(r.get("id", "")), title=r.get("title", ""), check=chk, clause=r.get("clause", r.get("title", "")),
                              severity=r.get("severity") if r.get("severity") in ("critical", "major", "minor") else "major"))
        return rules

    # ------------------------------------------------------------------ judge a free-text clause
    def judge(self, clause: str, art_text: str, png: Optional[bytes] = None):
        sys_ = ("You audit packaging artwork against ONE rule. Use only the artwork text and image provided. "
                "Quote the exact artwork text that supports your verdict (verbatim). If the rule cannot be decided from what you can see, answer unclear.")
        content = []
        if png:
            content.append(self._img(png)[0])
        content.append({"type": "text", "text": f"RULE: {clause}\n\nARTWORK TEXT (extracted):\n{art_text[:6000]}\n\n"
                        '{"verdict":"compliant|violation|unclear","confidence":0-1,"quotes":[str],"reason":str}'})
        return self._json(sys_, content)

    @staticmethod
    def verify_quotes(quotes: list, art_text: str, thresh=88) -> bool:
        """every quote must really occur in the artwork text (fuzzy, whitespace-insensitive)"""
        norm = re.sub(r"\s+", " ", art_text.lower())
        if not quotes:
            return False
        return all(fuzz.partial_ratio(re.sub(r"\s+", " ", q.lower()), norm) >= thresh for q in quotes if q.strip())

    # ------------------------------------------------------------------ follow-up questions
    def answer(self, finding: dict, question: str, history: Optional[list] = None, extra: str = "") -> str:
        sys_ = ("You are a packaging-compliance assistant. Answer the reviewer's question about ONE finding using only the finding data below. "
                "Be concrete and short. If the data does not settle the question, say what a person should check. Never invent regulations or numbers.")
        ctx = json.dumps({k: finding.get(k) for k in ("title", "verdict", "summary", "explanation", "found", "expected", "authority", "fix", "details", "evidence")}, ensure_ascii=False)[:6000]
        msgs = []
        for h in (history or [])[-6:]:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": f"FINDING:\n{ctx}\n{extra}\n\nQUESTION: {question}"})
        return self._chat(sys_, msgs, 600).strip()


def offline_answer(finding: dict, question: str) -> str:
    """No API key: answer the common follow-ups straight from the finding's own fields."""
    q = question.lower()
    f = finding
    parts = []
    if any(w in q for w in ("fix", "resolve", "correct", "do about", "what should")):
        parts.append("**Suggested fix:** " + (f.get("fix") or "Have the responsible reviewer confirm the source value, then correct the artwork or the approved data so they agree."))
    if any(w in q for w in ("where", "show", "locate", "evidence", "proof", "how do you know", "why")):
        ev = [e for e in f.get("evidence", []) if e.get("kind") == "artwork"]
        parts.append("**Evidence:** " + ("; ".join(f"“{e.get('text') or e.get('label')}” ({e.get('panel') or 'artwork'})" for e in ev[:4]) or "see the highlighted regions") + ".")
    if any(w in q for w in ("rule", "regulation", "law", "cite", "authority", "why")):
        parts.append("**Rule:** " + (f.get("authority") or "company standard") + ".")
    if any(w in q for w in ("sure", "confiden", "certain", "false alarm", "wrong")):
        parts.append(f"**Confidence:** {f.get('confidence', 0):.0%}. " + ("This is a deterministic check on text read from the PDF's own text layer and cross-checked with OCR." if f.get("method") == "deterministic" else "This one relies on an AI judgement and should be confirmed."))
    if not parts:
        parts.append(f.get("explanation") or f.get("summary", ""))
        parts.append("*(Connect an OpenAI API key to ask free-form follow-ups.)*")
    return "\n\n".join(parts)
