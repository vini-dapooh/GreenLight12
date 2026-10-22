"""Blocks -> structured label facts (nutrition table, ingredients, allergens, net quantity, claims, ...).

Every extracted value keeps the bounding box it came from, so every finding can point at pixels.
The LLM path (llm.py) can replace/augment these heuristics for layouts we have not seen; its output is
validated into the same structures.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, Any

from .extract import Artwork, union, _norm
from .models import Block


@dataclass
class Item:
    text: str
    bbox: Optional[tuple] = None
    panel: Optional[str] = None
    value: Any = None
    meta: dict = field(default_factory=dict)

    def ev(self, label="", kind="artwork"):
        from .models import Evidence
        return Evidence(kind=kind, label=label, text=self.text, bbox=list(self.bbox) if self.bbox else None, panel=self.panel)


@dataclass
class NFRow:
    key: str
    label: str
    amount: Optional[float]
    unit: str
    dv: Optional[float]
    text: str
    bbox: tuple
    amount_bbox: Optional[tuple] = None
    dv_bbox: Optional[tuple] = None
    panel: Optional[str] = None


@dataclass
class Parsed:
    nf_rows: dict = field(default_factory=dict)       # key -> NFRow
    nf_title: Optional[Item] = None
    nf_region: Optional[tuple] = None
    servings: Optional[Item] = None                   # value = int
    serving_size: Optional[Item] = None               # text as printed, value = float|None, meta.unit
    calories: Optional[NFRow] = None
    strip: dict = field(default_factory=dict)         # non-US "traffic light" strip: key -> (value, bbox)
    strip_region: Optional[tuple] = None
    ingredients: Optional[Item] = None
    allergens: Optional[Item] = None
    dietary: list = field(default_factory=list)
    storage: Optional[Item] = None
    net_qty: list = field(default_factory=list)
    identity: list = field(default_factory=list)
    manufacturer: Optional[Item] = None
    claims: dict = field(default_factory=dict)        # key -> [Item]
    marks: dict = field(default_factory=dict)         # key -> Item|None  (graphic marks)
    marketing: list = field(default_factory=list)
    hidden: list = field(default_factory=list)        # Blocks
    upc_text: Optional[Item] = None
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------- locate words
def locate(art: Artwork, phrase: str, within=None, pad=0.6):
    """Bounding box of a phrase inside the artwork (word-level), optionally constrained to a region."""
    toks = [_norm(t) for t in phrase.split() if _norm(t)]
    if not toks:
        return None
    words = art.words
    if within:
        x0, y0, x1, y1 = within
        words = [w for w in words if x0 - 1 <= (w["bbox"][0] + w["bbox"][2]) / 2 <= x1 + 1 and y0 - 1 <= (w["bbox"][1] + w["bbox"][3]) / 2 <= y1 + 1]
    # reading order: line (y) then x; justified text keeps baseline within a pt or two
    words = sorted(words, key=lambda w: (round(w["bbox"][1] / 3.0), w["bbox"][0]))
    nw = [_norm(w["text"]) for w in words]
    for i in range(len(words) - len(toks) + 1):
        if all(nw[i + k] == toks[k] or (k == len(toks) - 1 and nw[i + k].startswith(toks[k])) for k in range(len(toks))):
            bb = union([words[i + k]["bbox"] for k in range(len(toks))])
            return (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)
    return None


# --------------------------------------------------------------------------- nutrition facts
NF_ALIASES = [
    ("calories", r"calories"), ("total_fat", r"totalfat"), ("saturated_fat", r"saturatedfat"),
    ("trans_fat", r"transfat"), ("cholesterol", r"cholesterol"), ("sodium", r"sodium"),
    ("total_carbohydrate", r"totalcarb\w*"), ("dietary_fiber", r"dietaryfib\w*"), ("total_sugars", r"totalsugars"),
    ("added_sugars", r"(incl\w*|includes)?\d*\w*addedsugars?"), ("protein", r"protein"), ("vitamin_d", r"vitamind"),
    ("calcium", r"calcium"), ("iron", r"iron"), ("potassium", r"potassium"),
]
NF_LABEL = {"calories": "Calories", "total_fat": "Total Fat", "saturated_fat": "Saturated Fat", "trans_fat": "Trans Fat",
            "cholesterol": "Cholesterol", "sodium": "Sodium", "total_carbohydrate": "Total Carbohydrate",
            "dietary_fiber": "Dietary Fiber", "total_sugars": "Total Sugars", "added_sugars": "Added Sugars",
            "protein": "Protein", "vitamin_d": "Vitamin D", "calcium": "Calcium", "iron": "Iron", "potassium": "Potassium"}
AMT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mcg|µg|mg|g|kcal)\b", re.I)
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _rows(blocks, tol=3.0):
    """cluster blocks into visual rows by vertical centre"""
    blocks = sorted(blocks, key=lambda b: ((b.bbox[1] + b.bbox[3]) / 2, b.bbox[0]))
    rows, cur, cy = [], [], None
    for b in blocks:
        c = (b.bbox[1] + b.bbox[3]) / 2
        if cur and abs(c - cy) <= tol:
            cur.append(b); cy = sum((x.bbox[1] + x.bbox[3]) / 2 for x in cur) / len(cur)
        else:
            if cur: rows.append(sorted(cur, key=lambda x: x.bbox[0]))
            cur, cy = [b], c
    if cur: rows.append(sorted(cur, key=lambda x: x.bbox[0]))
    return rows


def _nf_key(label_norm):
    for key, pat in NF_ALIASES:
        if re.fullmatch(pat, label_norm):
            return key
    return None


def parse_nutrition(art: Artwork, P: Parsed):
    pb = art.print_blocks()
    titles = [b for b in pb if b.norm == "nutritionfacts" and b.size >= 10]
    title = None
    for t in titles:
        near = [b for b in pb if "servingspercontainer" in b.norm or "servingpercontainer" in b.norm]
        if any(abs(n.bbox[0] - t.bbox[0]) < 12 and 0 < n.bbox[1] - t.bbox[3] < 40 for n in near):
            title = t; break
    if title is None:
        return
    foot = [b for b in pb if b.norm.startswith("thedailyvalue") or b.norm.startswith("thedv") or b.norm.startswith("the%dailyvalue") or "dailyvalue(dv)" in b.text.lower().replace(" ", "")]
    foot_b = [b for b in pb if b.text.lstrip("* ").lower().startswith("the % daily value") and abs(b.bbox[0] - title.bbox[0]) < 15]
    y_end = max([b.bbox[3] for b in foot_b], default=title.bbox[3] + 280)
    x0, x1 = title.bbox[0] - 4, title.bbox[2] + 6
    region = (x0, title.bbox[1] - 2, x1, y_end + 3)
    P.nf_region = region
    P.nf_title = Item(title.text, title.bbox, title.panel)
    inside = [b for b in pb if b is not title and region[0] <= b.bbox[0] and b.bbox[2] <= region[2] + 3 and region[1] <= b.bbox[1] and b.bbox[3] <= region[3]
              and b.size < 30]
    body = [b for b in inside if b not in foot_b]
    for row in _rows(body):
        text = " ".join(b.text for b in row)
        tl = text.lower()
        bb = union([b.bbox for b in row])
        if "servings per container" in tl or "serving per container" in tl:
            m = re.search(r"(\d+(?:\.\d+)?)\s*servings?", tl)
            if m:
                P.servings = Item(text, bb, title.panel, value=float(m.group(1)))
            continue
        if tl.startswith("serving size"):
            rest = text[len("serving size"):].strip()
            um = re.search(r"(\d+(?:\.\d+)?)\s*(g|ml|oz|mL)\b", rest)
            val = re.search(r"\d+(?:\.\d+)?", rest)
            P.serving_size = Item(rest or "(blank)", bb, title.panel,
                                  value=float(val.group(0)) if val else None,
                                  meta=dict(has_unit=bool(um), unit=um.group(2) if um else None, grams=float(um.group(1)) if um and um.group(2) == "g" else None,
                                            full=text))
            continue
        if tl.startswith("amount per serving") or tl.startswith("% daily value"):
            continue
        # calories
        if tl.startswith("calories"):
            m = re.search(r"(\d+(?:\.\d+)?)", text)
            if m:
                numb = [b for b in row if re.fullmatch(r"\d+(?:\.\d+)?", b.text)]
                P.calories = NFRow("calories", "Calories", float(m.group(1)), "kcal", None, text, bb, numb[0].bbox if numb else None, None, title.panel)
                P.nf_rows["calories"] = P.calories
            continue
        # nutrient rows
        dvm = list(PCT_RE.finditer(text))
        dv = float(dvm[-1].group(1)) if dvm else None
        dv_b = None
        if dvm:
            for b in reversed(row):
                if "%" in b.text:
                    dv_b = b.bbox; break
        t2 = PCT_RE.sub("", text).strip() if dvm else text
        if "added sugars" in tl:
            am = re.search(r"(\d+(?:\.\d+)?)\s*(g)\b", t2, re.I)
            if am:
                key, amt, unit = "added_sugars", float(am.group(1)), "g"
                P.nf_rows[key] = NFRow(key, "Added Sugars", amt, unit, dv, text, bb, locate(art, am.group(0), bb), dv_b, title.panel)
            continue
        am = AMT_RE.search(t2)
        if not am:
            continue
        label = t2[:am.start()].strip()
        key = _nf_key(_norm(label))
        if not key:
            continue
        amt = float(am.group(1)); unit = am.group(2).lower()
        P.nf_rows[key] = NFRow(key, NF_LABEL[key], amt, unit, dv, text, bb, locate(art, am.group(0).replace(" ", ""), bb), dv_b, title.panel)


# --------------------------------------------------------------------------- non-US nutrition strip (back panel "traffic lights")
def parse_strip(art: Artwork, P: Parsed):
    pb = art.print_blocks()
    labs = [b for b in pb if b.norm in ("calories", "fat", "saturates", "sugars", "salt") and b.size <= 6.5 and b.rot == 0]
    if len(labs) < 3:
        return
    # same column
    groups = {}
    for b in labs:
        groups.setdefault(round(((b.bbox[0] + b.bbox[2]) / 2) / 25), []).append(b)
    col = max(groups.values(), key=len)
    if len(col) < 3:
        return
    col.sort(key=lambda b: b.bbox[1])
    x0 = min(b.bbox[0] for b in col) - 8; x1 = max(b.bbox[2] for b in col) + 30
    y0 = col[0].bbox[1] - 2; y1 = col[-1].bbox[3] + 28
    P.strip_region = (x0, y0, x1, y1)
    members = [b for b in pb if x0 <= b.bbox[0] <= x1 and y0 <= b.bbox[1] <= y1 and b not in col and "%" not in b.text and b.text.strip() != "%"
               and not re.fullmatch(r"\d+", b.text) or (b in pb and x0 <= b.bbox[0] <= x1 - 20 and y0 <= b.bbox[1] <= y1 and re.fullmatch(r"\d+(\.\d+)?", b.text) and b.size < 11)]
    members = [b for b in members if b not in col]
    for i, lab in enumerate(col):
        top = lab.bbox[1] - 1
        bot = col[i + 1].bbox[1] - 1 if i + 1 < len(col) else y1
        seg = [b for b in members if top <= (b.bbox[1] + b.bbox[3]) / 2 < bot and b.bbox[0] < x1 - 20 and b.size < 11]
        rows = _rows(seg, 2.5)
        vals = {}
        bbs = []
        for r in rows:
            txt = " ".join(b.text for b in r)
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(kj|kcal|g|mg)\b", txt, re.I):
                vals[m.group(2).lower()] = float(m.group(1))
            bbs += [b.bbox for b in r]
        name = lab.norm
        if vals:
            P.strip[name] = dict(values=vals, bbox=union(bbs + [lab.bbox]), label=lab.text)
    P.strip["_bbox"] = union([b.bbox for b in col] + [b.bbox for b in members]) if members else union([b.bbox for b in col])


# --------------------------------------------------------------------------- statements
HEADINGS = [("ingredients", r"^\s*ingredients?\s*:"), ("allergens", r"^\s*(allergen\s*information|allergens?|contains)\s*:"),
            ("dietary", r"^\s*dietary\s*information\s*:")]


def _paragraph(art: Artwork, pat: str):
    pb = art.print_blocks()
    stops = "|".join(h[1] for h in HEADINGS)
    for b in pb:
        if re.match(pat, b.text, re.I):
            txt, bb = b.text, b.bbox
            body_only = re.sub(pat, "", txt, flags=re.I).strip()
            if b.source == "ocr":
                # OCR gives one block per visual line: stitch the paragraph that follows (same left edge, tight line gap)
                prev = b
                for c in sorted([c for c in pb if c is not b and c.bbox[1] > b.bbox[1]], key=lambda c: c.bbox[1]):
                    lh = max(prev.bbox[3] - prev.bbox[1], 4)
                    if abs(c.bbox[0] - b.bbox[0]) <= 8 and 0 <= c.bbox[1] - prev.bbox[3] <= 1.1 * lh and not re.match(stops, c.text, re.I):
                        txt, bb, prev = txt + " " + c.text, union([bb, c.bbox]), c
                    elif c.bbox[1] - prev.bbox[3] > 2 * lh:
                        break
                return Item(re.sub(r"\s+", " ", txt).strip(), bb, b.panel)
            if len(body_only) < 3:   # heading alone: body is the block starting on the same line
                for c in pb:
                    if c is not b and abs(c.bbox[1] - b.bbox[1]) < 4 and c.bbox[2] > b.bbox[2] + 5 and abs(c.size - b.size) < 0.3 \
                            and not re.match(pat, c.text, re.I):
                        txt, bb = txt + " " + c.text, union([bb, c.bbox]); break
            return Item(re.sub(r"\s+", " ", txt).strip(), bb, b.panel)
    return None


CLAIMS = {
    "no_added_sugar": r"no\s+added\s+sugars?|no\s+sugars?\s+added|unsweetened",
    "gmo_free": r"non[\s-]?gmo|gmo[\s-]?free",
    "oven_baked": r"oven[\s-]?baked",
    "high_grain": r"high[\s-]?grain",
    "organic": r"\borganic\b",
    "vegan": r"\bvegans?\b",
    "vegetarian": r"\bvegetarians?\b",
    "gluten_free": r"gluten[\s-]?free",
    "no_artificial": r"no\s+artificial|without\s+artificial",
    "natural": r"\bnatural\b",
}


def parse_statements(art: Artwork, P: Parsed):
    P.ingredients = _paragraph(art, HEADINGS[0][1])
    P.allergens = _paragraph(art, HEADINGS[1][1])
    d = _paragraph(art, HEADINGS[2][1])
    if d:
        P.dietary.append(d)
    pb = art.print_blocks()
    for b in pb:
        if re.match(r"all our products", b.text, re.I):
            P.dietary.append(Item(b.text, b.bbox, b.panel, meta=dict(kind="footer")))
        if re.match(r"store in a cool", b.text, re.I):
            P.storage = Item(b.text, b.bbox, b.panel)
        if b.text.lower().startswith("start tapping"):
            P.marketing.append(Item(b.text, b.bbox, b.panel))
    # claims
    skip = []
    if P.nf_region: skip.append(P.nf_region)
    for b in pb:
        if P.ingredients and b.bbox == P.ingredients.bbox:
            continue
        for key, pat in CLAIMS.items():
            m = re.search(pat, b.text, re.I)
            if m:
                bb = locate(art, m.group(0), b.bbox) or b.bbox
                P.claims.setdefault(key, []).append(Item(b.text, bb, b.panel, meta=dict(match=m.group(0), block=b.id)))
    # graphic marks recognisable through OCR text (symbols without a text layer)
    P.marks = {}
    for w in art.ocr_words:
        t = _norm(w["text"])
        if t == "usda" and w["conf"] > 60:
            bb = w["bbox"]
            P.marks["usda_organic"] = Item("USDA seal (OCR: 'USDA')", (bb[0] - 14, bb[1] - 14, bb[2] + 18, bb[3] + 34), None, meta=dict(confidence=0.6, source="ocr"))
    for k, it in P.marks.items():
        it.panel = _panel(art, it.bbox)
    # identity: product common name on front panel / PDP
    for b in pb:
        if b.norm in ("granola", "berrycrunch") or "granola" == b.norm:
            P.identity.append(Item(b.text, b.bbox, b.panel, meta=dict(size=b.size)))
    # manufacturer / distributor line
    for b in pb:
        if re.match(r"(produced|manufactured|distributed|packed|made)\s+(by|for)", b.text, re.I):
            P.manufacturer = Item(b.text, b.bbox, b.panel)
            break


def _panel(art, bbox):
    from .extract import panel_of
    return panel_of(bbox, art.panels)


# --------------------------------------------------------------------------- net quantity
QTY_RE = re.compile(r"(?:net\s*(?:wt|weight|contents)\.?:?\s*)?(\d+(?:\.\d+)?)\s*(kg|g|lb|oz|ml|l|fl\.?\s*oz)\b", re.I)


def parse_net_qty(art: Artwork, P: Parsed):
    for b in art.print_blocks():
        if b.size < 8 or b.rot != 0:
            continue
        if P.nf_region and P.nf_region[0] <= b.bbox[0] <= P.nf_region[2] and P.nf_region[1] <= b.bbox[1] <= P.nf_region[3]:
            continue
        if len(b.text) > 40:
            continue
        for m in QTY_RE.finditer(b.text):
            P.net_qty.append(Item(m.group(0), locate(art, m.group(0).replace(" ", ""), b.bbox) or b.bbox, b.panel,
                                  value=float(m.group(1)), meta=dict(unit=m.group(2).lower().replace(" ", ""), size=b.size)))


# --------------------------------------------------------------------------- entry
def parse_artwork(art: Artwork) -> Parsed:
    P = Parsed()
    P.hidden = [b for b in art.blocks if b.kind == "hidden"]
    parse_nutrition(art, P)
    parse_strip(art, P)
    parse_statements(art, P)
    parse_net_qty(art, P)
    if art.barcode:
        P.upc_text = Item(art.barcode["text"], tuple(art.barcode["bbox"]), _panel(art, tuple(art.barcode["bbox"])), meta=art.barcode)
    return P


# --------------------------------------------------------------------------- ingredient list helpers
def split_ingredients(text: str):
    """Top-level comma split (parenthesis aware). Returns [(name, qualifier_text, percent|None)]"""
    t = re.sub(r"^\s*ingredients?\s*:\s*", "", text, flags=re.I).strip()
    t = re.sub(r"\.\s+(?=[A-Z])", ", ", t).rstrip(". ")
    out, depth, cur = [], 0, ""
    for ch in t:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            if cur.strip(): out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    items = []
    for raw in out:
        pct = re.search(r"\((\d+(?:\.\d+)?)\s*%\)", raw)
        name = re.sub(r"\s*\([^)]*\)", "", raw).strip()
        qual = " ".join(re.findall(r"\(([^)]*)\)", raw))
        items.append((name, qual, float(pct.group(1)) if pct else None, raw))
    return items
