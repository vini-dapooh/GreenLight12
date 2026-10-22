"""Artwork ingestion: PDF / image  ->  located text blocks.

Design choices (see docs/ARCHITECTURE.md):
  * Prefer the PDF text layer (exact characters, exact coordinates). OCR is the fallback for raster art
    AND an independent cross-check: text that exists in the file but is not visibly rendered is flagged.
  * Esko/ArtiosCAD dielines carry print-guide text ("Art orientation", "Front panel"...) on separate
    layers. We classify those as `guide` so they never become findings.
  * Panels (front/back/left/right) are inferred from the dieline guide labels so every finding can say
    "front panel" and checks such as "net weight must be on the PDP" are possible.
"""
from __future__ import annotations
import io, math, re
from dataclasses import dataclass, field
from typing import Optional

import pymupdf as fitz
from PIL import Image
from rapidfuzz import fuzz

from .models import Block

GUIDE_LAYER_RE = re.compile(r"dimension|glue|\bcut\b|crease|bleed|3d assist|\bard\b|artwork panel|inkfree|coating|overall|other", re.I)
DESIGN_LAYER_RE = re.compile(r"^(design|main design)$", re.I)
GUIDE_TEXT_RE = re.compile(
    r"^((art|front|back|left|right|bottom|top|side)\s+)?(panel|orientation)$|^principal display panel$|^copy free$|^gl\d$|^\d,\d\w*", re.I)
PANEL_LABEL_RE = re.compile(r"^(front|back|left|right) panel$", re.I)

OCR_DPI = 200


@dataclass
class PageInfo:
    index: int
    width: float
    height: float
    image_png: bytes = b""          # page as the designer sees it (dieline on)
    clean_png: bytes = b""          # guide layers off -> what OCR reads
    view_full: bytes = b""          # high-res JPEGs for the UI viewer (dieline on / off)
    view_clean: bytes = b""
    dpi: int = 130


@dataclass
class Artwork:
    filename: str
    pages: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    panels: dict = field(default_factory=dict)      # name -> bbox
    barcode: Optional[dict] = None                   # {text, format, bbox}
    ocr_words: list = field(default_factory=list)    # [{text, conf, bbox}]
    words: list = field(default_factory=list)        # text-layer words with boxes (for pin-point evidence)
    stats: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)        # human-readable pipeline log

    def print_blocks(self):
        return [b for b in self.blocks if b.kind == "print"]


# ----------------------------------------------------------------------------- helpers
def _hex(color) -> str:
    try:
        if isinstance(color, (int, float)):
            v = int(round(color * 255)); return f"{v:02x}{v:02x}{v:02x}"
        if len(color) == 1:
            v = int(round(color[0] * 255)); return f"{v:02x}{v:02x}{v:02x}"
        if len(color) == 3:
            return "".join(f"{int(round(c * 255)):02x}" for c in color)
        if len(color) == 4:  # CMYK -> rgb
            c, m, y, k = color
            return "".join(f"{int(round(255 * (1 - min(1, v + k)))):02x}" for v in (c, m, y))
    except Exception:
        pass
    return "000000"


def _iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _rot_of(d):
    ang = math.degrees(math.atan2(-d[1], d[0])) % 360
    for q in (0, 90, 180, 270, 360):
        if abs(ang - q) < 4:
            return int(q % 360)
    return -1


def _render(page, dpi, clip=None):
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    return pix.tobytes("png")


VIEW_DPI = 200


def _render_jpg(page, dpi=VIEW_DPI, q=86):
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return pix.tobytes("jpeg", jpg_quality=q)


# ----------------------------------------------------------------------------- spans -> blocks
def _spans_from_page(page):
    spans = []
    for t in page.get_texttrace():
        text = "".join(chr(c[0]) for c in t["chars"])
        if not text.strip():
            continue
        spans.append(dict(text=text, bbox=tuple(t["bbox"]), size=t["size"], color=_hex(t["color"]),
                          layer=t.get("layer") or "", dir=t["dir"], seq=t["seqno"], type=t["type"],
                          opacity=t.get("opacity", 1.0)))
    spans.sort(key=lambda s: s["seq"])
    return spans


def _dedupe(spans):
    """Outlined / shadowed type is drawn 3-5 times at the same place. Keep one."""
    out = []
    for s in spans:
        dup = False
        for o in out[-40:]:
            if o["text"].strip() == s["text"].strip() and _iou(o["bbox"], s["bbox"]) > 0.7:
                dup = True
                # keep the fill colour that is most legible (not white) as representative colour
                if o["color"] in ("ffffff", "fefefe") and s["color"] not in ("ffffff", "fefefe"):
                    o["color"] = s["color"]
                break
        if not dup:
            out.append(s)
    return out


def _gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return math.hypot(dx, dy)


def _merge(spans):
    """Merge glyph fragments into text runs. Returns list of dicts."""
    runs = []
    cur = None

    def flush():
        nonlocal cur
        if cur:
            runs.append(cur); cur = None

    for s in spans:
        rot = _rot_of(s["dir"])
        glyph = len(s["text"].strip()) <= 1
        if cur is not None:
            same_style = abs(cur["size"] - s["size"]) < 0.6 and cur["rot"] == rot and cur["layer"] == s["layer"]
            if same_style and rot == 0:
                same_line = abs(cur["bbox"][3] - s["bbox"][3]) < 1.6
                gap = s["bbox"][0] - cur["bbox"][2]
                if same_line and -2 <= gap <= 0.9 * s["size"]:
                    sep = " " if (gap > 0.22 * s["size"] and not cur["text"].endswith(" ") and not s["text"].startswith(" ")) else ""
                    cur["text"] += sep + s["text"]
                    cur["bbox"] = union([cur["bbox"], s["bbox"]])
                    continue
        if cur is not None and glyph and cur["glyph"] and abs(cur["size"] - s["size"]) < 0.8 and cur["layer"] == s["layer"] \
                and (rot != 0 or cur["rot"] != 0):
            lx, ly = (cur["last"][0] + cur["last"][2]) / 2, (cur["last"][1] + cur["last"][3]) / 2
            sx, sy = (s["bbox"][0] + s["bbox"][2]) / 2, (s["bbox"][1] + s["bbox"][3]) / 2
            dist = math.hypot(sx - lx, sy - ly)
            if dist < 2.2 * s["size"]:
                cur["text"] += (" " if dist > 1.35 * s["size"] else "") + s["text"]
                cur["bbox"] = union([cur["bbox"], s["bbox"]]); cur["last"] = s["bbox"]; cur["rot"] = -1
                continue
        flush()
        cur = dict(text=s["text"], bbox=s["bbox"], size=s["size"], color=s["color"], layer=s["layer"],
                   rot=rot, glyph=glyph, last=s["bbox"], type=s["type"])
    flush()
    for r in runs:
        r["text"] = re.sub(r"\s+", " ", r["text"]).strip()
    return [r for r in runs if r["text"]]


def _classify_guides(runs):
    guide_colors = set()
    for r in runs:
        if GUIDE_TEXT_RE.match(r["text"]) and not DESIGN_LAYER_RE.match(r["layer"] or "x"):
            guide_colors.add(r["color"])
    for r in runs:
        layer = r["layer"]
        by_layer = bool(layer) and not DESIGN_LAYER_RE.match(layer) and bool(GUIDE_LAYER_RE.search(layer))
        by_text = bool(GUIDE_TEXT_RE.match(r["text"])) and (not layer or by_layer or r["color"] in guide_colors)
        r["kind"] = "guide" if (by_layer or by_text) else "print"
    return runs


# ----------------------------------------------------------------------------- panels
def _infer_panels(blocks, page_w, page_h):
    """Use dieline panel labels ("Front panel", ...) to partition the sheet."""
    labels = [b for b in blocks if b.kind == "guide" and PANEL_LABEL_RE.match(b.text) and b.rot == 0]
    if len(labels) < 2:
        return {}
    # main row = the y with most labels
    rows = {}
    for b in labels:
        rows.setdefault(round(b.bbox[1] / 40), []).append(b)
    main = max(rows.values(), key=len)
    main.sort(key=lambda b: b.bbox[0])
    centers = [((b.bbox[0] + b.bbox[2]) / 2, b.text.split()[0].capitalize()) for b in main]
    if len(centers) < 2:
        return {}
    spacing = sorted(b - a for (a, _), (b, _) in zip(centers, centers[1:]))[len(centers) // 2 - 1 if len(centers) > 2 else 0]
    ycenter = sum((b.bbox[1] + b.bbox[3]) / 2 for b in main) / len(main) + 0.05 * spacing
    half_h = 0.72 * spacing
    panels = {}
    for i, (cx, name) in enumerate(centers):
        x0 = cx - spacing / 2 if i == 0 else (centers[i - 1][0] + cx) / 2
        x1 = cx + spacing / 2 if i == len(centers) - 1 else (centers[i + 1][0] + cx) / 2
        pname = f"{name} panel" + (" (principal display panel)" if name == "Front" else "")
        panels[pname] = (x0, ycenter - half_h, x1, ycenter + half_h)
    # flaps above / below the panel row
    gx0 = min(v[0] for v in panels.values()); gx1 = max(v[2] for v in panels.values())
    panels["Top flaps"] = (gx0, 0, gx1, ycenter - half_h)
    panels["Bottom flaps"] = (gx0, ycenter + half_h, gx1, page_h)
    return panels



def _panels_from_creases(page, blocks):
    """Exact panel boxes from the dieline's crease lines (vertical creases = panel edges), named by the guide labels."""
    labels = [b for b in blocks if b.kind == "guide" and PANEL_LABEL_RE.match(b.text) and b.rot == 0]
    if not labels:
        return {}
    vx, hy = [], []
    for d in page.get_drawings():
        lay = d.get("layer") or ""
        if "crease" not in lay.lower():
            continue
        r = d["rect"]
        if r.width < 2 and r.height > 100:
            vx.append((round(r.x0), r.y0, r.y1))
        elif r.height < 2 and r.width > 100:
            hy.append(round(r.y0))
    if len(vx) < 3 or not hy:
        return {}
    ly = sorted(((b.bbox[1] + b.bbox[3]) / 2 for b in labels))[len(labels) // 2]
    ly = max(((b.bbox[1] + b.bbox[3]) / 2 for b in labels if "panel" in b.text.lower()), default=ly)
    above = [y for y in hy if y < ly]; below = [y for y in hy if y > ly]
    if not above or not below:
        return {}
    y0, y1 = max(above), min(below)
    xs = sorted({x for x, a, b in vx if a <= y0 + 5 and b >= y1 - 5})
    # merge near-duplicates
    merged = []
    for x in xs:
        if not merged or x - merged[-1] > 3:
            merged.append(x)
    if len(merged) < 3:
        return {}
    panels = {}
    main_labels = [b for b in labels if y0 <= (b.bbox[1] + b.bbox[3]) / 2 <= y1]
    for a, b in zip(merged, merged[1:] + [page.rect.width]):
        lab = [l for l in main_labels if a <= (l.bbox[0] + l.bbox[2]) / 2 <= b]
        if not lab:
            continue
        name = lab[0].text.split()[0].capitalize() + " panel"
        if name.startswith("Front"):
            name += " (principal display panel)"
        panels[name] = (float(a), float(y0), float(b), float(y1))
    if len(panels) < 2:
        return {}
    gx0, gx1 = min(v[0] for v in panels.values()), max(v[2] for v in panels.values())
    panels["Top flaps"] = (gx0, 0.0, gx1, float(y0))
    panels["Bottom flaps"] = (gx0, float(y1), gx1, float(page.rect.height))
    return panels


def panel_of(bbox, panels):
    if not panels:
        return None
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for name, (x0, y0, x1, y1) in panels.items():
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return name
    return None


# ----------------------------------------------------------------------------- OCR
def _ocr(img: Image.Image, dpi: int, min_conf=40):
    import pytesseract
    data = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
    s = 72.0 / dpi
    words = []
    for i, t in enumerate(data["text"]):
        t = t.strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1
        if t and conf >= min_conf:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append(dict(text=t, conf=conf, bbox=(x * s, y * s, (x + w) * s, (y + h) * s)))
    return words


def _ocr_blocks(words):
    """Group OCR words into lines (used when there is no text layer at all)."""
    words = sorted(words, key=lambda w: (round(w["bbox"][1] / 4), w["bbox"][0]))
    lines, cur = [], []
    for w in words:
        if cur and abs(w["bbox"][1] - cur[-1]["bbox"][1]) < 4 and w["bbox"][0] - cur[-1]["bbox"][2] < 14:
            cur.append(w)
        else:
            if cur: lines.append(cur)
            cur = [w]
    if cur: lines.append(cur)
    return lines


def _norm(t):
    return "".join(ch for ch in t.lower() if ch.isalnum())


def _verify_visibility(blocks, ocr_words, text_words):
    """Position-accurate check: is every text-layer word actually *seen* by OCR at that spot?
    support = % of the block's words that OCR finds within a few points of where the PDF says they are."""
    for b in blocks:
        if b.kind != "print" or b.rot != 0 or len(b.norm) < 3 or not (3 <= b.size <= 30):
            continue
        x0, y0, x1, y1 = b.bbox
        bcy, ytol = (y0 + y1) / 2, 0.2 * (y1 - y0) + 0.8
        region = [w for w in text_words if x0 - 1 <= (w["bbox"][0] + w["bbox"][2]) / 2 <= x1 + 1
                  and abs((w["bbox"][1] + w["bbox"][3]) / 2 - bcy) <= ytol]
        considered, hit = 0, 0
        for tok in b.text.split():
            nt = _norm(tok)
            if not nt:
                continue
            cands = [w for w in region if _norm(w["text"]) == nt]
            if not cands:
                continue
            considered += 1
            ok = False
            for w in cands:
                wx, wy = (w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2
                tol = max(3.0, 0.7 * (w["bbox"][3] - w["bbox"][1]))
                for o in ocr_words:
                    ox, oy = (o["bbox"][0] + o["bbox"][2]) / 2, (o["bbox"][1] + o["bbox"][3]) / 2
                    if abs(ox - wx) <= tol + 0.25 * (w["bbox"][2] - w["bbox"][0]) and abs(oy - wy) <= tol:
                        no = _norm(o["text"])
                        if no and (fuzz.ratio(no, nt) >= 70 or (len(no) >= 2 and no in nt)):
                            ok = True; break
                if ok:
                    break
            hit += ok
        if considered == 0:
            # text layer stores this run as loose letters (letter-spaced type): compare whole-string instead
            reg = "".join(_norm(o["text"]) for o in sorted(ocr_words, key=lambda o: o["bbox"][0])
                          if x0 - 1 <= (o["bbox"][0] + o["bbox"][2]) / 2 <= x1 + 1 and y0 - 1 <= (o["bbox"][1] + o["bbox"][3]) / 2 <= y1 + 1)
            sup = float(fuzz.ratio(b.norm, reg)) if reg else 0.0
            # the first token must also START where the PDF says it does (catches near-duplicate rows)
            first = _norm(b.text.split()[0]) if b.text.split() else ""
            cy = (y0 + y1) / 2
            aligned = any(abs(o["bbox"][0] - x0) <= 4 and abs((o["bbox"][1] + o["bbox"][3]) / 2 - cy) <= 4
                          and (fuzz.ratio(_norm(o["text"]), first) >= 70 or first.startswith(_norm(o["text"])))
                          for o in ocr_words if _norm(o["text"]))
            b.ocr_support = sup if aligned else min(sup, 20.0)
            continue
        b.ocr_support = 100.0 * hit / considered


def _page_words(page):
    seen, out = set(), []
    for w in page.get_text("words"):
        key = (w[4], round(w[0]), round(w[1]))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(text=w[4], bbox=(w[0], w[1], w[2], w[3])))
    return out


def _overlap_frac(a, b):
    """how much of a's height is covered by b (only if their x-ranges genuinely intersect)"""
    xi = min(a[2], b[2]) - max(a[0], b[0])
    if xi < 10:
        return 0.0
    yi = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, yi) / max(1e-6, a[3] - a[1])


def _find_occluded(blocks):
    """A block is 'hidden' when it sits under another block AND the OCR sees the other one, not this one.
    (Requiring an overlapping, better-supported neighbour avoids false alarms from OCR misses on tiny or
    reversed-out type.)"""
    cand = [b for b in blocks if b.kind == "print" and b.ocr_support is not None]
    out = []
    for b in cand:
        if b.ocr_support >= 50:
            continue
        for a in cand:
            if a is b:
                continue
            if _overlap_frac(b.bbox, a.bbox) >= 0.3 and a.ocr_support > b.ocr_support + 20:
                out.append(b); break
    # propagate along the same column of letter-spaced rows (same size, same left edge)
    if len(out) >= 3:
        keys = {(round(b.size, 1), round(b.bbox[0])) for b in out}
        ys = (min(b.bbox[1] for b in out) - 12, max(b.bbox[3] for b in out) + 12)
        for b in cand:
            if b not in out and (round(b.size, 1), round(b.bbox[0])) in keys and ys[0] <= b.bbox[1] <= ys[1] and b.ocr_support < 90:
                out.append(b)
    return out


# ----------------------------------------------------------------------------- barcode
def _decode_barcode(page, dpi=300):
    try:
        import zxingcpp
    except Exception:
        return None
    rect = None
    for d in page.get_drawings():
        if (d.get("layer") or "").upper().startswith("UPC"):
            rect = d["rect"] if rect is None else rect | d["rect"]
    clips = [fitz.Rect(rect) + (-10, -10, 10, 10)] if rect is not None else [page.rect]
    for clip in clips:
        img = Image.open(io.BytesIO(_render(page, dpi, clip)))
        res = zxingcpp.read_barcodes(img)
        for r in res:
            if rect is not None:
                bb = tuple(rect)
            else:
                xs = [p.x for p in (r.position.top_left, r.position.bottom_right)]
                ys = [p.y for p in (r.position.top_left, r.position.bottom_right)]
                bb = (min(xs) * 72 / dpi, min(ys) * 72 / dpi, max(xs) * 72 / dpi, max(ys) * 72 / dpi)
            text = r.text
            fmt = str(r.format).split(".")[-1]
            if "EAN" in fmt and "13" in fmt and text.startswith("0") and len(text) == 13:
                text, fmt = text[1:], "UPC-A"
            return dict(text=text, format=fmt, bbox=bb)
    return None


# ----------------------------------------------------------------------------- public API
def load_artwork(path: str, filename: str | None = None, ocr: bool = True) -> Artwork:
    filename = filename or path.split("/")[-1]
    art = Artwork(filename=filename)
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext in ("png", "jpg", "jpeg", "tif", "tiff", "webp"):
        return _load_image(path, art, ocr)
    doc = fitz.open(path)
    page = doc[0]  # artwork sheets are single-page; multi-page support = loop (kept simple for the prototype)
    art.trace.append(f"Opened PDF: {len(doc)} page(s), {page.rect.width:.0f}x{page.rect.height:.0f} pt, "
                     f"{len(doc.get_ocgs())} layers")

    full_png = _render(page, 130)
    view_full = _render_jpg(page)
    spans = _dedupe(_spans_from_page(page))      # read the text layer BEFORE hiding any layers
    art.words = _page_words(page)

    runs = _classify_guides(_merge(spans))
    for i, r in enumerate(runs):
        art.blocks.append(Block(id=f"b{i}", text=r["text"], bbox=r["bbox"], page=0, size=round(r["size"], 1),
                                color=r["color"], layer=r["layer"], rot=r["rot"], kind=r["kind"]))
    # panel geometry comes from dieline vectors, so read it while the dieline layers are still on
    art.panels = _panels_from_creases(page, art.blocks) or _infer_panels(art.blocks, page.rect.width, page.rect.height)
    for b in art.blocks:
        b.panel = panel_of(b.bbox, art.panels)

    # clean render: dieline layers off, so neither OCR nor the reviewer is distracted by guide marks
    toggled = 0
    for c in doc.layer_ui_configs():
        if GUIDE_LAYER_RE.search(c["text"]) and not DESIGN_LAYER_RE.match(c["text"]):
            doc.set_layer_ui_config(c["number"], 2); toggled += 1
    page = doc[0]
    clean_png = _render(page, 130) if toggled else full_png
    view_clean = _render_jpg(page) if toggled else view_full
    clean_ocr_png = _render(page, OCR_DPI)
    art.pages.append(PageInfo(0, page.rect.width, page.rect.height, full_png, clean_png, view_full, view_clean, 130))
    if toggled:
        art.trace.append(f"Dieline/guide layers switched off for a clean render ({toggled} layers)")
    n_print = len(art.print_blocks())
    art.trace.append(f"Text layer: {len(spans)} unique glyph runs -> {len(runs)} text blocks "
                     f"({n_print} printable, {len(runs) - n_print} dieline-guide text ignored)")
    if art.panels:
        art.trace.append("Panels located from dieline crease lines + guide labels: " + ", ".join(k for k in art.panels if "flaps" not in k))

    if n_print < 5:  # outlined/raster artwork: OCR becomes the primary source
        art.trace.append("Text layer is (nearly) empty: falling back to OCR as primary source")
        words = _ocr(Image.open(io.BytesIO(clean_ocr_png)), OCR_DPI) if ocr else []
        art.ocr_words = words
        for j, line in enumerate(_ocr_blocks(words)):
            bb = union([w["bbox"] for w in line])
            art.blocks.append(Block(id=f"o{j}", text=" ".join(w["text"] for w in line), bbox=bb, source="ocr",
                                    size=round(bb[3] - bb[1], 1)))
    elif ocr:
        try:
            words = _ocr(Image.open(io.BytesIO(clean_ocr_png)), OCR_DPI)
            art.ocr_words = words
            _verify_visibility(art.blocks, words, art.words)
            hidden = _find_occluded(art.blocks)
            for b in hidden:
                b.kind = "hidden"
            art.trace.append(f"OCR cross-check: {len(words)} words read from the rendered page; "
                             f"{len(hidden)} text-layer blocks are NOT visible in the render (covered / non-printing)")
        except Exception as e:  # tesseract missing etc.
            art.trace.append(f"OCR unavailable ({e}); visibility cross-check skipped")

    art.barcode = _decode_barcode(page)
    if art.barcode:
        art.trace.append(f"Barcode decoded from bars: {art.barcode['format']} {art.barcode['text']}")
    art.stats = dict(blocks=len(art.blocks), printable=len([b for b in art.blocks if b.kind == "print"]),
                     guide=len([b for b in art.blocks if b.kind == "guide"]),
                     hidden=len([b for b in art.blocks if b.kind == "hidden"]), ocr_words=len(art.ocr_words))
    return art


def _load_image(path, art: Artwork, ocr=True) -> Artwork:
    img = Image.open(path).convert("RGB")
    dpi = 150                                  # assumed print resolution for converting pixels to points
    buf = io.BytesIO(); img.save(buf, "PNG")
    s = 72.0 / dpi
    jb = io.BytesIO(); img.save(jb, "JPEG", quality=88)
    art.pages.append(PageInfo(0, img.width * s, img.height * s, buf.getvalue(), buf.getvalue(), jb.getvalue(), jb.getvalue(), dpi))
    art.trace.append(f"Opened raster image {img.width}x{img.height}px; no text layer, so every word comes from OCR (lower reliability)")
    f = max(1.0, 3600 / max(img.size))
    big = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS) if f > 1.05 else img
    words = _ocr(big, dpi * (big.width / img.width)) if ocr else []
    art.ocr_words = words
    for j, line in enumerate(_ocr_blocks(words)):
        bb = union([w["bbox"] for w in line])
        art.blocks.append(Block(id=f"o{j}", text=" ".join(w["text"] for w in line), bbox=bb, source="ocr", size=round(bb[3] - bb[1], 1)))
    art.words = [dict(text=w["text"], bbox=w["bbox"]) for w in words]
    art.stats = dict(blocks=len(art.blocks), printable=len(art.blocks), guide=0, hidden=0, ocr_words=len(words))
    art.trace.append(f"OCR read {len(words)} words into {len(art.blocks)} text lines")
    return art
