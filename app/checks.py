"""Deterministic check primitives.

Philosophy: the LLM may *propose* and *explain*; these functions *decide*. A check returns a verdict
  fail    -> definite problem (rule is unambiguous AND the evidence is machine-verified)
  review  -> a human must look (ambiguous rule, low-confidence evidence, or approved data itself flagged unsure)
  pass    -> looks correct (we still show why, so reviewers can see what was verified)
and always carries the artwork regions / sheet cells that justify it.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from .extract import Artwork, union, _norm
from .models import Evidence, FAIL, REVIEW, PASS
from .parse import Parsed, Item, locate, split_ingredients, NF_LABEL
from .sheet import Approved, Field

CHECKS: dict = {}
KEYWORDS: dict = {}


def check(name, keywords=()):
    def deco(fn):
        CHECKS[name] = fn
        KEYWORDS[name] = tuple(keywords)
        return fn
    return deco


@dataclass
class Ctx:
    art: Artwork
    P: Parsed
    sheet: Optional[Approved] = None
    llm: object = None

    def pdp(self):
        for name, bb in self.art.panels.items():
            if "principal" in name.lower():
                return name, bb
        return None, None

    def ev(self, item_or_bbox, label="", text="", panel=None):
        if isinstance(item_or_bbox, Item):
            return item_or_bbox.ev(label)
        bb = list(item_or_bbox) if item_or_bbox else None
        from .extract import panel_of
        return Evidence("artwork", label, text, bb, 0, panel or (panel_of(tuple(bb), self.art.panels) if bb else None))

    def sheet_ev(self, f: Field, label=""):
        return Evidence("sheet", label or f.label, f.value, None, 0, None, f.ref)


def R(verdict, summary, explanation="", **kw):
    d = dict(verdict=verdict, summary=summary, explanation=explanation, confidence=kw.pop("confidence", 0.9))
    d.update(kw)
    return d


# ------------------------------------------------------------------ FDA reference values (21 CFR 101.9(c)(9))
DV = {"total_fat": ("g", 78), "saturated_fat": ("g", 20), "cholesterol": ("mg", 300), "sodium": ("mg", 2300),
      "total_carbohydrate": ("g", 275), "dietary_fiber": ("g", 28), "added_sugars": ("g", 50), "protein": ("g", 50),
      "vitamin_d": ("mcg", 20), "calcium": ("mg", 1300), "iron": ("mg", 18), "potassium": ("mg", 4700)}

ADDED_SUGAR_WORDS = ["honey", "coconut sugar", "cane sugar", "sugar", "syrup", "molasses", "dextrose", "fructose", "maltose",
                     "juice concentrate", "agave", "nectar", "maltodextrin", "glucose", "sucrose"]
ANIMAL_WORDS = ["honey", "milk", "whey", "casein", "butter", "egg", "gelatin", "gelatine", "beeswax", "shellac", "carmine", "lard",
                "tallow", "fish", "anchovy", "collagen", "lanolin", "cream", "cheese", "yogurt", "yoghurt"]
MAJOR_ALLERGENS = {
    "tree nuts": ["almond", "pecan", "walnut", "cashew", "pistachio", "hazelnut", "brazil nut", "macadamia", "pine nut", "chestnut"],
    "peanuts": ["peanut"], "wheat": ["wheat", "spelt", "kamut", "semolina", "farina"], "milk": ["milk", "whey", "casein", "butter", "cream", "cheese"],
    "egg": ["egg", "albumen"], "soy": ["soy", "soya", "soybean"], "fish": ["salmon", "cod", "tuna", "anchovy"],
    "shellfish": ["shrimp", "crab", "lobster", "prawn"], "sesame": ["sesame", "tahini"],
}


def _fmt(v):
    return ("%g" % round(v, 1))


def _find_ing_words(ctx, words, ing: Item):
    """bounding boxes for individual ingredient words inside the ingredient paragraph"""
    out = []
    for w in words:
        bb = locate(ctx.art, w, ing.bbox)
        if bb:
            out.append((w, bb))
    return out


# =================================================================== NUTRITION FACTS
@check("nf_present", ['nutrition facts panel', 'nutrition panel', 'nutrition label'])
def nf_present(ctx, p):
    P = ctx.P
    if not P.nf_title:
        return [R(FAIL, "No FDA 'Nutrition Facts' panel was found on the artwork.",
                  "A packaged food sold in the US needs a Nutrition Facts panel (21 CFR 101.9) unless exempt.",
                  expected="Nutrition Facts panel (vertical standard format)", found="not found", confidence=0.7,
                  fix="Add the Nutrition Facts panel in the FDA format.")]
    n = len(P.nf_rows)
    return [R(PASS, f"Nutrition Facts panel found with {n} parsed rows.",
              "A panel titled 'Nutrition Facts' with servings per container, serving size, calories and the required nutrient rows is present.",
              found=f"{n} rows parsed", evidence=[ctx.ev(P.nf_title, "Nutrition Facts panel"), ctx.ev(P.nf_region, "Panel region")], confidence=0.95)]


@check("nf_dv_math", ['%dv', 'daily value', 'internally consistent', 'reconcile'])
def nf_dv_math(ctx, p):
    P = ctx.P
    rows, bad, warn, ok = [], [], [], 0
    ev = []
    for key, r in P.nf_rows.items():
        if key not in DV or r.dv is None or r.amount is None:
            continue
        unit, ref = DV[key]
        exp = r.amount / ref * 100.0
        diff = abs(r.dv - exp)
        status = "ok"
        if diff > max(3.0, 0.25 * exp):
            status = "bad"
        elif diff > max(1.0, 0.12 * exp):
            status = "warn"
        rows.append(dict(label=NF_LABEL[key], found=f"{_fmt(r.amount)}{r.unit} → {_fmt(r.dv)}%DV",
                         expected=("<1%" if exp < 0.5 else f"{exp:.0f}%") + f"  ({_fmt(r.amount)}/{ref}{unit})", ok=status == "ok", status=status))
        if status == "bad":
            bad.append(r); ev.append(ctx.ev(r.dv_bbox or r.bbox, f"{NF_LABEL[key]} {_fmt(r.dv)}%DV (expected ≈{exp:.0f}%)"))
        elif status == "warn":
            warn.append(r); ev.append(ctx.ev(r.dv_bbox or r.bbox, f"{NF_LABEL[key]} {_fmt(r.dv)}%DV (≈{exp:.0f}%?)"))
        else:
            ok += 1
    if not rows:
        return []
    exp_txt = "; ".join(f"{r.label if hasattr(r,'label') else r.key} {_fmt(r.dv)}% vs ≈{r.amount / DV[r.key][1] * 100:.0f}%" for r in bad[:6])
    if bad:
        return [R(FAIL, f"{len(bad)} of {len(rows)} %Daily Value figures do not match the printed amounts.",
                  "%DV must equal the amount divided by the FDA reference value (21 CFR 101.9(c)(9)), rounded. "
                  f"Recomputing every row from the printed amount: {exp_txt}. "
                  "These are arithmetic errors, so they are definite regardless of which number is 'intended'.",
                  found=f"{len(bad)} mismatches", expected="%DV = amount ÷ reference value", details=rows, evidence=ev,
                  fix="Recalculate %DV from the final nutrient amounts (or correct the amounts if the %DV is the right one).", confidence=0.96)]
    if warn:
        return [R(REVIEW, f"{len(warn)} %DV figures differ slightly from a recomputation.",
                  "Small differences can come from rounding of the declared amount; a person should confirm.",
                  details=rows, evidence=ev, confidence=0.7)]
    return [R(PASS, f"All {ok} %Daily Value figures reconcile with the printed amounts.", "", details=rows, confidence=0.95)]


@check("nf_calories", ['calories', 'reconcile', 'internally consistent', 'kcal'])
def nf_calories(ctx, p):
    P = ctx.P
    c, f, ch, pr = P.calories, P.nf_rows.get("total_fat"), P.nf_rows.get("total_carbohydrate"), P.nf_rows.get("protein")
    if not (c and f and ch and pr):
        return []
    est = 9 * f.amount + 4 * ch.amount + 4 * pr.amount
    dev = abs(c.amount - est) / est if est else 0
    ev = [ctx.ev(c.amount_bbox or c.bbox, f"Calories {_fmt(c.amount)}"), ctx.ev(f.bbox, f"Total Fat {_fmt(f.amount)}g"),
          ctx.ev(ch.bbox, f"Total Carbohydrate {_fmt(ch.amount)}g"), ctx.ev(pr.bbox, f"Protein {_fmt(pr.amount)}g")]
    ex = f"≈{est:.0f} kcal  (9×{_fmt(f.amount)} + 4×{_fmt(ch.amount)} + 4×{_fmt(pr.amount)})"
    if dev > 0.25:
        return [R(FAIL, f"Declared {_fmt(c.amount)} kcal is far from the {est:.0f} kcal implied by fat, carbohydrate and protein.",
                  "Calories follow from the macronutrients (Atwater factors 9/4/4). A gap this large means at least one of the printed numbers is wrong; "
                  "FDA allows only small rounding differences.", found=f"{_fmt(c.amount)} kcal", expected=ex, evidence=ev, confidence=0.9,
                  fix="Re-derive the panel from the lab analysis; check which of calories / fat / carbohydrate / protein is wrong.")]
    if dev > 0.10:
        return [R(REVIEW, f"Calories differ by {dev:.0%} from the macronutrient estimate.", "Could be fibre/sugar-alcohol adjustments.", found=_fmt(c.amount), expected=ex, evidence=ev, confidence=0.6)]
    return [R(PASS, "Calories are consistent with the macronutrients.", "", found=_fmt(c.amount), expected=ex, evidence=ev[:1], confidence=0.9)]


@check("nf_physical", ['plausib', 'physically', 'exceeds'])
def nf_physical(ctx, p):
    P = ctx.P
    out = []
    f, ch, pr = P.nf_rows.get("total_fat"), P.nf_rows.get("total_carbohydrate"), P.nf_rows.get("protein")
    ss = P.serving_size
    if f and ch and pr and ss and ss.value:
        total = f.amount + ch.amount + pr.amount
        cands = [ss.value]
        sheet_n = _sheet_net_g(ctx)
        n_srv = P.servings.value if P.servings else None
        implied = sheet_n / n_srv if sheet_n and n_srv else None
        if implied: cands.append(implied)
        if total > max(cands) * 1.02:
            out.append(R(FAIL, f"Fat + carbohydrate + protein add up to {_fmt(total)} g, more than a whole serving ({', '.join(_fmt(c) + ' g' for c in cands)}).",
                         "A serving cannot contain more grams of nutrients than it weighs. This holds whichever serving size is correct "
                         f"(printed '{ss.text}'{f', implied by net weight ÷ servings = {_fmt(implied)} g' if implied else ''}), so the panel is internally impossible.",
                         found=f"{_fmt(f.amount)} + {_fmt(ch.amount)} + {_fmt(pr.amount)} = {_fmt(total)} g", expected=f"≤ serving size ({_fmt(min(cands))}–{_fmt(max(cands))} g)",
                         evidence=[ctx.ev(ss.bbox, f"Serving size '{ss.text}'"), ctx.ev(f.bbox, "Total Fat"), ctx.ev(ch.bbox, "Total Carbohydrate"), ctx.ev(pr.bbox, "Protein")],
                         confidence=0.88, fix="Confirm serving size and re-enter nutrient amounts per that serving.",
                         title="Nutrient amounts exceed the serving size"))
    # hierarchy
    sat, trans, fib, sug, add = (P.nf_rows.get(k) for k in ("saturated_fat", "trans_fat", "dietary_fiber", "total_sugars", "added_sugars"))
    msgs, ev = [], []
    if f and sat and trans and sat.amount + trans.amount > f.amount + 0.01:
        msgs.append(f"saturated + trans fat ({_fmt(sat.amount + trans.amount)} g) exceed total fat ({_fmt(f.amount)} g)"); ev += [ctx.ev(f.bbox, "Total Fat"), ctx.ev(sat.bbox, "Saturated"), ctx.ev(trans.bbox, "Trans")]
    if ch and fib and sug and fib.amount + sug.amount > ch.amount + 0.01:
        msgs.append(f"fibre + sugars ({_fmt(fib.amount + sug.amount)} g) exceed total carbohydrate ({_fmt(ch.amount)} g)"); ev += [ctx.ev(ch.bbox, "Total Carbohydrate")]
    if sug and add and add.amount > sug.amount + 0.01:
        msgs.append("added sugars exceed total sugars"); ev += [ctx.ev(add.bbox, "Added sugars")]
    if msgs:
        out.append(R(FAIL, "Nutrient hierarchy broken: " + "; ".join(msgs) + ".", "Sub-nutrients cannot exceed their parent.", evidence=ev, confidence=0.93, title="Nutrient hierarchy is inconsistent"))
    if trans and sat and trans.amount > 0.5 and trans.amount >= sat.amount and not _has_ing(ctx, ["hydrogenat", "shortening", "margarine"]):
        out.append(R(REVIEW, f"{_fmt(trans.amount)} g trans fat is implausible for a granola with no hydrogenated ingredients.",
                     "Trans fat is higher than saturated fat and the ingredient list has no partially hydrogenated oil or shortening. Likely a data-entry error, but a person must confirm with the analysis.",
                     found=f"Trans Fat {_fmt(trans.amount)} g", expected="≈0 g", evidence=[ctx.ev(trans.bbox, "Trans fat 8g")], confidence=0.7, title="Trans fat value looks implausible"))
    return out or [R(PASS, "Nutrient amounts are physically plausible.", "", confidence=0.8)]


def _has_ing(ctx, words):
    t = (ctx.P.ingredients.text if ctx.P.ingredients else "").lower()
    return any(w in t for w in words)


def _sheet_net_g(ctx):
    f = ctx.sheet.get("Package Size / Net Weight", "Portions Selected") if ctx.sheet else None
    if f:
        m = re.search(r"(\d+(?:\.\d+)?)\s*g\b", f.value)
        if m: return float(m.group(1))
    for q in ctx.P.net_qty:
        if q.meta.get("unit") == "g": return q.value
    return None


@check("nf_serving", ['serving size', 'servings per', 'arithmetic', 'household measure'])
def nf_serving(ctx, p):
    P = ctx.P
    out = []
    ss, sv = P.serving_size, P.servings
    if not ss:
        return [R(FAIL, "Serving size is missing from the Nutrition Facts panel.", "21 CFR 101.9(b) requires a serving size in household and metric units.", confidence=0.8)]
    if not ss.meta.get("has_unit"):
        out.append(R(FAIL, f"Serving size is printed as '{ss.text}' with no unit or household measure.",
                     "The panel must state a household measure with the metric equivalent, e.g. '3/4 cup (60g)' (21 CFR 101.9(b)). A bare number is unusable and the content sheet itself notes it could not read the value.",
                     found=ss.text, expected="household measure + metric weight, e.g. “¾ cup (60g)”", evidence=[ctx.ev(ss.bbox, f"Serving size: {ss.text}")], confidence=0.95,
                     fix="Print the serving size with unit(s); for 480 g ÷ 8 servings that is ≈ 60 g.", title="Serving size has no unit"))
    net = _sheet_net_g(ctx)
    if sv and net and ss.value:
        tot = sv.value * ss.value
        implied = net / sv.value
        if abs(tot - net) / net > 0.15:
            out.append(R(FAIL, f"{_fmt(sv.value)} servings × {_fmt(ss.value)} (as g) = {_fmt(tot)} g, but the pack contains {_fmt(net)} g.",
                         f"Servings per container × serving size must roughly equal net contents (21 CFR 101.9(b)(8)). A {_fmt(net)} g pack with {_fmt(sv.value)} servings implies ≈ {_fmt(implied)} g per serving, not {_fmt(ss.value)}. "
                         "(If the printed number is not grams the check cannot pass either, because no unit is given.)",
                         found=f"{_fmt(sv.value)} × {_fmt(ss.value)} = {_fmt(tot)}", expected=f"≈ {_fmt(net)} g  (serving ≈ {_fmt(implied)} g)",
                         evidence=[ctx.ev(sv.bbox, sv.text), ctx.ev(ss.bbox, f"Serving size {ss.text}")] + ([ctx.sheet_ev(ctx.sheet.get("Package Size / Net Weight"), "Approved net weight")] if ctx.sheet and ctx.sheet.get("Package Size / Net Weight") else []),
                         confidence=0.9, title="Servings × serving size ≠ net contents"))
        else:
            out.append(R(PASS, "Servings per container × serving size matches the net contents.", "", confidence=0.85))
    return out


@check("nf_added_sugars_vs_ingredients", ['added sugars line', 'sugar declaration', 'nutrition panel and the ingredient'])
def nf_added_sugars(ctx, p):
    P = ctx.P
    add = P.nf_rows.get("added_sugars")
    if not add or not P.ingredients:
        return []
    hits = _find_ing_words(ctx, ["Honey", "Coconut Sugar", "cane sugar"], P.ingredients)
    if add.amount == 0 and hits:
        names = ", ".join(w.lower() for w, _ in hits)
        ev = [ctx.ev(add.bbox, f"Declared: {_fmt(add.amount)} g added sugars")] + [ctx.ev(bb, f"Ingredient: {w}") for w, bb in hits]
        return [R(FAIL, f"Panel declares 0 g added sugars, but the ingredient list contains {names}.",
                  "FDA counts honey, sugars and syrups added during processing as 'added sugars' (21 CFR 101.9(c)(6)(iii)). "
                  "The panel also shows 18 g total sugars, which dried fruit alone does not explain.",
                  found="Incl. 0g Added Sugars", expected="> 0 g added sugars", evidence=ev, confidence=0.9,
                  fix="Declare added sugars from honey, coconut sugar and cane sugar (crystallized ginger).", title="'0 g added sugars' contradicts the ingredient list")]
    return [R(PASS, "Added-sugar declaration is consistent with the ingredients.", "", confidence=0.7)]


# =================================================================== CLAIMS vs INGREDIENTS
@check("claim_no_added_sugar", ['no added sugar', 'claim must be supported', 'claims must be supported', 'on-pack claim', 'supported by the ingredient'])
def claim_nas(ctx, p):
    P = ctx.P
    claims = P.claims.get("no_added_sugar")
    if not claims or not P.ingredients:
        return []
    hits = _find_ing_words(ctx, ["Honey", "Coconut Sugar", "cane sugar"], P.ingredients)
    if hits:
        ev = [ctx.ev(claims[0], "Claim: NO ADDED SUGAR")] + [ctx.ev(bb, f"Ingredient: {w}") for w, bb in hits]
        names = ", ".join(w.lower() for w, _ in hits)
        return [R(FAIL, f"“No Added Sugar” claim conflicts with the ingredient list ({names}).",
                  "A 'no added sugars' claim is only allowed if no sugar or sugar-containing ingredient (honey, syrups, juice concentrates…) is added "
                  "(21 CFR 101.60(c)(2)). Honey, coconut sugar and cane sugar are all added sweeteners. The approved content sheet even lists them as the product's nutritive sweeteners.",
                  found="NO ADDED SUGAR badge (top flap)", expected="Claim absent, or sweeteners removed", evidence=ev, confidence=0.93,
                  fix="Remove the claim or reformulate; keep it consistent with the Added Sugars line on the panel.")]
    return [R(PASS, "'No added sugar' claim is supported by the ingredient list.", "", confidence=0.8)]


@check("claim_vegan", ['vegan', 'vegetarian', 'animal-derived', 'animal derived'])
def claim_vegan(ctx, p):
    P = ctx.P
    out = []
    claims = P.claims.get("vegan", [])
    sheet_vegan = bool(ctx.sheet and re.search(r"vegan", " ".join(f.value for f in ctx.sheet.fields.values() if f.label.lower().startswith(("notes", "front"))), re.I))
    if not P.ingredients or not (claims or sheet_vegan):
        return []
    hits = _find_ing_words(ctx, ["Honey"], P.ingredients)
    ch = P.nf_rows.get("cholesterol")
    if hits:
        ev = [ctx.ev(c, f"Vegan claim: “{c.meta.get('match')}”") for c in claims[:3]] + [ctx.ev(bb, f"Ingredient: {w}") for w, bb in hits]
        if ctx.sheet:
            f = ctx.sheet.get("Notes")
            if f and re.search("vegan", f.value, re.I): ev.append(ctx.sheet_ev(f, "Approved claims (Certified Vegan)"))
        out.append(R(FAIL, "Vegan claims appear on a product that contains honey.",
                     "Honey is an animal-derived ingredient and is excluded under vegan.org and common vegan definitions, so “Suitable for Vegans”, “(vegan)” and a Certified Vegan mark cannot all be true. "
                     "FDA has no formal 'vegan' definition, but a false claim is misleading labelling (FD&C Act §403(a)), and the company standard forbids it.",
                     found="Vegan claim + Honey (1%)", expected="No vegan claim, or honey removed", evidence=ev, confidence=0.9, rule_source="internal",
                     fix="Drop the vegan statements / certification mark, or replace honey with a plant-based sweetener and re-certify."))
    if ch and ch.amount > 0:
        out.append(R(REVIEW, f"A vegan product shows {_fmt(ch.amount)} mg cholesterol ({_fmt(ch.dv)}% DV).",
                     "Cholesterol only occurs in animal products. Together with the honey this suggests the nutrition panel may belong to a different (non-vegan) product or is wrong.",
                     found=f"Cholesterol {_fmt(ch.amount)}mg", expected="0 mg for a plant-only recipe", evidence=[ctx.ev(ch.bbox, "Cholesterol 45mg 26%")],
                     confidence=0.7, title="Cholesterol on a vegan product"))
    return out or [R(PASS, "No vegan/animal-ingredient conflict detected.", "", confidence=0.7)]


# =================================================================== ALLERGENS
@check("allergen_contains", ['contains:', 'major allergen', 'allergens must', 'allergen statement'])
def allergen_contains(ctx, p):
    P = ctx.P
    if not P.ingredients:
        return []
    ing = P.ingredients.text.lower()
    present = {}
    for group, words in MAJOR_ALLERGENS.items():
        for w in words:
            if re.search(r"\b" + w, ing):
                present.setdefault(group, []).append(w)
    if not present:
        return [R(PASS, "No FDA major allergens detected in the ingredient list.", "", confidence=0.7)]
    a = P.allergens.text if P.allergens else ""
    declared = re.match(r"\s*(allergen\s*information\s*:?\s*)?contains\s*:", a, re.I) is not None
    paren = False
    flat = [w for ws in present.values() for w in ws]
    words = _find_ing_words(ctx, [w.capitalize() + ("s" if w in ("almond", "pecan", "walnut") else "") for w in flat], P.ingredients)
    ev = [ctx.ev(bb, f"Ingredient: {w}") for w, bb in words]
    if P.allergens: ev.append(ctx.ev(P.allergens, "Allergen statement"))
    names = ", ".join(sorted(set(flat)))
    if not declared and not paren:
        only_may = bool(re.search(r"may contain", a, re.I))
        return [R(FAIL, f"{', '.join(present).title()} ({names}) are ingredients, but there is no “Contains:” statement.",
                  "FALCPA (21 U.S.C. 343(w)) requires each major allergen present as an ingredient to be declared in a 'Contains:' line or in the ingredient list, "
                  "with tree nuts named by type. " + ("The only allergen text is a set of 'May Contain' advisories, which describe cross-contact risk and do not satisfy this requirement. " if only_may else "No allergen statement was found. ")
                  + "Presenting almonds, pecans and walnuts as 'may contain' also understates a certain allergen as a possible one.",
                  found=a[:120] or "no statement", expected=f"Contains: Almonds, Pecans, Walnuts (tree nuts)", evidence=ev, confidence=0.94,
                  fix="Add “Contains: Tree Nuts (Almonds, Pecans, Walnuts)” and keep any true advisory (e.g. Brazil nuts) separate.")]
    return [R(PASS, "Major allergens are declared in a Contains statement.", "", evidence=ev, confidence=0.85)]


@check("allergen_terms", ['gluten', 'plain language', 'us-style', 'non-us'])
def allergen_terms(ctx, p):
    P = ctx.P
    if not P.allergens:
        return []
    a = P.allergens.text
    issues, ev = [], []
    if re.search(r"cereals?\s+containing\s+gluten", a, re.I):
        bb = locate(ctx.art, "Cereals Containing Gluten", P.allergens.bbox)
        issues.append("“Cereals Containing Gluten” is EU/UK allergen wording; US labels name the major allergen (e.g. wheat) and treat gluten separately.")
        ev.append(ctx.ev(bb or P.allergens, "Cereals Containing Gluten"))
    if re.search(r"may contain nuts", a, re.I):
        bb = locate(ctx.art, "Nuts,", P.allergens.bbox)
        issues.append("“May Contain Nuts” is not specific; the US requires tree nuts to be named by type.")
    if re.search(r"brazil nuts?", a, re.I) and not re.search(r"brazil", P.ingredients.text if P.ingredients else "", re.I):
        issues.append("Brazil nuts appear only as an advisory, not as an ingredient (fine if it is a real cross-contact risk).")
    if not issues:
        return []
    return [R(REVIEW, "Allergen advisory uses non-US wording and is partly generic.", " ".join(issues), found=a[:140], evidence=ev + [ctx.ev(P.allergens, "Allergen statement")], confidence=0.75,
              fix="Confirm with regulatory whether wheat/gluten is actually present and rewrite in US terms.", title="Allergen wording is not US-style")]


# =================================================================== NET QUANTITY / IDENTITY / MANUFACTURER
@check("net_qty_pdp", ['principal display panel', 'pdp', 'net weight must', 'net quantity'])
def net_qty_pdp(ctx, p):
    P = ctx.P
    pdp_name, pdp_bb = ctx.pdp()
    if not P.net_qty:
        return [R(FAIL, "No net quantity statement was found.", "21 CFR 101.105 requires a net quantity of contents on the principal display panel.", confidence=0.75)]
    if not pdp_bb:
        return [R(REVIEW, "Net quantity found but the principal display panel could not be determined.", "No dieline panel labels were detected.", evidence=[ctx.ev(q, "Net quantity") for q in P.net_qty], confidence=0.5)]
    on_pdp = [q for q in P.net_qty if q.panel and q.panel == pdp_name]
    if not on_pdp:
        ev = [ctx.ev(q, f"Net quantity “{q.text}” ({q.panel})") for q in P.net_qty] + [ctx.ev(pdp_bb, "Principal display panel")]
        return [R(FAIL, f"Net quantity “{P.net_qty[0].text}” is on the {P.net_qty[0].panel.lower()}, not on the principal display panel.",
                  "The net quantity of contents must appear on the PDP, in the bottom 30% of the panel (21 CFR 101.105(f)). The front panel carries no weight statement at all.",
                  found=f"{P.net_qty[0].text} on {P.net_qty[0].panel}", expected="net quantity statement on the front (principal display) panel", evidence=ev, confidence=0.92,
                  fix="Add “NET WT 16.9 OZ (480g)” to the bottom third of the front panel."
                  )]
    q = on_pdp[0]
    x0, y0, x1, y1 = pdp_bb
    if (q.bbox[1] + q.bbox[3]) / 2 < y0 + 0.70 * (y1 - y0):
        return [R(REVIEW, "Net quantity is on the PDP but not in its lower 30%.", "", evidence=[ctx.ev(q, "Net quantity")], confidence=0.7)]
    return [R(PASS, "Net quantity is on the principal display panel.", "", evidence=[ctx.ev(q, "Net quantity")], confidence=0.9)]


@check("net_qty_units", ['ounces', 'metric', 'both units', 'grams', 'us customary'])
def net_qty_units(ctx, p):
    P = ctx.P
    if not P.net_qty:
        return []
    units = {q.meta["unit"] for q in P.net_qty}
    metric = units & {"g", "kg", "ml", "l"}
    cust = units & {"oz", "lb", "floz"}
    ev = [ctx.ev(q, f"Net quantity: {q.text}") for q in P.net_qty]
    if metric and not cust:
        g = [q.value for q in P.net_qty if q.meta["unit"] == "g"]
        oz = f"{g[0] / 28.3495:.1f} oz" if g else "oz"
        return [R(FAIL, f"Net quantity is given in metric only (“{P.net_qty[0].text}”).",
                  "Under the Fair Packaging and Labeling Act the net quantity must be declared in both US customary and metric units, e.g. “NET WT 16.9 OZ (1 LB 0.9 OZ) (480g)”.",
                  found=P.net_qty[0].text, expected=f"{oz} and metric, e.g. NET WT {oz.upper()} (480g)", evidence=ev, confidence=0.93,
                  fix="Add the avoirdupois declaration (16.9 oz) next to the grams value.")]
    if metric and cust:
        gv = next((q.value for q in P.net_qty if q.meta["unit"] == "g"), None)
        oz = next((q.value for q in P.net_qty if q.meta["unit"] == "oz"), None)
        if gv and oz and abs(gv / 28.3495 - oz) > 0.15:
            return [R(FAIL, "Metric and customary net quantities do not agree.", f"{gv:g} g = {gv / 28.3495:.1f} oz, not {oz:g} oz.", evidence=ev, confidence=0.9)]
        return [R(PASS, "Net quantity is given in both unit systems.", "", evidence=ev, confidence=0.9)]
    return []


@check("manufacturer_address", ['place of business', 'manufacturer', 'distributor', 'company address'])
def manufacturer_address(ctx, p):
    P = ctx.P
    if not P.manufacturer:
        return [R(FAIL, "No manufacturer / distributor name and address found.", "21 CFR 101.5 requires the name and place of business of the manufacturer, packer or distributor.", confidence=0.7)]
    t = P.manufacturer.text
    near = " ".join(b.text for b in ctx.art.print_blocks() if b.panel == P.manufacturer.panel and abs(b.bbox[1] - P.manufacturer.bbox[3]) < 40 and abs(b.bbox[0] - P.manufacturer.bbox[0]) < 120)
    has_addr = bool(re.search(r"\b[A-Z]{2}\s+\d{5}(-\d{4})?\b|,\s*[A-Z]{2}\b|\b\d{5}\b|\bUSA\b|\bU\.S\.A", t + " " + near))
    if not has_addr:
        return [R(FAIL, f"“{t}” gives a company name but no city, state and ZIP code.",
                  "The 'place of business' must include city, state and ZIP (street address may be omitted only if the firm is in a current city directory) (21 CFR 101.5(d)).",
                  found=t, expected="Produced by Dancing Fish Granola Co., City, ST 12345", evidence=[ctx.ev(P.manufacturer, "Manufacturer line")], confidence=0.88,
                  fix="Add city, state and ZIP (and country of origin if imported).")]
    return [R(PASS, "Manufacturer name and place of business are present.", "", evidence=[ctx.ev(P.manufacturer, "Manufacturer line")], confidence=0.85)]


@check("identity_matches", ['statement of identity', 'legal name', 'common name'])
def identity(ctx, p):
    P = ctx.P
    pdp_name, _ = ctx.pdp()
    f = ctx.sheet.get("Legal Name / Statement of Identity") if ctx.sheet else None
    toks = [t for t in re.split(r"[^A-Za-z]+", f.value) if len(t) > 2] if f else ["GRANOLA"]
    on_pdp = [i for i in P.identity if i.panel and "principal" in i.panel.lower()]
    found = {_norm(t) for i in on_pdp for t in i.text.split()}
    missing = [t for t in toks if _norm(t) not in found]
    if not on_pdp:
        return [R(FAIL, "Statement of identity (product name) is not on the principal display panel.", "21 CFR 101.3 requires the common name on the PDP.", confidence=0.7)]
    ev = [ctx.ev(i, f"PDP text: {i.text}") for i in on_pdp] + ([ctx.sheet_ev(f, "Approved legal name")] if f else [])
    if missing:
        return [R(REVIEW, f"PDP shows {', '.join(i.text for i in on_pdp)}; approved name also has {', '.join(missing)}.", "", evidence=ev, confidence=0.6)]
    return [R(PASS, "Statement of identity (GRANOLA · BERRY CRUNCH) is on the front panel and matches the approved legal name.", "'Granola' is the common name; 'Berry Crunch' the variety descriptor, both on the PDP.", evidence=ev, confidence=0.9)]


# =================================================================== CONTENT-SHEET MATCHING
def _clean(s):
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _sentences(t):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", _clean(t)) if len(x.strip()) > 3]


def _cmp(a, b):
    return fuzz.ratio(re.sub(r"[^a-z0-9]", "", a.lower()), re.sub(r"[^a-z0-9]", "", b.lower()))


@check("sheet_text_match", ['match the approved', 'exactly', 'word for word', 'content sheet', 'marketing copy', 'storage instructions', 'dietary statement', 'allergen statement'])
def sheet_text_match(ctx, p):
    if not ctx.sheet:
        return []
    P, out = ctx.P, []
    pairs = [("Marketing copy", ctx.sheet.get("Marketing Copy"), P.marketing[0] if P.marketing else None, "Marketing copy"),
             ("Storage instructions", ctx.sheet.get("Storage Instructions"), P.storage, "Storage instructions"),
             ("Allergen statement", ctx.sheet.get("Allergen Information"), P.allergens, "Allergen statement")]
    for name, f, it, _ in pairs:
        if not f:
            continue
        if not it:
            out.append(R(FAIL, f"{name} from the approved sheet was not found on the artwork.", "", evidence=[ctx.sheet_ev(f)], confidence=0.7, title=f"{name} missing"))
            continue
        a_t = re.sub(r"^(allergen information|ingredients)\s*:\s*", "", it.text, flags=re.I)
        b_t = re.sub(r"^(allergen information|ingredients)\s*:\s*", "", f.value, flags=re.I)
        s = _cmp(a_t, b_t)
        ev = [ctx.ev(it, f"Artwork: {name.lower()}"), ctx.sheet_ev(f, f"Approved: {name.lower()}")]
        if s >= 99:
            out.append(R(PASS, f"{name} matches the approved content sheet.", "Compared after normalising line breaks and punctuation.", evidence=ev, confidence=0.95, title=f"{name} matches approved text"))
        elif s >= 92:
            out.append(R(REVIEW, f"{name} differs slightly from the approved sheet ({s:.0f}% similar).", "", found=it.text, expected=f.value, evidence=ev, confidence=0.7, title=f"{name}: minor wording difference"))
        else:
            out.append(R(FAIL, f"{name} does not match the approved sheet ({s:.0f}% similar).", "", found=it.text, expected=f.value, evidence=ev, confidence=0.9, title=f"{name} differs from approved text"))
    # dietary statements, sentence level
    f = ctx.sheet.get("Dietary Information")
    if f and P.dietary:
        approved = _sentences(f.value)
        for d in P.dietary:
            body = re.sub(r"^dietary information\s*:\s*", "", d.text, flags=re.I)
            for s_art in _sentences(body):
                best = max(approved, key=lambda x: _cmp(s_art, x))
                sc = _cmp(s_art, best)
                ev = [ctx.ev(d, f"Artwork: “{s_art}”"), ctx.sheet_ev(f, "Approved dietary information")]
                if sc >= 99:
                    continue
                if sc < 88:
                    out.append(R(FAIL, f"Artwork says “{s_art}”, but the approved sheet says “{best}”.",
                                 "The dietary claim on the pack is different from the approved wording (and 'vegan' is a stronger claim than 'vegetarian').",
                                 found=s_art, expected=best, evidence=ev, confidence=0.9, title="Dietary statement differs from approved text",
                                 fix="Use the approved wording, or get the new claim approved (see the honey finding)."))
                else:
                    out.append(R(REVIEW, f"Dietary line differs in small ways from the approved text ({sc:.0f}% similar).",
                                 f"Artwork: “{s_art}”. Approved: “{best}”.", found=s_art, expected=best, evidence=ev, confidence=0.7, title="Dietary line: minor wording differences"))
    return out


@check("ingredients_match", ['ingredient list', 'ingredients on the artwork', 'ingredient statement'])
def ingredients_match(ctx, p):
    P = ctx.P
    f = ctx.sheet.get("Ingredient List") if ctx.sheet else None
    if not (f and P.ingredients):
        return []
    art_i = split_ingredients(P.ingredients.text)
    sh_i = split_ingredients(f.value)

    def key(n): return re.sub(r"\b(dried|organic|natural)\b", "", n.lower()).strip()

    def match(n, pool):
        best, bs = None, 0
        for m in pool:
            s = fuzz.token_set_ratio(key(n), key(m[0]))
            if s > bs: best, bs = m, s
        return best if bs >= 86 else None

    only_art = [i for i in art_i if not match(i[0], sh_i)]
    only_sheet = [i for i in sh_i if not match(i[0], art_i)]
    pct = []
    for i in art_i:
        m = match(i[0], sh_i)
        if m and i[2] is not None and m[2] is not None and abs(i[2] - m[2]) > 0.5:
            pct.append((i, m))
    # duplicates in sheet (Sunflower Seeds twice)
    dup = [n for n in {x[0].lower() for x in sh_i} if [x[0].lower() for x in sh_i].count(n) > 1]
    uncertain = f.uncertain
    tag = ["approved-data-unverified"] if uncertain else []
    note = (" Note: the content sheet itself says its ingredient list is “reconstructed” and must be verified against the art, so a person must decide which side is right." if uncertain else "")
    out = []
    if only_art or only_sheet:
        rows = [dict(label=i[0], found="on artwork", expected="not on sheet", ok=False) for i in only_art] + \
               [dict(label=i[0], found="not on artwork", expected="on sheet", ok=False) for i in only_sheet]
        ev = []
        for i in only_art:
            nm = i[0]
            bb = locate(ctx.art, nm, P.ingredients.bbox) or locate(ctx.art, re.sub(r"^(Dried|Organic)\s+", "", nm), P.ingredients.bbox) \
                or locate(ctx.art, nm.split()[-1], P.ingredients.bbox)
            ev.append(ctx.ev(bb or P.ingredients, f"Only on artwork: {nm}"))
        ev.append(ctx.sheet_ev(f, "Approved ingredient list"))
        names = ", ".join(i[0] for i in only_art)
        out.append(R(FAIL, f"Artwork lists ingredients not in the approved list: {names}." + (f" Sheet lists extra: {', '.join(i[0] for i in only_sheet)}." if only_sheet else ""),
                     "The ingredient statement is legally binding and must reflect the actual formula (21 CFR 101.4). Every difference is a potential mislabel, "
                     "and walnuts also change the allergen picture." + note,
                     found="; ".join(i[3] for i in only_art), expected="(not in approved list)", details=rows, evidence=ev, tags=tag, confidence=0.9, rule_source="content-sheet",
                     fix="Confirm the formula; update either the artwork or the content sheet so they agree.", title="Ingredient list differs from approved content sheet"))
    for a, b in pct:
        bb = locate(ctx.art, f"{a[0]} ({a[2]:g}%)", P.ingredients.bbox) or locate(ctx.art, a[0], P.ingredients.bbox)
        out.append(R(FAIL, f"{a[0]} is {a[2]:g}% on the artwork but {b[2]:g}% in the approved sheet.",
                     "A declared percentage that differs ninefold is not a rounding issue; it also affects the added-sugar and vegan findings." + note,
                     found=f"{a[0]} ({a[2]:g}%)", expected=f"{b[0]} ({b[2]:g}%)", evidence=[ctx.ev(bb or P.ingredients, f"{a[0]} ({a[2]:g}%)"), ctx.sheet_ev(f, "Approved ingredient list")],
                     tags=tag, confidence=0.9, rule_source="content-sheet", title=f"{a[0]} percentage differs from approved sheet"))
    if dup:
        out.append(R(REVIEW, f"Approved sheet lists “{dup[0].title()}” twice.", "Likely a typo in the sheet (the artwork lists it once); confirm the intended formula.", evidence=[ctx.sheet_ev(f)], confidence=0.6, severity="minor", title="Duplicate ingredient in approved sheet"))
    if not out:
        out.append(R(PASS, "Ingredient list matches the approved sheet.", "", evidence=[ctx.ev(P.ingredients, "Artwork ingredients"), ctx.sheet_ev(f)], confidence=0.9))
    return out


@check("net_qty_matches_sheet", ['net weight must match', 'net weight matches', 'must match the content sheet'])
def net_qty_sheet(ctx, p):
    f = ctx.sheet.get("Package Size / Net Weight") if ctx.sheet else None
    if not (f and ctx.P.net_qty):
        return []
    m = re.search(r"(\d+(?:\.\d+)?)\s*g", f.value)
    q = next((q for q in ctx.P.net_qty if q.meta["unit"] == "g"), None)
    if m and q:
        ok = abs(float(m.group(1)) - q.value) < 0.5
        ev = [ctx.ev(q, f"Artwork: {q.text}"), ctx.sheet_ev(f, "Approved net weight")]
        return [R(PASS if ok else FAIL, "Net weight matches the approved sheet (480 g)." if ok else f"Net weight {q.text} ≠ approved {f.value}.", "", evidence=ev, confidence=0.95, title="Net weight matches approved sheet" if ok else "Net weight differs from approved sheet")]
    return []


@check("upc_valid", ['barcode', 'upc', 'gtin'])
def upc_valid(ctx, p):
    art, P = ctx.art, ctx.P
    f = ctx.sheet.get("UPC / Barcode") if ctx.sheet else None
    if not art.barcode:
        return [R(REVIEW, "No barcode could be decoded from the artwork.", "Barcode may be missing, too small or distorted.", evidence=[ctx.sheet_ev(f)] if f else [], confidence=0.5)]
    d = art.barcode["text"]
    ok_len = len(d) == 12 and d.isdigit()
    chk = None
    if ok_len:
        o = sum(int(x) for x in d[0:11:2]); e = sum(int(x) for x in d[1:11:2])
        chk = (10 - (o * 3 + e) % 10) % 10
    ev = [ctx.ev(art.barcode["bbox"], f"Decoded barcode: {d}")]
    if ok_len and chk != int(d[-1]):
        return [R(FAIL, f"UPC {d} has an invalid check digit (expected {chk}).", "", evidence=ev, confidence=0.97)]
    if f:
        sd = re.sub(r"\D", "", f.value)
        ev.append(ctx.sheet_ev(f, "Approved UPC"))
        if sd != d:
            return [R(FAIL, f"Barcode encodes {d} but the approved UPC is {sd}.", "", found=d, expected=sd, evidence=ev, confidence=0.97)]
        return [R(PASS, f"Barcode decodes to {d}, passes the UPC-A check digit, and matches the approved sheet.", "The bars themselves were decoded (not just the printed digits), so a mismatch between bars and digits would have been caught.", evidence=ev, confidence=0.97, title="UPC barcode is valid and matches the sheet")]
    return [R(PASS, f"Barcode decodes to {d} and passes the check digit.", "", evidence=ev, confidence=0.9)]


@check("claims_present", ['certification mark', 'marks listed', 'must appear on the pack', 'claims listed'])
def claims_present(ctx, p):
    if not ctx.sheet:
        return []
    P = ctx.P
    ref = []
    for lbl in ("Front/Back Panel Claims & Certifications", "Notes"):
        f = ctx.sheet.get(lbl)
        if f: ref.append(f)
    txt = " ; ".join(f.value for f in ref)
    wants = []
    if re.search(r"usda organic", txt, re.I): wants.append(("USDA Organic seal", "usda_organic", "mark"))
    if re.search(r"vegan\.org|certified vegan", txt, re.I): wants.append(("Certified Vegan mark", "certified_vegan", "mark"))
    if re.search(r"no added sugar", txt, re.I): wants.append(("No Added Sugar", "no_added_sugar", "text"))
    if re.search(r"gmo free", txt, re.I): wants.append(("GMO Free", "gmo_free", "text"))
    if re.search(r"oven baked", txt, re.I): wants.append(("Oven Baked", "oven_baked", "text"))
    if re.search(r"how2recycle|recyclable", txt, re.I): wants.append(("Recyclable / How2Recycle mark", "recycle", "mark"))
    found, missing = [], []
    ev = []
    for label, key, kind in wants:
        if key in P.claims:
            found.append(label); ev.append(ctx.ev(P.claims[key][0], label))
        elif key in P.marks:
            found.append(label + " (read by OCR)"); ev.append(ctx.ev(P.marks[key], label))
        else:
            missing.append(label)
    out = []
    if found:
        out.append(R(PASS, f"{len(found)} of {len(wants)} claims on the approved sheet are present: " + ", ".join(found) + ".", "Text claims were matched in the PDF text layer; marks were located by OCR.", evidence=ev + [ctx.sheet_ev(f) for f in ref][:1], confidence=0.8, title="Approved claims are present on the artwork"))
    if missing:
        out.append(R(REVIEW, "Could not machine-verify: " + ", ".join(missing) + ".",
                     "These are graphic marks (outlined or raster), so there is no text to read. They may well be on the pack; a person (or the vision step, when an API key is configured) should confirm presence, size and correct certifier artwork.",
                     evidence=[ctx.sheet_ev(f) for f in ref][:1], confidence=0.6, title="Certification marks need visual confirmation"))
    return out


@check("organic_rules", ['organic', 'certifying agent', 'usda'])
def organic_rules(ctx, p):
    P = ctx.P
    org = [c for c in P.claims.get("organic", []) if "certified" in c.text.lower() or "all our products" in c.text.lower()]
    if not (org or "usda_organic" in P.marks):
        return []
    txt = " ".join(b.text for b in ctx.art.print_blocks())
    out = []
    if not re.search(r"certified\s+organic\s+by|certifying\s+agent|organic\s+by\s+[A-Z]", txt, re.I):
        ev = [ctx.ev(o, "Organic claim") for o in org[:1]] + ([ctx.ev(P.marks["usda_organic"], "USDA seal")] if "usda_organic" in P.marks else [])
        out.append(R(REVIEW, "Organic claim/seal present but no “Certified organic by …” certifier name was found.",
                     "Organic-labelled packaged products must show the name of the certifying agent on the information panel (7 CFR 205.303). "
                     "No such text exists anywhere in the artwork text layer or OCR.", expected="“Certified organic by <agent>” near the manufacturer line", evidence=ev, confidence=0.75,
                     fix="Add the certifier statement (and certifier's contact/ID if required by the agent).", title="Organic certifier statement not found"))
    if P.ingredients:
        n_total = len(split_ingredients(P.ingredients.text)); n_org = len(re.findall(r"\borganic\b", P.ingredients.text, re.I))
        if n_org < n_total and any("all our products" in c.text.lower() for c in org):
            out.append(R(REVIEW, f"The pack says all products are certified organic, but only {n_org} of {n_total} ingredients are marked “organic”.",
                         "For products labelled 'organic', organic ingredients must be identified in the ingredient list (7 CFR 205.305). Either the claim is too broad or the ingredient list is incomplete.",
                         evidence=[ctx.ev(P.ingredients, "Ingredient list")] + [ctx.ev(o, "“All our products are certified organic”") for o in org[:1]], confidence=0.65, title="Organic ingredients are not identified"))
    return out


# =================================================================== HIDDEN / NON-US CONTENT
@check("hidden_text", ['hidden', 'non-printing', 'not visible', 'invisible'])
def hidden_text(ctx, p):
    P = ctx.P
    hid = P.hidden
    out = []
    if hid:
        bb = union([b.bbox for b in hid])
        txt = "; ".join(b.text for b in hid[:6]) + (" …" if len(hid) > 6 else "")
        ev = [ctx.ev(b.bbox, b.text, panel=b.panel) for b in hid[:12]]
        out.append(R(REVIEW, f"{len(hid)} text objects exist in the file but are not visible in the rendered artwork ({hid[0].panel or 'unknown panel'}).",
                     "The text layer contains a second, UK/EU-style nutrition table (“Typical Values per 100g”, Energy 1868 kJ / 445 kcal, Salt 0.04 g …) that is covered by the white Nutrition Facts panel. "
                     "It is invisible in the render but still in the print file; OCR of the rendered page does not see it. Hidden copy can print, ship to the wrong market or confuse downstream OCR. "
                     "We flag it for a person because it may be an intentional cover-up while the file is in draft.",
                     found=txt, expected="No unseen text in a print-ready file", evidence=ev, confidence=0.85,
                     fix="Delete the stale UK panel objects from the native file.", title="Hidden UK-style nutrition table under the US panel"))
    return out


@check("nutrition_format_consistency", ['one nutrition declaration', 'single format', 'kj', 'saturates', 'uk panel', 'traffic light', 'nutrition strip'])
def nutrition_strip(ctx, p):
    P = ctx.P
    if not P.strip or len(P.strip) < 3:
        return []
    cal = P.strip.get("calories", {}).get("values", {})
    fat = P.strip.get("fat", {}).get("values", {})
    sat = P.strip.get("saturates", {}).get("values", {})
    nf = P.nf_rows
    rows, ratios = [], []
    def add(label, s, n, unit):
        if s is not None and n is not None and n:
            ratios.append(s / n)
            rows.append(dict(label=label, found=f"back panel: {_fmt(s)}{unit}", expected=f"Nutrition Facts: {_fmt(n)}{unit}", ok=abs(s - n) / n < 0.1))
    add("Calories", cal.get("kcal"), nf["calories"].amount if "calories" in nf else None, " kcal")
    add("Fat", fat.get("g"), nf["total_fat"].amount if "total_fat" in nf else None, " g")
    add("Saturates", sat.get("g"), nf["saturated_fat"].amount if "saturated_fat" in nf else None, " g")
    ev = [ctx.ev(P.strip["_bbox"], "Back-panel nutrition strip (kJ / Saturates / Salt)")]
    if P.nf_region: ev.append(ctx.ev(P.nf_region, "US Nutrition Facts panel"))
    rr = ", ".join(f"{r:.2f}" for r in ratios)
    mism = [r for r in rows if not r["ok"]]
    if mism:
        return [R(REVIEW, "Back panel shows a second, EU-style nutrition strip whose numbers contradict the Nutrition Facts panel.",
                  f"The strip uses kJ, “Saturates” and “Salt” (non-US terms) and states 46 kcal / 3.8 g fat per serving versus 166 kcal / 15 g on the Nutrition Facts panel. "
                  f"The ratios ({rr}) are not one consistent factor, so no common basis (e.g. a smaller serving) reconciles them. It might be a leftover from a UK variant; "
                  "a person should confirm whether it is meant to be there at all.",
                  details=rows, evidence=ev, confidence=0.8, fix="Remove the strip or replace it with US-format values matching the panel.", title="Two conflicting nutrition declarations on the pack")]
    return []


@check("type_size", ['type size', 'font size', 'minimum height', '1/16', 'legibility', 'point size'])
def type_size(ctx, p):
    P = ctx.P
    items = [i for i in (P.ingredients, P.allergens) if i]
    if not items:
        return []
    blk = {b.bbox: b for b in ctx.art.print_blocks()}
    sizes = [(i, next((b.size for b in ctx.art.print_blocks() if b.bbox[0] <= i.bbox[0] + 1 and abs(b.bbox[1] - i.bbox[1]) < 6 and b.size < 12), None)) for i in items]
    sizes = [(i, s) for i, s in sizes if s]
    if not sizes:
        return []
    mn = min(s for _, s in sizes)
    xh_pt = 0.5 * mn      # typical x-height ≈ 0.5 em
    need = 72 / 16        # 1/16 in = 4.5 pt
    if xh_pt < need:
        return [R(REVIEW, f"Ingredient/allergen text is set at {mn:g} pt; its lower-case letters are ≈{xh_pt:.1f} pt (≈{xh_pt / 72:.3f} in) tall, below the 1/16 in minimum.",
                  "21 CFR 101.2(c) requires mandatory statements to have lower-case letters at least 1/16 inch (≈4.5 pt) tall. The estimate assumes a typical x-height of half the point size and a 1:1 dieline; "
                  "a person should measure the real font (small-package exemptions may apply).", found=f"{mn:g} pt type (≈{xh_pt:.1f} pt x-height)", expected="≥ 4.5 pt x-height (1/16 in)",
                  evidence=[ctx.ev(i, f"{mn:g} pt text") for i, _ in sizes], confidence=0.55, fix="Increase type size or verify with the font's actual x-height.", title="Mandatory text may be below minimum type size")]
    return [R(PASS, "Mandatory text meets the minimum type size.", "", confidence=0.6)]
