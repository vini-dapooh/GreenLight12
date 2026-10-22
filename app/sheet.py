"""Content sheet (approved product data) loader.

The sample sheet is a human-formatted spreadsheet: label/value/note triples laid out side by side plus a
nutrition table. We read it generically (label -> value, note, cell address) so evidence can cite the cell.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import openpyxl

UNCERTAIN_RE = re.compile(r"verify|confirm|assum|not clearly|reconstruct|unverified|derived|check", re.I)


@dataclass
class Field:
    label: str
    value: str
    note: str = ""
    ref: str = ""            # e.g. "Content Sheet - US!B22"

    @property
    def uncertain(self) -> bool:
        return bool(self.note and UNCERTAIN_RE.search(self.note))


@dataclass
class Approved:
    filename: str
    fields: dict = field(default_factory=dict)            # normalised label -> Field
    nutrition: dict = field(default_factory=dict)          # nutrient -> {basis: (amount, dv)}
    nutrition_bases: list = field(default_factory=list)
    nutrition_ref: dict = field(default_factory=dict)

    def get(self, *labels):
        for l in labels:
            f = self.fields.get(_k(l))
            if f is not None and str(f.value).strip() not in ("", "N/A", "None"):
                return f
        return None


def _k(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def load_sheet(path: str, filename: str | None = None) -> Approved:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    ap = Approved(filename=filename or path.split("/")[-1])
    sheet = ws.title
    rows = list(ws.iter_rows())
    in_table = False
    for row in rows:
        vals = [c.value for c in row]
        if all(v is None for v in vals):
            continue
        first = str(vals[0]).strip() if vals[0] is not None else ""
        if first.startswith("Nutrient Composition"):
            in_table = True
            ap.nutrition_bases = [str(vals[i]) for i in (1, 3, 5) if i < len(vals) and vals[i] is not None]
            continue
        if in_table and first and not first.startswith("Nutrition Values"):
            name = first
            ap.nutrition[name] = {}
            for bi, basis in enumerate(ap.nutrition_bases):
                a_i, d_i = 1 + 2 * bi, 2 + 2 * bi
                amt = vals[a_i] if a_i < len(vals) else None
                dv = vals[d_i] if d_i < len(vals) else None
                ap.nutrition[name][basis] = (amt, dv)
            ap.nutrition_ref[name] = f"{sheet}!A{row[0].row}:G{row[0].row}"
            continue
        # label / value / note triples: (A,B,C) and (D,E,F)
        for lc in (0, 3):
            if lc + 1 < len(vals) and vals[lc] is not None and vals[lc + 1] is not None and str(vals[lc]).strip():
                label = str(vals[lc]).strip()
                if label in ("Field",):
                    continue
                v = vals[lc + 1]
                note = str(vals[lc + 2]).strip() if lc + 2 < len(vals) and vals[lc + 2] is not None else ""
                ref = f"{sheet}!{row[lc + 1].coordinate}"
                ap.fields[_k(label)] = Field(label, str(v).strip(), note, ref)
    return ap
