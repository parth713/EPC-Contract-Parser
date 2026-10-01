"""Step 2 - Leaf-unit anchor discovery (per division).

For each division from Step 1/1.5 this finds ONE level of structure - the coarsest numbered/titled unit
present - and stops there:

  * If the document is organised into titled SECTIONS/PARTS (typical of GCC), the SECTIONS are the leaves.
    Clauses inside a section are NOT separate nodes; they will be folded into the section's verbatim text
    in Step 3.
  * Otherwise the MAIN CLAUSES (Clause 1, 2, 3 - never 1.1 / 1.1.1) are the leaves; sub-clauses fold in.
  * If a division has no such structure (stamp paper, a 1-page BOQ summary, a drawing), the whole division
    is a single leaf.

Windows of a few pages (1-page overlap) are scanned WITHIN each division, never across boundaries. The
result is deduped, the leaf level is resolved per division, numbering is checked by a deterministic linter,
and leaf page ranges are asserted contiguous within the division (start..end, no gaps). Step 3 then reads
verbatim text between consecutive leaf starts.

Run (after Step 1 / 1.5 have produced out_macro/divisions.json):
    python -m epc_parser.stage2_anchors "SBUT04 Contract Agreement (3).pdf" -o out_macro/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import Settings
from .llm import GeminiBackend, LLMClient
from .numbering import clean, norm_text, normalize_marker, sibling_gaps
from .render import PageRenderer

log = logging.getLogger("epc.anchors")

ANCHOR_DPI = 200
ANCHOR_MEDIA = "medium"
WINDOW_SIZE = 4
STRIDE = 3          # 1-page overlap
MIN_SECTIONS = 2    # need at least this many section anchors to treat a division as section-structured


# =============================================================================================
# LLM schema
# =============================================================================================
class AnchorItem(BaseModel):
    pdf_page: int = Field(description="1-indexed physical PDF page where this unit BEGINS (from the label before each image).")
    kind: Literal["section", "clause"] = Field(description="'section' for a titled Section/Part; 'clause' for a main numbered clause.")
    marker: Optional[str] = Field(None, description="Marker exactly as printed, e.g. 'SECTION A', 'Part II', '4', 'Clause 5'. Empty if the section has only a title.")
    title: str = Field(description="The unit's heading/title text, verbatim.")


class WindowAnchors(BaseModel):
    anchors: list[AnchorItem] = Field(default_factory=list)


ANCHOR_SYSTEM = (
    "You locate the TOP-LEVEL structural units of ONE document inside an Indian EPC contract. "
    "A top-level unit is the COARSEST numbered or titled division of this document:\n"
    "  - If the document is organised into titled SECTIONS or PARTS (e.g. 'SECTION A - GENERAL', "
    "'PART II - EXECUTION'), return those SECTIONS. Do NOT descend into the clauses inside them.\n"
    "  - Otherwise return the MAIN CLAUSES (e.g. '1. DEFINITIONS', '5. SUSPENSION AND TERMINATION', "
    "'Clause 12'). \n"
    "  - NEVER return a sub-clause or lower level: not 1.1, not 5.2, not 5.2.3, not (a), not (i).\n"
    "  - A numbered list of defined terms inside a 'Definitions' clause, or a contents/index list of "
    "clause titles, is NOT a set of top-level clauses - it belongs to the ONE clause that contains it.\n"
    "  - Do NOT treat a number that merely appears as a CROSS-REFERENCE in body text (e.g. '...per "
    "clause 9.3...') as a heading.\n"
    "  - Ignore running headers/footers, page numbers, tables, figures and ordinary body text.\n"
    "For each unit report: the physical PDF page it BEGINS on (from the label printed before each image), "
    "its marker exactly as printed, its title verbatim, and whether it is a 'section' or a 'clause'. "
    "If no top-level unit begins on these pages, return an empty list."
)
ANCHOR_USER = (
    "Document: '{title}' (type {dtype}), spanning PDF pages {ds}-{de}. "
    "These images are pages {lo}-{hi} of it. Return the top-level units that BEGIN on these pages."
)

# Second, page-range-scoped pass: direct sub-clauses of ONE already-detected main clause.
SUBCLAUSE_SYSTEM = (
    "You are given the pages of ONE clause of an Indian EPC contract. Return ONLY its DIRECT "
    "sub-clauses - exactly ONE numbering level below this clause (for clause 5: 5.1, 5.2, 5.3; for "
    "clause 13: 13.1, 13.2 ...). Rules:\n"
    "  - Return a sub-clause ONLY where it is printed as its OWN numbered heading that BEGINS here.\n"
    "  - Do NOT return the clause's own number/title, nor anything deeper (5.2.1), nor lettered items "
    "((a),(i)), nor plain paragraphs, nor list items inside a definitions clause.\n"
    "  - Do NOT return a number that only appears as a CROSS-REFERENCE in the text (e.g. '...as set out "
    "in clause 9.3...'); only a heading that starts a sub-clause here.\n"
    "  - Ignore running headers/footers, page numbers and tables.\n"
    "For each sub-clause report its physical PDF page (from the label), its marker exactly as printed, "
    "and its title verbatim. If there are none, return an empty list."
)
SUBCLAUSE_USER = (
    "This is clause '{marker}' ({title}) of '{doc}', spanning PDF pages {a}-{b}. "
    "Return ONLY its direct sub-clauses, numbered '{marker}.x', that BEGIN on these pages."
)

# Page-range-scoped pass: the numbered CLAUSES directly inside ONE already-detected SECTION. These are
# read fresh from the section's pages (NOT reused from the main pass, which is told not to descend into a
# section) so clauses printed on the section's own heading page are recovered instead of suppressed.
SECTION_CLAUSE_SYSTEM = (
    "You are given the pages of ONE titled SECTION of an Indian EPC contract's General Conditions "
    "(e.g. 'SECTION 7 - INSURANCE'). Return ONLY the numbered CLAUSES that belong directly to THIS "
    "section - the main clause headings such as '1.0 ...', '2.0 ...', 'Clause 3 ...' that BEGIN on these "
    "pages. This section's own clause numbering starts at 1 and runs upward; a clause printed on the SAME "
    "page as the section heading still counts - include it, do NOT skip it. Rules:\n"
    "  - Return a clause ONLY where it is printed as its OWN numbered heading that BEGINS here.\n"
    "  - Do NOT return the SECTION's own heading or number, nor anything one level deeper (1.1, 2.3, "
    "5.2.1), nor lettered items ((a),(i)), nor plain paragraphs, nor list items inside a definitions clause.\n"
    "  - Do NOT return a number that only appears as a CROSS-REFERENCE in the text (e.g. '...per clause "
    "9.0...'); only a heading that starts a clause here.\n"
    "  - Ignore running headers/footers, page numbers and tables.\n"
    "For each clause report its physical PDF page (from the label), its marker exactly as printed, and "
    "its title verbatim. If there are none, return an empty list."
)
SECTION_CLAUSE_USER = (
    "This is section '{marker}' ({title}) of '{doc}', spanning PDF pages {a}-{b}. "
    "Return ONLY its own numbered clauses (starting at 1) that BEGIN on these pages."
)


# =============================================================================================
# Windows within a division
# =============================================================================================
def windows_in(lo: int, hi: int, size: int = WINDOW_SIZE, stride: int = STRIDE) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = lo
    while start <= hi:
        end = min(start + size - 1, hi)
        out.append((start, end))
        if end == hi:
            break
        start += stride
    return out


async def scan_anchor_window(llm: LLMClient, renderer: PageRenderer, div: dict, lo: int, hi: int
                             ) -> Optional[WindowAnchors]:
    imgs = await asyncio.gather(*[renderer.page(p, ANCHOR_DPI) for p in range(lo, hi + 1)])
    parts: list[str | bytes] = []
    for p, img in zip(range(lo, hi + 1), imgs):
        parts.append(f"\n[Physical PDF Page Number {p}]")
        parts.append(img)
    parts.append(ANCHOR_USER.format(title=clean(div["title"])[:120], dtype=div["division_type"],
                                    ds=div["start_page"], de=div["end_page"], lo=lo, hi=hi))
    res = await llm.call("anchor_scan", role="reasoner", system=ANCHOR_SYSTEM, parts=parts, schema=WindowAnchors,
                         thinking="low", media_resolution=ANCHOR_MEDIA, max_output_tokens=4096,
                         meta={"div": div["division_id"], "lo": lo, "hi": hi})
    return res.parsed if res.ok else None  # type: ignore[return-value]


# =============================================================================================
# Dedupe / leaf-level resolution / unit building
# =============================================================================================
def _anchor_key(a: AnchorItem) -> str:
    base = normalize_marker(a.marker) or norm_text(a.title)[:40] or f"p{a.pdf_page}"
    return f"{a.kind}:{base}"  # a Section and a clause can share a number ('SECTION 1' vs '1.0'); keep them apart


def _collect(anchors: list[AnchorItem], ds: int, de: int) -> tuple[list[AnchorItem], list[str]]:
    """Dedupe by unit key; keep the earliest page and the longest title. Returns (unique_anchors, notes)."""
    best: dict[str, AnchorItem] = {}
    notes: list[str] = []
    for a in anchors:
        if not ds <= a.pdf_page <= de:
            continue
        k = _anchor_key(a)
        prev = best.get(k)
        if prev is None:
            best[k] = a
            continue
        if a.pdf_page != prev.pdf_page:
            notes.append(f"unit '{k}' seen on pages {prev.pdf_page} and {a.pdf_page}")
        page = min(prev.pdf_page, a.pdf_page)
        title = a.title if len(clean(a.title)) > len(clean(prev.title)) else prev.title
        kind = prev.kind  # keep first-seen kind; resolved at division level below
        best[k] = AnchorItem(pdf_page=page, kind=kind, marker=prev.marker or a.marker, title=title)
    return sorted(best.values(), key=lambda x: x.pdf_page), notes


def _resolve_level(anchors: list[AnchorItem]) -> tuple[str, list[AnchorItem]]:
    """Pick the coarsest level present: sections where the doc uses them, else main clauses."""
    sections = [a for a in anchors if a.kind == "section"]
    clauses = [a for a in anchors if a.kind == "clause"]
    if len(sections) >= MIN_SECTIONS and len(sections) >= len(clauses):
        return "section", sorted(sections, key=lambda x: x.pdf_page)
    if clauses:
        return "clause", sorted(clauses, key=lambda x: x.pdf_page)
    if sections:
        return "section", sorted(sections, key=lambda x: x.pdf_page)
    return "whole", []


def _fold_subclauses(level: str, kept: list[AnchorItem]) -> tuple[list[AnchorItem], int]:
    """Main pass keeps only main clauses (bare integers) as level-1 units; anything dotted the main
    pass happened to return is dropped here (the scoped sub-pass finds the real X.Y separately). If a
    division has ONLY dotted markers (no integer level), keep them as leaves."""
    if level != "clause":
        return kept, 0
    main = [a for a in kept if re.fullmatch(r"\d+", normalize_marker(a.marker) or "")]
    if main:
        return main, len(kept) - len(main)
    return kept, 0


def _build_units(div: dict, level: str, kept: list[AnchorItem]) -> list[dict]:
    """Build the level-1 units (sections or main clauses). Children (X.Y) are filled later by the
    scoped sub-clause pass."""
    did, ds, de = div["division_id"], div["start_page"], div["end_page"]
    if not kept:
        return [{"unit_id": f"{did}-U01", "kind": "whole", "marker": None,
                 "title": clean(div["title"]) or None, "page_start": ds, "page_end": de,
                 "flags": [], "children": []}]
    units: list[dict] = []
    n = 0
    if kept[0].pdf_page > ds:  # front matter before the first unit
        n += 1
        units.append({"unit_id": f"{did}-U{n:02d}", "kind": "preamble", "marker": None, "title": "Preamble",
                      "page_start": ds, "page_end": kept[0].pdf_page - 1, "flags": [], "children": []})
    for i, a in enumerate(kept):
        ps = max(ds, a.pdf_page)
        pe = (kept[i + 1].pdf_page - 1) if i + 1 < len(kept) else de
        flags: list[str] = []
        if pe < ps:  # another unit starts on the same page
            pe = ps
            flags.append("shares_start_page")
        n += 1
        units.append({"unit_id": f"{did}-U{n:02d}", "kind": level, "marker": clean(a.marker) or None,
                      "title": clean(a.title) or None, "page_start": ps, "page_end": pe,
                      "flags": flags, "children": []})
    return units


def _trim_embedded_section(title: str) -> str:
    """A section-structured GCC's start page often carries BOTH the division banner and its first
    section, so the refined division title comes out as 'SECTION F - GENERAL CONDITIONS OF
    CONTRACTSECTION 1 - DEFINITIONS...'. Cut the embedded 'SECTION <n>' (the first section is now a unit
    of its own). Only trims when a SECOND, digit-numbered SECTION is glued on, so a legitimately
    'SECTION 4 - ...' division title is left untouched."""
    m = re.search(r"(.+?\S)\s*section\s*-?\s*\d", title, re.IGNORECASE)
    return m.group(1).strip() if m and len(m.group(1).strip()) >= 10 else title


async def scan_section_clauses(llm: LLMClient, renderer: PageRenderer, div: dict, unit: dict,
                               review: list[dict]) -> None:
    """Scoped pass for a section-structured division: read ONLY this SECTION's pages and attach its own
    numbered clauses (which restart per section, starting at 1) as sub-clause children. Read fresh - the
    main anchor pass deliberately does not descend into a section, so clauses on the section's heading
    page would otherwise be missing. Deeper levels (X.Y, (a)) fold into the child's text in Step 3."""
    a, b = unit["page_start"], unit["page_end"]
    wins = windows_in(a, b, size=8, stride=7)  # keep each vision call small
    results = await asyncio.gather(*[_scan_section_window(llm, renderer, div, unit, lo, hi) for lo, hi in wins])
    raw = [s for r in results if r for s in r.anchors]
    best: dict[str, AnchorItem] = {}
    for s in raw:
        key = normalize_marker(s.marker) or ""
        if not re.fullmatch(r"\d+", key):  # this section's main clauses only ('7.0'->'7'); drop 7.1 / lists
            continue
        if not a <= s.pdf_page <= b:
            continue
        prev = best.get(key)
        if prev is None or len(clean(s.title)) > len(clean(prev.title)):
            best[key] = AnchorItem(pdf_page=min(prev.pdf_page, s.pdf_page) if prev else s.pdf_page,
                                   kind="clause", marker=s.marker, title=s.title)
    slist = sorted(best.values(), key=lambda x: (x.pdf_page, int(normalize_marker(x.marker) or 0)))
    kids: list[dict] = []
    for j, s in enumerate(slist):
        cps = min(max(a, s.pdf_page), b)
        cpe = (slist[j + 1].pdf_page - 1) if j + 1 < len(slist) else b
        cflags: list[str] = []
        if cpe < cps:
            cpe = cps
            cflags.append("shares_start_page")
        kids.append({"unit_id": f"{unit['unit_id']}.{j + 1}", "kind": "subclause",
                     "marker": clean(s.marker) or None, "title": clean(s.title) or None,
                     "page_start": cps, "page_end": min(cpe, b), "flags": cflags})
    unit["children"] = kids
    for pm, missing, nm in sibling_gaps([k["marker"] for k in kids], max_gap=8):
        review.append({"code": "section_clause_gap", "division": div["division_id"], "level": "subclause",
                       "message": f"{clean(unit.get('title'))[:40]}: clause {missing} missing between "
                                  f"'{pm}' and '{nm}'"})


async def _scan_section_window(llm: LLMClient, renderer: PageRenderer, div: dict, unit: dict, lo: int, hi: int
                               ) -> Optional[WindowAnchors]:
    imgs = await asyncio.gather(*[renderer.page(p, ANCHOR_DPI) for p in range(lo, hi + 1)])
    parts: list[str | bytes] = []
    for p, img in zip(range(lo, hi + 1), imgs):
        parts.append(f"\n[Physical PDF Page Number {p}]")
        parts.append(img)
    parts.append(SECTION_CLAUSE_USER.format(marker=clean(unit["marker"]), title=clean(unit.get("title"))[:80],
                                            doc=clean(div["title"])[:80], a=lo, b=hi))
    res = await llm.call("section_clause_scan", role="reasoner", system=SECTION_CLAUSE_SYSTEM, parts=parts,
                         schema=WindowAnchors, thinking="low", media_resolution=ANCHOR_MEDIA,
                         max_output_tokens=4096, meta={"div": div["division_id"], "section": unit["marker"]})
    return res.parsed if res.ok else None  # type: ignore[return-value]


async def scan_subclauses(llm: LLMClient, renderer: PageRenderer, div: dict, unit: dict) -> None:
    """Scoped second pass: read ONLY this clause's pages and attach its direct sub-clauses (X.Y) as
    children. Prefix-filtered to the clause's own integer, so cross-references to other clauses and
    definition-list items can't leak in. Deeper levels (X.Y.Z) fold into the child's text later."""
    parent_int = normalize_marker(unit.get("marker"))
    if not parent_int or not re.fullmatch(r"\d+", parent_int):
        return
    a, b = unit["page_start"], unit["page_end"]
    wins = windows_in(a, b, size=8, stride=7)  # keep each vision call small
    results = await asyncio.gather(*[_scan_sub_window(llm, renderer, div, unit, lo, hi) for lo, hi in wins])
    raw = [s for r in results if r for s in r.anchors]
    # keep only genuine direct sub-clauses of THIS clause, deduped by normalised marker
    best: dict[str, AnchorItem] = {}
    for s in raw:
        key = normalize_marker(s.marker) or ""
        if not re.fullmatch(rf"{parent_int}\.\d+", key):
            continue
        if not a <= s.pdf_page <= b:
            continue
        prev = best.get(key)
        if prev is None or len(clean(s.title)) > len(clean(prev.title)):
            best[key] = AnchorItem(pdf_page=min(prev.pdf_page, s.pdf_page) if prev else s.pdf_page,
                                   kind="clause", marker=s.marker, title=s.title)
    slist = sorted(best.values(), key=lambda x: x.pdf_page)
    kids: list[dict] = []
    for j, s in enumerate(slist):
        cps = min(max(a, s.pdf_page), b)
        cpe = (slist[j + 1].pdf_page - 1) if j + 1 < len(slist) else b
        cflags: list[str] = []
        if cpe < cps:
            cpe = cps
            cflags.append("shares_start_page")
        kids.append({"unit_id": f"{unit['unit_id']}.{j + 1}", "kind": "subclause",
                     "marker": clean(s.marker) or None, "title": clean(s.title) or None,
                     "page_start": cps, "page_end": min(cpe, b), "flags": cflags})
    unit["children"] = kids


async def _scan_sub_window(llm: LLMClient, renderer: PageRenderer, div: dict, unit: dict, lo: int, hi: int
                           ) -> Optional[WindowAnchors]:
    imgs = await asyncio.gather(*[renderer.page(p, ANCHOR_DPI) for p in range(lo, hi + 1)])
    parts: list[str | bytes] = []
    for p, img in zip(range(lo, hi + 1), imgs):
        parts.append(f"\n[Physical PDF Page Number {p}]")
        parts.append(img)
    parts.append(SUBCLAUSE_USER.format(marker=clean(unit["marker"]), title=clean(unit.get("title"))[:80],
                                       doc=clean(div["title"])[:80], a=lo, b=hi))
    res = await llm.call("subclause_scan", role="reasoner", system=SUBCLAUSE_SYSTEM, parts=parts,
                         schema=WindowAnchors, thinking="low", media_resolution=ANCHOR_MEDIA,
                         max_output_tokens=4096, meta={"div": div["division_id"], "clause": unit["marker"]})
    return res.parsed if res.ok else None  # type: ignore[return-value]


def _leaf_count(units: list[dict]) -> int:
    """Deepest addressable nodes: a clause's children where it has any, else the clause itself."""
    return sum(len(u["children"]) if u.get("children") else 1 for u in units)


def _lint_sequence(kept: list[AnchorItem]) -> list[tuple[str, str, str]]:
    return sibling_gaps([a.marker for a in kept], max_gap=8)


def assert_division_coverage(units: list[dict], ds: int, de: int, did: str) -> None:
    """Units must run start..end with no gaps. Two units may share one page (a unit that starts partway
    down a page where the previous unit ends), so a start of `expected - 1` is allowed, never a jump."""
    expected = ds
    for u in units:
        ps, pe = u["page_start"], u["page_end"]
        if pe < ps or ps not in (expected, expected - 1):
            raise AssertionError(f"{did}: leaf coverage broken at {u['unit_id']}: expected {expected}, "
                                 f"got {ps}-{pe}")
        expected = pe + 1
    if expected != de + 1:
        raise AssertionError(f"{did}: leaf coverage ends at {expected - 1}, expected {de}")


# =============================================================================================
# Orchestration
# =============================================================================================
async def find_anchors(pdf_path: str | Path, out_dir: str | Path, settings: Settings | None = None) -> dict:
    started = datetime.now(timezone.utc)
    settings = settings or Settings()
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    ledger_path = out_dir / "divisions.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"{ledger_path} not found - run Step 1 (stage1_macro) first.")
    data = json.loads(ledger_path.read_text())
    divisions = data["divisions"]
    renderer = PageRenderer(pdf_path, settings)
    review: list[dict] = []
    try:
        llm = LLMClient(settings, GeminiBackend(settings))
        win_sem = asyncio.Semaphore(max(2, settings.concurrency // 3))

        async def one_division(div: dict) -> dict:
            ds, de = div["start_page"], div["end_page"]
            wins = windows_in(ds, de)

            async def one_win(lo: int, hi: int) -> Optional[WindowAnchors]:
                async with win_sem:
                    return await scan_anchor_window(llm, renderer, div, lo, hi)

            results = await asyncio.gather(*[one_win(lo, hi) for lo, hi in wins])
            raw = [a for r in results if r for a in r.anchors]
            unique, notes = _collect(raw, ds, de)
            for nt in notes:
                review.append({"code": "anchor_page_ambiguous", "division": div["division_id"], "message": nt})

            # Stage 1.5 decides the numbering scheme once, with document-level context. When it says
            # 'section', that is authoritative: the named SECTIONs are the clause tier (per the rule "if a
            # section name is there, treat that level as clause") and the per-section numbered clauses nest
            # under them. The window vote (_resolve_level) is only a fallback for un-classified divisions.
            scheme = div.get("numbering_scheme")
            sections = sorted([a for a in unique if a.kind == "section"], key=lambda x: x.pdf_page)
            if scheme == "section" and len(sections) >= MIN_SECTIONS:
                level, kept = "section", sections
            else:
                level, kept = _resolve_level(unique)

            title = clean(div["title"])
            if level == "section":
                for pm, missing, nm in _lint_sequence(kept):
                    review.append({"code": "anchor_numbering_gap", "division": div["division_id"], "level": "section",
                                   "message": f"{title[:50]}: section {missing} missing between '{pm}' and '{nm}'"})
                units = _build_units(div, "clause", kept)  # emit sections as clause-tier units
                assert_division_coverage(units, ds, de, div["division_id"])
                # Scoped pass per section: read its own pages fresh and attach its numbered clauses as
                # sub-clause children (recovers clauses on the section's heading page, which the main pass
                # suppresses because it is told not to descend into a section).
                await asyncio.gather(*[
                    scan_section_clauses(llm, renderer, div, u, review) for u in units if u["kind"] == "clause"])
                title = _trim_embedded_section(title)  # drop 'SECTION 1 ...' glued onto the division banner
            else:
                kept, folded = _fold_subclauses(level, kept)
                for pm, missing, nm in _lint_sequence(kept):  # gaps checked at the main level only
                    review.append({"code": "anchor_numbering_gap", "division": div["division_id"], "level": level,
                                   "message": f"{title[:50]}: {level} {missing} missing between '{pm}' and '{nm}'"})
                units = _build_units(div, level, kept)
                assert_division_coverage(units, ds, de, div["division_id"])
                # Scoped second pass: direct sub-clauses per main clause (clause-level divisions only).
                if level == "clause":
                    await asyncio.gather(*[
                        scan_subclauses(llm, renderer, div, u) for u in units if u["kind"] == "clause"])
            return {"division_id": div["division_id"], "division_type": div["division_type"],
                    "title": title, "start_page": ds, "end_page": de,
                    "leaf_level": level, "subclauses_nested": sum(len(u.get("children", [])) for u in units),
                    "unit_count": len(units), "leaf_count": _leaf_count(units), "units": units}

        out_divs = await asyncio.gather(*[one_division(d) for d in divisions])
        cost = llm.cost.report()
        finished = datetime.now(timezone.utc)
        result = {
            "source": data.get("source", {}),
            "generated_at": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 1),
            "cost_usd": cost["total_usd"],
            "stats": {"divisions": len(out_divs), "unit_count": sum(d["unit_count"] for d in out_divs),
                      "leaf_units": sum(d["leaf_count"] for d in out_divs), "review": len(review)},
            "divisions": list(out_divs),
            "review": review,
        }
        (out_dir / "leaves.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        log.info("done: %d divisions, %d leaf units, %d review, $%.3f -> %s", len(out_divs),
                 result["stats"]["leaf_units"], len(review), cost["total_usd"], out_dir / "leaves.json")
        return result
    finally:
        renderer.close()


def _print_leaves(result: dict) -> None:
    print(f"\n--- LEAF UNITS: {result['source'].get('file', '')} ---")
    for d in result["divisions"]:
        print(f"\n{d['division_id']} {d['division_type']} [{d['leaf_level']}] pp.{d['start_page']}-{d['end_page']} "
              f"- {d['title'][:60]}  ({d['unit_count']} units)")
        for u in d["units"]:
            rng = f"{u['page_start']}-{u['page_end']}"
            mk = f"{u['marker']} " if u.get("marker") else ""
            fl = f"  [{', '.join(u['flags'])}]" if u["flags"] else ""
            print(f"    {u['unit_id']:<14} {u['kind']:<9} {rng:<10} {mk}{(u['title'] or '')[:58]}{fl}")
            for c in u.get("children", []):
                crng = f"{c['page_start']}-{c['page_end']}"
                cmk = f"{c['marker']} " if c.get("marker") else ""
                print(f"        {c['unit_id']:<12} {'subclause':<9} {crng:<10} {cmk}{(c['title'] or '')[:52]}")
    r = result["review"]
    if r:
        print(f"\n{len(r)} review items:")
        for it in r[:40]:
            print(f"  [{it['code']}] {it.get('division','')}: {it['message']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epc_parser.stage2_anchors",
                                 description="Step 2: leaf-unit (section/clause) anchor discovery per division.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out_macro"), help="dir containing divisions.json")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--reasoner-model")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    s = Settings()
    if a.concurrency:
        s.concurrency = a.concurrency
    if a.reasoner_model:
        s.model_reasoner = a.reasoner_model
    if a.cache_dir:
        s.cache_dir = a.cache_dir

    result = asyncio.run(find_anchors(a.pdf, a.out, s))
    _print_leaves(result)


if __name__ == "__main__":
    main()
