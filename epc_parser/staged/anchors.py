"""Stage 2 - leaf-unit anchor discovery per division, TWO levels.

Pass A (main): the coarsest numbered/titled unit per division - a Section where sections exist, else a
main Clause. Clean top-level detection only (never sub-clauses), so a definitions list / contents list
can't be mistaken for clauses.

Pass B (scoped sub-clauses): for each main clause, a page-range-scoped read returns its DIRECT
sub-clauses (X.Y), prefix-filtered to that clause's own integer so cross-references and definition
items can't leak in. Sub-clauses become children of the clause; deeper levels fold into the child's
text at extraction. Level-1 page ranges stay contiguous within the division.
"""
from __future__ import annotations

import asyncio
import re
from typing import List, Optional, Tuple

import logging
logger = logging.getLogger("epc.staged.anchors")

from ..config import Settings as StagedConfig
from ..llm import LLMClient as StagedLLM
from .models import AnchorItem, WindowAnchors
from ..numbering import clean, norm_text, normalize_marker, sibling_gaps
from ..render import PageRenderer

ANCHOR_SYSTEM = (
    "You locate the TOP-LEVEL structural units of ONE document inside an Indian EPC contract. A "
    "top-level unit is the COARSEST numbered or titled division of this document:\n"
    "  - If it is organised into titled SECTIONS or PARTS (e.g. 'SECTION A - GENERAL', 'PART II'), "
    "return those SECTIONS. Do NOT descend into the clauses inside them.\n"
    "  - Otherwise return the MAIN CLAUSES (e.g. '1. DEFINITIONS', '5. SUSPENSION AND TERMINATION', "
    "'Clause 12'). An operative numbered clause counts whether it is printed '1.', '1)', or 'Clause 1' - "
    "a bare number with a bracket like '1)' '2)' is a MAIN CLAUSE, NOT a lettered sub-item like '(a)'. "
    "Return EVERY main clause, including the first ones.\n"
    "  - A LETTER-FORM document (Letter of Intent / LOI, Letter of Award or Acceptance / LOA, notice, "
    "minutes) IS structured when it sets out its terms as NUMBERED points or paragraphs: return each such "
    "numbered term as a MAIN CLAUSE (marker + title verbatim), exactly as for a contract's clauses. Only "
    "the letter FURNITURE is not a clause - the letterhead, reference/date line, addressee/'To' block, "
    "subject line and salutation. Do NOT collapse a letter that carries numbered terms into nothing and do "
    "NOT return it as one big blob: return its numbered terms as separate clauses. Return an EMPTY list "
    "ONLY when a letter/page genuinely has NO numbered terms at all (a pure transmittal/cover note, a "
    "title or divider page, a stamp/franking sheet).\n"
    "  - PREAMBLE vs clauses: an introductory PREAMBLE or set of RECITALS is NOT a clause - do NOT return "
    "it as a unit. The preamble may carry its OWN numbered recital points, and those recitals can OVERFLOW "
    "onto the next page(s) before the operative clauses begin - so the first main clause does NOT "
    "necessarily start at the top of the page after the preamble. Return a main clause only from where its "
    "OWN heading is actually printed (the operative clauses usually restart their numbering at 1 after the "
    "recitals end); do not treat a preamble recital point as a main clause, and do not start clause 1 "
    "before the recitals have finished.\n"
    "  - NEVER return a sub-clause or lower level: not 1.1, not 5.2, not 5.2.3, not (a), not (i).\n"
    "  - A numbered list of defined terms inside a 'Definitions' clause, or a contents/index list of "
    "clause titles, is NOT a set of top-level clauses - it belongs to the ONE clause that contains it.\n"
    "  - Do NOT treat a number that merely appears as a CROSS-REFERENCE in body text as a heading.\n"
    "  - Ignore running headers/footers, page numbers, tables, figures and ordinary body text.\n"
    "For each unit report the physical PDF page it BEGINS on (from the label before each image), its "
    "marker exactly as printed, its title verbatim, and whether it is a 'section' or a 'clause'. If no "
    "top-level unit begins on these pages, return an empty list."
)
ANCHOR_USER = ("Document: '{title}' (type {dtype}), spanning PDF pages {ds}-{de}. These images are "
               "pages {lo}-{hi} of it. Return the top-level units that BEGIN on these pages.")

def windows_in(lo: int, hi: int, size: int, stride: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    start = lo
    while start <= hi:
        end = min(start + size - 1, hi)
        out.append((start, end))
        if end == hi:
            break
        start += stride
    return out


async def _scan(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, div: dict, lo: int, hi: int,
                system: str, user: str) -> Optional[WindowAnchors]:
    imgs = await asyncio.gather(*[renderer.page(p, cfg.anchor_dpi) for p in range(lo, hi + 1)])
    parts: list = []
    for p, img in zip(range(lo, hi + 1), imgs):
        parts.append(f"\n[Physical PDF Page Number {p}]")
        parts.append(img)
    parts.append(user)
    res = await llm.call("anchors", role="reasoner", system=system, parts=parts, schema=WindowAnchors,
                         thinking="low", media_resolution=cfg.anchor_media, max_output_tokens=4096)
    return res.parsed if res.ok else None


def _anchor_key(a: AnchorItem) -> str:
    base = normalize_marker(a.marker) or norm_text(a.title)[:40] or f"p{a.pdf_page}"
    return f"{a.kind}:{base}"  # a Section and a clause can share a number ('SECTION 1' vs '1.0'); keep them apart


def _trim_embedded_section(title: str) -> str:
    """A section-structured GCC's start page often carries BOTH the division banner and its first
    section, so the refined title can come out as 'Section F - General Conditions of ContractSection 1 -
    Definitions...'. Cut the embedded 'SECTION <n>' (the first section is a unit of its own now). Only
    trims when a SECOND, digit-numbered SECTION is glued on, so a legitimately 'Section 4 - ...' title is
    left untouched."""
    m = re.search(r"(.+?\S)\s*section\s*-?\s*\d", title, re.IGNORECASE)
    return m.group(1).strip() if m and len(m.group(1).strip()) >= 10 else title


def _collect(anchors, ds, de):
    best: dict = {}
    notes: List[str] = []
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
        best[k] = AnchorItem(pdf_page=page, kind=prev.kind, marker=prev.marker or a.marker, title=title)
    return sorted(best.values(), key=lambda x: x.pdf_page), notes


def _resolve_level(anchors, min_sections):
    sections = [a for a in anchors if a.kind == "section"]
    clauses = [a for a in anchors if a.kind == "clause"]
    if len(sections) >= min_sections and len(sections) >= len(clauses):
        return "section", sorted(sections, key=lambda x: x.pdf_page)
    if clauses:
        return "clause", sorted(clauses, key=lambda x: x.pdf_page)
    if sections:
        return "section", sorted(sections, key=lambda x: x.pdf_page)
    return "whole", []


def _fold_subclauses(level, kept):
    """Main pass keeps only bare-integer clauses at level 1; dotted markers it happened to return are
    dropped (the scoped sub-pass finds real X.Y). If only dotted markers exist, keep them as leaves."""
    if level != "clause":
        return kept, 0
    # A main clause is a bare integer ('5') OR a numeric bracket ('5)' / '(5)', which normalizes to
    # '(5)') - operative clauses in deeds/guarantees are often printed '1)', '2)'. A LETTERED bracket
    # ('(a)', '(iv)') is a sub-item, not a main clause, and is still dropped.
    main = [a for a in kept if re.fullmatch(r"\(?\d+\)?", normalize_marker(a.marker) or "")]
    if main:
        return main, len(kept) - len(main)
    return kept, 0


def _build_units(div, level, kept):
    did, ds, de = div["division_id"], div["start_page"], div["end_page"]
    if not kept:
        return [{"unit_id": f"{did}-U01", "kind": "whole", "marker": None,
                 "title": clean(div["title"]) or None, "page_start": ds, "page_end": de,
                 "flags": [], "children": []}]
    units: List[dict] = []
    n = 0
    if kept[0].pdf_page > ds:
        n += 1
        units.append({"unit_id": f"{did}-U{n:02d}", "kind": "preamble", "marker": None, "title": "Preamble",
                      "page_start": ds, "page_end": kept[0].pdf_page - 1, "flags": [], "children": []})
    for i, a in enumerate(kept):
        ps = max(ds, a.pdf_page)
        pe = (kept[i + 1].pdf_page - 1) if i + 1 < len(kept) else de
        flags: List[str] = []
        if pe < ps:
            pe = ps
            flags.append("shares_start_page")
        n += 1
        units.append({"unit_id": f"{did}-U{n:02d}", "kind": level, "marker": clean(a.marker) or None,
                      "title": clean(a.title) or None, "page_start": ps, "page_end": pe,
                      "flags": flags, "children": []})
    return units


def assert_division_coverage(units, ds, de, did):
    expected = ds
    for u in units:
        ps, pe = u["page_start"], u["page_end"]
        if pe < ps or ps not in (expected, expected - 1):
            raise AssertionError(f"{did}: leaf coverage broken at {u['unit_id']}: expected {expected}, got {ps}-{pe}")
        expected = pe + 1
    if expected != de + 1:
        raise AssertionError(f"{did}: leaf coverage ends at {expected - 1}, expected {de}")


def _leaf_count(units):
    return sum(len(u["children"]) if u.get("children") else 1 for u in units)


async def run(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, divisions: List[dict]
              ) -> Tuple[List[dict], List[dict]]:
    win_sem = asyncio.Semaphore(max(2, cfg.concurrency // 3))
    review: List[dict] = []

    async def one_div(div):
        ds, de = div["start_page"], div["end_page"]
        wins = windows_in(ds, de, cfg.anchor_window, cfg.anchor_stride)

        async def one_win(lo, hi):
            async with win_sem:
                user = ANCHOR_USER.format(title=clean(div["title"])[:120], dtype=div["division_type"],
                                          ds=ds, de=de, lo=lo, hi=hi)
                return await _scan(llm, renderer, cfg, div, lo, hi, ANCHOR_SYSTEM, user)

        results = await asyncio.gather(*[one_win(lo, hi) for lo, hi in wins])
        raw = [a for r in results if r for a in r.anchors]
        unique, notes = _collect(raw, ds, de)
        for nt in notes:
            review.append({"code": "anchor_page_ambiguous", "division": div["division_id"], "message": nt})

        # Stage 1.5 decides the numbering scheme once, with document-level context. When it says
        # 'section', that is authoritative: the named SECTIONs are the clause tier (per the rule "if a
        # section name is there, treat that level as clause") and each section's own numbered clauses
        # (which restart per section) become its children, derived in Stage 3 from the block markers of
        # the verbatim read - so clauses on a section's heading page are captured, not suppressed. The
        # window vote (_resolve_level) is only a fallback for un-classified divisions.
        scheme = div.get("numbering_scheme")
        sections = sorted([a for a in unique if a.kind == "section"], key=lambda x: x.pdf_page)
        if scheme == "section" and len(sections) >= cfg.min_sections:
            level, kept = "section", sections
        else:
            level, kept = _resolve_level(unique, cfg.min_sections)

        title = clean(div["title"])
        if level == "section":
            for pm, missing, nm in sibling_gaps([a.marker for a in kept], max_gap=cfg.max_gap):
                review.append({"code": "anchor_numbering_gap", "division": div["division_id"], "level": "section",
                               "message": f"{title[:50]}: section {missing} missing between '{pm}' and '{nm}'"})
            units = _build_units(div, "clause", kept)  # emit sections as clause-tier units
            title = _trim_embedded_section(title)      # drop 'Section 1 ...' glued onto the division banner
        else:
            kept, folded = _fold_subclauses(level, kept)
            for pm, missing, nm in sibling_gaps([a.marker for a in kept], max_gap=cfg.max_gap):
                review.append({"code": "anchor_numbering_gap", "division": div["division_id"], "level": level,
                               "message": f"{title[:50]}: {level} {missing} missing between '{pm}' and '{nm}'"})
            units = _build_units(div, level, kept)
        assert_division_coverage(units, ds, de, div["division_id"])
        # Sub-clauses (X.Y for clause divisions, and each section's own numbered clauses for section
        # divisions) are NOT scanned here — they are derived deterministically from the verbatim extract
        # pass's block markers (Stage 3), which reads the same pages at higher resolution. This removes
        # ~1 LLM call per clause with no quality loss.
        return {"division_id": div["division_id"], "division_type": div["division_type"],
                "title": title, "running_header": div.get("running_header"),
                "start_page": ds, "end_page": de, "leaf_level": level,
                "unit_count": len(units), "leaf_count": _leaf_count(units), "units": units}

    leaves = list(await asyncio.gather(*[one_div(d) for d in divisions]))
    logger.info(f"[staged.anchors] {len(leaves)} divisions, {sum(d['leaf_count'] for d in leaves)} leaves, "
                f"{len(review)} review")
    return leaves, review
