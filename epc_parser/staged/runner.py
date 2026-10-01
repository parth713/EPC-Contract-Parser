"""Staged pipeline entry point (coarse-to-fine). Runs macro -> refine -> anchors -> gap-hunt ->
extract -> dates -> enrich in memory over a PDF and returns the full structure plus a flat clause
projection. Self-contained in epc: it uses epc's own `Settings`, cached `LLMClient` (disk cache + cost
report) and `PageRenderer`, and carries NO external application coupling (no database ids or foreign keys).

Leaf identity (product rule): the clause number IS the leaf node — a Section where the document uses
sections, otherwise a main Clause — with the clause/section title given in brackets, formatted:
"5 (The Contractor: Obligations, Duties And Other Essentials)", "SECTION A (General)".
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from ..config import Settings
from ..llm import Backend, GeminiBackend, LLMClient
from ..numbering import clean
from ..render import PageRenderer
from . import anchors, dates, enrich, extract, gaphunt, macro, refine

logger = logging.getLogger("epc.staged.runner")

_ACRONYMS = {"GCC", "SCC", "BOQ", "LOI", "LOA", "NIT", "ITB", "ITT", "EHS", "NSC", "RCC", "PHE",
             "HVAC", "FF", "CCTV", "PVC", "MEP", "SLA", "VAT", "GST", "TDS", "EPC", "PDF"}
_TITLE_MAX = 200
_ID_TITLE_MAX = 160
# A 'whole'-type unit (unstructured cover/title/divider matter) is dropped from the clause projection
# only when it starts within this many pages of the document start — i.e. genuine front matter. A 'whole'
# division that begins later is kept as a clause row, since it more likely holds real content.
_WHOLE_CLAUSE_MAX_PAGE = 5


def _pretty(s: Optional[str]) -> str:
    """Readable Title Case for a heading, preserving known acronyms. Leaves already-mixed-case text
    (which is usually fine) alone."""
    s = clean(s)
    if not s:
        return ""
    letters = [c for c in s if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        words = []
        for w in s.split():
            core = re.sub(r"[^A-Za-z0-9]", "", w)
            words.append(w.upper() if core.upper() in _ACRONYMS else w.capitalize())
        s = " ".join(words)
    return s


def _display_number(marker: Optional[str], pretty_title: str) -> str:
    m = clean(marker)
    t = pretty_title[:_ID_TITLE_MAX]
    if m and t and t.lower() != m.lower():
        return f"{m} ({t})"
    return m or t or "Untitled"


def _drop_from_clauses(kind: Optional[str], page_start: Optional[int] = None) -> bool:
    """Units that are NOT clause rows, dropped from the flat projection though the full structure keeps
    the record.

    A 'preamble' is the recitals/boilerplate before the first real clause — never a clause row, wherever
    it sits. A 'whole' unit is an ENTIRE division with no clause structure — stamp paper, franking/cover
    sheets, and divider/title-only pages. Those are front matter, so a 'whole' unit is dropped ONLY when
    it starts within the first _WHOLE_CLAUSE_MAX_PAGE pages of the document; a 'whole' division that
    begins later is KEPT (it more likely holds real, if unstructured, content)."""
    kind = kind or ""
    if kind == "preamble":
        return True
    if kind == "whole":
        return (page_start or 1) <= _WHOLE_CLAUSE_MAX_PAGE
    return False


def _to_clause_dicts(leaves: List[dict]) -> List[dict]:
    """Flat clause projection: one row per non-dropped leaf unit. The bracket next to a clause number
    names its DIVISION, not the clause itself."""
    out: List[dict] = []
    for d in leaves:
        div_title = _pretty(d.get("title"))
        for u in d.get("units", []):
            if _drop_from_clauses(u.get("kind"), u.get("page_start")):
                continue
            content = (u.get("text") or "").strip()
            if not content:
                continue  # skip empty leaves (cover sheets, blank dividers)
            pretty = _pretty(u.get("title"))
            out.append({
                "clause_id": _display_number(u.get("marker"), div_title),
                "clause_number": _display_number(u.get("marker"), div_title),
                "clause_title": (pretty or clean(u.get("marker")) or "No Title found")[:_TITLE_MAX],
                "clause_description": u.get("clause_description") or "",
                "clause_content": content,
                "priority": u.get("priority", "Medium"),
                "risk_level": u.get("risk_level", "medium"),
                "clause_type": u.get("clause_type", "General"),
                "page": u.get("page_start"),
                "division_id": d.get("division_id"),
                "division_title": div_title,
            })
    return out


async def run(pdf_path: str | Path, settings: Optional[Settings] = None,
              backend: Optional[Backend] = None) -> dict:
    """Run the staged pipeline over a PDF. Returns the full structure (divisions + leaves with units and
    verbatim text), the flat clause projection, the contract date-anchor registry + principal parties,
    the review queue, and the cost report. Nothing is written to disk here — see `staged.output`."""
    settings = settings or Settings()
    pdf_path = Path(pdf_path)
    renderer = PageRenderer(pdf_path, settings)
    review: List[dict] = []
    try:
        total = renderer.page_count
        source_sha256 = renderer.sha256()
        if total <= 0:
            return {"source": str(pdf_path), "source_sha256": source_sha256, "page_count": 0,
                    "divisions": [], "leaves": [], "clauses": [],
                    "date_anchors": {"registry": {}, "anchors": [], "conflicts": [], "parties": []},
                    "parties": [], "review": [],
                    "extraction_metadata": {"pipeline": "staged", "pages": 0},
                    "cost": {}, "status": "completed", "failed_page_ranges": [], "notes": "empty document"}
        llm = LLMClient(settings, backend or GeminiBackend(settings))

        divisions, toc_pages = await macro.run(llm, renderer, settings, total)
        await refine.run(llm, renderer, settings, divisions)
        macro.fold_internal_tocs(divisions)  # doc-internal TOC -> front matter of the doc it indexes
        leaves, anchor_review = await anchors.run(llm, renderer, settings, divisions)
        review += anchor_review
        review += await gaphunt.run(llm, renderer, settings, leaves)
        review += await extract.run(llm, renderer, settings, leaves)
        # Date-anchor engine AFTER extract (independent of TOC/structure): covers read as images
        # (handwritten/stamped dates), body candidates as text, with conflict resolution. Defensive —
        # never raises, never blocks clause parsing.
        date_anchors = await dates.run(llm, renderer, settings, leaves)
        await enrich.run(llm, settings, leaves)

        clauses = _to_clause_dicts(leaves)
        parties = date_anchors.get("parties", [])
        failed_pages = sorted({p for r in review if r.get("code") == "page_unreadable" for p in r.get("pages", [])})
        failed_ranges = [str(p) for p in failed_pages]
        status = "INCOMPLETE" if (failed_ranges and not clauses) else "completed"
        llm_calls = sum(int(m.get("calls", 0)) for m in llm.cost.by_model.values())
        cost = llm.cost.report()
        notes = (f"staged pipeline: {len(divisions)} divisions, {len(leaves)} leaf groups, "
                 f"{len(clauses)} clauses, {len(failed_ranges)} unreadable page(s), "
                 f"llm_calls={llm_calls}, ${cost.get('total_usd', 0):.3f}")
        logger.info("[staged.runner] done: %s", notes)

        return {
            "source": str(pdf_path),
            "source_sha256": source_sha256,
            "page_count": total,
            "divisions": divisions,
            "leaves": leaves,
            "clauses": clauses,
            "date_anchors": date_anchors,
            "parties": parties,
            "review": review,
            "extraction_metadata": {
                "pipeline": "staged",
                "pages": total,
                "divisions": len(divisions),
                "leaf_groups": len(leaves),
                "total_clauses_extracted": len(clauses),
                "master_toc_pages": toc_pages,
                "llm_calls": llm_calls,
                "pipeline_status": status,
                "failed_page_ranges": failed_ranges,
                "review": review,
            },
            "cost": cost,
            "status": status,
            "failed_page_ranges": failed_ranges,
            "notes": notes,
        }
    finally:
        renderer.close()
