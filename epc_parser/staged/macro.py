"""Stage 1 - macro division sweep. Finds where each major partition begins; stitches a contiguous,
gap-free division map. 15-page sliding windows (1-page overlap), deterministic stitch."""
from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

import logging
logger = logging.getLogger("epc.staged.macro")

from ..config import Settings as StagedConfig
from ..llm import LLMClient as StagedLLM
from .models import BatchDivisionResponse, BoundaryItem
from ..numbering import clean
from ..render import PageRenderer

MACRO_SYSTEM = (
    "You are a strict boundary parser for scanned Indian EPC tender/contract bundles. Your ONLY task is "
    "to find where each MAJOR macro-document (a top-level partition of the bundle) begins - never clauses "
    "or sub-clauses.\n"
    "Macro-documents include: e-stamp paper, Contract Agreement, Recitals, Letter of Award/Intent, Pre-Bid "
    "Meeting Minutes, Notice Inviting Tender (NIT/TENDER_NOTICE), Instructions to Tenderers/Bidders "
    "(ITB/ITT = INSTRUCTIONS_TO_TENDERERS), Form of Tender, General Conditions of Contract (GCC), Special "
    "Conditions of Contract (SCC), Technical Specifications, Scope of Work, Schedules, Annexures, Appendices, "
    "Bill of Quantities (BoQ), Drawings, Bank Guarantees, proforma/format samples (PROFORMA_OR_FORM), and "
    "printed Index/Table-of-Contents pages.\n\n"
    "USE THE RUNNING HEADER/FOOTER. Most pages carry a running header/footer naming the document or volume. "
    "A change in the running header/volume almost always marks a new macro-document (report it even without "
    "an obvious cover); a constant header means the SAME document (do NOT split on internal sub-headings). "
    "Report the running header for each boundary.\n\n"
    "RULES:\n"
    "  - Report a boundary for a genuine document partition: a cover/title/divider page, a top-level section "
    "or volume heading, a running-header change, or the first page of a clearly new document.\n"
    "  - Also report a boundary for any clearly separate, distinctly titled document with its own cover/title "
    "(e.g. a titled Annexure/Appendix/Schedule), even inside the same volume.\n"
    "  - Do NOT split one document into several divisions just because it spans many pages.\n"
    "  - NEVER report an ordinary clause/sub-clause heading (e.g. '5. TERMINATION', '5.1') as a boundary.\n"
    "  - Use the exact physical PDF page number printed in the label before each image.\n"
    "  - If a window has no new macro-document, return an empty boundaries list. Report pages only in-window."
)
_HEAD = ("--- BEGIN WINDOW (physical PDF pages {lo} to {hi}) ---\nInspect the page images below (each is "
         "preceded by its exact physical PDF page number) and identify every point where a NEW macro-document begins.")
_TAIL = "Return the structured JSON division breakdown for pages {lo} to {hi} only."


def make_windows(total: int, size: int, stride: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    start = 1
    while start <= total:
        end = min(start + size - 1, total)
        windows.append((start, end))
        if end == total:
            break
        start += stride
    return windows


async def _scan(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, lo: int, hi: int
                ) -> Optional[BatchDivisionResponse]:
    imgs = await asyncio.gather(*[renderer.page(p, cfg.macro_dpi) for p in range(lo, hi + 1)])
    parts: list = [_HEAD.format(lo=lo, hi=hi)]
    for p, img in zip(range(lo, hi + 1), imgs):
        parts.append(f"\n[IMAGE RECORD: Physical PDF Page Number {p}]")
        parts.append(img)
    parts.append(_TAIL.format(lo=lo, hi=hi))
    res = await llm.call("macro", role="reasoner", system=MACRO_SYSTEM, parts=parts, schema=BatchDivisionResponse,
                         thinking="low", media_resolution=cfg.macro_media, max_output_tokens=4096)
    return res.parsed if res.ok else None


def _collect(batches, total):
    by_page: dict = {}
    toc: set = set()
    for batch in batches:
        if batch is None:
            continue
        if batch.contains_master_toc:
            toc.update(p for p in batch.master_toc_pages if 1 <= p <= total)
        for b in batch.boundaries_detected:
            if not 1 <= b.pdf_page_number <= total:
                continue
            prev = by_page.get(b.pdf_page_number)
            if prev is None or len(clean(b.exact_title_text)) > len(clean(prev.exact_title_text)):
                by_page[b.pdf_page_number] = b
    return by_page, toc


def stitch(batches, total) -> Tuple[List[dict], List[int]]:
    by_page, toc = _collect(batches, total)
    items = [(by_page[p], []) for p in sorted(by_page)]
    if not items or items[0][0].pdf_page_number > 1:
        front = BoundaryItem(pdf_page_number=1, division_type="OTHER_DOCUMENT",
                             exact_title_text="PRELIMINARY / FRONT MATTER", position_on_page="TOP",
                             is_standalone_cover_sheet=False)
        items.insert(0, (front, ["inferred_front_matter"]))
    starts = [b.pdf_page_number for b, _ in items]
    ledger: List[dict] = []
    for i, (b, flags) in enumerate(items):
        sp = b.pdf_page_number
        ep = (starts[i + 1] - 1) if i + 1 < len(items) else total
        if b.position_on_page in ("MIDDLE", "BOTTOM"):
            flags.append("mid_page_start")
        if b.is_standalone_cover_sheet:
            flags.append("cover_sheet")
        if i + 1 < len(items) and b.division_type == items[i + 1][0].division_type and ep - sp + 1 <= 2:
            flags.append("possible_overlap_split")
        ledger.append({
            "division_id": f"DIV-{i + 1:02d}",
            "division_type": b.division_type,
            "title": clean(b.exact_title_text) or f"[{b.division_type}]",
            "running_header": clean(b.running_header) or None,
            "start_page": sp, "end_page": ep, "page_count": ep - sp + 1,
            "start_position": b.position_on_page, "is_cover_sheet": b.is_standalone_cover_sheet,
            "confidence": "inferred" if "inferred_front_matter" in flags else "detected",
            "flags": flags,
        })
    _assert_coverage(ledger, total)
    return ledger, sorted(toc)


def _assert_coverage(ledger, total):
    expected = 1
    for d in ledger:
        if d["start_page"] != expected or d["end_page"] < d["start_page"]:
            raise AssertionError(f"coverage broken at {d['division_id']}: expected {expected}")
        expected = d["end_page"] + 1
    if expected != total + 1:
        raise AssertionError(f"coverage ends at {expected - 1}, expected {total}")


# A document-internal Table-of-Contents/Index is folded back into whichever substantive document it
# indexes. These are the types that own a body worth indexing; a TOC sitting after one of them is its
# contents page, not a partition of its own.
_FOLDABLE_INTO = {
    "RECITALS", "CONTRACT_AGREEMENT", "LETTER_OF_AWARD", "LETTER_OF_INTENT", "PRE_BID_MINUTES",
    "TENDER_NOTICE", "INSTRUCTIONS_TO_TENDERERS", "FORM_OF_TENDER", "GCC", "SCC", "TECHNICAL_SPECS",
    "SCOPE_OF_WORK", "SCHEDULE", "ANNEXURE", "APPENDIX", "BOQ", "DRAWINGS", "BANK_GUARANTEE",
    "PROFORMA_OR_FORM",
}


def _headers_conflict(a: Optional[str], b: Optional[str]) -> bool:
    """True only when both running headers are present AND name different documents (neither contains
    the other). A missing header on either side is not treated as a conflict, so the fold still runs."""
    a, b = clean(a).casefold(), clean(b).casefold()
    if not a or not b:
        return False
    return a not in b and b not in a


def fold_internal_tocs(divisions: List[dict]) -> int:
    """A printed Table-of-Contents / Index page that sits INSIDE a document (the document's own
    contents list) is not a top-level partition of the bundle - but the macro sweep reports it as a
    boundary anyway (its prompt lists Index/TOC pages as macro-documents). Left alone that boundary
    opens an INDEX_OR_TOC division which then owns every page up to the next document - the whole body
    it merely indexes - so all those clauses get bracketed with the wrong parent ('2 (Table of
    Contents)' instead of '2 (General Conditions of Contract)').

    Fold each such internal TOC back into the substantive division right before it: that division
    reclaims the TOC's pages as front matter. A front/master TOC (no substantive division before it)
    and a TOC under a conflicting running header (a genuinely separate contents volume) are left
    standing. Runs after `refine` so division types/headers are trustworthy. Mutates `divisions` in
    place; returns the number folded."""
    kept: List[dict] = []
    folded = 0
    for d in divisions:
        prev = kept[-1] if kept else None
        if (d["division_type"] == "INDEX_OR_TOC" and prev is not None
                and prev["division_type"] in _FOLDABLE_INTO
                and not _headers_conflict(prev.get("running_header"), d.get("running_header"))):
            prev["end_page"] = d["end_page"]
            prev["page_count"] = prev["end_page"] - prev["start_page"] + 1
            prev.setdefault("flags", []).append("absorbed_internal_toc")
            folded += 1
            continue
        kept.append(d)
    if folded:
        for i, d in enumerate(kept):
            d["division_id"] = f"DIV-{i + 1:02d}"
        divisions[:] = kept
        _assert_coverage(divisions, divisions[-1]["end_page"])
        logger.info(f"[staged.macro] folded {folded} internal TOC/index page(s) into their parent division")
    return folded


async def run(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, total: int
              ) -> Tuple[List[dict], List[int]]:
    windows = make_windows(total, cfg.macro_window, cfg.macro_stride)
    logger.info(f"[staged.macro] {total} pages, {len(windows)} windows")
    win_sem = asyncio.Semaphore(max(2, cfg.concurrency // 4))

    async def one(lo, hi):
        async with win_sem:
            return await _scan(llm, renderer, cfg, lo, hi)

    batches = await asyncio.gather(*[one(lo, hi) for lo, hi in windows])
    divisions, toc = stitch(batches, total)
    logger.info(f"[staged.macro] {len(divisions)} divisions ({sum(1 for b in batches if b is None)} windows failed)")
    return divisions, toc
