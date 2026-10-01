"""Step 1 - Macro division sweep.

The only job of this pass is *namespace protection*: find where each major legal partition of the
bundle begins, and stitch a contiguous, gap-free division map covering every page. It does NOT read
clauses or verbatim text (that is Step 2+). Because the model only scans macro layout (big section
headers, cover sheets, stamped divider pages), 15 low-resolution page images fit comfortably in one call.

Design:
  * 15-page sliding windows, stride 14 (1-page overlap so a boundary on a window edge is seen twice).
  * Rendering and the LLM call reuse the existing plumbing: PageRenderer (threaded + cached, so overlap
    pages are not re-rendered) and LLMClient (content-hash disk cache -> re-runs are free, backoff on
    429/5xx, cost tracking, JSON repair).
  * The stitch is deterministic: divisions are deduped by page (distinct pages are never merged, so a
    real boundary is never destroyed), and the result is ASSERTED to cover 1..N with no gaps.

Run:
    python -m epc_parser.stage1_macro "SBUT04 Contract Agreement (3).pdf" -o out_macro/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import Settings
from .llm import GeminiBackend, LLMClient
from .numbering import clean
from .render import PageRenderer

log = logging.getLogger("epc.macro")

# Render lightweight: title headers are huge, so 150 DPI is plenty. Cost is driven by media_resolution
# (see config.py note), not DPI; "medium" reads section titles reliably. Drop to "low" to cut cost if
# your scans have very large headers.
MACRO_DPI = 200
MACRO_MEDIA = "medium"
WINDOW_SIZE = 15
STRIDE = 14  # 1-page overlap

# Indian EPC-specific macro document types.
DivisionType = Literal[
    "RECITALS", "CONTRACT_AGREEMENT", "LETTER_OF_AWARD", "LETTER_OF_INTENT", "PRE_BID_MINUTES",
    "TENDER_NOTICE", "INSTRUCTIONS_TO_TENDERERS", "FORM_OF_TENDER",
    "GCC", "SCC", "TECHNICAL_SPECS", "SCOPE_OF_WORK", "SCHEDULE", "ANNEXURE", "APPENDIX", "BOQ",
    "DRAWINGS", "BANK_GUARANTEE", "PROFORMA_OR_FORM", "INDEX_OR_TOC", "OTHER_DOCUMENT",
]
Position = Literal["TOP", "MIDDLE", "BOTTOM", "FULL_PAGE"]


# =============================================================================================
# LLM schemas (flat + enum-based so Gemini structured output stays reliable)
# =============================================================================================
class BoundaryItem(BaseModel):
    pdf_page_number: int = Field(description="1-indexed physical PDF page where this division STARTS.")
    division_type: DivisionType = Field(description="Recognised macro legal document type.")
    exact_title_text: str = Field(description="Verbatim title/headline on the page, e.g. 'SECTION IV: SPECIAL CONDITIONS OF CONTRACT'.")
    position_on_page: Position = Field(description="Where the title appears on the page.")
    is_standalone_cover_sheet: bool = Field(description="True if the page is mostly blank and acts purely as a section divider sheet.")
    running_header: Optional[str] = Field(None, description="Running header/footer text on this start page identifying its document or volume, e.g. 'SBUT Volume 1 - SCC'. Empty if none.")


class BatchDivisionResponse(BaseModel):
    boundaries_detected: list[BoundaryItem] = Field(default_factory=list, description="All division starts detected in this window.")
    contains_master_toc: bool = Field(default=False, description="True if any page holds a printed master Table of Contents / Index.")
    master_toc_pages: list[int] = Field(default_factory=list, description="Pages that hold the printed contents/index.")
    running_header_text_observed: Optional[str] = Field(default=None, description="Dominant running header seen across the window.")


# =============================================================================================
# Prompts
# =============================================================================================
MACRO_SYSTEM = (
    "You are a strict boundary parser for scanned Indian EPC tender/contract bundles. "
    "Your ONLY task is to find where each MAJOR macro-document (a top-level partition of the bundle) begins "
    "- never clauses or sub-clauses.\n"
    "Macro-documents include: e-stamp paper, Contract Agreement, Recitals, Letter of Award/Intent, Pre-Bid "
    "Meeting Minutes, Notice Inviting Tender (NIT/TENDER_NOTICE), Instructions to Tenderers/Bidders "
    "(ITB/ITT = INSTRUCTIONS_TO_TENDERERS), Form of Tender, General Conditions of Contract (GCC), Special "
    "Conditions of Contract (SCC), Technical Specifications, Scope of Work, Schedules, Annexures, Appendices, "
    "Bill of Quantities (BoQ), Drawings, Bank Guarantees, proforma/format samples (PROFORMA_OR_FORM), and "
    "printed Index/Table-of-Contents pages.\n"
    "\n"
    "USE THE RUNNING HEADER/FOOTER. Most pages carry a running header or footer naming the document or volume "
    "they belong to (e.g. 'SBUT Volume 1 - SCC', 'Technical Specifications', 'Section 4 - GCC'). Read it on "
    "every page and treat it as a primary signal:\n"
    "  - When the running header or volume CHANGES from the previous page, that almost always marks the start "
    "of a new macro-document - report a boundary there even if the page has no obvious cover or title.\n"
    "  - When the running header or volume STAYS THE SAME, the pages belong to the SAME macro-document. Do NOT "
    "start a new division for internal sub-section headings, continuation pages, tables, or list items that "
    "carry the same header.\n"
    "  - For every boundary you report, also return the running header/footer text seen on that start page.\n"
    "\n"
    "RULES:\n"
    "  - Report a boundary for a genuine document partition: a cover/title/divider page, a top-level "
    "section or volume heading, a running-header/volume change, or the first page of a clearly new document.\n"
    "  - Also report a boundary for any clearly separate, distinctly titled document that has its own cover or "
    "title page (for example a titled Annexure, Appendix, or Schedule), even when it sits inside the same volume.\n"
    "  - Do NOT split one document into several divisions just because it spans many pages or has internal "
    "headings - a constant running header is your evidence that it is still one document.\n"
    "  - NEVER report an ordinary clause or sub-clause heading (e.g. '5. TERMINATION', '5.1', 'Article 12') "
    "as a boundary.\n"
    "  - Use the exact physical PDF page number printed in the label immediately before each image.\n"
    "  - If a window contains no new macro-document, return an empty boundaries list.\n"
    "  - Report page numbers only within the range of this window."
)

MACRO_USER_HEAD = (
    "--- BEGIN WINDOW (physical PDF pages {lo} to {hi}) ---\n"
    "Inspect the page images below (each is preceded by its exact physical PDF page number) and identify "
    "every point where a NEW macro-document begins."
)
MACRO_USER_TAIL = "Return the structured JSON division breakdown for pages {lo} to {hi} only."


# =============================================================================================
# Windows
# =============================================================================================
def make_windows(total: int, size: int = WINDOW_SIZE, stride: int = STRIDE) -> list[tuple[int, int]]:
    """Overlapping windows over 1..total, e.g. (1,15),(15,29),(29,43)..."""
    windows: list[tuple[int, int]] = []
    start = 1
    while start <= total:
        end = min(start + size - 1, total)
        windows.append((start, end))
        if end == total:
            break
        start += stride
    return windows


# =============================================================================================
# Batch scan
# =============================================================================================
async def scan_window(ctx_llm: LLMClient, renderer: PageRenderer, lo: int, hi: int) -> Optional[BatchDivisionResponse]:
    imgs = await asyncio.gather(*[renderer.page(p, MACRO_DPI) for p in range(lo, hi + 1)])
    parts: list[str | bytes] = [MACRO_USER_HEAD.format(lo=lo, hi=hi)]
    for p, img in zip(range(lo, hi + 1), imgs):
        parts.append(f"\n[IMAGE RECORD: Physical PDF Page Number {p}]")
        parts.append(img)
    parts.append(MACRO_USER_TAIL.format(lo=lo, hi=hi))

    res = await ctx_llm.call("macro_sweep", role="reasoner", system=MACRO_SYSTEM, parts=parts,
                             schema=BatchDivisionResponse, thinking="low", media_resolution=MACRO_MEDIA,
                             max_output_tokens=4096, meta={"lo": lo, "hi": hi})
    if not res.ok:
        log.warning("window %d-%d failed: %s", lo, hi, res.error or res.finish_reason)
        return None
    return res.parsed  # type: ignore[return-value]


# =============================================================================================
# Stitch -> contiguous division ledger
# =============================================================================================
def _clamp_and_collect(batches: list[Optional[BatchDivisionResponse]], total: int
                       ) -> tuple[dict[int, BoundaryItem], set[int]]:
    """Dedupe boundaries by start page (keep the most descriptive title). Also union master-TOC pages."""
    by_page: dict[int, BoundaryItem] = {}
    toc_pages: set[int] = set()
    for batch in batches:
        if batch is None:
            continue
        if batch.contains_master_toc:
            toc_pages.update(p for p in batch.master_toc_pages if 1 <= p <= total)
        for b in batch.boundaries_detected:
            if not 1 <= b.pdf_page_number <= total:
                continue
            prev = by_page.get(b.pdf_page_number)
            if prev is None or len(clean(b.exact_title_text)) > len(clean(prev.exact_title_text)):
                by_page[b.pdf_page_number] = b
    return by_page, toc_pages


def stitch_divisions(batches: list[Optional[BatchDivisionResponse]], total: int
                    ) -> tuple[list[dict], list[int]]:
    by_page, toc_pages = _clamp_and_collect(batches, total)
    # Each boundary carries a mutable flag list. Dedup-by-page already collapses the only overlap case
    # that occurs with a 1-page stride (a boundary landing on a shared page, reported at the same page
    # number by both windows). We never merge distinct pages - that could destroy a real division.
    items: list[tuple[BoundaryItem, list[str]]] = [(by_page[p], []) for p in sorted(by_page)]

    # Front matter fallback: if nothing starts at page 1, the run from page 1 is its own division.
    if not items or items[0][0].pdf_page_number > 1:
        front = BoundaryItem(pdf_page_number=1, division_type="OTHER_DOCUMENT",
                             exact_title_text="PRELIMINARY / FRONT MATTER", position_on_page="TOP",
                             is_standalone_cover_sheet=False)
        items.insert(0, (front, ["inferred_front_matter"]))

    starts = [b.pdf_page_number for b, _ in items]
    ledger: list[dict] = []
    for i, (b, flags) in enumerate(items):
        start_p = b.pdf_page_number
        end_p = (starts[i + 1] - 1) if i + 1 < len(items) else total
        if b.position_on_page in ("MIDDLE", "BOTTOM"):
            flags.append("mid_page_start")  # a later step may need to split the shared page
        if b.is_standalone_cover_sheet:
            flags.append("cover_sheet")
        # Advisory only: a very short division sharing its successor's type may be an overlap off-by-one
        # (model misread a page label). Flagged for review, never auto-merged.
        if i + 1 < len(items) and b.division_type == items[i + 1][0].division_type and end_p - start_p + 1 <= 2:
            flags.append("possible_overlap_split")
        ledger.append({
            "division_id": f"DIV-{i + 1:02d}",
            "division_type": b.division_type,
            "title": clean(b.exact_title_text) or f"[{b.division_type}]",
            "running_header": clean(b.running_header) or None,
            "start_page": start_p,
            "end_page": end_p,
            "page_count": end_p - start_p + 1,
            "start_position": b.position_on_page,
            "is_cover_sheet": b.is_standalone_cover_sheet,
            "confidence": "inferred" if "inferred_front_matter" in flags else "detected",
            "flags": flags,
        })
    assert_coverage(ledger, total)
    return ledger, sorted(toc_pages)


def assert_coverage(ledger: list[dict], total: int) -> None:
    expected = 1
    for d in ledger:
        if d["start_page"] != expected or d["end_page"] < d["start_page"]:
            raise AssertionError(f"coverage broken at {d['division_id']}: expected start {expected}, "
                                 f"got {d['start_page']}-{d['end_page']}")
        expected = d["end_page"] + 1
    if expected != total + 1:
        raise AssertionError(f"coverage ends at {expected - 1}, expected {total}")


# =============================================================================================
# Orchestration
# =============================================================================================
async def run_macro_sweep(pdf_path: str | Path, out_dir: str | Path, settings: Settings | None = None) -> dict:
    started = datetime.now(timezone.utc)
    settings = settings or Settings()
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = PageRenderer(pdf_path, settings)
    try:
        total = renderer.page_count
        llm = LLMClient(settings, GeminiBackend(settings))
        windows = make_windows(total)
        log.info("macro sweep: %d pages, %d windows (size %d, stride %d)", total, len(windows), WINDOW_SIZE, STRIDE)

        win_sem = asyncio.Semaphore(max(2, settings.concurrency // 4))  # bound images held in memory

        async def one(lo: int, hi: int) -> Optional[BatchDivisionResponse]:
            async with win_sem:
                r = await scan_window(llm, renderer, lo, hi)
            log.info("window %d-%d done (running cost $%.3f)", lo, hi, llm.cost.report()["total_usd"])
            return r

        batches = await asyncio.gather(*[one(lo, hi) for lo, hi in windows])
        failed = sum(1 for b in batches if b is None)
        if failed:
            log.warning("%d/%d windows failed; coverage is still contiguous but may be coarse", failed, len(windows))

        divisions, toc_pages = stitch_divisions(batches, total)
        cost = llm.cost.report()
        finished = datetime.now(timezone.utc)
        result = {
            "source": {"file": pdf_path.name, "sha256": renderer.sha256(), "page_count": total},
            "generated_at": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 1),
            "cost_usd": cost["total_usd"],
            "stats": {"divisions": len(divisions), "batches": len(windows), "batches_failed": failed},
            "master_toc_pages": toc_pages,
            "divisions": divisions,
        }
        (out_dir / "divisions.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        log.info("done: %d divisions over %d pages, $%.3f -> %s",
                 len(divisions), total, cost["total_usd"], out_dir / "divisions.json")
        return result
    finally:
        renderer.close()


def _print_ledger(result: dict) -> None:
    print(f"\n--- MACRO DIVISION LEDGER: {result['source']['file']} ({result['source']['page_count']} pages) ---")
    print(f"{'ID':<7} {'TYPE':<20} {'PAGES':<12} {'TITLE'}")
    for d in result["divisions"]:
        rng = f"{d['start_page']}-{d['end_page']}"
        flags = f"  [{', '.join(d['flags'])}]" if d["flags"] else ""
        hdr = f'  hdr="{d["running_header"][:40]}"' if d.get("running_header") else ""
        print(f"{d['division_id']:<7} {d['division_type']:<20} {rng:<12} {d['title'][:60]}{hdr}{flags}")
    if result["master_toc_pages"]:
        print(f"\nPrinted index/TOC pages detected: {result['master_toc_pages']}")
    print(f"\n{result['stats']['divisions']} divisions | {result['stats']['batches']} batches "
          f"({result['stats']['batches_failed']} failed) | ${result['cost_usd']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epc_parser.stage1_macro",
                                 description="Step 1: macro division sweep of a scanned EPC bundle.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out_macro"))
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--reasoner-model", help="override the model used for the sweep")
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

    result = asyncio.run(run_macro_sweep(a.pdf, a.out, s))
    _print_ledger(result)


if __name__ == "__main__":
    main()
