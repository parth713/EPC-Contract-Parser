"""Stage 1.5 - title/type refinement ("local zoom"). Re-reads each division's start page at high
resolution to fix the title, type and running header the low-res macro sweep got wrong. Boundaries
are never touched, so coverage stays valid. Mutates the division dicts in place."""
from __future__ import annotations

import asyncio
from typing import List

import logging
logger = logging.getLogger("epc.staged.refine")

from ..config import Settings as StagedConfig
from ..llm import LLMClient as StagedLLM
from .models import DivisionTitleLLM
from ..numbering import clean
from ..render import PageRenderer

TITLE_BAND = (0, 0, 560, 1000)

REFINE_SYSTEM = (
    "You are shown a high-resolution image of the START page (or its title region) of ONE document or "
    "section inside a scanned Indian EPC contract bundle. Read what is actually printed and return:\n"
    "  - title: the document's OWN top-level title/heading, cleaned and LOGICALLY FORMATTED - not a "
    "verbatim copy of the source's typesetting defects. Correct obvious errors such as missing spaces or "
    "run-together words (e.g. read 'CONTRACTSECTION' as two words). If the printed heading runs the "
    "top-level title together with the FIRST sub-section or clause heading beneath it (e.g. 'SECTION F - "
    "GENERAL CONDITIONS OF CONTRACTSECTION 1 - DEFINITIONS, AND INTERPRETATION...'), return ONLY the "
    "top-level title ('Section F - General Conditions of Contract') and drop the glued-on sub-heading. "
    "Present the title in Title Case (e.g. 'Section F - General Conditions of Contract', not "
    "'SECTION F - GENERAL CONDITIONS OF CONTRACT'), but keep genuine acronyms and identifiers as printed "
    "(GCC, BOQ, FSSAI, EPC, roman numerals, 'A 1.4'). Do NOT return the running header/footer, trust "
    "name, or volume name as the title.\n"
    "  - running_header: the running header/footer text, if any.\n"
    "  - numbering_scheme: 'section' if this document is organised into titled SECTION/PART headings as "
    "its top level (e.g. 'SECTION 1 - DEFINITIONS', 'SECTION 2 - INSURANCE'); 'clause' if its top level "
    "is plain numbered clauses ('1. DEFINITIONS', '2. ...') with no section headings; 'flat' if it has no "
    "internal numbering. A General Conditions of Contract that shows a 'SECTION 1 - ...' heading is "
    "'section'; one that starts straight at '1. DEFINITIONS' is 'clause'.\n"
    "  - division_type: classify by the document's NATURE, not just a label word. A priced schedule / "
    "bill of quantities / summary of quantities is BOQ even when labelled 'Annexure'. A list/index of "
    "drawings/contents is INDEX_OR_TOC. General/Special Conditions are GCC/SCC. A Notice Inviting Tender "
    "is TENDER_NOTICE; Instructions to Tenderers/Bidders is INSTRUCTIONS_TO_TENDERERS; a Form of Tender is "
    "FORM_OF_TENDER; proforma/format samples are PROFORMA_OR_FORM. Do NOT default a tender notice or "
    "instructions to GCC/SCC/LETTER_OF_INTENT.\n"
    "  - is_cover_sheet: true if the page is mostly a blank divider.\n"
    "Report exactly what is printed; do not invent or infer beyond the image."
)
REFINE_USER = (
    "This is division {div_id}, PDF page {page}. The automated first pass guessed type='{type_guess}' and "
    "title='{title_guess}', which may be wrong (it often confused the running header with the title). "
    "Correct them from what you actually see. Return the structured JSON."
)


async def _refine_one(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, div: dict):
    sp = div["start_page"]
    full = "mid_page_start" in div.get("flags", []) or div.get("is_cover_sheet")
    img = await (renderer.page(sp, cfg.crop_dpi) if full else renderer.crop(sp, TITLE_BAND, cfg.crop_dpi))
    res = await llm.call("refine", role="reader", system=REFINE_SYSTEM,
                         parts=[img, REFINE_USER.format(div_id=div["division_id"], page=sp,
                                                        type_guess=div["division_type"],
                                                        title_guess=clean(div["title"])[:120])],
                         schema=DivisionTitleLLM, thinking="low", media_resolution=cfg.refine_media,
                         max_output_tokens=1024)
    return res.parsed if res.ok else None


async def run(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, divisions: List[dict]) -> None:
    sem = asyncio.Semaphore(max(2, cfg.concurrency // 2))

    async def one(div):
        async with sem:
            refined = await _refine_one(llm, renderer, cfg, div)
        if refined is None:
            div.setdefault("flags", []).append("title_refine_failed")
            return
        div["title_raw"] = div.get("title")
        div["division_type_raw"] = div.get("division_type")
        new_title = clean(refined.title) or div["title_raw"]
        changed = (new_title != clean(div["title_raw"] or "")) or (refined.division_type != div["division_type_raw"])
        div["title"] = new_title
        div["division_type"] = refined.division_type
        div["running_header"] = clean(refined.running_header) or div.get("running_header")
        div["is_cover_sheet"] = refined.is_cover_sheet
        div["numbering_scheme"] = refined.numbering_scheme
        div.setdefault("flags", []).append("title_refined" if changed else "title_confirmed")

    await asyncio.gather(*[one(d) for d in divisions])
    n = sum(1 for d in divisions if "title_refined" in d.get("flags", []))
    logger.info(f"[staged.refine] refined {len(divisions)} titles ({n} changed)")
