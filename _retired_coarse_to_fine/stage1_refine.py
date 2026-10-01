"""Step 1.5 - Title & type refinement ("local zoom").

The macro sweep (Step 1) reads whole pages at low resolution, so division titles and types are sometimes
wrong: the running header gets mistaken for the title (BOQ parts came out titled "Annexure A 1.4" with
header "BILL OF QUANTITIES"), and priced schedules were typed ANNEXURE instead of BOQ.

This pass re-reads ONLY each division's start page at high resolution (a zoomed crop of the title region,
or the full page when the document starts mid-page or is a cover sheet) and rewrites the title, type and
running header. It never touches page boundaries, so the contiguous 1..N coverage from Step 1 still holds.

The original macro values are kept as `title_raw` / `division_type_raw` for traceability.

Run (after Step 1 has produced out_macro/divisions.json):
    python -m epc_parser.stage1_refine "SBUT04 Contract Agreement (3).pdf" -o out_macro/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import Settings
from .llm import GeminiBackend, LLMClient
from .numbering import clean
from .render import PageRenderer
from .stage1_macro import DivisionType

log = logging.getLogger("epc.refine")

REFINE_DPI = 300          # zoom: read small title/header text reliably
REFINE_MEDIA = "high"
TITLE_BAND = (0, 0, 560, 1000)  # top ~56% of the page, full width (markers/titles sit here)


class DivisionTitleLLM(BaseModel):
    division_type: DivisionType = Field(description="Classify by the document's NATURE, not just a label word.")
    title: str = Field(description="The document/section's OWN top-level title, cleaned and logically formatted in Title Case (keep genuine acronyms/identifiers as printed) - correct obvious typesetting defects (missing spaces, run-together words); do NOT copy the source's mistakes verbatim, and do NOT glue on the first sub-heading. NOT the running header or volume name.")
    running_header: Optional[str] = Field(None, description="Running header/footer text (often a volume or trust name), separate from the title. Empty if none.")
    is_cover_sheet: bool = Field(False, description="True if the page is mostly a blank section divider.")
    numbering_scheme: Literal["section", "clause", "flat"] = Field(
        "clause",
        description="How this division is structured at its TOP level. 'section': organised into titled "
        "SECTION/PART headings (e.g. 'SECTION 1 - DEFINITIONS', 'SECTION 2 - INSURANCE', 'PART II') - the "
        "sections are the units and clause numbers usually RESTART inside each one. 'clause': the top "
        "level is plain numbered clauses ('1. DEFINITIONS', '2. CONDITIONS PRECEDENT') with NO section "
        "headings. 'flat': no internal numbering at all (a form, cover sheet, stamp paper).")


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
    "(GCC, BOQ, FSSAI, EPC, roman numerals, 'A 1.4'). Do NOT return the running header/footer, the trust "
    "name, or the volume name as the title - those are separate.\n"
    "  - running_header: the running header/footer text (often a volume or organisation name), if any.\n"
    "  - division_type: classify by the document's NATURE, not just a label word. A priced schedule, bill "
    "of quantities, or summary of quantities is BOQ even when it is labelled 'Annexure'. A list or index "
    "of drawings/contents is INDEX_OR_TOC. General/Special Conditions are GCC/SCC. A Notice Inviting Tender "
    "is TENDER_NOTICE; Instructions to Tenderers/Bidders is INSTRUCTIONS_TO_TENDERERS; a Form of Tender is "
    "FORM_OF_TENDER; proforma/format samples are PROFORMA_OR_FORM. Do NOT default a tender notice or "
    "instructions to GCC/SCC/LETTER_OF_INTENT.\n"
    "  - is_cover_sheet: true if the page is mostly a blank divider.\n"
    "  - numbering_scheme: 'section' if this document is organised into titled SECTION/PART headings as "
    "its top level (e.g. 'SECTION 1 - DEFINITIONS', 'SECTION 2 - INSURANCE'); 'clause' if its top level "
    "is plain numbered clauses ('1. DEFINITIONS', '2. ...') with no section headings; 'flat' if it has no "
    "internal numbering. A General Conditions of Contract that shows a 'SECTION 1 - ...' heading is "
    "'section'; one that starts straight at '1. DEFINITIONS' is 'clause'.\n"
    "Report exactly what is printed; do not invent or infer beyond the image."
)
REFINE_USER = (
    "This is division {div_id}, PDF page {page}. The automated first pass guessed type='{type_guess}' and "
    "title='{title_guess}', which may be wrong (it often confused the running header with the title). "
    "Correct them from what you actually see. Return the structured JSON."
)


async def refine_one(llm: LLMClient, renderer: PageRenderer, div: dict) -> Optional[DivisionTitleLLM]:
    sp = div["start_page"]
    full = "mid_page_start" in div.get("flags", []) or div.get("is_cover_sheet")
    img = await (renderer.page(sp, REFINE_DPI) if full else renderer.crop(sp, TITLE_BAND, REFINE_DPI))
    res = await llm.call("title_refine", role="reader", system=REFINE_SYSTEM,
                         parts=[img, REFINE_USER.format(div_id=div["division_id"], page=sp,
                                                        type_guess=div["division_type"],
                                                        title_guess=clean(div["title"])[:120])],
                         schema=DivisionTitleLLM, thinking="low", media_resolution=REFINE_MEDIA,
                         max_output_tokens=1024, meta={"div": div["division_id"], "page": sp})
    return res.parsed if res.ok else None  # type: ignore[return-value]


async def refine_divisions(pdf_path: str | Path, out_dir: str | Path, settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    ledger_path = out_dir / "divisions.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"{ledger_path} not found - run Step 1 (stage1_macro) first.")
    data = json.loads(ledger_path.read_text())
    divisions = data["divisions"]
    renderer = PageRenderer(pdf_path, settings)
    try:
        llm = LLMClient(settings, GeminiBackend(settings))
        sem = asyncio.Semaphore(max(2, settings.concurrency // 2))

        async def one(div: dict) -> None:
            async with sem:
                refined = await refine_one(llm, renderer, div)
            if refined is None:
                div.setdefault("flags", []).append("title_refine_failed")
                return
            div["title_raw"] = div.get("title")
            div["division_type_raw"] = div.get("division_type")
            new_title = clean(refined.title) or div["title_raw"]
            new_type = refined.division_type
            changed = (new_title != clean(div["title_raw"] or "")) or (new_type != div["division_type_raw"])
            div["title"] = new_title
            div["division_type"] = new_type
            div["running_header"] = clean(refined.running_header) or div.get("running_header")
            div["is_cover_sheet"] = refined.is_cover_sheet
            div["numbering_scheme"] = refined.numbering_scheme
            flags = div.setdefault("flags", [])
            flags.append("title_refined" if changed else "title_confirmed")

        await asyncio.gather(*[one(d) for d in divisions])

        cost = llm.cost.report()
        data.setdefault("stats", {})["title_refine_cost_usd"] = cost["total_usd"]
        data["title_refined"] = True
        ledger_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        n_changed = sum(1 for d in divisions if "title_refined" in d.get("flags", []))
        log.info("refined %d/%d titles (%d changed), $%.3f -> %s", len(divisions), len(divisions), n_changed,
                 cost["total_usd"], ledger_path)
        return data
    finally:
        renderer.close()


def _print_ledger(data: dict) -> None:
    print(f"\n--- REFINED DIVISION LEDGER: {data['source']['file']} ({data['source']['page_count']} pages) ---")
    print(f"{'ID':<7} {'TYPE':<18} {'PAGES':<11} {'TITLE'}")
    for d in data["divisions"]:
        rng = f"{d['start_page']}-{d['end_page']}"
        was = ""
        if "title_refined" in d.get("flags", []):
            was = f'   (was: {d.get("division_type_raw")} / "{clean(d.get("title_raw") or "")[:40]}")'
        print(f"{d['division_id']:<7} {d['division_type']:<18} {rng:<11} {clean(d['title'])[:62]}{was}")
    print(f"\ntitle-refine cost: ${data['stats'].get('title_refine_cost_usd', 0)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epc_parser.stage1_refine",
                                 description="Step 1.5: high-res title/type refinement of the division ledger.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out_macro"), help="dir containing divisions.json")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--reader-model")
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
    if a.reader_model:
        s.model_reader = a.reader_model
    if a.cache_dir:
        s.cache_dir = a.cache_dir

    data = asyncio.run(refine_divisions(a.pdf, a.out, s))
    _print_ledger(data)


if __name__ == "__main__":
    main()
