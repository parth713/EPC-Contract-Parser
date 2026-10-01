"""Step 3 - Verbatim text extraction per leaf unit.

Each page is read ONCE, verbatim, into ordered blocks (marker + kind + text). Per division the blocks form
a single stream; blocks are assigned to leaf units by matching the unit's marker - which splits a page that
holds several units (LOA clauses 5-11 all on p13) and welds a unit that spans pages (GCC Clause 5 over
pp105-126) with the same logic. Sub-clause text (5.1, 5.2.3) folds into its parent clause automatically,
because those blocks fall between the parent's marker and the next main marker.

Provenance is preserved: every unit records the pages it drew text from. The verbatim page reads are cached,
so re-runs and crash-resumes cost nothing.

Outputs (into out_macro/):
  * contract.json  - the division -> leaf-unit tree, each unit with its full verbatim `text`
  * clauses.jsonl  - one line per leaf unit (id, division, marker, title, pages, text) - ready for search/RAG

Run (after Steps 1, 1.5, 2, 2.5):
    python -m epc_parser.stage3_extract "SBUT04 Contract Agreement (3).pdf" -o out_macro/
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
from .numbering import clean, normalize_marker
from .render import PageRenderer

log = logging.getLogger("epc.extract")

EXTRACT_DPI = 300
EXTRACT_MEDIA = "high"
MAX_OUT = 24000


class TextBlockLLM(BaseModel):
    kind: Literal["heading", "clause", "section", "paragraph", "list_item", "table", "signature", "stamp", "other"]
    marker: Optional[str] = Field(None, description="If this block STARTS a numbered/titled item, its marker exactly as printed (e.g. '5', '5.2', 'SECTION A', '(a)'); else null.")
    text: str = Field(description="Verbatim text of the block. Render tables as GitHub-flavoured markdown. Do not summarise, correct, or omit anything.")


class PageTextLLM(BaseModel):
    first_block_continues_previous_page: bool = Field(False, description="True if the first block continues a paragraph/sentence from the previous page.")
    blocks: list[TextBlockLLM] = Field(default_factory=list)


EXTRACT_SYSTEM = (
    "You transcribe one scanned page of an Indian EPC contract VERBATIM. Output every block of content on "
    "the page, in natural reading order, exactly as printed - clause numbers, sub-clauses, lists, and tables. "
    "Rules:\n"
    "  - Do NOT summarise, paraphrase, correct, translate, or omit anything. Reproduce the text exactly.\n"
    "  - For each block, if it begins with a clause/section/item marker or a heading number, put that marker "
    "in 'marker' (e.g. '5', '5.2', '5.2.1', 'SECTION A', '(a)', 'ARTICLE 3'); otherwise leave it null.\n"
    "  - Render any table as a GitHub-flavoured markdown table in the block's text (kind='table').\n"
    "  - Put stamps/seals as kind='stamp' and signature blocks as kind='signature'; keep their text.\n"
    "  - Ignore running headers, running footers and bare page numbers.\n"
    "  - If part of the page is illegible, write '[illegible]' in place of the unreadable words."
)
EXTRACT_USER = "Transcribe PDF page {page} of the contract verbatim into ordered blocks."


# =============================================================================================
# Page read
# =============================================================================================
async def read_page(llm: LLMClient, renderer: PageRenderer, page: int) -> tuple[Optional[PageTextLLM], Optional[str]]:
    img = await renderer.page(page, EXTRACT_DPI)
    res = await llm.call("verbatim_read", role="reader", system=EXTRACT_SYSTEM,
                         parts=[img, EXTRACT_USER.format(page=page)], schema=PageTextLLM, thinking="minimal",
                         media_resolution=EXTRACT_MEDIA, max_output_tokens=MAX_OUT, meta={"page": page})
    if res.ok:
        return res.parsed, ("truncated" if res.truncated else None)  # type: ignore[return-value]
    return None, (res.error or res.finish_reason)


# =============================================================================================
# Assign blocks to units + stitch
# =============================================================================================
def _unit_starts_here(unit: dict, page: int, block: TextBlockLLM, is_first_on_page: bool) -> bool:
    m = unit.get("marker")
    if m:
        return page >= unit["page_start"] and normalize_marker(block.marker) == normalize_marker(m)
    # markerless unit (preamble / whole / titled section without a number): advance when its page is reached
    return is_first_on_page and page >= unit["page_start"]


def _assemble(blocks: list[tuple[TextBlockLLM, bool, bool]]) -> str:
    """blocks: (block, is_first_on_page, page_continues). Join with paragraph breaks; weld a page-continuing
    first block onto the previous text (de-hyphenating)."""
    out = ""
    for k, (b, is_first, page_cont) in enumerate(blocks):
        piece = b.text.strip()
        if not piece:
            continue
        if not out:
            out = piece
        elif k > 0 and is_first and page_cont and b.kind not in ("table", "signature", "stamp"):
            if out.endswith("-") and piece[:1].islower():
                out = out[:-1] + piece
            else:
                out = out + " " + piece
        else:
            out = out + "\n\n" + piece
    return out


def extract_division_text(div: dict, pages_text: dict[int, PageTextLLM]) -> None:
    """Assign each page's blocks to the division's leaf units (in order) and set each unit's `text`."""
    ds, de = div["start_page"], div["end_page"]
    units = div["units"]
    buckets: dict[str, list[tuple[TextBlockLLM, bool, bool]]] = {u["unit_id"]: [] for u in units}
    pages_used: dict[str, set[int]] = {u["unit_id"]: set() for u in units}

    i = 0
    for p in range(ds, de + 1):
        pt = pages_text.get(p)
        if not pt:
            continue
        for j, b in enumerate(pt.blocks):
            is_first = j == 0
            while i + 1 < len(units) and _unit_starts_here(units[i + 1], p, b, is_first):
                i += 1
            buckets[units[i]["unit_id"]].append((b, is_first, pt.first_block_continues_previous_page))
            pages_used[units[i]["unit_id"]].add(p)

    for u in units:
        u["text"] = _assemble(buckets[u["unit_id"]])
        u["char_count"] = len(u["text"])
        pu = sorted(pages_used[u["unit_id"]])
        u["text_pages"] = [pu[0], pu[-1]] if pu else []
        flags = u.setdefault("flags", [])
        if not u["text"].strip():
            flags.append("empty_text")


# =============================================================================================
# Orchestration
# =============================================================================================
async def extract_all(pdf_path: str | Path, out_dir: str | Path, settings: Settings | None = None,
                      only_ids: list[str] | None = None) -> dict:
    started = datetime.now(timezone.utc)
    settings = settings or Settings()
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    leaves_path = out_dir / "leaves.json"
    if not leaves_path.exists():
        raise FileNotFoundError(f"{leaves_path} not found - run Step 2 (stage2_anchors) first.")
    data = json.loads(leaves_path.read_text())
    divisions = data["divisions"]
    if only_ids:
        divisions = [d for d in divisions if d["division_id"] in only_ids]
        if not divisions:
            raise ValueError(f"none of {only_ids} found in leaves.json")
    partial = bool(only_ids)
    pages_to_read = sorted({p for d in divisions for p in range(d["start_page"], d["end_page"] + 1)})
    renderer = PageRenderer(pdf_path, settings)
    review: list[dict] = [] if partial else list(data.get("review", []))
    try:
        llm = LLMClient(settings, GeminiBackend(settings))
        page_sem = asyncio.Semaphore(max(4, settings.concurrency // 2))
        pages_text: dict[int, PageTextLLM] = {}
        done = 0
        total = len(pages_to_read)

        async def one(p: int) -> None:
            nonlocal done
            async with page_sem:
                pt, problem = await read_page(llm, renderer, p)
            done += 1
            if done % 25 == 0 or done == total:
                log.info("pages %d/%d read (cost so far $%.3f)", done, total, llm.cost.report()["total_usd"])
            if pt is None:
                review.append({"code": "page_unreadable", "severity": "error", "pages": [p],
                               "message": f"page {p} could not be transcribed ({problem}); units on it lose text."})
                return
            if problem == "truncated":
                review.append({"code": "page_truncated", "severity": "warning", "pages": [p],
                               "message": f"page {p} transcription hit the output limit; text may be cut off."})
            pages_text[p] = pt

        await asyncio.gather(*[one(p) for p in pages_to_read])

        for div in divisions:
            extract_division_text(div, pages_text)

        cost = llm.cost.report()
        finished = datetime.now(timezone.utc)
        all_units = [u for d in divisions for u in d["units"]]
        result = {
            "source": data.get("source", {}),
            "generated_at": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 1),
            "cost_usd": cost["total_usd"],
            "stats": {
                "divisions": len(divisions), "leaf_units": len(all_units),
                "units_with_text": sum(1 for u in all_units if u.get("text", "").strip()),
                "empty_units": sum(1 for u in all_units if not u.get("text", "").strip()),
                "total_chars": sum(u.get("char_count", 0) for u in all_units),
                "pages_read": len(pages_text), "pages_failed": total - len(pages_text),
            },
            "divisions": divisions,
            "review": review,
        }
        contract_name = "contract.partial.json" if partial else "contract.json"
        clauses_name = "clauses.partial.jsonl" if partial else "clauses.jsonl"
        (out_dir / contract_name).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        with open(out_dir / clauses_name, "w", encoding="utf-8") as f:
            for d in divisions:
                for u in d["units"]:
                    f.write(json.dumps({
                        "unit_id": u["unit_id"], "division_id": d["division_id"], "division_type": d["division_type"],
                        "division_title": d["title"], "leaf_level": d.get("leaf_level"), "kind": u["kind"],
                        "marker": u.get("marker"), "title": u.get("title"),
                        "page_start": u["page_start"], "page_end": u["page_end"],
                        "text": u.get("text", ""),
                    }, ensure_ascii=False) + "\n")
        log.info("done: %d units, %d with text, %d empty, %s pages failed, $%.3f -> %s",
                 result["stats"]["leaf_units"], result["stats"]["units_with_text"], result["stats"]["empty_units"],
                 result["stats"]["pages_failed"], cost["total_usd"], out_dir / contract_name)
        return result
    finally:
        renderer.close()


def _print_summary(result: dict) -> None:
    s = result["stats"]
    print(f"\n--- EXTRACTION SUMMARY: {result['source'].get('file','')} ---")
    print(f"divisions: {s['divisions']} | leaf units: {s['leaf_units']} | with text: {s['units_with_text']} | "
          f"empty: {s['empty_units']}")
    print(f"pages read: {s['pages_read']} | pages failed: {s['pages_failed']} | total chars: {s['total_chars']:,} | "
          f"cost: ${result['cost_usd']}")
    empties = [u for d in result["divisions"] for u in d["units"] if not u.get("text", "").strip()]
    if empties:
        print(f"\nempty units ({len(empties)}) - check these:")
        for u in empties[:25]:
            print(f"  {u['unit_id']:<14} {u.get('marker') or '-':<8} {(u.get('title') or '')[:60]}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epc_parser.stage3_extract", description="Step 3: verbatim text extraction per leaf unit.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out_macro"), help="dir containing leaves.json")
    ap.add_argument("--only", help="comma-separated division ids to extract (trial), e.g. DIV-17. Writes *.partial.*")
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

    only_ids = [x.strip() for x in a.only.split(",")] if a.only else None
    result = asyncio.run(extract_all(a.pdf, a.out, s, only_ids=only_ids))
    _print_summary(result)


if __name__ == "__main__":
    main()
