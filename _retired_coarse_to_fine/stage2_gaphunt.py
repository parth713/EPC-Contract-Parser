"""Step 2.5 - Targeted numbering-gap hunt.

Step 2's linter flags where clause numbering skips (e.g. GCC 3 -> 5, ITT 1 -> 3). This pass zooms the pages
between the two neighbours of each gap and checks whether the missing clause is ACTUALLY printed as a
heading (not merely referenced in body text):

  * found + confirmed  -> insert the recovered clause as a leaf unit, shrink the preceding unit's page range,
                          keep division coverage contiguous, flag `gap_recovered`.
  * not found          -> the gap is a genuine numbering skip; downgrade the review item to info
                          (`numbering_gap_confirmed`).

It rewrites out_macro/leaves.json in place. Hunts run only on clause-level divisions (sections/whole have
no clause numbering to check).

Run (after Step 2):
    python -m epc_parser.stage2_gaphunt "SBUT04 Contract Agreement (3).pdf" -o out_macro/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .config import Settings
from .llm import GeminiBackend, LLMClient
from .numbering import clean, normalize_marker, sibling_gaps
from .render import PageRenderer
from .stage2_anchors import assert_division_coverage

log = logging.getLogger("epc.gaphunt")

HUNT_DPI = 300
HUNT_MEDIA = "high"
MAX_HUNT_PAGES = 6  # a wider span is more likely renumbering than a genuinely missed single clause


class GapHuntLLM(BaseModel):
    found: bool = Field(description="True ONLY if a top-level clause with the target number begins as a heading on one of these pages.")
    pdf_page: Optional[int] = Field(None, description="The physical PDF page where it begins.")
    marker: Optional[str] = Field(None, description="The clause marker exactly as printed.")
    title: Optional[str] = Field(None, description="The clause heading/title, verbatim.")


HUNT_SYSTEM = (
    "You check whether a specific top-level clause is PRINTED as a heading in a scanned Indian EPC contract. "
    "You are given the target clause number and a few page images (each labelled with its physical PDF page). "
    "Return found=true ONLY if that clause number actually BEGINS a clause/heading on one of these pages "
    "(its own numbered heading, at the top level - not a sub-clause like 4.1, and NOT a mere mention or "
    "cross-reference to the clause inside body text). If it only appears in running text, or the numbering "
    "simply jumps past it, return found=false. When found, give the physical PDF page, the marker exactly as "
    "printed, and the heading title."
)
HUNT_USER = (
    "Target clause number: '{missing}'. It should sit between clause '{prev}' and clause '{next}' of the "
    "document '{title}'. Does clause '{missing}' begin as a heading on any of the pages below?"
)


async def hunt_gap(llm: LLMClient, renderer: PageRenderer, div_title: str, prev_m: str, missing: str,
                   next_m: str, pages: list[int]) -> Optional[GapHuntLLM]:
    imgs = await asyncio.gather(*[renderer.page(p, HUNT_DPI) for p in pages])
    parts: list[str | bytes] = []
    for p, img in zip(pages, imgs):
        parts.append(f"\n[Physical PDF Page Number {p}]")
        parts.append(img)
    parts.append(HUNT_USER.format(missing=missing, prev=prev_m, next=next_m, title=clean(div_title)[:100]))
    res = await llm.call("gap_hunt", role="reasoner", system=HUNT_SYSTEM, parts=parts, schema=GapHuntLLM,
                         thinking="low", media_resolution=HUNT_MEDIA, max_output_tokens=1024,
                         meta={"missing": missing, "pages": [pages[0], pages[-1]]})
    return res.parsed if res.ok else None  # type: ignore[return-value]


def _clause_units(div: dict) -> list[dict]:
    return [u for u in div["units"] if u["kind"] == "clause" and u.get("marker")]


def _clip_children(u: dict) -> None:
    """Keep a level-1 unit's nested sub-clauses inside its (possibly reshaped) page range after a
    recovery. Sub-clauses that fell out of the range are dropped; the rest re-partition it."""
    kids = u.get("children")
    if not kids:
        return
    kept = [c for c in kids if c["page_start"] <= u["page_end"]]
    for k, c in enumerate(kept):
        c["page_start"] = min(max(c["page_start"], u["page_start"]), u["page_end"])
        c["page_end"] = (kept[k + 1]["page_start"] - 1) if k + 1 < len(kept) else u["page_end"]
        if c["page_end"] < c["page_start"]:
            c["page_end"] = c["page_start"]
    u["children"] = kept


def _rebuild_ranges(units: list[dict], ds: int, de: int) -> None:
    """Recompute every unit's page_end from the (ascending) unit starts, so coverage stays contiguous by
    construction: a unit runs to the page before the next unit's start (or to `de` for the last one).
    Nested sub-clauses are re-clipped into their parent's new range."""
    for i, u in enumerate(units):
        if i + 1 < len(units):
            u["page_end"] = max(u["page_start"], units[i + 1]["page_start"] - 1)
        else:
            u["page_end"] = de
        _clip_children(u)


def _recover_insert(div: dict, next_unit: dict, missing: str, title: Optional[str], found_page: int) -> dict:
    """Insert the recovered clause before next_unit (in reading order) and rebuild all page ranges."""
    units = div["units"]
    prev = units[units.index(next_unit) - 1]
    found_page = min(max(found_page, prev["page_start"]), next_unit["page_start"])  # clamp into the gap span
    new_unit = {"unit_id": f"{prev['unit_id']}R", "kind": "clause", "marker": clean(missing),
                "title": clean(title) or None, "page_start": found_page, "page_end": found_page,
                "flags": ["gap_recovered"], "children": []}
    units.insert(units.index(next_unit), new_unit)
    _rebuild_ranges(units, div["start_page"], div["end_page"])
    return new_unit


async def run_gap_hunt(pdf_path: str | Path, out_dir: str | Path, settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    leaves_path = out_dir / "leaves.json"
    if not leaves_path.exists():
        raise FileNotFoundError(f"{leaves_path} not found - run Step 2 (stage2_anchors) first.")
    data = json.loads(leaves_path.read_text())
    renderer = PageRenderer(pdf_path, settings)
    outcomes: list[dict] = []
    try:
        llm = LLMClient(settings, GeminiBackend(settings))
        sem = asyncio.Semaphore(max(2, settings.concurrency // 3))

        async def hunt_division(div: dict) -> None:
            if div.get("leaf_level") != "clause":
                return
            units = _clause_units(div)
            markers = [u["marker"] for u in units]
            existing = {normalize_marker(u["marker"]) for u in units}
            for prev_m, missing, next_m in sibling_gaps(markers, max_gap=8):
                key = normalize_marker(missing)
                if key in existing:  # the "missing" clause already exists elsewhere (out-of-order marker); a linter artifact
                    outcomes.append({"code": "numbering_gap_artifact", "division": div["division_id"], "severity": "info",
                                     "message": f"{div['title'][:40]}: clause {missing} flagged as a gap but already "
                                                f"present out of order; no hunt needed."})
                    continue
                prev_u = next((u for u in units if normalize_marker(u["marker"]) == normalize_marker(prev_m)), None)
                next_u = next((u for u in units if normalize_marker(u["marker"]) == normalize_marker(next_m)), None)
                if not prev_u or not next_u:
                    continue
                lo, hi = prev_u["page_start"], next_u["page_start"]
                pages = list(range(lo, hi + 1))
                if len(pages) > MAX_HUNT_PAGES:
                    outcomes.append({"code": "numbering_gap_unhunted", "division": div["division_id"], "severity": "warning",
                                     "message": f"{div['title'][:40]}: clause {missing} spans {len(pages)} pages "
                                                f"({lo}-{hi}); too wide to hunt, likely renumbering."})
                    continue
                async with sem:
                    res = await hunt_gap(llm, renderer, div["title"], prev_m, missing, next_m, pages)
                ok = bool(res and res.found and res.pdf_page in pages
                          and normalize_marker(res.marker) == key)
                if ok:
                    new_u = _recover_insert(div, next_u, missing, res.title, res.pdf_page)  # type: ignore[arg-type]
                    outcomes.append({"code": "gap_recovered", "division": div["division_id"], "severity": "info",
                                     "message": f"{div['title'][:40]}: recovered clause {missing} "
                                                f"('{clean(res.title or '')[:40]}') on page {res.pdf_page}."})
                    log.info("recovered %s clause %s on p%d", div["division_id"], missing, res.pdf_page)
                else:
                    outcomes.append({"code": "numbering_gap_confirmed", "division": div["division_id"], "severity": "info",
                                     "message": f"{div['title'][:40]}: clause {missing} is not printed between "
                                                f"'{prev_m}' and '{next_m}' (genuine numbering skip)."})
            div["unit_count"] = len(div["units"])
            assert_division_coverage(div["units"], div["start_page"], div["end_page"], div["division_id"])

        await asyncio.gather(*[hunt_division(d) for d in data["divisions"]])

        # rebuild the review list: keep everything except the old (now-resolved) numbering gaps
        kept = [r for r in data.get("review", []) if r.get("code") != "anchor_numbering_gap"]
        data["review"] = kept + outcomes
        recovered = sum(1 for o in outcomes if o["code"] == "gap_recovered")
        data.setdefault("stats", {})["gap_hunt_cost_usd"] = llm.cost.report()["total_usd"]
        data["stats"]["leaf_units"] = sum(len(d["units"]) for d in data["divisions"])
        data["stats"]["gaps_recovered"] = recovered
        leaves_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        log.info("gap hunt: %d gaps checked, %d recovered, $%.3f -> %s", len(outcomes), recovered,
                 llm.cost.report()["total_usd"], leaves_path)
        return data
    finally:
        renderer.close()


def _print_outcomes(data: dict) -> None:
    print(f"\n--- GAP HUNT: {data['source'].get('file','')} ---")
    for o in [r for r in data["review"] if r["code"] in ("gap_recovered", "numbering_gap_confirmed", "numbering_gap_unhunted")]:
        tag = {"gap_recovered": "RECOVERED", "numbering_gap_confirmed": "genuine skip", "numbering_gap_unhunted": "not hunted"}[o["code"]]
        print(f"  [{tag}] {o['division']}: {o['message']}")
    print(f"\nrecovered: {data['stats'].get('gaps_recovered',0)} | hunt cost: ${data['stats'].get('gap_hunt_cost_usd',0)} | "
          f"total leaf units now: {data['stats'].get('leaf_units')}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epc_parser.stage2_gaphunt", description="Step 2.5: targeted numbering-gap hunt.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out_macro"), help="dir containing leaves.json")
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

    data = asyncio.run(run_gap_hunt(a.pdf, a.out, s))
    _print_outcomes(data)


if __name__ == "__main__":
    main()
