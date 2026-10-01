"""Stage 2.5 - targeted numbering-gap hunt. For each clause-level gap, zoom the pages between the two
neighbours and check whether the missing clause is actually printed; recover it (insert + rebuild
ranges) or confirm the gap is a genuine skip. Mutates the leaf divisions in place."""
from __future__ import annotations

import asyncio
from typing import List, Optional

import logging
logger = logging.getLogger("epc.staged.gaphunt")

from .anchors import assert_division_coverage
from ..config import Settings as StagedConfig
from ..llm import LLMClient as StagedLLM
from .models import GapHuntLLM
from ..numbering import clean, normalize_marker, sibling_gaps
from ..render import PageRenderer

HUNT_SYSTEM = (
    "You check whether a specific top-level clause is PRINTED as a heading in a scanned Indian EPC "
    "contract. You are given the target clause number and a few page images (each labelled with its "
    "physical PDF page). Return found=true ONLY if that clause number actually BEGINS a clause/heading on "
    "one of these pages (its own numbered heading, at the top level - not a sub-clause like 4.1, and NOT a "
    "mere mention or cross-reference in body text). If it only appears in running text, or the numbering "
    "simply jumps past it, return found=false. When found, give the physical PDF page, the marker exactly "
    "as printed, and the heading title."
)
HUNT_USER = ("Target clause number: '{missing}'. It should sit between clause '{prev}' and clause '{next}' "
             "of the document '{title}'. Does clause '{missing}' begin as a heading on any of the pages below?")


async def _hunt(llm, renderer, cfg, div_title, prev_m, missing, next_m, pages) -> Optional[GapHuntLLM]:
    imgs = await asyncio.gather(*[renderer.page(p, cfg.hunt_dpi) for p in pages])
    parts: list = []
    for p, img in zip(pages, imgs):
        parts.append(f"\n[Physical PDF Page Number {p}]")
        parts.append(img)
    parts.append(HUNT_USER.format(missing=missing, prev=prev_m, next=next_m, title=clean(div_title)[:100]))
    res = await llm.call("gaphunt", role="reasoner", system=HUNT_SYSTEM, parts=parts, schema=GapHuntLLM,
                         thinking="low", media_resolution=cfg.hunt_media, max_output_tokens=1024)
    return res.parsed if res.ok else None


def _clause_units(div):
    return [u for u in div["units"] if u["kind"] == "clause" and u.get("marker")]


def _clip_children(u):
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


def _rebuild_ranges(units, de):
    for i, u in enumerate(units):
        if i + 1 < len(units):
            u["page_end"] = max(u["page_start"], units[i + 1]["page_start"] - 1)
        else:
            u["page_end"] = de
        _clip_children(u)


def _recover_insert(div, next_unit, missing, title, found_page):
    units = div["units"]
    prev = units[units.index(next_unit) - 1]
    found_page = min(max(found_page, prev["page_start"]), next_unit["page_start"])
    new_unit = {"unit_id": f"{prev['unit_id']}R", "kind": "clause", "marker": clean(missing),
                "title": clean(title) or None, "page_start": found_page, "page_end": found_page,
                "flags": ["gap_recovered"], "children": []}
    units.insert(units.index(next_unit), new_unit)
    _rebuild_ranges(units, div["end_page"])
    return new_unit


async def run(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, leaves: List[dict]) -> List[dict]:
    if not cfg.gap_hunts:
        return []
    sem = asyncio.Semaphore(max(2, cfg.concurrency // 3))
    outcomes: List[dict] = []

    async def hunt_div(div):
        if div.get("leaf_level") != "clause":
            return
        units = _clause_units(div)
        existing = {normalize_marker(u["marker"]) for u in units}
        for prev_m, missing, next_m in sibling_gaps([u["marker"] for u in units], max_gap=cfg.max_gap):
            key = normalize_marker(missing)
            if key in existing:
                outcomes.append({"code": "numbering_gap_artifact", "division": div["division_id"],
                                 "message": f"{div['title'][:40]}: clause {missing} already present out of order"})
                continue
            prev_u = next((u for u in units if normalize_marker(u["marker"]) == normalize_marker(prev_m)), None)
            next_u = next((u for u in units if normalize_marker(u["marker"]) == normalize_marker(next_m)), None)
            if not prev_u or not next_u:
                continue
            lo, hi = prev_u["page_start"], next_u["page_start"]
            pages = list(range(lo, hi + 1))
            if len(pages) > cfg.max_hunt_pages:
                outcomes.append({"code": "numbering_gap_unhunted", "division": div["division_id"],
                                 "message": f"{div['title'][:40]}: clause {missing} spans {len(pages)} pages, likely renumbering"})
                continue
            async with sem:
                res = await _hunt(llm, renderer, cfg, div["title"], prev_m, missing, next_m, pages)
            ok = bool(res and res.found and res.pdf_page in pages and normalize_marker(res.marker) == key)
            if ok:
                _recover_insert(div, next_u, missing, res.title, res.pdf_page)
                outcomes.append({"code": "gap_recovered", "division": div["division_id"],
                                 "message": f"{div['title'][:40]}: recovered clause {missing} on page {res.pdf_page}"})
            else:
                outcomes.append({"code": "numbering_gap_confirmed", "division": div["division_id"],
                                 "message": f"{div['title'][:40]}: clause {missing} not printed (genuine skip)"})
        div["unit_count"] = len(div["units"])
        assert_division_coverage(div["units"], div["start_page"], div["end_page"], div["division_id"])

    await asyncio.gather(*[hunt_div(d) for d in leaves])
    logger.info(f"[staged.gaphunt] {len(outcomes)} gaps checked, "
                f"{sum(1 for o in outcomes if o['code'] == 'gap_recovered')} recovered")
    return outcomes
