"""Stage 3 - clause hierarchy for each document.

* Candidates are heading/clause blocks already read (and cross-verified) from the page images.
* The reasoner only classifies candidate IDs (keep / level / number / title); it cannot invent entries.
* Verified trees of standard documents (e.g. a PSU GCC edition) are stored as templates and reused on later contracts.
* Numbering gaps (7.2 -> 7.4) trigger a targeted image hunt; a found clause must be confirmed by a second blind read.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import prompts as P
from .context import RunContext
from .models import FLAT_DOC_TYPES, TREE_IF_STRUCTURED, Block, GapHuntLLM, HeadingScanLLM, PageRecord, Segment, TreeLLM
from .numbering import clean, depth_hint, extract_leading_marker, norm_text, normalize_marker, sibling_gaps
from .stage_pages import promote_or_insert

log = logging.getLogger("epc.tree")
STRUCTURAL = {"heading", "clause"}


@dataclass
class Decision:
    keep: bool
    level: int
    number: str | None
    title: str | None
    source: str  # llm | template | fallback | hunt


# =============================================================================================
# Templates
# =============================================================================================
class TemplateStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._templates: list[dict] | None = None

    @staticmethod
    def key(b: Block) -> str:
        marker = normalize_marker(b.number) or normalize_marker(extract_leading_marker(b.text)) or ""
        return f"{marker}|{norm_text(b.title or b.text)[:60]}"

    def _load(self) -> list[dict]:
        if self._templates is None:
            self._templates = []
            for f in sorted(self.dir.glob("*.json")):
                try:
                    self._templates.append(json.loads(f.read_text()))
                except Exception:
                    log.warning("ignoring unreadable template %s", f)
        return self._templates

    def match(self, doc_type: str, cands: list[Block]) -> tuple[dict | None, float]:
        keys = [self.key(c) for c in cands]
        best, best_cov = None, 0.0
        for t in self._load():
            if t.get("doc_type") != doc_type or not keys:
                continue
            cov = sum(k in t["entries"] for k in keys) / len(keys)
            if cov > best_cov:
                best, best_cov = t, cov
        return best, best_cov

    def save(self, seg: Segment, cands: list[Block], decisions: dict[str, Decision]) -> None:
        entries = {self.key(c): [decisions[c.id].keep, decisions[c.id].level, decisions[c.id].number, decisions[c.id].title]
                   for c in cands if c.id in decisions}
        if len(entries) < 15:  # tiny documents are not worth templating
            return
        digest = hashlib.sha256(json.dumps(sorted(entries), ensure_ascii=False).encode()).hexdigest()[:16]
        path = self.dir / f"{seg.doc_type}_{digest}.json"
        if not path.exists():
            path.write_text(json.dumps({"doc_type": seg.doc_type, "title": seg.title, "entries": entries}, ensure_ascii=False))
            self._templates = None
            log.info("saved template %s (%d entries)", path.name, len(entries))


# =============================================================================================
# Candidates and hierarchy decisions
# =============================================================================================
def collect_candidates(seg: Segment, records: dict[int, PageRecord]) -> list[Block]:
    out = []
    for p in range(seg.start_page, seg.end_page + 1):
        r = records[p]
        if r.duplicate_of:
            continue
        out.extend(b for b in r.blocks if b.kind in STRUCTURAL)
    return out


def needs_tree(seg: Segment, cands: list[Block]) -> bool:
    if seg.doc_type in FLAT_DOC_TYPES:
        return False
    if seg.doc_type in TREE_IF_STRUCTURED:
        return len(cands) >= 2
    return len(cands) >= 1


def _line(b: Block, fixed: Decision | None = None) -> str:
    text = clean(b.text).replace("|", "/")[:160]
    base = f"{b.id} | p{b.page} | {b.kind} | {clean(b.number) or '-'} | {text}"
    if fixed:
        return f"{base} | FIXED k={str(fixed.keep).lower()} l={fixed.level} n={fixed.number or '-'}"
    return f"{base} | {depth_hint(b.number or (extract_leading_marker(b.text) if b.kind == 'clause' else None))}"


def _fallback_decision(b: Block, prev_level: int, max_depth: int) -> Decision:
    hint = depth_hint(b.number or extract_leading_marker(b.text))
    if hint.startswith("dotted-"):
        level = int(hint.split("-")[1])
    elif b.kind == "heading":
        level = 1
    else:
        level = prev_level + 1
    return Decision(True, max(1, min(level, max_depth)), clean(b.number) or None, clean(b.title) or None, "fallback")


async def decide_hierarchy(ctx: RunContext, seg: Segment, cands: list[Block], store: TemplateStore) -> dict[str, Decision]:
    s = ctx.settings
    decisions: dict[str, Decision] = {}
    tmpl, cov = store.match(seg.doc_type, cands)
    if tmpl and cov >= s.template_min_coverage:
        for c in cands:
            e = tmpl["entries"].get(store.key(c))
            if e:
                decisions[c.id] = Decision(e[0], e[1], e[2], e[3], "template")
        ctx.flag("info", "template_reused", f"{seg.id} matched stored template '{tmpl.get('title')}' ({cov:.0%} of headings).",
                 [seg.start_page, seg.end_page])

    i = 0
    while i < len(cands):
        group, pending = [], 0
        while i < len(cands) and pending < s.tree_chunk_size:
            group.append(cands[i])
            pending += cands[i].id not in decisions
            i += 1
        ids = [c.id for c in group if c.id not in decisions]
        if not ids:
            continue
        start = cands.index(group[0])
        context = [c for c in cands[max(0, start - s.tree_context_items):start] if c.id in decisions]
        lines = [_line(c, decisions[c.id]) for c in context] + [_line(c, decisions.get(c.id)) for c in group]
        got = await _tree_call(ctx, seg, lines, ids)
        missing = [x for x in ids if x not in got]
        if missing:
            got.update(await _tree_call(ctx, seg, lines, missing))
        prev_level = 1
        for c in group:
            if c.id in decisions:
                prev_level = decisions[c.id].level if decisions[c.id].keep else prev_level
                continue
            d = got.get(c.id)
            if d is None:
                d = _fallback_decision(c, prev_level, s.max_tree_depth)
                ctx.flag("warning", "hierarchy_fallback", f"Level for '{c.text[:60]}' assigned by numbering rules.", [c.page], c.id)
            decisions[c.id] = d
            if d.keep:
                prev_level = d.level
    return decisions


async def _tree_call(ctx: RunContext, seg: Segment, lines: list[str], ids: list[str]) -> dict[str, Decision]:
    max_out = min(65000, 90 * len(ids) + 1024)
    res = await ctx.llm.call("hierarchy", role="reasoner", system=P.TREE_SYSTEM,
                             parts=[P.TREE_USER.format(doc_type=seg.doc_type, title=seg.title, max_depth=ctx.settings.max_tree_depth,
                                                       lines="\n".join(lines), ids=", ".join(ids))],
                             schema=TreeLLM, thinking="low", max_output_tokens=max_out, meta={"segment": seg.id})
    if not res.ok:
        return {}
    wanted = set(ids)
    return {it.id: Decision(it.k, max(1, it.l), clean(it.n) or None, clean(it.t) or None, "llm")
            for it in res.parsed.items if it.id in wanted}  # type: ignore[union-attr]


# =============================================================================================
# Numbering gap hunts
# =============================================================================================
def _sibling_groups(cands: list[Block], decisions: dict[str, Decision]) -> list[list[Block]]:
    groups: dict[str, list[Block]] = {}
    stack: list[tuple[int, str]] = []  # (level, id)
    for c in cands:
        d = decisions[c.id]
        if not d.keep:
            continue
        while stack and stack[-1][0] >= d.level:
            stack.pop()
        parent = stack[-1][1] if stack else "root"
        groups.setdefault(f"{parent}@{d.level}", []).append(c)
        stack.append((d.level, c.id))
    return list(groups.values())


async def hunt_gaps(ctx: RunContext, seg: Segment, records: dict[int, PageRecord], cands: list[Block],
                    decisions: dict[str, Decision]) -> list[Block]:
    """Returns the (possibly extended) candidate list. Mutates page records when a clause is recovered."""
    s = ctx.settings
    jobs = []
    for group in _sibling_groups(cands, decisions):
        markers = [decisions[c.id].number or c.number or extract_leading_marker(c.text) for c in group]
        for prev_m, missing, next_m in sibling_gaps(markers, s.max_gap_size):
            prev_b = group[markers.index(prev_m)]
            next_b = group[markers.index(next_m)]
            jobs.append((prev_b, missing, next_b))
    if not jobs:
        return cands

    async def hunt(prev_b: Block, missing: str, next_b: Block) -> Block | None:
        pages = list(range(prev_b.page, next_b.page + 1))
        pages = [p for p in pages if records[p].status != "blank"]
        if not s.gap_hunts or len(pages) > s.hunt_max_pages:
            ctx.flag("warning", "numbering_gap", f"{seg.title}: clause {missing} not found between "
                     f"'{prev_b.text[:40]}' and '{next_b.text[:40]}'.", [prev_b.page, next_b.page], prev_b.id)
            return None
        parts: list[str | bytes] = []
        legend = []
        for k, p in enumerate(pages, 1):
            parts.append(await ctx.renderer.page(p))
            legend.append(f"Image {k} = PDF page {p}")
        parts.append(P.GAP_HUNT_USER.format(title=seg.title, prev=clean(prev_b.text)[:60], next=clean(next_b.text)[:60],
                                            missing=missing, page_legend="\n".join(legend)))
        res = await ctx.llm.call("gap_hunt", role="reasoner", system=P.GAP_HUNT_SYSTEM, parts=parts, schema=GapHuntLLM,
                                 thinking="low", media_resolution="high", max_output_tokens=1024,
                                 meta={"segment": seg.id, "missing": missing})
        hunt_res: GapHuntLLM | None = res.parsed  # type: ignore[assignment]
        if hunt_res is None or not hunt_res.found or hunt_res.pdf_page not in pages or not hunt_res.first_line:
            ctx.flag("warning", "numbering_gap", f"{seg.title}: clause {missing} is not printed between "
                     f"'{prev_b.text[:40]}' and '{next_b.text[:40]}' (numbering skips it, or pages are missing).",
                     pages, prev_b.id)
            return None
        if hunt_res.obscured:
            ctx.flag("warning", "gap_found_obscured", f"Clause {missing} appears on page {hunt_res.pdf_page} but is partly "
                     "hidden; confirm manually.", [hunt_res.pdf_page], prev_b.id)
            return None
        # Blind confirmation: a fresh structural scan of that page must show the marker.
        img = await ctx.renderer.page(hunt_res.pdf_page, ctx.settings.crop_dpi)
        conf = await ctx.llm.call("gap_confirm", role="reader", system=P.HEADING_SCAN_SYSTEM,
                                  parts=[img, P.HEADING_SCAN_USER], schema=HeadingScanLLM, thinking="low",
                                  media_resolution="high", max_output_tokens=8192, meta={"page": hunt_res.pdf_page})
        key = normalize_marker(missing)
        line = next((l for l in (conf.parsed.lines if conf.ok else [])  # type: ignore[union-attr]
                     if normalize_marker(l.marker or extract_leading_marker(l.text)) == key), None)
        if line is None:
            ctx.flag("warning", "gap_found_unconfirmed", f"Hunt reported clause {missing} on page {hunt_res.pdf_page}, "
                     "but a second read did not confirm it.", [hunt_res.pdf_page], prev_b.id)
            return None
        rec = records[hunt_res.pdf_page]
        promote_or_insert(rec, line.text if len(line.text) > len(hunt_res.first_line) else hunt_res.first_line,
                          line.box[0] if line.box else None, ctx, verification="hunted", source="hunt")
        recovered = next((b for b in rec.blocks if b.source == "hunt" and normalize_marker(b.number) == key), None)
        if recovered:
            decisions[recovered.id] = Decision(True, decisions[prev_b.id].level, missing, None, "hunt")
            ctx.flag("info", "gap_recovered", f"{seg.title}: recovered clause {missing} on page {rec.page}.", [rec.page],
                     recovered.id)
        return recovered

    await asyncio.gather(*[hunt(*j) for j in jobs])
    return collect_candidates(seg, records)


# =============================================================================================
# Stage entry point
# =============================================================================================
async def build_hierarchies(ctx: RunContext, segments: list[Segment], records: dict[int, PageRecord]
                            ) -> dict[str, dict[str, Decision]]:
    store = TemplateStore(ctx.settings.template_dir)

    async def one(seg: Segment) -> tuple[str, dict[str, Decision]]:
        cands = collect_candidates(seg, records)
        if not needs_tree(seg, cands):
            return seg.id, {}
        decisions = await decide_hierarchy(ctx, seg, cands, store)
        cands = await hunt_gaps(ctx, seg, records, cands, decisions)
        for c in cands:  # blocks created by arbitration after the hierarchy call
            if c.id not in decisions:
                decisions[c.id] = _fallback_decision(c, 1, ctx.settings.max_tree_depth)
        clean_run = not any(r.node_id in decisions and r.severity != "info" for r in ctx.review)
        if clean_run and all(d.source != "fallback" for d in decisions.values()):
            store.save(seg, cands, decisions)
        return seg.id, decisions

    return dict(await asyncio.gather(*[one(s) for s in segments]))

