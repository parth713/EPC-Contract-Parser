"""Stage 1 - read every page.

Per page:
  Pass A  full transcription + page card, climbing an escalation ladder on failure
  Pass B  blind structural-line scan with a different model and different wording
  Code    align A and B headings / page labels
  Arb.    blind high-resolution crop reads resolve every disagreement
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from . import prompts as P
from .context import RunContext
from .llm import CallResult
from .models import (Block, Dispute, HeadingLineLLM, HeadingScanLLM, PageLabelLLM, PageReadLLM, PageRecord,
                     RegionReadLLM)
from .numbering import (clean, extract_leading_marker, jaccard, looks_degenerate, norm_text, normalize_marker,
                        shingles, similarity)

log = logging.getLogger("epc.pages")

STRUCTURAL_KINDS = {"heading", "clause"}
NON_BODY_KINDS = {"header", "footer", "page_number"}
MAX_OUT_FULL = 32768
MAX_OUT_HALF = 20000


@dataclass(frozen=True)
class Rung:
    name: str
    role: str
    media: str
    thinking: str
    mode: str  # full | halves


LADDER = [
    Rung("L0_flash_medium", "reader", "medium", "minimal", "full"),
    Rung("L1_flash_high", "reader", "high", "minimal", "full"),
    Rung("L2_flash_halves", "reader", "high", "minimal", "halves"),
    Rung("L3_pro_halves", "escalation", "high", "low", "halves"),
]


# =============================================================================================
# Pass A
# =============================================================================================
def _diagnose(res: CallResult, ink: float, ctx: RunContext) -> str | None:
    if not res.ok:
        if res.truncated:
            return "truncated"
        if res.blocked:
            return f"blocked:{res.finish_reason}"
        return f"error:{(res.error or res.finish_reason)[:120]}"
    read: PageReadLLM = res.parsed  # type: ignore[assignment]
    body = "\n".join(b.text for b in read.blocks)
    if looks_degenerate(body):
        return "degenerate_repetition"
    if not read.blocks and ink > ctx.settings.blank_ink_threshold * 4 and read.page_type != "blank":
        return "empty_output_for_inked_page"
    return None


def _remap_box(box: list[int] | None, y_off: int) -> list[int] | None:
    if not box or len(box) != 4:
        return None
    y0, x0, y1, x1 = box
    return [int(y_off + y0 * 0.55), x0, int(y_off + y1 * 0.55), x1]


def merge_halves(top: PageReadLLM, bottom: PageReadLLM) -> PageReadLLM:
    tb = [b.model_copy(update={"box": _remap_box(b.box, 0)}) for b in top.blocks]
    bb = [b.model_copy(update={"box": _remap_box(b.box, 450)}) for b in bottom.blocks]
    tail = [b for b in tb if not b.box or b.box[2] > 380] or tb[-4:]
    kept = []
    for b in bb:
        in_overlap = not b.box or b.box[0] < 570
        if in_overlap and any(similarity(b.text, t.text) > 0.85 for t in tail):
            continue
        kept.append(b)
    worst = max([top.overall_legibility, bottom.overall_legibility], key=["clear", "partial", "illegible"].index)
    return PageReadLLM(
        page_type=top.page_type if top.page_type != "other" else bottom.page_type,
        is_document_start=top.is_document_start,
        document_title=top.document_title,
        printed_page_label=top.printed_page_label or bottom.printed_page_label,
        running_header=top.running_header,
        running_footer=bottom.running_footer or top.running_footer,
        first_block_continues_previous_page=top.first_block_continues_previous_page,
        blocks=tb + kept,
        cross_references=list(dict.fromkeys(top.cross_references + bottom.cross_references)),
        contract_documents_list=list(dict.fromkeys(top.contract_documents_list + bottom.contract_documents_list)),
        stamp_paper=top.stamp_paper or bottom.stamp_paper,
        languages=list(dict.fromkeys(top.languages + bottom.languages)),
        overall_legibility=worst,
    )


async def _read_halves(ctx: RunContext, pno: int, rung: Rung, ink: float) -> tuple[PageReadLLM | None, str | None]:
    top_img, bottom_img = await ctx.renderer.halves(pno)
    specs = [(top_img, "TOP", P.EDGE_RULE_TOP), (bottom_img, "BOTTOM", P.EDGE_RULE_BOTTOM)]
    results = await asyncio.gather(*[
        ctx.llm.call("page_read_half", role=rung.role, system=P.PAGE_READ_SYSTEM,
                     parts=[img, P.PAGE_READ_HALF_USER.format(part=part, pct=55, edge_rule=rule)],
                     schema=PageReadLLM, thinking=rung.thinking, media_resolution=rung.media,
                     max_output_tokens=MAX_OUT_HALF, meta={"page": pno, "mode": part.lower()})
        for img, part, rule in specs])
    for r in results:
        problem = _diagnose(r, ink / 2, ctx)
        if problem:
            return None, problem
    return merge_halves(results[0].parsed, results[1].parsed), None  # type: ignore[arg-type]


async def read_page(ctx: RunContext, pno: int, ink: float) -> tuple[PageReadLLM | None, str, list[str], bool]:
    """Returns (read, rung_name, failure_notes, text_available)."""
    notes: list[str] = []
    skip_full = False
    for rung in LADDER:
        if rung.mode == "full":
            if skip_full:
                continue
            img = await ctx.renderer.page(pno)
            res = await ctx.llm.call("page_read", role=rung.role, system=P.PAGE_READ_SYSTEM, parts=[img, P.PAGE_READ_USER],
                                     schema=PageReadLLM, thinking=rung.thinking, media_resolution=rung.media,
                                     max_output_tokens=MAX_OUT_FULL, meta={"page": pno, "mode": "full"})
            problem = _diagnose(res, ink, ctx)
            if problem is None:
                return res.parsed, rung.name, notes, True  # type: ignore[return-value]
        else:
            read, problem = await _read_halves(ctx, pno, rung, ink)
            if problem is None:
                return read, rung.name, notes, True
        notes.append(f"{rung.name}: {problem}")
        if problem == "truncated":
            skip_full = True  # a bigger image will not produce fewer tokens

    # Last resort: headings + clause openings only (survives RECITATION blocks on standard-form text).
    img = await ctx.renderer.page(pno)
    for role in ("reader", "escalation"):
        res = await ctx.llm.call("page_structure_only", role=role, system=P.PAGE_READ_SYSTEM,
                                 parts=[img, P.PAGE_STRUCTURE_ONLY_USER], schema=PageReadLLM, thinking="low",
                                 media_resolution="high", max_output_tokens=12000, meta={"page": pno, "mode": "structure"})
        if _diagnose(res, ink, ctx) is None:
            return res.parsed, f"structure_only_{role}", notes, False  # type: ignore[return-value]
        notes.append(f"structure_only_{role}: {res.finish_reason}")
    return None, "failed", notes, False


# =============================================================================================
# Pass B
# =============================================================================================
async def scan_page(ctx: RunContext, pno: int) -> HeadingScanLLM | None:
    img = await ctx.renderer.page(pno)
    for role in ("scanner", "reader"):
        res = await ctx.llm.call("heading_scan", role=role, system=P.HEADING_SCAN_SYSTEM, parts=[img, P.HEADING_SCAN_USER],
                                 schema=HeadingScanLLM, thinking="minimal", media_resolution="high",
                                 max_output_tokens=8192, meta={"page": pno})
        if res.ok:
            return res.parsed  # type: ignore[return-value]
    return None


# =============================================================================================
# Record building and A/B comparison
# =============================================================================================
def build_record(pno: int, read: PageReadLLM, level: str, text_available: bool, ink: float) -> PageRecord:
    blocks: list[Block] = []
    for i, b in enumerate(read.blocks):
        if not clean(b.text) and not b.number:
            continue
        blocks.append(Block(id=f"p{pno:04d}_b{i:03d}", page=pno, index=len(blocks), kind=b.kind,
                            number=clean(b.number) or None, title=clean(b.title) or None, inline_title=b.inline_title,
                            text=b.text.strip(), box=b.box if b.box and len(b.box) == 4 else None,
                            legibility=b.legibility, unreadable_note=b.unreadable_note))
    return PageRecord(
        page=pno, status="ok" if text_available else "structure_only", read_level=level, page_type=read.page_type,
        is_document_start=read.is_document_start, document_title=clean(read.document_title) or None,
        printed_label=clean(read.printed_page_label) or None, running_header=clean(read.running_header) or None,
        running_footer=clean(read.running_footer) or None, first_continues=read.first_block_continues_previous_page,
        blocks=blocks, cross_references=[clean(x) for x in read.cross_references if clean(x)],
        contract_documents_list=[clean(x) for x in read.contract_documents_list if clean(x)],
        stamp_paper=read.stamp_paper, languages=read.languages, legibility=read.overall_legibility, ink_ratio=ink,
    )


def _a_marker(b: Block) -> str | None:
    return normalize_marker(b.number) or (normalize_marker(extract_leading_marker(b.text)) if b.kind == "clause" else None)


def _b_marker(line: HeadingLineLLM) -> str | None:
    return normalize_marker(line.marker) or normalize_marker(extract_leading_marker(line.text))


def _a_compare_text(b: Block) -> str:
    return b.text if b.kind == "heading" else (b.text[:160])


def compare_reads(rec: PageRecord, scan: HeadingScanLLM, threshold: float) -> list[Dispute]:
    a_items = [b for b in rec.blocks if b.kind in STRUCTURAL_KINDS]
    label = norm_text(scan.printed_page_label)
    b_items = [l for l in scan.lines if norm_text(l.text) and not (label and norm_text(l.text) == label)]
    used: set[int] = set()
    disputes: list[Dispute] = []
    last_j = -1
    for a in a_items:
        ka = _a_marker(a)
        best_j, best_s = None, 0.0
        # reading order is shared, so search forward from the last match first (cheap alignment)
        order = list(range(last_j + 1, len(b_items))) + list(range(0, last_j + 1))
        for j in order:
            if j in used:
                continue
            bl = b_items[j]
            s = similarity(_a_compare_text(a), bl.text)
            if ka and ka == _b_marker(bl):
                s = max(s, 0.5 + 0.5 * similarity(a.title or _a_compare_text(a), bl.text))
            if s > best_s:
                best_j, best_s = j, s
        if best_j is not None and best_s >= threshold:
            used.add(best_j)
            last_j = best_j
            bl = b_items[best_j]
            kb = _b_marker(bl)
            if (ka or kb) and ka != kb:
                disputes.append(Dispute(type="marker_mismatch", block_id=a.id, a_marker=a.number, a_text=a.text[:200],
                                        b_marker=bl.marker or extract_leading_marker(bl.text), b_text=bl.text))
            elif bl.obscured or a.legibility != "clear":
                a.verification = "disputed"
                disputes.append(Dispute(type="a_only", block_id=a.id, a_marker=a.number, a_text=a.text[:200],
                                        b_marker=bl.marker, b_text=bl.text, resolution="obscured_in_scan"))
            else:
                a.verification = "agreed"
        else:
            disputes.append(Dispute(type="a_only", block_id=a.id, a_marker=a.number, a_text=a.text[:200]))
    for j, bl in enumerate(b_items):
        if j not in used:
            disputes.append(Dispute(type="b_only", b_marker=bl.marker, b_text=bl.text))
    if norm_text(rec.printed_label) != norm_text(scan.printed_page_label):
        disputes.append(Dispute(type="label", a_text=rec.printed_label, b_text=scan.printed_page_label))

    def priority(d: Dispute) -> int:
        marker = d.a_marker or d.b_marker
        return {"marker_mismatch": 0, "label": 1}.get(d.type, 2 if marker else 3)

    return sorted(disputes, key=priority)


# =============================================================================================
# Arbitration
# =============================================================================================
def _best_line(lines, target: str):
    best, best_s = None, 0.0
    candidates = [(l.text, l.obscured) for l in lines]
    candidates += [(f"{a.text} {b.text}", a.obscured or b.obscured) for a, b in zip(lines, lines[1:])]  # wrapped headings
    for text, obscured in candidates:
        s = similarity(target, text)
        if s > best_s:
            best, best_s = (text, obscured), s
    return best, best_s


def promote_or_insert(rec: PageRecord, arb_text: str, y: int | None, ctx: RunContext,
                      verification: str = "recovered", source: str = "arbitration") -> str:
    """Make sure a confirmed heading exists as its own block: promote a paragraph, split a merged block, or insert."""
    marker = extract_leading_marker(arb_text)
    kind = "clause" if marker else "heading"
    probe = clean(arb_text)[:40].casefold()
    for b in rec.blocks:
        if b.kind in STRUCTURAL_KINDS or b.kind in NON_BODY_KINDS:
            continue
        if similarity(b.text[: len(arb_text) + 10], arb_text) >= 0.8:
            b.kind, b.number, b.verification, b.source = kind, marker, verification, source  # type: ignore[assignment]
            return "promoted_block"
        pos = clean(b.text).casefold().find(probe) if len(probe) >= 12 else -1
        if pos > 0:  # heading was merged into the middle of a paragraph: split it out
            flat = clean(b.text)
            new = Block(id=f"{b.id}s{len(rec.blocks)}", page=b.page, index=b.index + 1, kind=kind, number=marker, text=flat[pos:],
                        box=b.box, verification=verification, source=source)  # type: ignore[arg-type]
            b.text = flat[:pos].rstrip()
            rec.blocks.insert(rec.blocks.index(b) + 1, new)
            _reindex(rec)
            return "split_block"
    y = 1000 if y is None else y
    pos = next((i for i, b in enumerate(rec.blocks) if b.box and b.box[0] > y), len(rec.blocks))
    rec.blocks.insert(pos, Block(id=f"p{rec.page:04d}_r{pos:03d}_{len(rec.blocks)}", page=rec.page, index=pos, kind=kind,
                                 number=marker, text=clean(arb_text), box=None, verification=verification,  # type: ignore[arg-type]
                                 source=source))  # type: ignore[arg-type]
    _reindex(rec)
    ctx.flag("warning", "heading_inserted", f"Heading '{clean(arb_text)[:80]}' was missed by the page read and inserted; "
             "its clause body may still sit inside the preceding paragraph.", [rec.page])
    return "inserted_block"


def _reindex(rec: PageRecord) -> None:
    for i, b in enumerate(rec.blocks):
        b.index = i


async def arbitrate(ctx: RunContext, rec: PageRecord, scan: HeadingScanLLM, disputes: list[Dispute]) -> None:
    limit = ctx.settings.max_disputes_per_page
    if len(disputes) > limit:
        ctx.flag("warning", "too_many_disputes", f"{len(disputes)} heading disagreements; only {limit} arbitrated.", [rec.page])
    blocks = {b.id: b for b in rec.blocks}
    scan_lines = {l.text: l for l in scan.lines}

    async def resolve(d: Dispute) -> None:
        if d.resolution == "obscured_in_scan":
            ctx.flag("warning", "obscured_heading", f"Heading partly hidden: '{(d.a_text or '')[:80]}'", [rec.page], d.block_id)
            return
        if d.type == "label":
            top = await ctx.renderer.crop(rec.page, (0, 0, 140, 1000))
            bottom = await ctx.renderer.crop(rec.page, (860, 0, 1000, 1000))
            res = await ctx.llm.call("label_arbitration", role="reader", system=P.PAGE_LABEL_SYSTEM,
                                     parts=[top, bottom, P.PAGE_LABEL_USER], schema=PageLabelLLM, thinking="minimal",
                                     media_resolution="high", max_output_tokens=512, meta={"page": rec.page})
            if res.ok:
                lab: PageLabelLLM = res.parsed  # type: ignore[assignment]
                rec.printed_label = clean(lab.printed_page_label) or None
                d.resolution = "obscured" if lab.obscured else "resolved"
                if lab.obscured:
                    ctx.flag("warning", "obscured_page_label", "Printed page label partly hidden.", [rec.page])
            return

        block = blocks.get(d.block_id or "")
        line = scan_lines.get(d.b_text or "")
        box = (block.box if block else None) or (line.box if line else None)
        if not box:
            d.resolution = "no_box"
            if block:
                block.verification = "disputed"
            return
        img = await ctx.renderer.band(rec.page, box[0], box[2])
        res = await ctx.llm.call("arbitration", role="reader", system=P.REGION_READ_SYSTEM, parts=[img, P.REGION_READ_USER],
                                 schema=RegionReadLLM, thinking="low", media_resolution="high", max_output_tokens=2048,
                                 meta={"page": rec.page, "box": box})
        if not res.ok:
            d.resolution = "arbitration_failed"
            if block:
                block.verification = "disputed"
            return
        lines = res.parsed.lines  # type: ignore[union-attr]
        target = (d.a_text if d.type != "b_only" else d.b_text) or ""
        best, score = _best_line(lines, target)
        if best is None or score < 0.6:
            if d.type == "b_only":
                d.resolution = "scan_not_confirmed"
            else:
                d.resolution = "not_confirmed"
                if block:
                    block.verification = "disputed"
                ctx.flag("warning", "heading_not_confirmed",
                         f"Heading could not be confirmed on re-read: '{target[:80]}'", [rec.page], d.block_id)
            return
        arb_text, obscured = best
        if obscured:
            d.resolution = "obscured"
            if block:
                block.verification, block.legibility = "disputed", "partial"
            ctx.flag("warning", "obscured_heading", f"Heading partly hidden: '{arb_text[:80]}'", [rec.page], d.block_id)
            return
        if d.type == "a_only" and block:
            block.verification, d.resolution = "confirmed_by_arbitration", "confirmed_a"
        elif d.type == "b_only" and line:
            d.resolution = promote_or_insert(rec, arb_text, line.box[0] if line.box else None, ctx)
        elif d.type == "marker_mismatch" and block:
            arb_marker_raw = extract_leading_marker(arb_text)
            ka, kb, karb = normalize_marker(d.a_marker), normalize_marker(d.b_marker), normalize_marker(arb_marker_raw)
            if karb == ka:
                block.verification, d.resolution = "confirmed_by_arbitration", "a_correct"
            else:
                old = block.number
                block.number = arb_marker_raw
                if old and block.text.lstrip().startswith(old):
                    block.text = block.text.replace(old, arb_marker_raw or old, 1)
                block.verification = "corrected_by_arbitration"
                d.resolution = "b_correct" if karb == kb else "arbitration_value"
                if karb != kb:
                    ctx.flag("warning", "marker_uncertain",
                             f"Three reads disagree on a clause number: '{d.a_marker}' / '{d.b_marker}' / '{arb_marker_raw}'",
                             [rec.page], block.id)

    await asyncio.gather(*[resolve(d) for d in disputes[:limit]])


# =============================================================================================
# Orchestration
# =============================================================================================
async def process_page(ctx: RunContext, pno: int) -> PageRecord:
    ink = await ctx.renderer.ink_ratio(pno)
    if ink < ctx.settings.blank_ink_threshold:
        return PageRecord(page=pno, status="blank", page_type="blank", ink_ratio=ink, read_level="pixels")

    (read, level, notes, text_ok), scan = await asyncio.gather(read_page(ctx, pno, ink), scan_page(ctx, pno))
    if read is None:
        rec = PageRecord(page=pno, status="failed", ink_ratio=ink, read_level="failed", flags=notes)
        ctx.flag("error", "page_unreadable", f"All read attempts failed: {'; '.join(notes)[:300]}", [pno])
        return rec
    rec = build_record(pno, read, level, text_ok, ink)
    rec.flags.extend(notes)
    if not text_ok:
        ctx.flag("error", "text_unavailable", "Verbatim text could not be extracted (blocked or failed); "
                 "only headings and clause openings are available.", [pno])
    if scan is None:
        ctx.flag("warning", "unverified_page", "Blind heading scan failed; headings on this page are unverified.", [pno])
        return rec
    rec.scan_verified = True
    disputes = compare_reads(rec, scan, ctx.settings.heading_match_threshold)
    rec.disputes = disputes
    if disputes and ctx.settings.arbitration:
        await arbitrate(ctx, rec, scan, disputes)
    for b in rec.blocks:
        if b.kind in STRUCTURAL_KINDS and b.legibility != "clear":
            ctx.flag("warning", "illegible_heading", f"{b.legibility} heading: '{b.text[:80]}' ({b.unreadable_note or ''})",
                     [pno], b.id)
    return rec


def mark_duplicates(ctx: RunContext, records: dict[int, PageRecord]) -> None:
    sigs: dict[int, set[str]] = {}
    for pno, r in records.items():
        body = " ".join(b.text for b in r.blocks if b.kind not in NON_BODY_KINDS | {"stamp_or_seal", "signature_block"})
        sigs[pno] = shingles(body)
    pages = sorted(records)
    for i, p in enumerate(pages):
        if len(sigs[p]) < 30:
            continue
        for q in pages[:i]:
            if records[q].duplicate_of or not 0.7 < len(sigs[q]) / len(sigs[p]) < 1.43:
                continue
            if jaccard(sigs[p], sigs[q]) >= ctx.settings.duplicate_jaccard:
                records[p].duplicate_of = q
                ctx.flag("warning", "duplicate_page", f"Page {p} appears to duplicate page {q}.", [q, p])
                break


async def process_all_pages(ctx: RunContext) -> dict[int, PageRecord]:
    done = 0
    total = len(ctx.pages)
    page_sem = asyncio.Semaphore(max(4, ctx.settings.concurrency // 2))  # bounds images held in memory

    async def one(pno: int) -> PageRecord:
        nonlocal done
        async with page_sem:
            rec = await process_page(ctx, pno)
        done += 1
        if done % 25 == 0 or done == total:
            log.info("pages %d/%d read (cost so far $%.3f)", done, total, ctx.llm.cost.report()["total_usd"])
        return rec

    results = await asyncio.gather(*[one(p) for p in ctx.pages])
    records = {r.page: r for r in results}
    mark_duplicates(ctx, records)
    return records
