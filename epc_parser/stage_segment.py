"""Stage 2 - split the bundle into documents.

The reasoner sees compact page cards (about 100 tokens per page) rather than full text, so a 400-page bundle fits in one
call. Code then enforces the coverage invariant, and uncertain boundaries are re-checked on the actual page images.
"""
from __future__ import annotations

import asyncio
import logging
import typing

from . import prompts as P
from .context import RunContext
from .models import BoundaryLLM, DocType, PageRecord, Segment, SegmentationLLM, SegmentLLM
from .numbering import clean, parse_label

log = logging.getLogger("epc.segment")

NON_BODY = {"header", "footer", "page_number"}
STRUCTURAL = {"heading", "clause"}


def _q(s: str | None, n: int) -> str:
    return clean(s).replace('"', "'")[:n]


def card_line(r: PageRecord) -> str:
    if r.status == "blank":
        return f"[p{r.page}] BLANK"
    if r.status == "failed":
        return f"[p{r.page}] UNREADABLE"
    if r.duplicate_of:
        return f"[p{r.page}] DUPLICATE of p{r.duplicate_of}"
    body = [b for b in r.blocks if b.kind not in NON_BODY]
    first = _q(body[0].text, 90) if body else ""
    heads = "; ".join(_q(b.text, 50) for b in body if b.kind in STRUCTURAL)
    heads = heads[:260]
    refs = ", ".join(_q(x, 40) for x in r.cross_references[:5])
    return (f'[p{r.page}] type={r.page_type} start={"Y" if r.is_document_start else "N"} label="{_q(r.printed_label, 20)}" '
            f'hdr="{_q(r.running_header, 50)}" cont={"Y" if r.first_continues else "N"} title="{_q(r.document_title, 90)}" '
            f'first="{first}" heads={heads} refs={refs}')


def _prev_content_page(records: dict[int, PageRecord], p: int) -> int | None:
    q = p - 1
    while q in records and (records[q].status in ("blank", "failed") or records[q].duplicate_of):
        q -= 1
    return q if q in records else None


def _label_continues(prev: PageRecord | None, cur: PageRecord) -> bool:
    if not prev:
        return False
    a, b = parse_label(prev.printed_label), parse_label(cur.printed_label)
    return bool(a and b and a[0] == b[0] and b[1] == a[1] + 1)


def _heuristic_starts(records: dict[int, PageRecord], pages: list[int]) -> dict[int, SegmentLLM]:
    """Fallback when the segmentation call fails: title blocks, label resets and header changes."""
    starts: dict[int, SegmentLLM] = {}
    for p in pages:
        r = records[p]
        if r.status == "blank" or r.duplicate_of:
            continue
        prev = records.get(_prev_content_page(records, p) or -1)
        lab = parse_label(r.printed_label)
        reset = bool(lab and lab[1] == 1 and not _label_continues(prev, r))
        header_change = bool(prev and r.running_header and prev.running_header and
                             clean(r.running_header).casefold() != clean(prev.running_header).casefold())
        if p == pages[0] or r.is_document_start or reset or header_change:
            starts[p] = SegmentLLM(start_page=p, end_page=p, doc_type=r.page_type, confidence=0.5,
                                   title=r.document_title or f"[{r.page_type.replace('_', ' ')}]",
                                   evidence="heuristic: " + ", ".join(x for x, ok in [("title", r.is_document_start),
                                                                                     ("label reset", reset),
                                                                                     ("header change", header_change)] if ok))
    return starts


async def _segment_window(ctx: RunContext, records: dict[int, PageRecord], lo: int, hi: int) -> SegmentationLLM | None:
    cards = "\n".join(card_line(records[p]) for p in range(lo, hi + 1) if p in records)
    res = await ctx.llm.call("segmentation", role="reasoner", system=P.SEGMENT_SYSTEM,
                             parts=[P.SEGMENT_USER.format(first=lo, last=hi, cards=cards)], schema=SegmentationLLM,
                             thinking="medium", max_output_tokens=16384, meta={"lo": lo, "hi": hi})
    return res.parsed if res.ok else None  # type: ignore[return-value]


async def segment_bundle(ctx: RunContext, records: dict[int, PageRecord]) -> list[Segment]:
    s = ctx.settings
    pages = sorted(records)
    first, last = pages[0], pages[-1]
    win, ov = s.segmentation_window, s.segmentation_overlap
    windows = [(first, last)] if len(pages) <= win else \
        [(lo, min(last, lo + win - 1)) for lo in range(first, last + 1, win - ov) if lo <= last - ov or lo == first]
    results = await asyncio.gather(*[_segment_window(ctx, records, lo, hi) for lo, hi in windows])

    starts: dict[int, SegmentLLM] = {}
    for (lo, hi), res in zip(windows, results):
        if res is None:
            ctx.flag("warning", "segmentation_fallback", f"Segmentation call failed for pages {lo}-{hi}; used heuristics.",
                     [lo, hi])
            proposals = list(_heuristic_starts(records, list(range(lo, hi + 1))).values())
        else:
            proposals = res.segments
        margin = ov // 2
        for seg in proposals:
            p = seg.start_page
            if not lo <= p <= hi:
                continue
            if (lo != first and p < lo + margin) or (hi != last and p > hi - margin):
                continue  # boundary near a window edge: trust the neighbouring window instead
            if p not in starts or seg.confidence > starts[p].confidence:
                starts[p] = seg

    # Blank pages never start a document: move such starts to the next content page.
    for p in sorted(starts):
        if p != first and records[p].status == "blank":
            seg = starts.pop(p)
            q = p + 1
            while q <= last and records[q].status == "blank":
                q += 1
            if q <= last and q not in starts:
                starts[q] = seg
    if first not in starts:
        r = records[first]
        starts[first] = SegmentLLM(start_page=first, end_page=first, doc_type=r.page_type, confidence=0.5,
                                   title=r.document_title or f"[{r.page_type}]", evidence="first page")

    verified = await _verify_boundaries(ctx, records, starts, first)
    ordered = sorted(starts)
    segments = []
    for i, p in enumerate(ordered):
        meta = starts[p]
        end = ordered[i + 1] - 1 if i + 1 < len(ordered) else last
        segments.append(Segment(id=f"D{i + 1:02d}", start_page=p, end_page=end, doc_type=meta.doc_type,
                                title=clean(meta.title) or f"[{meta.doc_type}]", reference=clean(meta.reference) or None,
                                date=clean(meta.date) or None, confidence=meta.confidence, evidence=meta.evidence,
                                boundary_verified=verified.get(p)))
    assert_coverage(segments, first, last)
    log.info("segmentation: %d documents", len(segments))
    return segments


async def _verify_boundaries(ctx: RunContext, records: dict[int, PageRecord], starts: dict[int, SegmentLLM],
                             first: int) -> dict[int, bool]:
    s = ctx.settings
    to_check: set[int] = set()
    for p, seg in starts.items():
        if p == first:
            continue
        prev = records.get(_prev_content_page(records, p) or -1)
        conflict = (not records[p].is_document_start) or _label_continues(prev, records[p])
        if seg.confidence < s.boundary_confidence_threshold or conflict:
            to_check.add(p)
    for p, r in records.items():
        if p != first and p not in starts and r.status == "ok" and not r.duplicate_of and r.is_document_start:
            to_check.add(p)  # page looks like a start but segmentation merged it
    checks = sorted(to_check)
    if len(checks) > s.max_boundary_checks:
        ctx.flag("warning", "boundary_checks_capped", f"{len(checks)} uncertain boundaries; checked {s.max_boundary_checks}.")
        checks = checks[: s.max_boundary_checks]
    types = ", ".join(typing.get_args(DocType))

    async def check(p: int) -> tuple[int, BoundaryLLM | None]:
        a = _prev_content_page(records, p)
        if a is None:
            return p, None
        img_a, img_b = await ctx.renderer.page(a), await ctx.renderer.page(p)
        res = await ctx.llm.call("boundary_check", role="reasoner", system=P.BOUNDARY_SYSTEM,
                                 parts=[img_a, img_b, P.BOUNDARY_USER.format(a=a, b=p, types=types)], schema=BoundaryLLM,
                                 thinking="low", media_resolution="medium", max_output_tokens=1024, meta={"a": a, "b": p})
        return p, res.parsed if res.ok else None  # type: ignore[return-value]

    verified: dict[int, bool] = {}
    for p, res in await asyncio.gather(*[check(p) for p in checks]):
        r = records[p]
        if res is None or res.verdict == "unclear":
            verified[p] = False
            ctx.flag("warning", "boundary_unclear", f"Could not confirm whether page {p} starts a new document.", [p])
            continue
        if res.verdict == "new_document":
            verified[p] = True
            if p not in starts:
                doc_type = res.new_document_type or r.page_type
                starts[p] = SegmentLLM(start_page=p, end_page=p, doc_type=doc_type, confidence=0.85,
                                       title=res.new_document_title or r.document_title or f"[{doc_type}]",
                                       evidence=f"boundary check: {res.evidence}")
                ctx.flag("info", "boundary_added", f"New document detected at page {p}: {starts[p].title}", [p])
        elif p in starts:
            removed = starts.pop(p)
            ctx.flag("info", "boundary_removed", f"Page {p} continues the previous document (was '{removed.title}').", [p])
    return verified


def assert_coverage(segments: list[Segment], first: int, last: int) -> None:
    expected = first
    for seg in segments:
        if seg.start_page != expected or seg.end_page < seg.start_page:
            raise AssertionError(f"coverage broken at {seg.id}: expected start {expected}, got {seg.start_page}-{seg.end_page}")
        expected = seg.end_page + 1
    if expected != last + 1:
        raise AssertionError(f"coverage ends at {expected - 1}, expected {last}")
