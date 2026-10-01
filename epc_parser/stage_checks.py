"""Stage 5 - prove completeness.

Code checks (free): page-label continuity, cross-references vs. what was found, attachment series gaps, the contract's own
list of contract documents vs. detected documents. One optional LLM audit proposes omissions for human review only.
"""
from __future__ import annotations

import logging
import re

from . import prompts as P
from .context import RunContext
from .models import AuditLLM, Node, PageRecord, Segment
from difflib import SequenceMatcher

from .numbering import attachment_key, clean, norm_text, parse_label, similarity
from .stage_segment import card_line
from .stage_text import iter_nodes

log = logging.getLogger("epc.checks")

DOC_KEYWORDS = {
    "letter of award": "letter_of_award", "loa": "letter_of_award", "letter of acceptance": "letter_of_award",
    "letter of intent": "letter_of_intent", "loi": "letter_of_intent", "notification of award": "notification_of_award",
    "notice of award": "notification_of_award", "general conditions": "general_conditions", "gcc": "general_conditions",
    "special conditions": "special_conditions", "scc": "special_conditions", "particular conditions": "special_conditions",
    "technical specification": "technical_specification", "specification": "technical_specification",
    "scope of work": "scope_of_work", "bill of quantities": "boq_or_price_schedule", "boq": "boq_or_price_schedule",
    "price schedule": "boq_or_price_schedule", "schedule of rates": "boq_or_price_schedule",
    "integrity pact": "integrity_pact", "minutes": "minutes_of_meeting", "amendment": "amendment_or_corrigendum",
    "corrigendum": "amendment_or_corrigendum", "addendum": "amendment_or_corrigendum",
    "instructions to bidders": "tender_notice_or_itb", "notice inviting tender": "tender_notice_or_itb",
    "nit": "tender_notice_or_itb", "bank guarantee": "bank_guarantee", "power of attorney": "power_of_attorney",
    "agreement": "contract_agreement", "deviation": "deviation_list",
}


def _found_attachment_keys(segments: list[Segment], docs: list[Node]) -> dict[tuple[str, str], list[int]]:
    found: dict[tuple[str, str], list[int]] = {}
    for seg in segments:
        for text in (seg.title, seg.reference):
            k = attachment_key(text)
            if k:
                found.setdefault(k, []).append(seg.start_page)
    for doc in docs:
        for n in iter_nodes(doc):
            if n.kind in ("heading", "clause") and n.level <= 2:
                k = attachment_key(f"{n.number or ''} {n.title or n.heading_text or ''}")
                if k and norm_text(n.heading_text or n.title or "").startswith(k[0]):
                    found.setdefault(k, []).append(n.page_start)
    return found


def check_labels(ctx: RunContext, segments: list[Segment], records: dict[int, PageRecord]) -> None:
    for seg in segments:
        prev_page, prev_lab = None, None
        for p in range(seg.start_page, seg.end_page + 1):
            r = records[p]
            if r.status in ("blank", "failed") or r.duplicate_of:
                continue
            lab = parse_label(r.printed_label)
            if lab and prev_lab and lab[0] == prev_lab[0]:
                expected = prev_lab[1] + (p - prev_page)  # blank pages may be unnumbered but still counted
                if lab[1] < prev_lab[1]:
                    ctx.flag("warning", "label_backwards", f"{seg.title}: printed page '{r.printed_label}' after "
                             f"'{records[prev_page].printed_label}' (pages out of order?)", [prev_page, p])
                elif lab[1] > expected and lab[1] > prev_lab[1] + 1:
                    ctx.flag("warning", "label_jump", f"{seg.title}: printed page jumps from "
                             f"'{records[prev_page].printed_label}' to '{r.printed_label}' (pages missing from scan?)",
                             [prev_page, p])
            if lab:
                prev_page, prev_lab = p, lab


def check_references(ctx: RunContext, segments: list[Segment], docs: list[Node], records: dict[int, PageRecord]) -> None:
    found = _found_attachment_keys(segments, docs)
    referenced: dict[tuple[str, str], list[int]] = {}
    for p, r in sorted(records.items()):
        for ref in r.cross_references:
            k = attachment_key(ref)
            if k:
                referenced.setdefault(k, []).append(p)
    for k, pages in sorted(referenced.items()):
        if k not in found:
            ctx.flag("warning", "reference_not_found",
                     f"{k[0].title()} {k[1].upper()} is referenced but was not found in the scan.", sorted(set(pages))[:8])
    for k, pages in sorted(found.items()):
        if k not in referenced:
            ctx.flag("info", "attachment_not_referenced", f"{k[0].title()} {k[1].upper()} is present but never referenced.",
                     pages[:3])
    by_kind: dict[str, list[int]] = {}
    for kind, idx in found:
        if idx.isdigit():
            by_kind.setdefault(kind, []).append(int(idx))
    for kind, nums in by_kind.items():
        nums = sorted(set(nums))
        for a, b in zip(nums, nums[1:]):
            if 1 < b - a <= 5:
                missing = ", ".join(str(i) for i in range(a + 1, b))
                ctx.flag("warning", "attachment_series_gap", f"{kind.title()} numbering skips {missing} (found {a} and {b}).")


def check_contract_documents_list(ctx: RunContext, segments: list[Segment], records: dict[int, PageRecord]) -> None:
    items: list[tuple[int, str]] = []
    for p, r in sorted(records.items()):
        items.extend((p, it) for it in r.contract_documents_list)
    if not items:
        ctx.flag("info", "no_contract_documents_list", "No list of contract documents was detected in the agreement.")
        return
    found_keys = _found_attachment_keys(segments, [])
    types_present = {s.doc_type for s in segments}
    for page, item in items:
        low = clean(item).casefold()
        k = attachment_key(item)
        if k:  # numbered attachments must match exactly: "Annexure-II" is not "Annexure-III"
            ok = k in found_keys
        else:
            ok = any(re.search(rf"\b{re.escape(kw)}\b", low) and dt in types_present for kw, dt in DOC_KEYWORDS.items())
            ok = ok or any(SequenceMatcher(None, norm_text(item), norm_text(s.title)).ratio() >= 0.8 for s in segments)
        if not ok:
            ctx.flag("warning", "listed_document_missing", f"Contract lists '{clean(item)[:120]}' but no matching document "
                     "was identified in the scan.", [page])


def check_index_pages(ctx: RunContext, segments: list[Segment], docs: list[Node], records: dict[int, PageRecord]) -> None:
    """If the bundle contains printed contents/index pages (e.g. a GCC index), every listed entry should exist in the tree."""
    titles = [norm_text(f"{n.number or ''} {n.title or n.heading_text or n.text[:80]}") for d in docs for n in iter_nodes(d)]
    titles += [norm_text(n.title) for d in docs for n in iter_nodes(d) if n.title]
    for p, r in sorted(records.items()):
        if r.page_type != "index_or_contents" or r.duplicate_of:
            continue
        entries: list[str] = []
        for b in r.blocks:
            if b.kind == "table":
                entries += [c for row in b.text.splitlines()[2:] for c in row.split("|")[1:2]]
            elif b.kind in ("clause", "heading", "paragraph"):
                entries.append(b.text)
        for e in entries:
            e = re.sub(r"[.\s]{4,}\S*\s*$", "", clean(e))  # drop dot leaders and trailing page numbers
            e = re.sub(r"\s+\d{1,4}$", "", e)
            ne = norm_text(e)
            if len(ne) < 6 or ne in ("contents", "index", "table of contents", "clause", "page", "title"):
                continue
            if not any(SequenceMatcher(None, ne, t).ratio() >= 0.8 or (len(ne) > 12 and ne in t) for t in titles):
                ctx.flag("warning", "index_entry_not_found",
                         f"Printed index on page {p} lists '{e[:100]}' but no matching entry was found in the extracted tree.", [p])


def _outline(docs: list[Node]) -> str:
    lines = []
    for doc in docs:
        lines.append(f"{doc.id} {doc.title} [{doc.doc_type}] pp.{doc.page_start}-{doc.page_end}")
        for n in iter_nodes(doc):
            if n.kind != "document" and n.level <= 2:
                label = clean(f"{n.number or ''} {n.title or n.heading_text or n.text[:60]}")
                lines.append(f"{'  ' * n.level}{label[:100]} p.{n.page_start}")
    return "\n".join(lines)


async def final_audit(ctx: RunContext, docs: list[Node], records: dict[int, PageRecord]) -> None:
    if not ctx.settings.final_audit:
        return
    outline = _outline(docs)
    cards = "\n".join(card_line(records[p]) for p in sorted(records))
    if len(outline) + len(cards) > 600_000:  # keep the audit well inside the context window
        cards = "\n".join(line.split(" first=")[0] for line in cards.splitlines())
    res = await ctx.llm.call("final_audit", role="reasoner", system=P.AUDIT_SYSTEM,
                             parts=[P.AUDIT_USER.format(outline=outline, cards=cards)], schema=AuditLLM,
                             thinking="medium", max_output_tokens=8192)
    if not res.ok:
        ctx.flag("info", "audit_failed", "Final audit call failed; completeness relies on code checks only.")
        return
    titles = [re.sub(r"\s+(pp?\.[\d-]+|\[\w+\])+$", "", line.strip()) for line in outline.splitlines()]
    for item in res.parsed.missing:  # type: ignore[union-attr]
        if any(similarity(item.title, t, prefix=False) >= 0.85 for t in titles):
            continue
        ctx.flag("warning", "audit_possible_omission", f"Audit suggests a missing entry: '{item.title}' - {item.reason}",
                 [item.pdf_page] if item.pdf_page else [])


async def run_checks(ctx: RunContext, segments: list[Segment], docs: list[Node], records: dict[int, PageRecord]) -> None:
    check_labels(ctx, segments, records)
    check_references(ctx, segments, docs, records)
    check_contract_documents_list(ctx, segments, records)
    check_index_pages(ctx, segments, docs, records)
    await final_audit(ctx, docs, records)
