"""Stage 4 - build the document tree and attach verbatim text to every node.

Text is assembled by code from the blocks read in stage 1, in reading order across pages:
  * a kept heading/clause block opens a node; every following body block belongs to it until the next node opens
  * a paragraph continuing from the previous page is joined without a paragraph break (with de-hyphenation)
  * tables split across pages are stitched, dropping a repeated header row
  * stamps and signatures become annotations (not clause text); handwritten insertions stay inline, marked
Every node keeps the block ids it was built from, so each character can be traced to a page.
"""
from __future__ import annotations

import asyncio
import logging
import re
from difflib import SequenceMatcher

from . import prompts as P
from .context import RunContext
from .models import Annotation, Block, Node, PageRecord, Segment, SummaryLLM
from .numbering import clean, norm_text
from .stage_tree import Decision

log = logging.getLogger("epc.text")

SKIP_KINDS = {"header", "footer", "page_number"}
ANNOTATION_KINDS = {"stamp_or_seal", "signature_block"}
LEGIBILITY_ORDER = ["clear", "partial", "illegible"]


def _page_of(block_id: str) -> int:
    return int(block_id[1:5])


def _render(b: Block) -> str:
    if b.kind == "handwritten":
        return f"[Handwritten: {b.text.strip()}]"
    return b.text.strip()


def _is_md_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line.strip()))


class _Assembler:
    def __init__(self) -> None:
        self.last_node: Node | None = None
        self.last_kind: str | None = None
        self.last_table_header: str | None = None

    def append(self, node: Node, b: Block, continues: bool) -> None:
        piece = _render(b)
        if not piece:
            return
        same_flow = continues and self.last_node is node
        if b.kind == "table":
            lines = piece.splitlines()
            if same_flow and self.last_kind == "table" and lines and self.last_table_header and \
                    norm_text(lines[0]) == norm_text(self.last_table_header):
                lines = lines[2:] if len(lines) > 1 and _is_md_separator(lines[1]) else lines[1:]
                piece = "\n".join(lines)
                joiner = "\n"
            else:
                joiner = "\n" if same_flow and self.last_kind == "table" else "\n\n"
                self.last_table_header = lines[0] if lines else None
        elif same_flow and self.last_kind in ("paragraph", "clause", "recital", "heading"):
            if node.text.endswith("-") and piece[:1].islower():
                node.text = node.text[:-1]
                joiner = ""
            else:
                joiner = " "
        else:
            joiner = "\n\n"
        node.text = piece if not node.text else f"{node.text}{joiner}{piece}"
        node.block_ids.append(b.id)
        if b.legibility != "clear" and LEGIBILITY_ORDER.index(b.legibility) > LEGIBILITY_ORDER.index(node.legibility):
            node.legibility = b.legibility
        self.last_node, self.last_kind = node, b.kind


def build_document(ctx: RunContext, index: int, seg: Segment, records: dict[int, PageRecord],
                   decisions: dict[str, Decision]) -> Node:
    s = ctx.settings
    pages = [records[p] for p in range(seg.start_page, seg.end_page + 1)]
    first_content = next((r for r in pages if r.status not in ("blank", "failed")), pages[0])
    doc = Node(
        id=seg.id, kind="document", structure=str(index), level=0, title=seg.title, doc_type=seg.doc_type,
        page_start=seg.start_page, page_end=seg.end_page, printed_label_start=first_content.printed_label,
        verification="agreed" if seg.boundary_verified or (
            seg.boundary_verified is None and seg.confidence >= s.boundary_confidence_threshold) else "unverified",
        metadata={
            "reference": seg.reference, "date": seg.date, "segmentation_confidence": seg.confidence,
            "segmentation_evidence": seg.evidence, "boundary_verified": seg.boundary_verified,
            "stamp_papers": [dict(page=r.page, **r.stamp_paper.model_dump()) for r in pages if r.stamp_paper],
            "blank_pages": [r.page for r in pages if r.status == "blank"],
            "duplicate_pages": {str(r.page): r.duplicate_of for r in pages if r.duplicate_of},
            "text_unavailable_pages": [r.page for r in pages if r.status in ("structure_only", "failed")],
            "printed_labels": {str(r.page): r.printed_label for r in pages if r.printed_label},
            "languages": sorted({lang for r in pages for lang in r.languages}),
        },
    )
    kept = {bid: d for bid, d in decisions.items() if d.keep}
    stack: list[Node] = [doc]
    asm = _Assembler()
    recitals: Node | None = None

    for r in pages:
        if r.duplicate_of or r.status in ("blank", "failed"):
            continue
        body = [b for b in r.blocks if b.kind not in SKIP_KINDS]
        for pos, b in enumerate(body):
            continues = pos == 0 and r.first_continues
            d = kept.get(b.id)
            if d is not None and d.level > s.max_tree_depth:
                d = None  # deeper than the configured depth: stays as body text of its parent
            if d is not None:
                parent_level = stack[-1].level
                level = max(1, min(d.level, parent_level + 1))
                while stack[-1].level >= level:
                    stack.pop()
                parent = stack[-1]
                path = f"{parent.structure}.{len(parent.children) + 1}"
                number = d.number or b.number
                title = d.title or b.title
                node = Node(
                    id=f"{seg.id}.{path.split('.', 1)[1]}", kind="heading" if b.kind == "heading" else "clause",
                    structure=path, level=level, number=number, title=title,
                    heading_text=clean(b.text) if b.kind == "heading" else clean(f"{number or ''} {title or ''}") or None,
                    page_start=b.page, page_end=b.page, printed_label_start=r.printed_label, starts_at_top=pos == 0,
                    verification="template" if d.source == "template" and b.verification == "unverified" else b.verification,
                    legibility=b.legibility,
                )
                if d.source == "hunt":
                    node.verification = "hunted"
                if b.kind == "heading":
                    node.block_ids.append(b.id)
                else:
                    asm.append(node, b, continues=False)
                asm.last_node, asm.last_kind = node, b.kind
                parent.children.append(node)
                stack.append(node)
                continue

            current = stack[-1]
            if b.kind == "heading" and current is doc and not doc.text and \
                    SequenceMatcher(None, norm_text(b.text), norm_text(seg.title)).ratio() >= 0.85:
                doc.heading_text = clean(b.text)  # the document's own title block, not body text
                doc.block_ids.append(b.id)
                continue
            if b.kind in ANNOTATION_KINDS:
                current.annotations.append(Annotation(kind=b.kind, page=b.page, text=clean(b.text)))
                current.block_ids.append(b.id)
                continue
            if b.kind == "recital" and current is doc:
                if recitals is None:
                    rpath = f"{index}.{len(doc.children) + 1}"
                    recitals = Node(id=f"{seg.id}.{rpath.split('.', 1)[1]}", kind="recitals", structure=rpath, level=1,
                                    title="Recitals",
                                    page_start=b.page, page_end=b.page, printed_label_start=r.printed_label,
                                    verification="agreed")
                    doc.children.append(recitals)
                asm.append(recitals, b, continues)
                continue
            asm.append(current, b, continues)

    _finalise(doc)
    return doc


def _finalise(node: Node) -> int:
    pages = [_page_of(b) for b in node.block_ids] + [node.page_start]
    for a in node.annotations:
        pages.append(a.page)
    for child in node.children:
        pages.append(_finalise(child))
    if node.kind != "document":
        node.page_end = max(pages)
    return node.page_end


def full_text(node: Node) -> str:
    """Own text plus all descendants, in document order (use for retrieval / grounding)."""
    parts = []
    if node.kind == "heading" and node.heading_text:
        parts.append(node.heading_text)
    if node.text:
        parts.append(node.text)
    parts.extend(full_text(c) for c in node.children)
    return "\n\n".join(p for p in parts if p)


def iter_nodes(node: Node):
    yield node
    for c in node.children:
        yield from iter_nodes(c)


def build_all(ctx: RunContext, segments: list[Segment], records: dict[int, PageRecord],
              hierarchies: dict[str, dict[str, Decision]]) -> list[Node]:
    docs = [build_document(ctx, i + 1, seg, records, hierarchies.get(seg.id, {})) for i, seg in enumerate(segments)]
    for doc in docs:
        incomplete = set(doc.metadata.get("text_unavailable_pages", []))
        for n in iter_nodes(doc):
            if incomplete and any(_page_of(b) in incomplete for b in n.block_ids):
                n.flags.append("text_incomplete")
            if n.kind != "document" and n.verification in ("disputed", "unverified"):
                n.flags.append(n.verification)
            if n.legibility != "clear":
                n.flags.append(f"legibility:{n.legibility}")
            if n.kind == "clause" and not n.text.strip():
                n.flags.append("empty_text")
    return docs


async def summarise(ctx: RunContext, docs: list[Node]) -> None:
    mode = ctx.settings.summaries
    if mode == "none":
        return
    targets: list[tuple[Node, Node]] = []
    for doc in docs:
        if mode == "top":
            targets.append((doc, doc))
            targets.extend((doc, c) for c in doc.children)
        else:
            targets.extend((doc, n) for n in iter_nodes(doc))

    async def one(doc: Node, node: Node) -> None:
        text = full_text(node)
        if len(text) < 200:
            return
        section = node.title or node.heading_text or node.number or doc.title or ""
        res = await ctx.llm.call("summary", role="summarizer", system=P.SUMMARY_SYSTEM,
                                 parts=[P.SUMMARY_USER.format(doc_title=doc.title, section=section, text=text[:60000])],
                                 schema=SummaryLLM, thinking="minimal", max_output_tokens=1024, meta={"node": node.id})
        if res.ok:
            node.summary = res.parsed.summary  # type: ignore[union-attr]
            node.key_obligations = res.parsed.key_obligations  # type: ignore[union-attr]

    await asyncio.gather(*[one(d, n) for d, n in targets])
