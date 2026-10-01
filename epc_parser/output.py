"""Write results. The original PDF is never modified; bookmarks go into a separate copy."""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from .context import RunContext
from .models import Node, PageRecord, Segment
from .numbering import clean
from .stage_text import full_text, iter_nodes

log = logging.getLogger("epc.output")
SCHEMA_VERSION = "1.0"


def _label(n: Node, width: int = 110) -> str:
    if n.kind == "document":
        return clean(n.title)[:width]
    head = clean(f"{n.number or ''} {n.title or ''}")
    if not n.title:
        snippet = clean(n.text if n.kind == "clause" else (n.heading_text or n.text))
        if n.number and snippet.startswith(n.number):
            snippet = snippet[len(n.number):].strip(" .:-")
        head = clean(f"{n.number or ''} {snippet[:70]}{'...' if len(snippet) > 70 else ''}")
    return head[:width] or "(untitled)"


def write_outputs(ctx: RunContext, out_dir: Path, pdf_path: Path, segments: list[Segment], docs: list[Node],
                  records: dict[int, PageRecord], started: datetime) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_nodes = [n for d in docs for n in iter_nodes(d)]
    status = Counter(r.status for r in records.values())
    severity = Counter(r.severity for r in ctx.review)
    cost = ctx.llm.cost.report()
    finished = datetime.now(timezone.utc)

    index = {
        "schema_version": SCHEMA_VERSION,
        "source": {"file": pdf_path.name, "sha256": ctx.renderer.sha256(), "page_count": ctx.renderer.page_count,
                   "pages_processed": list(ctx.page_range)},
        "generated_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 1),
        "models": {"reader": ctx.settings.model_reader, "scanner": ctx.settings.model_scanner,
                   "reasoner": ctx.settings.model_reasoner, "escalation": ctx.settings.model_escalation},
        "cost": cost,
        "stats": {
            "documents": len(docs), "nodes": len(all_nodes) - len(docs),
            "pages": dict(status), "duplicate_pages": sum(1 for r in records.values() if r.duplicate_of),
            "review": dict(severity),
            "verification": dict(Counter(n.verification for n in all_nodes if n.kind != "document")),
        },
        "documents": [d.model_dump(mode="json") for d in docs],
        "review_queue": [r.model_dump(mode="json") for r in ctx.review],
    }
    (out_dir / "contract.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    (out_dir / "pages.json").write_text(json.dumps([records[p].model_dump(mode="json") for p in sorted(records)],
                                                   ensure_ascii=False, indent=1))
    (out_dir / "review_queue.json").write_text(json.dumps(index["review_queue"], ensure_ascii=False, indent=2))

    with open(out_dir / "clauses.jsonl", "w", encoding="utf-8") as f:
        for doc in docs:
            for n in iter_nodes(doc):
                if n.kind == "document" and n.children and not n.text.strip():
                    continue  # its content lives in the child records
                f.write(json.dumps({
                    "id": n.id, "document_id": doc.id, "document_title": doc.title, "doc_type": doc.doc_type,
                    "structure": n.structure, "level": n.level, "number": n.number, "title": n.title,
                    "path_titles": _path_titles(doc, n), "page_start": n.page_start, "page_end": n.page_end,
                    "printed_label_start": n.printed_label_start, "verification": n.verification, "flags": n.flags,
                    "text": n.text, "full_text": n.text if n.kind == "document" else full_text(n), "summary": n.summary,
                }, ensure_ascii=False) + "\n")

    (out_dir / "toc.md").write_text(_toc_markdown(index, docs, ctx), encoding="utf-8")
    if ctx.settings.write_bookmarked_pdf:
        _bookmarked_pdf(pdf_path, out_dir / f"{pdf_path.stem}.bookmarked.pdf", docs)
    log.info("outputs written to %s", out_dir)
    return index


def _path_titles(doc: Node, target: Node) -> list[str]:
    def walk(n: Node, trail: list[str]) -> list[str] | None:
        trail = trail + [_label(n, 80)]
        if n is target:
            return trail
        for c in n.children:
            got = walk(c, trail)
            if got:
                return got
        return None
    return walk(doc, []) or []


def _toc_markdown(index: dict, docs: list[Node], ctx: RunContext) -> str:
    src = index["source"]
    lines = [f"# Table of Contents - {src['file']}", "",
             f"SHA-256 of source: `{src['sha256']}`  ",
             f"Generated: {index['generated_at']}  |  Documents: {index['stats']['documents']}  |  "
             f"Entries: {index['stats']['nodes']}  |  LLM cost: ${index['cost']['total_usd']}", "",
             "PDF pages are physical pages of the scanned file. Printed labels are the page numbers printed on the page.", "",
             "| # | Entry | Type | PDF pages | Printed | Check |", "|---|---|---|---|---|---|"]
    max_level = min(3, ctx.settings.max_tree_depth)
    for doc in docs:
        for n in iter_nodes(doc):
            if n.level > max_level:
                continue
            indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * n.level
            entry = f"**{_label(n)}**" if n.kind == "document" else f"{indent}{_label(n)}"
            pages = f"{n.page_start}" if n.page_start == n.page_end else f"{n.page_start}-{n.page_end}"
            check = "" if not n.flags and n.verification not in ("disputed", "unverified") else "review"
            lines.append(f"| {n.structure} | {entry.replace('|', '/')} | {doc.doc_type if n.kind == 'document' else ''} | "
                         f"{pages} | {n.printed_label_start or ''} | {check} |")
    problems = [r for r in ctx.review if r.severity != "info"]
    if problems:
        lines += ["", "## Items needing review", "", "| Severity | Code | Pages | Message |", "|---|---|---|---|"]
        for r in sorted(problems, key=lambda x: (x.severity != "error", x.pages[:1] or [0])):
            lines.append(f"| {r.severity} | {r.code} | {', '.join(map(str, r.pages))} | {r.message.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def _bookmarked_pdf(src: Path, dst: Path, docs: list[Node]) -> None:
    toc = []
    for doc in docs:
        for n in iter_nodes(doc):
            toc.append([n.level + 1, _label(n, 120), n.page_start])
    with pymupdf.open(src) as pdf:
        pdf.set_toc(toc)
        pdf.save(dst, garbage=0, deflate=True)
