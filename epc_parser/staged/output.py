"""File outputs for the staged pipeline: contract.json (full structure), clauses.jsonl (flat, one row
per clause — ready for search/RAG), and toc.md (human-readable outline + review table). Kept dependency-
free (json + pathlib). The DB persistence path (epc_parser.db) is separate and optional; these files are
always written so the staged pipeline has the same "inspect the output" affordance as the flagship one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def _unit_tree(u: dict) -> dict:
    """Serialisable view of one leaf unit and its sub-clauses (children)."""
    return {
        "unit_id": u.get("unit_id"),
        "kind": u.get("kind"),
        "marker": u.get("marker"),
        "title": u.get("title"),
        "page_start": u.get("page_start"),
        "page_end": u.get("page_end"),
        "text": u.get("text") or "",
        "clause_description": u.get("clause_description") or "",
        "priority": u.get("priority"),
        "risk_level": u.get("risk_level"),
        "clause_type": u.get("clause_type"),
        "flags": u.get("flags") or [],
        "children": [
            {"marker": c.get("marker"), "title": c.get("title"), "text": c.get("text") or ""}
            for c in (u.get("children") or [])
        ],
    }


def _division_tree(d: dict) -> dict:
    return {
        "division_id": d.get("division_id"),
        "division_type": d.get("division_type"),
        "title": d.get("title"),
        "running_header": d.get("running_header"),
        "start_page": d.get("start_page"),
        "end_page": d.get("end_page"),
        "flags": d.get("flags") or [],
        "units": [_unit_tree(u) for u in d.get("units", [])],
    }


def write_outputs(result: dict, out_dir: str | Path) -> dict:
    """Write contract.json, clauses.jsonl and toc.md into out_dir. Returns a small summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    leaves: List[dict] = result.get("leaves", [])
    clauses: List[dict] = result.get("clauses", [])
    review: List[dict] = result.get("review", [])

    contract = {
        "source": result.get("source"),
        "source_sha256": result.get("source_sha256"),
        "page_count": result.get("page_count"),
        "status": result.get("status"),
        "notes": result.get("notes"),
        "stats": {
            "divisions": len(result.get("divisions", [])),
            "leaf_groups": len(leaves),
            "clauses": len(clauses),
            "review_items": len(review),
        },
        "cost": result.get("cost", {}),
        "extraction_metadata": result.get("extraction_metadata", {}),
        "date_anchors": result.get("date_anchors", {}),
        "parties": result.get("parties", []),
        "documents": [_division_tree(d) for d in leaves],
        "review_queue": review,
    }
    (out / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2))

    with (out / "clauses.jsonl").open("w", encoding="utf-8") as f:
        for c in clauses:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    (out / "toc.md").write_text(_render_toc(result))

    return {"out_dir": str(out), "files": ["contract.json", "clauses.jsonl", "toc.md"],
            "clauses": len(clauses), "review_items": len(review)}


def _render_toc(result: dict) -> str:
    lines: List[str] = []
    review: List[dict] = result.get("review", [])
    src = Path(result.get("source") or "contract").name
    lines.append(f"# Table of contents — {src}")
    lines.append("")
    md = result.get("extraction_metadata", {})
    lines.append(f"- Pages: {result.get('page_count')}  ·  Divisions: {md.get('divisions')}  ·  "
                 f"Clauses: {md.get('total_clauses_extracted')}  ·  Status: {result.get('status')}")
    da = result.get("date_anchors", {}) or {}
    registry = da.get("registry") or {}
    if registry:
        populated = {k: v for k, v in registry.items() if v}
        lines.append(f"- Named dates: {len(registry)} ({len(populated)} resolved)")
    parties = result.get("parties") or []
    if parties:
        who = "; ".join(f"{p.get('side')}: {p.get('name')}" for p in parties if p.get("name"))
        if who:
            lines.append(f"- Parties: {who}")
    lines.append("")

    for d in result.get("leaves", []):
        title = d.get("title") or d.get("division_type") or "Division"
        lines.append(f"## {d.get('division_id')} — {title}  "
                     f"(p{d.get('start_page')}–{d.get('end_page')}, {d.get('division_type')})")
        for u in d.get("units", []):
            if u.get("kind") in ("preamble", "whole") and not (u.get("text") or "").strip():
                continue
            marker = (u.get("marker") or "").strip()
            utitle = (u.get("title") or "").strip()
            label = f"{marker} — {utitle}" if marker and utitle else (marker or utitle or "(untitled)")
            lines.append(f"- {label}  _(p{u.get('page_start')})_")
            for c in (u.get("children") or []):
                cm = (c.get("marker") or "").strip()
                ct = (c.get("title") or "").strip()
                clabel = f"{cm} {ct}".strip() or "(sub-clause)"
                lines.append(f"  - {clabel}")
        lines.append("")

    if review:
        lines.append("## Review queue")
        lines.append("")
        lines.append("| Severity | Code | Pages | Message |")
        lines.append("|---|---|---|---|")
        for r in review:
            pages = ",".join(str(p) for p in (r.get("pages") or []))
            msg = (r.get("message") or "").replace("|", "\\|")
            lines.append(f"| {r.get('severity','')} | {r.get('code','')} | {pages} | {msg} |")
        lines.append("")
    return "\n".join(lines)
