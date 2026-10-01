"""Command line interface: python -m epc_parser contract.pdf -o out/"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .config import Settings
from .pipeline import parse_contract


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epc_parser", description="LLM-only structured extraction of scanned EPC contracts.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--pages", help="page range to process, e.g. 1-40 (for trials)")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--summaries", choices=["none", "top", "all"])
    ap.add_argument("--no-audit", action="store_true", help="skip the final LLM completeness audit")
    ap.add_argument("--no-hunts", action="store_true", help="flag numbering gaps without image hunts")
    ap.add_argument("--no-arbitration", action="store_true", help="flag A/B disagreements without crop re-reads")
    ap.add_argument("--max-depth", type=int, help="deepest clause level to split into nodes (default 5)")
    ap.add_argument("--reader-model")
    ap.add_argument("--scanner-model")
    ap.add_argument("--reasoner-model")
    ap.add_argument("--escalation-model")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--template-dir", type=Path)
    ap.add_argument("--no-bookmarks", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    s = Settings()
    if a.concurrency: s.concurrency = a.concurrency
    if a.summaries: s.summaries = a.summaries
    if a.no_audit: s.final_audit = False
    if a.no_hunts: s.gap_hunts = False
    if a.no_arbitration: s.arbitration = False
    if a.max_depth: s.max_tree_depth = a.max_depth
    if a.reader_model: s.model_reader = a.reader_model
    if a.scanner_model: s.model_scanner = a.scanner_model
    if a.reasoner_model: s.model_reasoner = a.reasoner_model
    if a.escalation_model: s.model_escalation = a.escalation_model
    if a.cache_dir: s.cache_dir = a.cache_dir
    if a.template_dir: s.template_dir = a.template_dir
    if a.no_bookmarks: s.write_bookmarked_pdf = False

    page_range = None
    if a.pages:
        lo, _, hi = a.pages.partition("-")
        page_range = (int(lo), int(hi or lo))
    asyncio.run(parse_contract(a.pdf, a.out, s, page_range=page_range))


if __name__ == "__main__":
    main()
