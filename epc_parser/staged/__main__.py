"""CLI for the staged coarse-to-fine pipeline.

    python -m epc_parser.staged contract.pdf -o out/
    python -m epc_parser.staged contract.pdf -o out/ --db-url sqlite:///epc.db
    python -m epc_parser.staged contract.pdf -o out/ --db-url postgresql://user:pw@localhost/epc

Writes contract.json / clauses.jsonl / toc.md into the output dir, and (when --db-url or EPC_DB_URL is
set) also persists the parsed tree into that SQL database. Re-runs are cheap: every LLM response is cached
on disk by content hash (see --cache-dir), so only changed calls cost anything.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from ..config import Settings
from . import output, runner


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epc_parser.staged",
                                 description="Staged clause extraction (macro -> refine -> anchors -> "
                                             "gap-hunt -> extract -> dates -> enrich), with optional SQL persistence.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out_staged"))
    ap.add_argument("--db-url", default=os.getenv("EPC_DB_URL"),
                    help="SQLAlchemy URL of any SQL database to persist into, e.g. sqlite:///epc.db or "
                         "postgresql://user:pw@host/db. Defaults to $EPC_DB_URL; omit to skip persistence.")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--reader-model")
    ap.add_argument("--reasoner-model")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--no-dates", action="store_true", help="skip the contract date-anchor / parties engine")
    ap.add_argument("--no-enrich", action="store_true", help="skip clause summary/title/priority/risk enrichment")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "httpcore", "aiosqlite", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    s = Settings()
    if a.concurrency:
        s.concurrency = a.concurrency
    if a.reader_model:
        s.model_reader = a.reader_model
    if a.reasoner_model:
        s.model_reasoner = a.reasoner_model
    if a.cache_dir:
        s.cache_dir = a.cache_dir
    if a.no_dates:
        s.dates_enabled = False
    if a.no_enrich:
        s.enrich_enabled = False

    result = asyncio.run(runner.run(a.pdf, s))
    summary = output.write_outputs(result, a.out)
    print(f"\n{result['notes']}")
    print(f"wrote {', '.join(summary['files'])} to {summary['out_dir']}")

    if a.db_url:
        from ..db import persist
        info = asyncio.run(persist(result, a.db_url))
        print(f"persisted document #{info['document_id']} "
              f"({info['divisions']} divisions, {info['clauses']} clauses) to {info['db_url'].split('://', 1)[0]}")


if __name__ == "__main__":
    main()
