"""One command to (re)build the full table of contents in a single run.

Runs, in order and in one process, the four TOC stages that were previously separate commands:

    Stage 1   macro division sweep        (stage1_macro)
    Stage 1.5 title/type refinement       (stage1_refine)
    Stage 2   leaf-unit anchors           (stage2_anchors)  - now TWO levels: main clause + X.X
    Stage 2.5 numbering-gap hunt          (stage2_gaphunt)

The TOC leaf is the deepest structural node available: a main clause's direct sub-clause (X.X) where
one is printed, otherwise the main clause (or section) itself. Nothing else in the pipeline changes;
verbatim extraction (Stage 3) still runs separately.

    python -m epc_parser.build_toc "contract.pdf" -o out_macro/
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .config import Settings
from .stage1_macro import run_macro_sweep
from .stage1_refine import refine_divisions
from .stage2_anchors import find_anchors, _print_leaves
from .stage2_gaphunt import run_gap_hunt

log = logging.getLogger("epc.toc")


async def build_toc(pdf_path: str | Path, out_dir: str | Path, settings: Settings, gap_hunt: bool = True) -> dict:
    log.info("stage 1/4: macro division sweep")
    await run_macro_sweep(pdf_path, out_dir, settings)
    log.info("stage 2/4: title & type refinement")
    await refine_divisions(pdf_path, out_dir, settings)
    log.info("stage 3/4: leaf-unit anchors (main clause + sub-clause)")
    await find_anchors(pdf_path, out_dir, settings)
    if gap_hunt:
        log.info("stage 4/4: numbering-gap hunt")
        data = await run_gap_hunt(pdf_path, out_dir, settings)
    else:
        import json
        data = json.loads((Path(out_dir) / "leaves.json").read_text())
    return data


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epc_parser.build_toc",
                                 description="Build the two-level TOC (macro -> refine -> anchors -> gap hunt) in one run.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out_macro"))
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--reader-model")
    ap.add_argument("--reasoner-model")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--template-dir", type=Path)
    ap.add_argument("--no-gap-hunt", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "httpcore"):
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
    if a.template_dir:
        s.template_dir = a.template_dir

    data = asyncio.run(build_toc(a.pdf, a.out, s, gap_hunt=not a.no_gap_hunt))
    _print_leaves(data)


if __name__ == "__main__":
    main()
