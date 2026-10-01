"""End-to-end orchestration."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .context import RunContext
from .llm import Backend, GeminiBackend, LLMClient
from .output import write_outputs
from .render import PageRenderer
from .stage_checks import run_checks
from .stage_pages import process_all_pages
from .stage_segment import segment_bundle
from .stage_text import build_all, summarise
from .stage_tree import build_hierarchies

log = logging.getLogger("epc")


async def parse_contract(pdf_path: str | Path, out_dir: str | Path, settings: Settings | None = None,
                         backend: Backend | None = None, page_range: tuple[int, int] | None = None) -> dict:
    started = datetime.now(timezone.utc)
    settings = settings or Settings()
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    renderer = PageRenderer(pdf_path, settings)
    try:
        first, last = page_range or (1, renderer.page_count)
        if not 1 <= first <= last <= renderer.page_count:
            raise ValueError(f"page range {first}-{last} outside 1-{renderer.page_count}")
        llm = LLMClient(settings, backend or GeminiBackend(settings))
        ctx = RunContext(settings=settings, renderer=renderer, llm=llm, page_range=(first, last))

        log.info("stage 1/5: reading %d pages", last - first + 1)
        records = await process_all_pages(ctx)

        log.info("stage 2/5: segmenting bundle")
        segments = await segment_bundle(ctx, records)

        log.info("stage 3/5: building clause hierarchies")
        hierarchies = await build_hierarchies(ctx, segments, records)

        log.info("stage 4/5: assembling clause text")
        docs = build_all(ctx, segments, records, hierarchies)
        await summarise(ctx, docs)

        log.info("stage 5/5: completeness checks")
        await run_checks(ctx, segments, docs, records)

        result = write_outputs(ctx, out_dir, pdf_path, segments, docs, records, started)
        log.info("done: %d documents, %d entries, %s review items, $%.3f", result["stats"]["documents"],
                 result["stats"]["nodes"], result["stats"]["review"], result["cost"]["total_usd"])
        return result
    finally:
        renderer.close()
