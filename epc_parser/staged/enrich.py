"""Stage 4 - enrichment. A cheap text-only pass that gives each leaf unit a one-sentence summary,
priority, risk level and clause type (the response fields the downstream DB/API expect). Batched per
division to bound calls; defaults fill in on failure so enrichment can never block extraction."""
from __future__ import annotations

import asyncio
from typing import List

import logging
logger = logging.getLogger("epc.staged.enrich")

from ..config import Settings as StagedConfig
from ..llm import LLMClient as StagedLLM
from .models import EnrichResponse
from ..numbering import clean

ENRICH_SYSTEM = (
    "You classify contract clauses for a CLM platform. For each clause given (identified by its index), "
    "return: clause_description (ONE original sentence, max ~30 words, summarising what the clause "
    "governs - do not copy the text); priority (High|Medium|Low); risk_level (low|medium|high|critical); "
    "clause_type, one of: Legal & Compliance, Financial, Risk & Liability, Termination, Confidentiality, "
    "Intellectual Property, Performance & Delivery, Dispute Resolution, General. "
    "A clause line is shown as '[index N] <marker> <title>' followed by its body. If the <title> is BLANK "
    "(the line has only a marker, or nothing, before the body), you MUST return clause_title - a short 3-6 "
    "word Title Case name for the clause drawn from its content. Never leave it empty and never echo the "
    "marker: every titleless clause must get a real generated name. Only when the clause already shows a "
    "title, return null for clause_title. Echo each clause's index back. Return one item per clause, no more."
)
_SNIPPET = 1200
# A genuine clause heading is short. When anchor detection finds a numbered point with no heading of its
# own (common in minutes / letters / prose clauses), it fills the title with the clause's opening sentence
# instead - long text that is really the body, not a name. Treat such a pseudo-title as "no title" so
# enrich names the clause. Kept deliberately loose (length only) to avoid clobbering real long headings.
_TITLE_WORD_MAX = 10
_TITLE_CHAR_MAX = 80


def _is_heading(title: str) -> bool:
    """True when `title` reads like a real clause heading rather than the clause's own body sentence."""
    t = clean(title)
    return bool(t) and len(t) <= _TITLE_CHAR_MAX and len(t.split()) <= _TITLE_WORD_MAX


def _clause_line(i: int, u: dict) -> str:
    raw = clean(u.get("title") or "")
    # Only show the model a title it should keep; a body-sentence pseudo-title is blanked so the model
    # is asked to generate a proper clause_title for it (matching the apply rule below).
    title = raw if _is_heading(raw) else ""
    body = (u.get("text") or "")[:_SNIPPET]
    return f"[index {i}] {u.get('marker') or ''} {title}\n{body}"


async def _enrich_batch(llm: StagedLLM, cfg: StagedConfig, units: List[dict], base: int, doc_title: str) -> None:
    lines = "\n\n---\n\n".join(_clause_line(base + k, u) for k, u in enumerate(units))
    prompt = (f"Document: {clean(doc_title)[:80]}. Classify the following {len(units)} clause(s). "
              f"Return exactly {len(units)} items.\n\n{lines}")
    res = await llm.call("enrich", role="reasoner", system=ENRICH_SYSTEM, parts=[prompt], schema=EnrichResponse,
                         thinking="minimal", max_output_tokens=min(8192, 300 * len(units) + 512))
    by_index = {}
    if res.ok:
        for it in res.parsed.items:
            by_index[it.index] = it
    for k, u in enumerate(units):
        it = by_index.get(base + k)
        u["clause_description"] = (clean(it.clause_description) if it else "") or ""
        u["priority"] = (it.priority if it else "Medium")
        u["risk_level"] = (it.risk_level if it else "medium")
        u["clause_type"] = (it.clause_type if it else "General")
        # Title fallback: fill when the clause has no genuine heading of its own - either truly blank, or
        # a body-sentence pseudo-title that anchor detection left behind (see _is_heading). A real
        # upstream heading stays authoritative. Flows into clause_title via _persist_summaries / _to_clause_dicts.
        if not _is_heading(u.get("title") or ""):
            gen = clean(it.clause_title) if it else ""
            # Prefer the generated short title; never keep a body sentence as the title - fall back to the
            # marker (persist turns a None title into the marker / "No Title found").
            u["title"] = gen or None


async def run(llm: StagedLLM, cfg: StagedConfig, leaves: List[dict]) -> None:
    if not cfg.enrich_enabled:
        for d in leaves:
            for u in d["units"]:
                u.setdefault("clause_description", "")
                u.setdefault("priority", "Medium")
                u.setdefault("risk_level", "medium")
                u.setdefault("clause_type", "General")
        return

    sem = asyncio.Semaphore(max(2, cfg.concurrency))
    jobs = []

    async def batched(div):
        units = [u for u in div["units"] if (u.get("text") or "").strip()]
        # units with no text still need the fields set (defaults)
        for u in div["units"]:
            if not (u.get("text") or "").strip():
                u.setdefault("clause_description", "")
                u.setdefault("priority", "Medium")
                u.setdefault("risk_level", "medium")
                u.setdefault("clause_type", "General")
        for start in range(0, len(units), cfg.enrich_batch):
            chunk = units[start:start + cfg.enrich_batch]

            async def go(chunk=chunk, start=start, div=div):
                async with sem:
                    await _enrich_batch(llm, cfg, chunk, start, div["title"])
            jobs.append(go())

    for d in leaves:
        await batched(d)
    await asyncio.gather(*jobs)
    logger.info(f"[staged.enrich] enriched {sum(len(d['units']) for d in leaves)} units")
