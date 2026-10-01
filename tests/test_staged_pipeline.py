"""Offline end-to-end wiring test for the staged pipeline + file outputs + SQL persistence.

A mock Backend returns minimal, schema-valid responses per call, so the whole chain (macro -> refine ->
anchors -> gap-hunt -> extract -> dates -> enrich -> projection -> contract.json/clauses.jsonl/toc.md ->
SQLite) runs without any network or Gemini spend. It asserts the wiring holds and rows land in the DB,
not extraction quality.
"""
import asyncio
import json

import pymupdf
import pytest

from epc_parser.config import Settings
from epc_parser.llm import RawResponse
from epc_parser.staged import output, runner

# Minimal valid JSON per response schema (by class name).
_RESPONSES = {
    "BatchDivisionResponse": '{"boundaries_detected": []}',  # -> one inferred front-matter division
    "DivisionTitleLLM": '{"division_type": "CONTRACT_AGREEMENT", "title": "Contract Agreement", '
                        '"numbering_scheme": "clause"}',
    "WindowAnchors": '{"anchors": [{"pdf_page": 1, "kind": "clause", "marker": "1", "title": "Definitions"}]}',
    "GapHuntLLM": '{"found": false}',
    "PageTextLLM": '{"first_block_continues_previous_page": false, "blocks": '
                   '[{"kind": "clause", "marker": "1", "text": "1. DEFINITIONS In this Contract the following apply."}]}',
    "EnrichResponse": '{"items": [{"index": 0, "clause_description": "Defines contract terms.", '
                      '"priority": "Medium", "risk_level": "medium", "clause_type": "General"}]}',
    "HeadFacts": '{"date_anchors": [], "parties": []}',
}


class MockBackend:
    def __init__(self):
        self.calls = 0

    async def generate(self, *, model, system, parts, schema, thinking, media_resolution,
                       max_output_tokens, meta, temperature=None):
        self.calls += 1
        name = schema.__name__ if schema else ""
        text = _RESPONSES.get(name, "{}")
        return RawResponse(text, "STOP", input_tokens=100, output_tokens=20)


@pytest.fixture
def synthetic_pdf(tmp_path):
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 100), f"1. DEFINITIONS In this Contract the following apply. (page {i + 1})")
    path = tmp_path / "synthetic.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def _settings(tmp_path):
    s = Settings(api_key="fake")
    s.cache_dir = tmp_path / "cache"
    s.concurrency = 2
    return s


def test_runner_end_to_end_and_outputs(synthetic_pdf, tmp_path):
    s = _settings(tmp_path)
    result = asyncio.run(runner.run(synthetic_pdf, s, backend=MockBackend()))

    assert result["status"] == "completed"
    assert result["page_count"] == 3
    assert result["source_sha256"]
    assert result["leaves"], "no divisions produced"
    assert isinstance(result["clauses"], list)
    assert "registry" in result["date_anchors"]  # date engine ran, returned a (possibly empty) registry
    # a clause was extracted and enriched
    assert any("DEFINITIONS" in c["clause_content"] for c in result["clauses"])

    summary = output.write_outputs(result, tmp_path / "out")
    for f in ("contract.json", "clauses.jsonl", "toc.md"):
        assert (tmp_path / "out" / f).exists()
    contract = json.loads((tmp_path / "out" / "contract.json").read_text())
    assert contract["documents"], "contract.json has no document tree"
    assert contract["source_sha256"] == result["source_sha256"]


def test_persist_to_sqlite(synthetic_pdf, tmp_path):
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from epc_parser.db import persist, to_async_url
    from epc_parser.db.models import Clause, Document

    s = _settings(tmp_path)
    result = asyncio.run(runner.run(synthetic_pdf, s, backend=MockBackend()))

    db_url = f"sqlite:///{tmp_path / 'epc.db'}"
    info = asyncio.run(persist(result, db_url))
    assert info["document_id"] == 1
    assert info["divisions"] >= 1

    async def _counts():
        engine = create_async_engine(to_async_url(db_url))
        try:
            maker = async_sessionmaker(engine)
            async with maker() as session:
                docs = (await session.execute(select(func.count()).select_from(Document))).scalar_one()
                clauses = (await session.execute(select(func.count()).select_from(Clause))).scalar_one()
            return docs, clauses
        finally:
            await engine.dispose()

    docs, clauses = asyncio.run(_counts())
    assert docs == 1
    assert clauses >= 1


def test_persist_url_normalisation():
    from epc_parser.db import to_async_url
    assert to_async_url("sqlite:///x.db") == "sqlite+aiosqlite:///x.db"
    assert to_async_url("postgresql://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"
    assert to_async_url("mysql://u:p@h/db") == "mysql+aiomysql://u:p@h/db"
    assert to_async_url("postgresql+asyncpg://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"  # unchanged
