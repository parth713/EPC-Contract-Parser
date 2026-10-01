import asyncio
import json

from epc_parser import Settings, parse_contract
from epc_parser.stage_text import iter_nodes
from tests.mock_backend import MockBackend, make_pdf


def _settings(tmp_path):
    s = Settings(api_key="test")
    s.cache_dir = tmp_path / "cache"
    s.template_dir = tmp_path / "templates"
    s.concurrency = 8
    return s


def _find(node, number):
    return next(n for n in iter_nodes(node) if n.get("number") == number) if isinstance(node, dict) else None


def _iter(d):
    yield d
    for c in d["children"]:
        yield from _iter(c)


def test_end_to_end(tmp_path):
    pdf = tmp_path / "bundle.pdf"
    make_pdf(pdf)
    backend = MockBackend()
    result = asyncio.run(parse_contract(pdf, tmp_path / "out", _settings(tmp_path), backend=backend))

    pages = {p["page"]: p for p in json.loads((tmp_path / "out" / "pages.json").read_text())}
    assert pages[8]["status"] == "blank"
    assert pages[6]["read_level"] == "L2_flash_halves"            # truncated full read -> halves
    b24 = next(b for b in pages[7]["blocks"] if b["number"] == "2.4")
    assert b24["verification"] == "corrected_by_arbitration"      # A read 2.5, scan read 2.4, crop confirmed 2.4
    assert pages[11]["duplicate_of"] == 10

    docs = result["documents"]
    assert [(d["page_start"], d["page_end"]) for d in docs] == [(1, 1), (2, 3), (4, 4), (5, 8), (9, 9), (10, 11)]
    assert docs[5]["title"] == "ANNEXURE-III"                     # boundary check split the merged annexures

    agreement = docs[1]
    assert agreement["children"][0]["kind"] == "recitals"
    c11 = next(n for n in _iter(agreement) if n["number"] == "1.1")
    assert "in the order listed above" in c11["text"]               # paragraph continued across pages 2 -> 3

    gcc = docs[3]
    tops = [c["number"] for c in gcc["children"]]
    assert tops == ["1", "2"]
    two = gcc["children"][1]
    assert [c["number"] for c in two["children"]] == ["2.1", "2.3", "2.4"]
    c23 = two["children"][1]
    assert "| Commissioning | 10% |" in c23["text"] and c23["text"].count("| Milestone | Share |") == 1  # table stitched
    assert c23["page_start"] == 6 and c23["page_end"] == 7
    c24 = two["children"][2]
    assert [c["number"] for c in c24["children"]] == ["(a)", "(b)"]
    assert any(a["kind"] == "stamp_or_seal" for a in c24["children"][1]["annotations"])

    codes = {r["code"] for r in result["review_queue"]}
    for expected in ("numbering_gap", "reference_not_found", "attachment_series_gap", "duplicate_page",
                     "listed_document_missing", "boundary_added"):
        assert expected in codes, expected
    gap = next(r for r in result["review_queue"] if r["code"] == "numbering_gap")
    assert "2.2" in gap["message"]

    out = tmp_path / "out"
    for f in ("contract.json", "pages.json", "clauses.jsonl", "toc.md", "review_queue.json", "bundle.bookmarked.pdf"):
        assert (out / f).exists(), f
    import pymupdf
    with pymupdf.open(out / "bundle.bookmarked.pdf") as d:
        assert len(d.get_toc()) == result["stats"]["nodes"] + result["stats"]["documents"]

    # second run: everything that succeeded is served from cache
    backend2 = MockBackend()
    asyncio.run(parse_contract(pdf, tmp_path / "out2", _settings(tmp_path), backend=backend2))
    assert backend2.calls.get("heading_scan", 0) == 0 and backend2.calls.get("hierarchy", 0) == 0
