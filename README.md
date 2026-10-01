# epc-contract-parser

LLM-only structured extraction of scanned Indian EPC contract bundles (stamp paper, agreement, LOI/LOA, GCC, SCC,
annexures, forms, BoQ, with no printed TOC). It produces a verified table of contents **and the verbatim text of every
document, clause and sub-clause**, stored as JSON with page-level provenance.

No Tesseract or other OCR engine is used. All reading is done by Gemini on page images; code does rendering,
comparison, validation and assembly.

## Install

```bash
pip install -r requirements.txt          # google-genai, pymupdf, pydantic
export GEMINI_API_KEY=...                # or GOOGLE_GENAI_USE_VERTEXAI=true with Vertex credentials
```

## Run

```bash
python -m epc_parser contract.pdf -o out/

# trial on a slice first
python -m epc_parser contract.pdf -o out_trial/ --pages 1-40 --no-audit --summaries none
```

Useful flags: `--concurrency 32`, `--summaries none|top|all`, `--max-depth 4`, `--no-hunts`, `--no-arbitration`,
`--reader-model`, `--scanner-model`, `--reasoner-model`, `--escalation-model`, `--cache-dir`, `--template-dir`, `-v`.

From Python:

```python
import asyncio
from epc_parser import Settings, parse_contract
result = asyncio.run(parse_contract("contract.pdf", "out/", Settings(summaries="none")))
```

Re-runs are cheap: every successful LLM response is cached by content hash in `.epc_cache/`, so a crashed or
re-tuned run only pays for calls that changed.

## Staged pipeline (coarse-to-fine) + SQL persistence

Alongside the flagship pipeline above, there is a second, **coarse-to-fine** pipeline that runs entirely
in memory and can persist its result to any SQL database. It is the resynced descendant of the original
`build_toc` + `stage3_extract` lineage, with two net-new passes added:

```
macro      coarse division sweep (sliding windows)        ->  where each major document begins
refine     per-division title / type / numbering scheme
anchors    leaf-unit anchors (section or main clause)      ->  the clause tree
gap-hunt   numbering-gap image hunt + blind confirm
extract    verbatim text per unit, with RECITATION recovery (sentinel re-read, then PDF text layer)
dates      contract-level named-date registry + the two principal parties   (net-new)
enrich     per-clause summary, title, priority, risk, type                   (net-new)
```

```bash
# file outputs only
python -m epc_parser.staged contract.pdf -o out/

# also persist into a SQL database (SQLite works out of the box; zero setup)
python -m epc_parser.staged contract.pdf -o out/ --db-url sqlite:///epc.db

# any SQL backend via its URL (install the matching driver: pip install -e ".[postgres]" / ".[mysql]")
python -m epc_parser.staged contract.pdf -o out/ --db-url postgresql://user:pw@localhost/epc
export EPC_DB_URL=sqlite:///epc.db   # or set it once in the environment
```

Flags: `--db-url`, `--concurrency`, `--reader-model`, `--reasoner-model`, `--cache-dir`, `--no-dates`,
`--no-enrich`, `-v`. From Python: `from epc_parser.staged import run; result = await run("contract.pdf")`,
then `epc_parser.staged.output.write_outputs(result, "out/")` and/or
`await epc_parser.db.persist(result, "sqlite:///epc.db")`.

The persisted schema mirrors a parsed document — `epc_documents` → `epc_divisions` → `epc_clauses` →
`epc_subclauses`, with the date-anchor registry and principal parties stored as JSON on the document row.
It is **standalone**: no external ids, no application coupling, portable JSON columns across SQLite /
PostgreSQL / MySQL. Tables are created automatically on first use.

## Outputs

| File | Contents |
|---|---|
| `contract.json` | The full index: source SHA-256, models, cost, stats, the document tree with text, and the review queue |
| `clauses.jsonl` | One record per node: ids, path titles, pages, `text` (own text) and `full_text` (with sub-clauses). Ready for search/RAG |
| `pages.json` | Every page's blocks as read (kind, marker, text, box, legibility, verification), A/B disputes and their resolutions |
| `toc.md` | Human-readable TOC with PDF pages **and** printed page labels, plus the review table |
| `review_queue.json` | Everything a human should look at |
| `<name>.bookmarked.pdf` | Copy of the PDF with the TOC as bookmarks (the original is never modified) |

### Node schema (in `contract.json` → `documents[]`, recursive `children`)

```jsonc
{
  "id": "D04.2.3",                 // document id + structural path
  "kind": "clause",                // document | heading | clause | recitals
  "structure": "4.2.3",            // synthetic position in the tree
  "level": 2,
  "number": "2.4",                 // clause number as printed (after verification)
  "title": "Retention",
  "heading_text": "2.4 Retention",
  "page_start": 7, "page_end": 7,  // physical PDF pages, always set by code
  "printed_label_start": "GCC-3",  // page label printed on the scan
  "starts_at_top": false,
  "text": "2.4 Retention: The Employer shall retain",   // verbatim own text, sub-clauses excluded
  "block_ids": ["p0007_b001"],     // provenance -> pages.json
  "annotations": [{"kind": "stamp_or_seal", "page": 7, "text": "Blue stamp: ABC Infra Ltd."}],
  "verification": "corrected_by_arbitration",
  "legibility": "clear",
  "flags": [],
  "summary": null, "key_obligations": [],
  "metadata": {},                  // documents: reference, date, stamp papers, blank/duplicate pages, printed labels ...
  "children": [ ... ]
}
```

`verification` values: `agreed` (two independent reads matched), `confirmed_by_arbitration`,
`corrected_by_arbitration`, `recovered` (missed by the page read, confirmed by crop re-read), `hunted` (found by a
numbering-gap hunt and confirmed by a second read), `template`, `disputed`, `unverified`.

## How it works

```
Stage 1  pages      Pass A  Flash reads each page image: verbatim blocks + page card     (escalation ladder)
                    Pass B  Flash-Lite, blind, different prompt: structural lines only
                    Code    align A/B headings and page labels
                    Arb.    full-width high-res crops re-read blindly for every disagreement
                    Code    blank pages (pixels), duplicate pages (shingles)
Stage 2  segment    reasoner splits the bundle using ~100-token page cards (windowed for huge bundles)
                    code enforces contiguous coverage; weak boundaries re-checked on two page images
Stage 3  hierarchy  reasoner classifies heading candidate IDs only (keep/level/number/title), chunked with context
                    stored templates of standard GCC editions are reused; numbering gaps -> image hunt + blind confirm
Stage 4  text       code joins blocks into nodes: cross-page paragraphs, stitched tables, recitals, annotations
Stage 5  checks     label continuity, cross-references, annexure series, contract-documents list, printed index pages,
                    optional LLM audit (suggestions go to review only)
```

Key rules:

* Page numbers come from code, never from the model.
* Reading prompts forbid correction and inference; hidden text becomes `[illegible]` and a review item.
* Verification reads are blind and use a different model or a crop, so errors are not simply repeated.
* Structuring calls can only reference IDs they were given, so they cannot invent clauses.
* Nothing is silently dropped: anything uncertain lands in the review queue.

Escalation ladder for Pass A: Flash at medium resolution → Flash high → Flash on two overlapping half-page crops →
Pro on half pages → structure-only mode (headings and clause openings, used when verbatim output is blocked, e.g.
RECITATION on standard-form text) → page marked failed. Truncation jumps straight to half pages; repetition loops
and empty output for inked pages count as failures.

All prompts are in `epc_parser/prompts.py`, each with its design rationale at the top of the file.

## Review codes

| Code | Meaning |
|---|---|
| `page_unreadable`, `text_unavailable` | Page could not be read verbatim; text for those nodes is incomplete (`text_incomplete` flag) |
| `obscured_heading`, `illegible_heading`, `obscured_page_label` | Stamp/signature/fading hides part of a heading or label |
| `heading_not_confirmed`, `marker_uncertain` | Reads disagree and arbitration could not settle it |
| `heading_inserted` | A missed heading was inserted; check the preceding paragraph for its body |
| `numbering_gap`, `gap_found_unconfirmed`, `gap_found_obscured` | Clause numbering skips a number |
| `boundary_unclear` | Could not confirm where a document starts |
| `duplicate_page`, `label_backwards`, `label_jump` | Scan order or completeness problems |
| `reference_not_found`, `attachment_series_gap`, `listed_document_missing`, `index_entry_not_found` | Something referenced or listed is not in the scan |
| `audit_possible_omission` | LLM audit suggestion (unverified) |
| `boundary_added`, `boundary_removed`, `gap_recovered`, `template_reused` (info) | Automatic repairs, for traceability |

## Cost and time

Rough expectations for a 400-page bundle with default settings (verify prices on Google's pricing page; model IDs and
prices live in `epc_parser/config.py`):

| Part | Approx. cost |
|---|---|
| Pass A verbatim reads (Flash, minimal thinking) | $1.8–2.6 |
| Pass B blind scans (Flash-Lite) | ~$0.2 |
| Arbitration, label checks | ~$0.1–0.2 |
| Segmentation + boundary checks | ~$0.15 |
| Hierarchy, gap hunts, audit, escalations | ~$0.4–0.8 |
| Top-level summaries | ~$0.05 |
| **Total** | **~$3–4.5**, about 10–20 min at concurrency 24 (rate limits permitting) |

Full verbatim text is the dominant cost because output tokens cost six times input tokens. Levers:

* `--summaries none` and `--no-audit` for bulk runs.
* Build up the template library (`.epc_templates/`): documents matching a stored edition skip the hierarchy call.
  Only entries from clean runs with at least 15 headings are saved; delete a template file to retire it.
* Text for non-urgent runs can be moved to the Gemini Batch API (50% off); the `Backend` protocol in `llm.py` is the
  place to add a batch backend.
* If Flash-Lite causes too many A/B disputes on your scans (watch `by_task.arbitration` in the cost report), set
  `--scanner-model gemini-3-flash-preview`.

## Tests

```bash
pip install pytest && python -m pytest -q
```

`tests/mock_backend.py` simulates two imperfect readers on a synthetic 11-page bundle: a misread clause number,
a truncated page read, a missed document boundary, a genuinely missing clause, a missing annexure and a duplicate
page. The end-to-end test asserts that each is repaired or flagged correctly, and that re-runs hit the cache.
`tests/test_gemini_backend.py` checks the real Gemini adapter against a stubbed client (no network).

## Before production use

* Gemini 3 models are preview models: re-check model IDs, prices and `media_resolution` / `thinking_level` behaviour.
* Build a regression set of 50–100 hand-annotated pages from real bundles (stamps over clause numbers, Hindi stamp
  paper, landscape BoQ, faded carbon copies) and measure dispute, escalation and review rates before changing models
  or prompts.
* The code has been tested offline with mock responses only; run a trial slice (`--pages 1-40`) on a real contract
  and inspect `pages.json` before processing full bundles.
