"""A fake Gemini backend driven by a synthetic contract spec.

It behaves like two imperfect readers so the verification machinery is exercised:
  * Pass A misreads clause "2.4" as "2.5" on page 7 (arbitration must correct it)
  * Pass A output for page 6 is truncated on full-page reads (the ladder must fall back to halves)
  * segmentation merges Annexure-III into Annexure-I with low confidence (boundary check must split it)
  * clause 2.2 is genuinely absent (gap hunt must report it, not invent it)
  * Annexure-II is referenced but absent, page 11 duplicates page 10
"""
from __future__ import annotations

import json
import re

import pymupdf

from epc_parser.llm import RawResponse
from epc_parser.numbering import depth_hint, norm_text

FILLER = ("The Contractor shall execute the Works in accordance with the Contract and shall provide all labour, "
          "materials, plant and equipment required for the completion of the Works within the Time for Completion. ")


def B(kind, text, number=None, title=None, inline=False):
    return {"kind": kind, "text": text, "number": number, "title": title, "inline_title": inline}


SPEC = {
    1: dict(type="stamp_paper", start=True, title="e-Stamp Certificate", label=None, cont=False, blocks=[
        B("heading", "INDIA NON JUDICIAL"),
        B("paragraph", "Certificate No. IN-UP12345678901234X  Stamp Duty Amount Rs. 500  Purchased by ABC Infra Ltd. " + FILLER),
        B("stamp_or_seal", "Stock Holding Corporation of India seal"),
    ], stamp={"certificate_no": "IN-UP12345678901234X", "stamp_duty_amount": "Rs. 500", "state": "Uttar Pradesh"}),
    2: dict(type="contract_agreement", start=True, title="CONTRACT AGREEMENT", label="1", cont=False, blocks=[
        B("heading", "CONTRACT AGREEMENT"),
        B("recital", "WHEREAS the Employer invited bids for the 2x660 MW EPC package. " + FILLER),
        B("recital", "AND WHEREAS the Contractor submitted its bid. " + FILLER),
        B("heading", "ARTICLE 1 - CONTRACT DOCUMENTS", number="Article 1", title="CONTRACT DOCUMENTS"),
        B("clause", "1.1 The following documents shall be deemed to form and be read and construed as part of this "
                    "Agreement: (a) Letter of Award; (b) General Conditions of Contract; (c) Annexure-I; (d) Annexure-II. "
                    "In case of any conflict the documents shall prevail in the order", number="1.1"),
    ], docs_list=["Letter of Award", "General Conditions of Contract", "Annexure-I", "Annexure-II"],
       refs=["Annexure-I", "Annexure-II"]),
    3: dict(type="contract_agreement", start=False, title=None, label="2", cont=True, blocks=[
        B("paragraph", "listed above, the earlier document prevailing over the later one. " + FILLER),
        B("heading", "ARTICLE 2 - CONTRACT PRICE", number="Article 2", title="CONTRACT PRICE"),
        B("clause", "2.1 The Contract Price is Rs. 1,000 crore inclusive of all taxes. " + FILLER, number="2.1"),
        B("signature_block", "For and on behalf of the Employer (signed)"),
    ]),
    4: dict(type="letter_of_award", start=True, title="LETTER OF AWARD", label=None, cont=False, blocks=[
        B("heading", "LETTER OF AWARD"),
        B("paragraph", "Ref. No. EMP/CS/2024/77 dated 12.03.2024. Subject: Award of EPC Contract. " + FILLER),
        B("clause", "1. Scope: design, engineering, supply, erection and commissioning. " + FILLER, number="1", title="Scope", inline=True),
        B("clause", "2. Price: as per Annexure-I. " + FILLER, number="2", title="Price", inline=True),
    ], refs=["Annexure-I"]),
    5: dict(type="general_conditions", start=True, title="GENERAL CONDITIONS OF CONTRACT", label="GCC-1", cont=False, blocks=[
        B("heading", "GENERAL CONDITIONS OF CONTRACT"),
        B("heading", "1. DEFINITIONS", number="1", title="DEFINITIONS"),
        B("clause", "1.1 'Contract' means the Contract Agreement and the documents listed in Article 1. " + FILLER, number="1.1"),
        B("clause", "1.2 'Works' means the permanent and temporary works. " + FILLER, number="1.2"),
    ]),
    6: dict(type="general_conditions", start=False, title=None, label="GCC-2", cont=False, truncate_full=True, blocks=[
        B("clause", "1.3 Headings shall not affect interpretation. " + FILLER, number="1.3"),
        B("heading", "2. PAYMENT", number="2", title="PAYMENT"),
        B("clause", "2.1 Advance: ten percent of the Contract Price against an advance bank guarantee. " + FILLER, number="2.1"),
        B("clause", "2.3 Running bills shall be paid within 30 days of certification. The milestone schedule is:", number="2.3"),
        B("table", "| Milestone | Share |\n|---|---|\n| Supply | 60% |\n| Erection | 30% |"),
    ]),
    7: dict(type="general_conditions", start=False, title=None, label="GCC-3", cont=True, blocks=[
        B("table", "| Milestone | Share |\n|---|---|\n| Commissioning | 10% |"),
        B("clause", "2.4 Retention: The Employer shall retain", number="2.4", title="Retention", inline=True),
        B("clause", "(a) five percent of each running bill; and", number="(a)"),
        B("clause", "(b) release the retention on issue of the Taking Over Certificate. " + FILLER, number="(b)"),
        B("stamp_or_seal", "Blue stamp: ABC Infra Ltd."),
    ], misread={"2.4": "2.5"}),
    8: dict(type="blank", start=False, title=None, label=None, cont=False, blocks=[]),
    9: dict(type="annexure", start=True, title="ANNEXURE-I", label=None, cont=False, blocks=[
        B("heading", "ANNEXURE-I", number="Annexure-I"),
        B("heading", "PRICE SCHEDULE SUMMARY"),
        B("table", "| Item | Amount |\n|---|---|\n| Supply | Rs. 600 crore |\n| Services | Rs. 400 crore |"),
        B("paragraph", FILLER * 2),
    ]),
    10: dict(type="annexure", start=True, title="ANNEXURE-III", label=None, cont=False, blocks=[
        B("heading", "ANNEXURE-III", number="Annexure-III"),
        B("heading", "FORMAT OF PERFORMANCE BANK GUARANTEE"),
        B("paragraph", "We, the Bank, hereby irrevocably undertake to pay the Employer on first demand. " + FILLER * 3),
    ]),
}
SPEC[11] = dict(SPEC[10], start=False)  # rescanned duplicate of page 10

STRUCTURAL = {"heading", "clause"}


def make_pdf(path) -> None:
    doc = pymupdf.open()
    for p in sorted(SPEC):
        page = doc.new_page(width=595, height=842)
        y = 60
        for b in SPEC[p]["blocks"]:
            for chunk in re.findall(r".{1,85}(?:\s|$)", b["text"].replace("\n", " ")):
                page.insert_text((50, y), chunk, fontsize=10)
                y += 13
            y += 8
        if SPEC[p]["label"]:
            page.insert_text((280, 815), SPEC[p]["label"], fontsize=9)
        if p == 11:  # a rescan is never byte-identical to the original page
            page.insert_text((450, 30), "scan 2", fontsize=7)
    doc.save(path)


def _boxes(blocks):
    n = max(1, len(blocks))
    return [[60 + i * (880 // n), 40, 60 + (i + 1) * (880 // n) - 10, 960] for i in range(len(blocks))]


def _page_read(p, blocks_filter=None, box_map=None, misread=True):
    s = SPEC[p]
    out_blocks = []
    for b, box in zip(s["blocks"], _boxes(s["blocks"])):
        if blocks_filter and not blocks_filter(box):
            continue
        bb = dict(b, box=box_map(box) if box_map else box)
        if misread and b["number"] in s.get("misread", {}):
            wrong = s["misread"][b["number"]]
            bb["number"] = wrong
            bb["text"] = b["text"].replace(b["number"], wrong, 1)
        out_blocks.append(bb)
    return {"page_type": s["type"], "is_document_start": s["start"], "document_title": s["title"] if s["start"] else None,
            "printed_page_label": s["label"], "first_block_continues_previous_page": s["cont"], "blocks": out_blocks,
            "cross_references": s.get("refs", []), "contract_documents_list": s.get("docs_list", []),
            "stamp_paper": s.get("stamp"), "languages": ["en"], "overall_legibility": "clear"}


def _scan(p):
    s = SPEC[p]
    lines = []
    for b, box in zip(s["blocks"], _boxes(s["blocks"])):
        if b["kind"] in STRUCTURAL:
            lines.append({"marker": b["number"], "text": re.split(r"(?<=[a-z])\. ", b["text"], maxsplit=1)[0][:120], "box": box})
    return {"printed_page_label": s["label"], "is_blank": not s["blocks"], "lines": lines}


class MockBackend:
    def __init__(self):
        self.calls: dict[str, int] = {}

    async def generate(self, *, model, system, parts, schema, thinking, media_resolution, max_output_tokens, meta,
                        temperature=None):
        task = meta["task"].split(":")[0]
        self.calls[task] = self.calls.get(task, 0) + 1
        text = next((x for x in reversed(parts) if isinstance(x, str)), "")
        p = meta.get("page")
        finish = "STOP"

        if task == "page_read":
            if SPEC[p].get("truncate_full"):
                return RawResponse('{"page_type": "general_conditions", "blocks": [', "MAX_TOKENS", 1000, 32768)
            data = _page_read(p)
        elif task == "page_read_half":
            if meta["mode"] == "top":
                data = _page_read(p, lambda b: b[0] < 500, lambda b: [int(b[0] / 0.55), b[1], int(min(1000, b[2] / 0.55)), b[3]])
            else:
                data = _page_read(p, lambda b: b[0] >= 500, lambda b: [int((b[0] - 450) / 0.55), b[1], int((b[2] - 450) / 0.55), b[3]])
        elif task in ("heading_scan", "gap_confirm"):
            data = _scan(p)
        elif task == "label_arbitration":
            data = {"printed_page_label": SPEC[p]["label"], "obscured": False}
        elif task == "arbitration":
            y0, y1 = meta["box"][0], meta["box"][2]
            lines = [{"text": b["text"][:120], "obscured": False}
                     for b, box in zip(SPEC[p]["blocks"], _boxes(SPEC[p]["blocks"])) if box[0] <= y1 and box[2] >= y0]
            data = {"lines": lines}
        elif task == "segmentation":
            segs = []
            starts = [q for q in sorted(SPEC) if SPEC[q]["start"] and q != 10]  # deliberately misses page 10
            for i, q in enumerate(starts):
                end = starts[i + 1] - 1 if i + 1 < len(starts) else max(SPEC)
                segs.append({"start_page": q, "end_page": end, "doc_type": SPEC[q]["type"], "title": SPEC[q]["title"],
                             "confidence": 0.95, "evidence": "title block"})
            data = {"segments": segs}
        elif task == "boundary_check":
            b = meta["b"]
            is_new = SPEC[b]["start"] and not (b == 11)
            data = {"verdict": "new_document" if is_new else "continuation",
                    "new_document_title": SPEC[b]["title"] if is_new else None,
                    "new_document_type": SPEC[b]["type"] if is_new else None, "evidence": "mock"}
        elif task == "hierarchy":
            doc_title = re.search(r"Document title: (.*)", text).group(1)
            ids = [x.strip() for x in text.rsplit("Return items for these ids only:", 1)[1].split(",")]
            items = []
            prev_level = 1
            for line in text.splitlines():
                cols = [c.strip() for c in line.split("|")]
                if len(cols) < 6 or cols[0] not in ids:
                    continue
                marker = None if cols[3] == "-" else cols[3]
                hint = depth_hint(marker)
                if norm_text(cols[4]).startswith(norm_text(doc_title)) and len(norm_text(cols[4])) <= len(norm_text(doc_title)) + 2:
                    items.append({"id": cols[0], "k": False, "l": 1})
                    continue
                if hint.startswith("dotted-"):
                    level = int(hint.split("-")[1])
                elif hint in ("alpha", "roman-or-alpha"):
                    level = prev_level + 1
                else:
                    level = 1
                items.append({"id": cols[0], "k": True, "l": level, "n": marker})
                prev_level = level if not hint.startswith("alpha") else prev_level
            data = {"items": items}
        elif task == "gap_hunt":
            data = {"found": False, "evidence": "numbering skips it"}
        elif task == "final_audit":
            data = {"missing": []}
        elif task == "summary":
            data = {"summary": "Mock summary of the section.", "key_obligations": []}
        else:
            raise AssertionError(f"unexpected task {task}")
        return RawResponse(json.dumps(data), finish, 1500, 400, 0)
