"""Data models.

Two families:
  * *LLM schemas* (suffix ``LLM``): what the model must return. Kept flat and Gemini-schema friendly
    (no dicts, no unions other than Optional) so structured output works reliably.
  * *Internal / output models*: what the pipeline stores. Every piece of text keeps its provenance
    (page, block id, verification status) so any clause in the final JSON can be traced to pixels.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------------
DocType = Literal[
    "stamp_paper", "contract_agreement", "letter_of_intent", "letter_of_award", "notification_of_award",
    "correspondence", "tender_notice_or_itb", "general_conditions", "special_conditions",
    "technical_specification", "scope_of_work", "annexure", "appendix", "schedule", "form_or_format",
    "bank_guarantee", "power_of_attorney", "board_resolution", "integrity_pact", "minutes_of_meeting",
    "deviation_list", "amendment_or_corrigendum", "boq_or_price_schedule", "drawing",
    "certificate_or_affidavit", "index_or_contents", "separator_sheet", "blank", "other",
]

BlockKind = Literal[
    "heading", "clause", "paragraph", "recital", "table", "signature_block", "stamp_or_seal",
    "handwritten", "header", "footer", "page_number", "other",
]

Legibility = Literal["clear", "partial", "illegible"]

# Doc types whose content is not a clause hierarchy. They become a single node with full text.
FLAT_DOC_TYPES = {"stamp_paper", "boq_or_price_schedule", "drawing", "blank", "separator_sheet"}
# Doc types that only get a tree if they actually contain several structural headings.
TREE_IF_STRUCTURED = {"letter_of_intent", "letter_of_award", "notification_of_award", "correspondence",
                      "bank_guarantee", "power_of_attorney", "board_resolution", "certificate_or_affidavit",
                      "form_or_format", "index_or_contents", "other"}


# --------------------------------------------------------------------------------------------
# LLM schemas
# --------------------------------------------------------------------------------------------
class BlockLLM(BaseModel):
    kind: BlockKind
    number: Optional[str] = Field(None, description="Clause/item marker exactly as printed, e.g. '14.1', '(a)', 'Annexure-III'")
    title: Optional[str] = Field(None, description="Heading title words (for headings and inline-titled clauses)")
    inline_title: bool = False
    text: str = Field(description="Verbatim text of the block. Tables as a markdown table.")
    box: Optional[list[int]] = Field(None, description="[ymin, xmin, ymax, xmax] normalised 0-1000")
    legibility: Legibility = "clear"
    unreadable_note: Optional[str] = None


class StampPaperLLM(BaseModel):
    certificate_no: Optional[str] = None
    stamp_duty_amount: Optional[str] = None
    state: Optional[str] = None
    issue_date: Optional[str] = None
    purchased_by: Optional[str] = None
    first_party: Optional[str] = None
    second_party: Optional[str] = None
    article_or_description: Optional[str] = None


class PageReadLLM(BaseModel):
    page_type: DocType
    is_document_start: bool
    document_title: Optional[str] = None
    printed_page_label: Optional[str] = None
    running_header: Optional[str] = None
    running_footer: Optional[str] = None
    first_block_continues_previous_page: bool = False
    blocks: list[BlockLLM] = Field(default_factory=list)
    cross_references: list[str] = Field(default_factory=list)
    contract_documents_list: list[str] = Field(default_factory=list)
    stamp_paper: Optional[StampPaperLLM] = None
    languages: list[str] = Field(default_factory=list)
    overall_legibility: Legibility = "clear"


class HeadingLineLLM(BaseModel):
    marker: Optional[str] = None
    text: str
    obscured: bool = False
    box: Optional[list[int]] = None


class HeadingScanLLM(BaseModel):
    printed_page_label: Optional[str] = None
    is_blank: bool = False
    lines: list[HeadingLineLLM] = Field(default_factory=list)


class RegionLineLLM(BaseModel):
    text: str
    obscured: bool = False


class RegionReadLLM(BaseModel):
    lines: list[RegionLineLLM] = Field(default_factory=list)


class PageLabelLLM(BaseModel):
    printed_page_label: Optional[str] = None
    obscured: bool = False


class SegmentLLM(BaseModel):
    start_page: int
    end_page: int
    doc_type: DocType
    title: str
    reference: Optional[str] = None
    date: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


class SegmentationLLM(BaseModel):
    segments: list[SegmentLLM]


class BoundaryLLM(BaseModel):
    verdict: Literal["new_document", "continuation", "unclear"]
    new_document_title: Optional[str] = None
    new_document_type: Optional[DocType] = None
    evidence: str = ""


class TreeItemLLM(BaseModel):
    id: str
    k: bool = Field(description="keep: real structural heading/clause of this document")
    l: int = Field(description="level, 1 = top division of this document")
    n: Optional[str] = Field(None, description="normalised marker")
    t: Optional[str] = Field(None, description="short title")


class TreeLLM(BaseModel):
    items: list[TreeItemLLM]


class GapHuntLLM(BaseModel):
    found: bool
    pdf_page: Optional[int] = None
    first_line: Optional[str] = None
    obscured: bool = False
    evidence: str = ""


class AuditItemLLM(BaseModel):
    title: str
    pdf_page: Optional[int] = None
    reason: str


class AuditLLM(BaseModel):
    missing: list[AuditItemLLM] = Field(default_factory=list)


class SummaryLLM(BaseModel):
    summary: str
    key_obligations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Internal / output models
# --------------------------------------------------------------------------------------------
Verification = Literal[
    "agreed", "confirmed_by_arbitration", "corrected_by_arbitration", "recovered", "hunted",
    "disputed", "unverified", "template",
]


class Block(BaseModel):
    id: str
    page: int
    index: int
    kind: BlockKind
    number: Optional[str] = None
    title: Optional[str] = None
    inline_title: bool = False
    text: str = ""
    box: Optional[list[int]] = None
    legibility: Legibility = "clear"
    unreadable_note: Optional[str] = None
    verification: Verification = "unverified"
    source: Literal["read", "arbitration", "hunt"] = "read"


class Dispute(BaseModel):
    type: Literal["a_only", "b_only", "marker_mismatch", "label"]
    block_id: Optional[str] = None
    a_marker: Optional[str] = None
    a_text: Optional[str] = None
    b_marker: Optional[str] = None
    b_text: Optional[str] = None
    resolution: Optional[str] = None


class PageRecord(BaseModel):
    page: int
    status: Literal["ok", "blank", "structure_only", "failed"] = "ok"
    read_level: Optional[str] = None
    page_type: DocType = "other"
    is_document_start: bool = False
    document_title: Optional[str] = None
    printed_label: Optional[str] = None
    running_header: Optional[str] = None
    running_footer: Optional[str] = None
    first_continues: bool = False
    blocks: list[Block] = Field(default_factory=list)
    cross_references: list[str] = Field(default_factory=list)
    contract_documents_list: list[str] = Field(default_factory=list)
    stamp_paper: Optional[StampPaperLLM] = None
    languages: list[str] = Field(default_factory=list)
    legibility: Legibility = "clear"
    ink_ratio: float = 0.0
    duplicate_of: Optional[int] = None
    scan_verified: bool = False
    disputes: list[Dispute] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class Segment(BaseModel):
    id: str
    start_page: int
    end_page: int
    doc_type: DocType
    title: str
    reference: Optional[str] = None
    date: Optional[str] = None
    confidence: float = 0.0
    evidence: str = ""
    boundary_verified: Optional[bool] = None


class Annotation(BaseModel):
    kind: str
    page: int
    text: str


class Node(BaseModel):
    id: str
    kind: Literal["document", "heading", "clause", "preamble", "recitals"]
    structure: str
    level: int
    number: Optional[str] = None
    title: Optional[str] = None
    heading_text: Optional[str] = None
    doc_type: Optional[DocType] = None
    page_start: int
    page_end: int
    printed_label_start: Optional[str] = None
    starts_at_top: Optional[bool] = None
    text: str = ""
    block_ids: list[str] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    verification: Verification = "unverified"
    legibility: Legibility = "clear"
    flags: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    key_obligations: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    children: list["Node"] = Field(default_factory=list)


class ReviewItem(BaseModel):
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    pages: list[int] = Field(default_factory=list)
    node_id: Optional[str] = None


Node.model_rebuild()
