"""Pydantic response schemas for the staged pipeline's Gemini calls. Kept flat/enum-based so
structured output stays reliable."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

DivisionType = Literal[
    "RECITALS", "CONTRACT_AGREEMENT", "LETTER_OF_AWARD", "LETTER_OF_INTENT", "PRE_BID_MINUTES",
    "TENDER_NOTICE", "INSTRUCTIONS_TO_TENDERERS", "FORM_OF_TENDER",
    "GCC", "SCC", "TECHNICAL_SPECS", "SCOPE_OF_WORK", "SCHEDULE", "ANNEXURE", "APPENDIX", "BOQ",
    "DRAWINGS", "BANK_GUARANTEE", "PROFORMA_OR_FORM", "INDEX_OR_TOC", "OTHER_DOCUMENT",
]
Position = Literal["TOP", "MIDDLE", "BOTTOM", "FULL_PAGE"]


# ---- macro sweep -------------------------------------------------------------------------------
class BoundaryItem(BaseModel):
    pdf_page_number: int = Field(description="1-indexed physical PDF page where this division STARTS.")
    division_type: DivisionType
    exact_title_text: str = Field(description="Verbatim title/headline on the page.")
    position_on_page: Position
    is_standalone_cover_sheet: bool
    running_header: Optional[str] = Field(None, description="Running header/footer identifying the document/volume.")


class BatchDivisionResponse(BaseModel):
    boundaries_detected: List[BoundaryItem] = Field(default_factory=list)
    contains_master_toc: bool = False
    master_toc_pages: List[int] = Field(default_factory=list)
    running_header_text_observed: Optional[str] = None


# ---- title refine ------------------------------------------------------------------------------
class DivisionTitleLLM(BaseModel):
    division_type: DivisionType
    title: str = Field(description="The document/section's OWN top-level title, cleaned and logically formatted in Title Case (keep genuine acronyms/identifiers as printed) - correct obvious typesetting defects (missing spaces, run-together words); do NOT copy the source's mistakes verbatim, and do NOT glue on the first sub-heading. NOT the running header or volume name.")
    running_header: Optional[str] = None
    is_cover_sheet: bool = False
    numbering_scheme: Literal["section", "clause", "flat"] = Field(
        "clause",
        description="How this division is structured at its TOP level. 'section': organised into titled "
        "SECTION/PART headings (e.g. 'SECTION 1 - DEFINITIONS', 'SECTION 2 - INSURANCE', 'PART II') - the "
        "sections are the units and clause numbers usually RESTART inside each one. 'clause': the top "
        "level is plain numbered clauses ('1. DEFINITIONS', '2. CONDITIONS PRECEDENT') with NO section "
        "headings. 'flat': no internal numbering at all (a form, cover sheet, stamp paper).")


# ---- anchors -----------------------------------------------------------------------------------
class AnchorItem(BaseModel):
    pdf_page: int = Field(description="1-indexed physical PDF page where this unit BEGINS.")
    kind: Literal["section", "clause"]
    marker: Optional[str] = Field(None, description="Marker exactly as printed, e.g. 'SECTION A', '4', 'Clause 5'.")
    title: str


class WindowAnchors(BaseModel):
    anchors: List[AnchorItem] = Field(default_factory=list)


# ---- gap hunt ----------------------------------------------------------------------------------
class GapHuntLLM(BaseModel):
    found: bool
    pdf_page: Optional[int] = None
    marker: Optional[str] = None
    title: Optional[str] = None


# ---- verbatim extract --------------------------------------------------------------------------
class TextBlockLLM(BaseModel):
    kind: Literal["heading", "clause", "section", "paragraph", "list_item", "table", "signature", "stamp", "other"]
    marker: Optional[str] = Field(None, description="Marker this block starts (e.g. '5', '5.2', 'SECTION A'), else null.")
    text: str = Field(description="Verbatim text; tables as markdown.")


class PageTextLLM(BaseModel):
    first_block_continues_previous_page: bool = False
    blocks: List[TextBlockLLM] = Field(default_factory=list)


# ---- enrich (summary / priority / risk / type) -------------------------------------------------
class EnrichItem(BaseModel):
    index: int = Field(description="0-based index of the unit within the batch, echoed back.")
    clause_description: str = Field(description="One-sentence summary (max ~30 words) of what the clause governs.")
    clause_title: Optional[str] = Field(
        None,
        description="A SHORT title (3-6 words, Title Case) naming what this clause is about, derived from "
        "its content. REQUIRED for any clause shown to you WITHOUT a title of its own (marker-only or blank) "
        "- always generate a real name, never leave it empty or echo the marker. ONLY when the clause "
        "already has a title of its own, return null - do not restate or rephrase it.")
    priority: Literal["High", "Medium", "Low"] = "Medium"
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    clause_type: str = "General"


class EnrichResponse(BaseModel):
    items: List[EnrichItem] = Field(default_factory=list)
