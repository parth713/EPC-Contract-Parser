"""Runtime settings. Everything tunable lives here so the pipeline code stays free of magic numbers."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelPrice:
    input_per_m: float
    output_per_m: float  # thinking tokens are billed as output


# Prices in USD per 1M tokens. Verify against Google's pricing page before relying on cost reports.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "gemini-3-flash-preview": ModelPrice(0.50, 3.00),
    "gemini-3.1-flash-lite": ModelPrice(0.25, 1.50),
    "gemini-3.1-pro-preview": ModelPrice(2.00, 12.00),
    "gemini-3.5-flash": ModelPrice(2.00, 12.00)
}


@dataclass
class Settings:
    # --- credentials -------------------------------------------------------------------------
    api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    use_vertex: bool = field(default_factory=lambda: os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"))

    # --- model roles -------------------------------------------------------------------------
    # reader:    Pass A full page transcription + page card (the expensive, important call)
    # scanner:   Pass B blind heading read. A *different* model decorrelates errors from Pass A.
    # reasoner:  segmentation, boundary checks, hierarchy, gap hunts, audit
    # escalation: last rung of the ladder for pages the cheaper models cannot read
    model_reader: str = "gemini-3-flash-preview"
    model_scanner: str = "gemini-3-flash-preview"
    model_reasoner: str = "gemini-3-flash-preview"
    model_escalation: str = "gemini-3-flash-preview"
    model_summarizer: str = "gemini-3-flash-preview"
    prices: dict[str, ModelPrice] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    use_response_schema: bool = True  # structured output; falls back to plain JSON mode automatically
    temperature: float = 0.0  # deterministic by default: extraction/classification, not creative generation

    # --- rendering ---------------------------------------------------------------------------
    render_dpi: int = 200          # page images sent to the model (token cost is set by media_resolution, not DPI)
    crop_dpi: int = 300            # arbitration crops: more pixels where it matters
    jpeg_quality: int = 85
    max_image_side_px: int = 3000
    blank_ink_threshold: float = 0.0003  # dark-pixel fraction below which a page is blank (no LLM call); keep low so a lone "ANNEXURE-IV" separator is still read

    # --- execution ---------------------------------------------------------------------------
    concurrency: int = 24
    max_retries: int = 5
    request_timeout_s: float = 180.0
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("EPC_CACHE_DIR", ".epc_cache")))
    template_dir: Path = field(default_factory=lambda: Path(os.getenv("EPC_TEMPLATE_DIR", ".epc_templates")))

    # --- verification & repair ---------------------------------------------------------------
    arbitration: bool = True
    max_disputes_per_page: int = 10
    heading_match_threshold: float = 0.75
    boundary_confidence_threshold: float = 0.9
    max_boundary_checks: int = 120
    gap_hunts: bool = True
    hunt_max_pages: int = 8
    max_gap_size: int = 4          # 7.2 -> 7.7 is probably a real numbering jump, not four missed clauses
    duplicate_jaccard: float = 0.9
    final_audit: bool = True

    # --- structure ---------------------------------------------------------------------------
    max_tree_depth: int = 5
    tree_chunk_size: int = 300     # heading candidates per hierarchy call
    tree_context_items: int = 25   # already-decided items shown as context to the next chunk
    segmentation_window: int = 220 # pages per segmentation call (with overlap) for very large bundles
    segmentation_overlap: int = 20
    template_min_coverage: float = 0.9

    # --- output extras -----------------------------------------------------------------------
    summaries: str = "top"         # none | top | all
    write_bookmarked_pdf: bool = True

    # --- staged pipeline (coarse-to-fine: macro -> refine -> anchors -> gaphunt -> extract -----
    #     -> enrich -> dates). These are only
    #     read by `epc_parser.staged`; the flagship pipeline (pipeline.py) ignores them. --------
    seed: int = 42
    retry_base: float = 1.0        # backoff base seconds: delay in [0, base*2^(attempt-1)], capped 60s
    # macro sweep (coarse division boundaries)
    macro_dpi: int = 150
    macro_media: str = "medium"
    macro_window: int = 15
    macro_stride: int = 14
    # title / type refine
    refine_media: str = "high"
    # leaf-unit anchors (main clause + sub-clause)
    anchor_dpi: int = 200
    anchor_media: str = "medium"
    anchor_window: int = 4
    anchor_stride: int = 3
    min_sections: int = 2
    # numbering-gap hunt (staged names; flagship uses hunt_max_pages / max_gap_size)
    hunt_dpi: int = 300
    hunt_media: str = "high"
    max_hunt_pages: int = 6
    max_gap: int = 8
    # verbatim extract
    extract_dpi: int = 300
    extract_media: str = "high"
    extract_max_out: int = 65536   # headroom so a dense page's full verbatim (big BOQ/spec tables) isn't truncated
    recitation_retries: int = 2    # RECITATION blocks on boilerplate are non-deterministic — retry perturbed
    sentinel_recovery: bool = True # RECITATION fallback: re-read with a throwaway sentinel interleaved between words
    recovery_passes: int = 2       # sequential re-read attempts for pages the parallel read left unreadable
    # enrich (summary / title / priority / risk / type)
    enrich_enabled: bool = True
    enrich_batch: int = 20
    # contract-level named-date registry + principal parties (post-extract)
    dates_enabled: bool = True
    dates_judge_enabled: bool = True   # final LLM curation pass over grounded date candidates (keep/drop/fix)
    dates_max_chars: int = 90000       # cap on body text fed to the text date read (bounds cost)
    dates_near: int = 200              # proximity (chars) between anchor-term and date-literal to select a page
    dates_cover_max_pages: int = 10    # max pages per cover division rendered as images for the date read

    def price_for(self, model: str) -> ModelPrice:
        return self.prices.get(model, ModelPrice(0.0, 0.0))

    def model_for(self, role: str) -> str:
        """Role -> model id. The staged pipeline uses only reader/reasoner."""
        return {"reader": self.model_reader, "scanner": self.model_scanner,
                "reasoner": self.model_reasoner, "escalation": self.model_escalation,
                "summarizer": self.model_summarizer}.get(role, self.model_reader)
