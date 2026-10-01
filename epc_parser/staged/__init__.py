"""Staged coarse-to-fine clause-extraction pipeline for epc-contract-parser.

macro -> refine -> anchors -> gap-hunt -> extract -> dates -> enrich, run in memory over a PDF. A
self-contained, in-memory pipeline (epc's own Settings / cached LLMClient / PageRenderer; no external
application coupling).

Entry point: `from epc_parser.staged import run` (async) or `python -m epc_parser.staged contract.pdf`.
"""
from .runner import run

__all__ = ["run"]
