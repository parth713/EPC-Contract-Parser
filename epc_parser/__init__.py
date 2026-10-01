"""LLM-only structured extraction of scanned Indian EPC contract bundles."""
from .config import Settings
from .pipeline import parse_contract

__all__ = ["Settings", "parse_contract"]
__version__ = "0.1.0"
