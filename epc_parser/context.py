from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import Settings
from .llm import LLMClient
from .models import ReviewItem
from .render import PageRenderer

log = logging.getLogger("epc")


@dataclass
class RunContext:
    settings: Settings
    renderer: PageRenderer
    llm: LLMClient
    page_range: tuple[int, int]
    review: list[ReviewItem] = field(default_factory=list)

    def flag(self, severity: str, code: str, message: str, pages: list[int] | None = None, node_id: str | None = None) -> None:
        self.review.append(ReviewItem(severity=severity, code=code, message=message, pages=pages or [], node_id=node_id))
        getattr(log, "warning" if severity != "info" else "info")("[%s] %s %s", code, message, pages or "")

    @property
    def pages(self) -> range:
        return range(self.page_range[0], self.page_range[1] + 1)
