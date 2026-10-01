"""Page rendering. Pure pixel work: nothing here reads text."""
from __future__ import annotations

import asyncio
import hashlib
import threading
from collections import OrderedDict
from pathlib import Path

import pymupdf

from .config import Settings

Box = tuple[int, int, int, int]  # ymin, xmin, ymax, xmax in 0-1000


class PageRenderer:
    """Thread-safe renderer (MuPDF documents are not thread-safe, so access is serialised)."""

    def __init__(self, pdf_path: Path, settings: Settings):
        self.path = Path(pdf_path)
        self.s = settings
        self._doc = pymupdf.open(self.path)
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._cache_max = 48

    @property
    def page_count(self) -> int:
        return self._doc.page_count

    def sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # ---- sync primitives ------------------------------------------------------------------
    def _pixmap(self, pno: int, dpi: int, box: Box | None = None) -> pymupdf.Pixmap:
        page = self._doc[pno - 1]
        rect = page.rect
        clip = None
        if box:
            y0, x0, y1, x1 = box
            clip = pymupdf.Rect(rect.x0 + rect.width * x0 / 1000, rect.y0 + rect.height * y0 / 1000,
                                rect.x0 + rect.width * x1 / 1000, rect.y0 + rect.height * y1 / 1000)
        zoom = dpi / 72
        target = clip or rect
        longest = max(target.width, target.height) * zoom
        if longest > self.s.max_image_side_px:
            zoom *= self.s.max_image_side_px / longest
        return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)

    def _jpeg(self, pno: int, dpi: int, box: Box | None = None) -> bytes:
        key = (pno, dpi, box)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            data = self._pixmap(pno, dpi, box).tobytes("jpeg", jpg_quality=self.s.jpeg_quality)
            self._cache[key] = data
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
            return data

    def _ink_ratio(self, pno: int) -> float:
        with self._lock:
            pix = self._doc[pno - 1].get_pixmap(matrix=pymupdf.Matrix(40 / 72, 40 / 72), colorspace=pymupdf.csGRAY, alpha=False)
        samples = pix.samples
        if not samples:
            return 0.0
        # Ignore a 4% border: scanner edges and punch holes are not content.
        w, h = pix.width, pix.height
        bx, by = max(1, w // 25), max(1, h // 25)
        dark = total = 0
        for y in range(by, h - by):
            row = samples[y * w + bx: y * w + w - bx]
            total += len(row)
            dark += sum(1 for v in row if v < 110)
        return dark / total if total else 0.0

    # ---- async API --------------------------------------------------------------------------
    async def page(self, pno: int, dpi: int | None = None) -> bytes:
        return await asyncio.to_thread(self._jpeg, pno, dpi or self.s.render_dpi, None)

    async def crop(self, pno: int, box: Box, dpi: int | None = None) -> bytes:
        y0, x0, y1, x1 = (max(0, min(1000, v)) for v in box)
        if y1 - y0 < 20:
            y0, y1 = max(0, y0 - 10), min(1000, y1 + 10)
        return await asyncio.to_thread(self._jpeg, pno, dpi or self.s.crop_dpi, (y0, x0, y1, x1))

    async def band(self, pno: int, y0: int, y1: int, pad: int = 25) -> bytes:
        """Full-width horizontal strip: clause markers sit at the left margin, so never crop them away."""
        return await self.crop(pno, (y0 - pad, 0, y1 + pad, 1000))

    async def halves(self, pno: int) -> tuple[bytes, bytes]:
        top = await self.crop(pno, (0, 0, 550, 1000), self.s.render_dpi)
        bottom = await self.crop(pno, (450, 0, 1000, 1000), self.s.render_dpi)
        return top, bottom

    async def ink_ratio(self, pno: int) -> float:
        return await asyncio.to_thread(self._ink_ratio, pno)

    def page_text_blocks(self, pno: int) -> list[str]:
        """The page's EMBEDDED text-layer paragraphs, in reading order (top-to-bottom, left-to-right).
        Empty when the PDF has no text layer for this page (a pure scan). Used only as a last-resort
        fallback when the vision read cannot transcribe the page (e.g. a RECITATION block), so the
        page's content is preserved from the file itself instead of being lost. MuPDF documents are not
        thread-safe, so this holds the same lock as rendering."""
        with self._lock:
            page = self._doc[pno - 1]
            # (x0, y0, x1, y1, text, block_no, block_type); block_type 0 = text, 1 = image.
            raw = page.get_text("blocks")
        out: list[str] = []
        for b in sorted(raw, key=lambda r: (round(r[1]), round(r[0]))):
            if len(b) >= 7 and b[6] != 0:
                continue  # skip image blocks
            txt = (b[4] or "").strip()
            if txt:
                out.append(txt)
        return out

    def close(self) -> None:
        self._doc.close()
