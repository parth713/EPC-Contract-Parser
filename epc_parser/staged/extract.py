"""Stage 3 - verbatim extraction + sub-clause derivation (single pass).

Each page is read ONCE, verbatim, into ordered blocks that each carry their own marker. Blocks are
assigned to level-1 units (main clause / section) by marker; then each clause's DIRECT sub-clauses
(X.Y) are derived DETERMINISTICALLY from the markers the SAME read already produced - no extra LLM
call. This is strictly higher fidelity than a separate scoped scan: the extract read is high-res, has
full-page context, and a block's marker is set only when it STARTS a heading (so a cross-reference in
body text can never be mistaken for a sub-clause). Deeper levels (X.Y.Z) fold into the child's text;
the clause keeps own_text (chapeau) and full_text (consolidated).
"""
from __future__ import annotations

import asyncio
import re
from typing import List, Optional

import logging
logger = logging.getLogger("epc.staged.extract")

from ..config import Settings as StagedConfig
from ..llm import LLMClient as StagedLLM
from .models import PageTextLLM, TextBlockLLM
from ..numbering import clean, normalize_marker
from ..render import PageRenderer

EXTRACT_SYSTEM = (
    "You transcribe ONE scanned page of an Indian EPC contract COMPLETELY and VERBATIM into ordered "
    "blocks. Structure (clause / sub-clause boundaries) is derived from your block markers, so both the "
    "text AND the markers must be exact.\n"
    "THE PRINTED PAGE IS THE ONLY SOURCE OF TRUTH:\n"
    "  - You may be given a short STRUCTURAL HINT about where this page sits (its division, whether it is "
    "section- or clause-organised, and which unit it falls within). The hint is ONLY to help you apply the "
    "right marking style - it is NOT the content of the page.\n"
    "  - Transcribe exactly what is physically printed on THIS page, in the order printed. NEVER add, "
    "invent, renumber, re-letter, relocate, complete or omit any marker, heading, clause, word or table to "
    "make the page agree with the hint.\n"
    "  - The hint NEVER tells you which numbers or headings to expect. Do NOT emit a marker or heading "
    "that is not actually printed on this page. If the page disagrees with the hint in any way (a "
    "different number, a different or absent section, a heading where none was hinted, or one where a hint "
    "was given but nothing is printed), OBEY THE PAGE and ignore the hint.\n"
    "COMPLETENESS - leave nothing out:\n"
    "  - Output every block of content on the page, in natural reading order, top to bottom.\n"
    "  - Reproduce the text EXACTLY as printed: do not summarise, paraphrase, correct, translate, "
    "re-order, reformat, or omit any word, sentence, list item, footnote or table cell.\n"
    "  - Include every clause, sub-clause, lettered/roman item, and continuation paragraph. If text "
    "wraps from the previous page, still transcribe what is on THIS page.\n"
    "BLOCKS & MARKERS - one heading per block:\n"
    "  - START A NEW BLOCK at every numbered or lettered heading (a clause, sub-clause, or numbered "
    "item that has its own marker). NEVER merge two differently-numbered items into one block.\n"
    "  - For every block that begins with such a marker, set 'marker' to that marker EXACTLY as printed "
    "- including sub-clauses and deeper levels (e.g. '5', '5.1', '5.2.1', 'ARTICLE 3', 'SECTION A', "
    "'(a)', '(iv)').\n"
    "  - GCC organised into titled SECTIONS: a section heading such as 'SECTION 6 - LABOUR' starts its "
    "OWN block with marker 'SECTION 6' - even when it is printed at the bottom of a page beneath the "
    "previous section's last clause or a signature block, never fold it into the block above. The "
    "numbered clauses beneath a section ('1.0', '2.0', ... which RESTART inside each section) each start "
    "their own block with their number as the marker.\n"
    "  - For a continuation paragraph or plain prose with no leading marker, set marker to null.\n"
    "  - A number that only appears INSIDE running text as a cross-reference (e.g. '... as per clause "
    "9.3 ...') is NOT a marker - it stays inside the block's text, marker null.\n"
    "PREAMBLE / RECITALS vs OPERATIVE CLAUSES - do NOT merge two number series:\n"
    "  - An introductory PREAMBLE or set of RECITALS may carry its OWN numbered points (1, 2, 3 ...), and "
    "those points can run across more than one page.\n"
    "  - The main OPERATIVE clauses that follow the preamble usually begin their OWN numbering again from "
    "1, so the SAME number can appear twice (once as a recital point, once as a clause).\n"
    "  - When that happens, keep the two DISTINCT: mark a preamble/recital numbered point kind='list_item' "
    "and an operative clause heading kind='clause'. Both keep their printed number verbatim; never merge, "
    "renumber, or relabel one into the other.\n"
    "  - RECITAL BOUNDARY (very common in Indian contracts): recitals usually open after 'WHEREAS' and END "
    "at a testimonium phrase such as 'NOW THIS AGREEMENT WITNESSETH', 'NOW IT IS HEREBY AGREED AS FOLLOWS', "
    "'NOW THEREFORE ... AS FOLLOWS', 'NOW THIS DEED WITNESSETH'. EVERY numbered point at or before that "
    "phrase is a recital -> kind='list_item'; the phrase itself is recital prose -> kind='paragraph'. Only "
    "the FIRST numbered heading printed AFTER that phrase is operative clause 1 -> kind='clause'. This holds "
    "even when the recitals (2, 3, 4 ...) continue at the TOP of the page and that operative '1' appears "
    "lower down the SAME page: still label the recitals list_item and only that operative heading clause, "
    "so it is never mislabelled as one more recital point.\n"
    "KIND: a heading that STARTS a top-level clause is kind='clause'; a titled section heading is "
    "kind='section'; a recital point or a numbered/lettered list item that lives inside a clause's prose "
    "or in the preamble is kind='list_item'; ordinary prose is kind='paragraph'.\n"
    "FORMATTING:\n"
    "  - Render any table as a GitHub-flavoured markdown table in the block's text (kind='table'), "
    "keeping every row and cell.\n"
    "  - Stamps/seals -> kind='stamp'; signature blocks -> kind='signature'. Transcribe their text as "
    "usual (keep the page verbatim), but always give them their OWN block with the right kind - never fold "
    "a signature or seal into an adjacent clause/paragraph block - so they can be told apart from clause "
    "text.\n"
    "  - A LETTER's CLOSING / sign-off area is likewise NOT clause content -> kind='signature': the "
    "courtesy close ('We trust you will find ...', 'Thanking you', 'Yours faithfully', 'Yours truly'), the "
    "signatory's name / designation / company, and the trailing office-address, contact, Tel/Fax/Email, "
    "CIN or website FOOTER block. Give each such block kind='signature' (transcribe verbatim as above) so "
    "it is separated from the last clause - the clause must keep ONLY its own substantive terms, never the "
    "sign-off or the letterhead/footer block that follows it.\n"
    "  - Ignore running headers, running footers and bare page numbers. A signature / initialling strip or "
    "seal that RECURS in the same page margin, header or footer across pages (e.g. a 'For <party> "
    "____' line or an initial box printed on every page) is running execution matter, NOT clause content "
    "- treat it like a running footer and leave it out. A genuine one-off execution/signature area at the "
    "end of a document or annexure still gets its own kind='signature'/'stamp' block as above.\n"
    "  - If part of the page is genuinely unreadable, write '[illegible]' in place of the missing words "
    "- never drop the surrounding text.\n"
    "CROSS-PAGE CONTINUATION:\n"
    "  - If the FIRST block on this page continues a sentence, clause or sub-clause from the previous "
    "page, set first_block_continues_previous_page = true and give that block marker = null (it is a "
    "continuation, not a new heading). Transcribe the continuing text in full; do not repeat text from "
    "the previous page.\n"
    "Do not add commentary or anything not printed on the page."
)
EXTRACT_USER = "Transcribe PDF page {page} of the contract verbatim into ordered blocks. Omit nothing."

# RECITATION fallback prompt. Gemini's RECITATION guard fires on long verbatim runs that match memorised
# boilerplate; interleaving a throwaway marker between words shatters every run so the match can't form,
# WITHOUT changing which real words are transcribed. The marker (`_SENTINEL`) is stripped back out of the
# blocks before they leave `_read_page_sentinel`.
#
# The marker MUST be plain ASCII: an exotic codepoint (a Private-Use-Area char) is NOT reliably emitted -
# Gemini substitutes a NUL (U+0000) or other control byte for it, which then survives an exact-match strip
# and can abort the DB write (Postgres rejects NUL). `[[BRK]]` is ASCII (emitted verbatim), can't occur in
# real contract prose, and is matched tolerantly (case / inner spacing) on the way out; `_strip_sentinel`
# ALSO drops any stray C0 control byte, so a mis-emitted marker can never leak downstream.
_SENTINEL = "[[BRK]]"
SENTINEL_ADDENDUM = (
    "\nOUTPUT SEPARATOR (MANDATORY FOR THIS READ ONLY): after roughly every 4-6 words of transcribed prose, "
    f"insert the literal ASCII marker {_SENTINEL} as a standalone separator between two words. It is a "
    "mechanical marker that is stripped out afterwards; it does NOT relax the verbatim rule - transcribe "
    "every real word EXACTLY as printed and only ADD the separator between existing words.\n"
    f"  - Emit the marker EXACTLY as the seven ASCII characters {_SENTINEL} - never a Unicode symbol, a "
    "control character, or any other substitute.\n"
    f"  - Place {_SENTINEL} ONLY at a space between two words. NEVER inside a word, number, date, decimal, "
    "clause reference, or a marker.\n"
    f"  - NEVER put {_SENTINEL} in a 'marker' value - markers stay exact. Insert it only inside 'text'.\n"
    f"  - Do NOT insert {_SENTINEL} inside a table block (kind='table') - leave table text clean.\n"
)
SENTINEL_USER = (
    f"IMPORTANT: interleave the literal ASCII marker {_SENTINEL} between words as instructed, roughly every "
    "4-6 words, so the output never reproduces a long verbatim run. Transcribe every real word exactly."
)
# Match the marker tolerantly: case-insensitive, and forgiving of stray inner whitespace the model may add.
_SENTINEL_RE = re.compile(r"\[\[\s*BRK\s*\]\]", re.IGNORECASE)
# Any C0 control byte except tab / newline / CR - a NUL or other control the model may substitute for the
# marker. Postgres rejects NUL outright and the rest are junk, so they never reach a block's text.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SPACE_RUN = re.compile(r"[ \t]{2,}")


def _strip_sentinel(text: str) -> str:
    """Remove the RECITATION-fallback marker and heal the spacing. A marker that sat BETWEEN words leaves a
    double space once dropped (``w [[BRK]] w`` -> ``w  w``) which collapses to one; a bare marker the model
    wrongly placed mid-word just rejoins the word. Any stray C0 control byte the model emitted instead of the
    ASCII marker (e.g. a NUL) is dropped too, so nothing invalid reaches the store. Newlines are preserved so
    markdown tables and line structure survive."""
    out = _SENTINEL_RE.sub("", text)
    out = _CONTROL_RE.sub("", out)          # drop any NUL / control byte substituted for the marker
    out = _SPACE_RUN.sub(" ", out)          # collapse the gap left where "w [[BRK]] w" became "w  w"
    out = re.sub(r"[ \t]+\n", "\n", out)    # tidy any space left before a newline
    return out.strip()


def _desentinel_page(pt: PageTextLLM) -> PageTextLLM:
    """Strip the sentinel out of every block's text (and, defensively, any marker) in place."""
    for b in pt.blocks:
        b.text = _strip_sentinel(b.text)
        if b.marker:
            b.marker = _strip_sentinel(b.marker) or None
    return pt


async def _read_page_sentinel(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, page: int,
                              context: Optional[str] = None):
    """RECITATION fallback: re-read the WHOLE page but ask the model to interleave a throwaway sentinel
    between words. Splintering every verbatim run defeats Gemini's memorised-text (RECITATION) block while
    keeping the high-quality vision transcription — so unlike the text-layer fallback this preserves markers,
    kinds and block structure. The sentinel is stripped back out before the blocks are returned."""
    img = await renderer.page(page, cfg.extract_dpi)
    user = f"{EXTRACT_USER.format(page=page)}\n\n{SENTINEL_USER}"
    if context:
        user = f"{user}\n\n{context}"
    res = await llm.call("extract:sentinel", role="reader", system=f"{EXTRACT_SYSTEM}{SENTINEL_ADDENDUM}",
                         parts=[img, user], schema=PageTextLLM, thinking="minimal",
                         media_resolution=cfg.extract_media, max_output_tokens=cfg.extract_max_out)
    if res.ok:
        return _desentinel_page(res.parsed), ("truncated" if res.truncated else None)
    return None, (res.error or res.finish_reason)


async def _read_page(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, page: int,
                     context: Optional[str] = None):
    img = await renderer.page(page, cfg.extract_dpi)
    user = EXTRACT_USER.format(page=page)
    if context:
        user = f"{user}\n\n{context}"
    res = await llm.call("extract", role="reader", system=EXTRACT_SYSTEM,
                         parts=[img, user], schema=PageTextLLM, thinking="minimal",
                         media_resolution=cfg.extract_media, max_output_tokens=cfg.extract_max_out,
                         recitation_retries=cfg.recitation_retries)
    if res.ok:
        return res.parsed, ("truncated" if res.truncated else None)
    return None, (res.error or res.finish_reason)


def _looks_overflow(reason: Optional[str]) -> bool:
    """A read that failed because its verbatim output exceeded the token cap. A whole-page re-read can
    never fit either, so recovery must split the page instead of retrying it unchanged."""
    return bool(reason) and "MAX_TOKENS" in reason


def _is_recitation(reason: Optional[str]) -> bool:
    """A read blocked by Gemini's RECITATION / IMAGE_RECITATION guard (near-verbatim reproduction of
    memorised boilerplate). Splitting the page can't clear it — the block is about the TEXT content, not
    the image size — so recovery goes straight to the PDF's own text layer for these."""
    return bool(reason) and "RECITATION" in reason


_MIN_TEXTLAYER_CHARS = 40  # below this a page has no usable embedded text (a pure scan) — don't use it


def _merge_halves(top: PageTextLLM, bottom: PageTextLLM) -> PageTextLLM:
    """Stitch a top-half read and a bottom-half read of the SAME page into one page result. The halves
    are rendered with a small vertical overlap so a line on the seam is never lost; a block the overlap
    made the model transcribe in BOTH halves is dropped once when the top's last block and the bottom's
    first block are identical, so the overlap can only duplicate, never delete."""
    blocks = list(top.blocks)
    tail = blocks[-1] if blocks else None
    for b in bottom.blocks:
        if tail is not None and b.kind == tail.kind and b.marker == tail.marker \
                and b.text.strip() == tail.text.strip():
            continue  # exact seam duplicate — keep the top copy only
        blocks.append(b)
        tail = b
    return PageTextLLM(first_block_continues_previous_page=top.first_block_continues_previous_page,
                       blocks=blocks)


async def _read_page_split(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, page: int,
                           context: Optional[str] = None):
    """Overflow fallback: read the page as two vertically-overlapping halves and merge the blocks. Each
    half's verbatim comfortably fits the output cap, so a page whose full text overflows a single read
    (a dense BOQ/spec table) is still transcribed completely instead of being lost."""
    halves = ((0, 0, 520, 1000), (480, 0, 1000, 1000))  # (ymin,xmin,ymax,xmax) 0-1000, ~4% seam overlap
    parts_out = []
    for i, box in enumerate(halves):
        img = await renderer.crop(page, box, cfg.extract_dpi)
        where = "TOP" if i == 0 else "BOTTOM"
        user = (f"{EXTRACT_USER.format(page=page)} You are shown ONLY the {where} portion of the page; "
                f"transcribe just what is visible in this portion.")
        if context:
            user = f"{user}\n\n{context}"
        res = await llm.call("extract:split", role="reader", system=EXTRACT_SYSTEM, parts=[img, user],
                             schema=PageTextLLM, thinking="minimal",
                             media_resolution=cfg.extract_media, max_output_tokens=cfg.extract_max_out)
        if not res.ok:
            return None, (res.error or res.finish_reason)
        parts_out.append(res.parsed)
    return _merge_halves(parts_out[0], parts_out[1]), None


def _textlayer_blocks(paras: List[str]) -> List[TextBlockLLM]:
    """Wrap the PDF's embedded text-layer paragraphs into PLAIN paragraph blocks — no markers. This is a
    last-resort content-preservation path for a page the vision read refused; it does NOT try to detect
    clause structure. Deriving markers from raw text-layer paragraphs is unsafe (a number could be a
    restarting point, a cross-reference or an address), so every paragraph is kept as marker=None text and
    simply lands, in order, in whatever clause owns the page. Not a full vision transcription."""
    return [TextBlockLLM(kind="paragraph", marker=None, text=para) for para in paras]


async def _read_textlayer(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, page: int,
                          context: Optional[str] = None):
    """Last-resort fallback: use the PDF's OWN embedded text layer for a page the vision read cannot
    transcribe. A RECITATION block can't be cleared by re-reading or splitting, but the exact text is
    sitting in the file, so pull it directly (no model call -> no recitation/safety block) instead of
    losing the page. Returns None when the page has no usable text layer (a pure scan)."""
    paras = await asyncio.to_thread(renderer.page_text_blocks, page)
    if sum(len(p) for p in paras) < _MIN_TEXTLAYER_CHARS:
        return None, "no_text_layer"
    return PageTextLLM(blocks=_textlayer_blocks(paras)), None


async def _recover_pages(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, failures: dict,
                         page_ctx: dict, pages_text: dict) -> None:
    """Second-pass recovery for pages the parallel read could not transcribe — driving `failed_pages`
    toward zero. Runs SEQUENTIALLY (no fan-out) so it never competes for rate-limit budget, and escalates
    per failure kind:

      1. a plain re-read — transient exhaustion and INVALID_JSON are non-deterministic and usually clear
         on a fresh attempt (failed calls are never cached, so this really re-hits the model);
      2. a half-page split read — for a page whose verbatim overflows the output cap (MAX_TOKENS) a
         whole-page read can NEVER fit, so the split is tried first for those (and as a fallback for a
         generic failure). A RECITATION page SKIPS the split — cropping can't clear a recitation block; it
         goes to `_read_page_sentinel` FIRST — a whole-page re-read with a throwaway sentinel interleaved
         between words, which splinters the memorised-text run so the RECITATION guard can't match, while
         keeping full vision structure (markers/kinds) that the text layer cannot;
      3. the PDF's embedded TEXT LAYER (`_read_textlayer`) — a last resort with no model call, so a
         RECITATION/SAFETY block can never fire. It rescues a page whose content is in the file even when
         the vision read refuses it; it no-ops on a pure scan (no text layer).

    A recovered page is written into `pages_text` and removed from `failures`; whatever stays in
    `failures` is a genuine unreadable page. Idempotent w.r.t. pages already read (skips them)."""
    if not failures:
        return
    logger.info(f"[staged.extract] recovery pass over {len(failures)} failed page(s): {sorted(failures)}")
    for p in sorted(failures):
        ctx = page_ctx.get(p)
        for _ in range(max(1, cfg.recovery_passes)):
            reason = failures.get(p)
            if _is_recitation(reason):
                # Sentinel re-read defeats the memorised-text match while keeping vision structure; a plain
                # re-read alone can't (the block is deterministic on the input). Split can't clear it either.
                strategies = [_read_page_sentinel, _read_page, _read_textlayer] if cfg.sentinel_recovery \
                    else [_read_page, _read_textlayer]
            elif _looks_overflow(reason):
                strategies = [_read_page_split, _read_page, _read_textlayer]
            else:
                strategies = [_read_page, _read_page_split, _read_textlayer]
            for strat in strategies:
                pt, problem = await strat(llm, renderer, cfg, p, ctx)
                if pt is not None:
                    pages_text[p] = pt
                    logger.info(f"[staged.extract] recovered page {p} via {strat.__name__} "
                                f"(was: {failures.get(p)})")
                    break
                failures[p] = problem  # remember the latest reason for the next strategy/pass
            if p in pages_text:
                break
    recovered = [p for p in list(failures) if p in pages_text]
    for p in recovered:
        failures.pop(p, None)
    logger.info(f"[staged.extract] recovery rescued {len(recovered)} page(s); "
                f"{len(failures)} still unreadable: {sorted(failures)}")


def _page_contexts(leaves: List[dict]) -> dict:
    """Coarse, ADVISORY per-page structural context, computed once from the static Stage-2 structure
    (never from a prior read, so page reads stay independent and parallel). It carries only where the
    page sits - division, organisation, containing unit, whether a heading may begin here - and NEVER an
    expected marker value or numeric sequence, so it cannot seed a hallucination. The read prompt treats
    it strictly as a hint and obeys the printed page."""
    ctx: dict = {}
    for d in leaves:
        dtype = clean(d.get("division_type") or "")
        title = clean(d.get("title") or "") or "this document"
        scheme = d.get("leaf_level")
        if scheme == "section":
            org = ("organised into titled SECTIONS; each section's own clause numbers RESTART at 1, and a "
                   "section heading can sit at the bottom of a page under the previous section's last clause")
        elif scheme == "clause":
            org = "organised as numbered MAIN CLAUSES, each optionally holding its own sub-clauses"
        else:
            org = "a single continuous document"
        units = d.get("units", [])
        for p in range(d["start_page"], d["end_page"] + 1):
            covering = [u for u in units if u["page_start"] <= p <= u["page_end"]]
            begins = any(u["page_start"] == p for u in units)
            here = next((u for u in covering if u.get("kind") not in ("preamble", "whole")), None)
            within = ""
            if here:
                lbl = " ".join(x for x in [clean(here.get("marker")), clean(here.get("title") or "")[:60]] if x)
                within = f" It falls within '{lbl}'." if lbl else ""
            begin_txt = (" A new heading may begin on this page." if begins else
                         " No new top-level heading is expected here (content continues within that unit); "
                         "still transcribe any heading that is actually printed.")
            # Transition page: the preamble (which may carry its OWN numbered recital points) and the
            # first main clause meet here, both numbering from 1 - warn the read to keep them separate.
            preamble_here = any(u.get("kind") == "preamble" and u["page_start"] <= p <= u["page_end"] for u in units)
            clause_begins = any(u.get("kind") not in ("preamble", "whole") and u["page_start"] == p for u in units)
            transition = (" The preamble/recitals and the main clauses meet around here: the preamble may have "
                          "its OWN numbered recital points (kind='list_item') and the operative clauses number "
                          "from 1 too (kind='clause') - keep the two series SEPARATE, do not merge them."
                          if (preamble_here and clause_begins) else "")
            dt = f" ({dtype})" if dtype else ""
            ctx[p] = (f"STRUCTURAL HINT (a location hint only, NOT authoritative - obey the printed page): "
                      f"this page is inside the division '{title}'{dt}, {org}.{within}{begin_txt}{transition} "
                      f"Mark every numbered or lettered clause and sub-clause exactly as it is printed.")
    return ctx


_NON_CLAUSE_KINDS = ("signature", "stamp")  # execution artefacts (signatures, stamps/seals) — not clauses


def _assemble(entries) -> str:
    """entries: (block, is_first_on_page, page_continues). Join blocks with paragraph breaks; weld a
    page-continuing first block onto the previous text (de-hyphenating). Each block's own marker (the
    clause number x / x.x / x.x.x, or a lettered item) is kept in front of its text, so the consolidated
    clause content carries the numbering. Continuation blocks have no marker, so welding is unaffected.

    Signature/stamp/seal blocks are DROPPED here: they are execution artefacts (party signatures,
    stamps, seals, the running signature/initial space repeated in a page's header/footer), not clause
    content - we want the clauses, not a gold-standard OCR of every seal. They remain in
    the raw document_pages record; this only keeps them out of clause own_text/full_text."""
    out = ""
    for k, (b, is_first, page_cont) in enumerate(entries):
        if b.kind in _NON_CLAUSE_KINDS:
            continue
        piece = b.text.strip()
        mk = clean(b.marker)
        if mk and b.kind not in ("table", "signature", "stamp") and not piece.lower().startswith(mk.lower()):
            piece = f"{mk} {piece}".strip()  # keep the clause number (x / x.x / x.x.x) in the consolidated text
        if not piece:
            continue
        if not out:
            out = piece
        elif k > 0 and is_first and page_cont and b.kind not in ("table", "signature", "stamp"):
            if out.endswith("-") and piece[:1].islower():
                out = out[:-1] + piece
            else:
                out = out + " " + piece
        else:
            out = out + "\n\n" + piece
    return out


def _heading_title(marker: Optional[str], text: str) -> Optional[str]:
    """Best-effort sub-clause title from its heading block: first line, marker stripped, cut at the
    first sentence boundary. Text integrity is unaffected - this is only the display title."""
    raw = (text or "").strip()
    if not raw:
        return None
    line = clean(raw.splitlines()[0])
    mk = clean(marker)
    if mk and line.lower().startswith(mk.lower()):
        line = line[len(mk):].lstrip(" .:-)")
    cut = re.split(r"(?<=[.:;])\s", line, 1)[0]
    title = (cut if len(cut) <= 90 else line[:90]).strip()
    return title[:200] or None


def _derive_children(unit: dict, entries: list) -> None:
    """Split a clause's blocks into its DIRECT sub-clauses (X.Y) by their markers. Blocks before the
    first sub-clause become own_text (chapeau); deeper markers (X.Y.Z) stay inside the child's text."""
    parent = normalize_marker(unit.get("marker"))
    if not parent or not re.fullmatch(r"\d+", parent):
        unit["children"] = []
        unit["own_text"] = unit.get("text", "")
        return
    own: list = []
    groups: list = []
    for (b, pg, isf, pc) in entries:
        key = normalize_marker(b.marker) or ""
        if re.fullmatch(rf"{parent}\.\d+", key):
            groups.append({"marker": clean(b.marker), "page": pg, "last": pg, "entries": []})
        if not groups:
            own.append((b, isf, pc))
        else:
            g = groups[-1]
            g["entries"].append((b, isf, pc))
            g["last"] = pg                    # track the real last page this sub-clause's text reaches
    kids: list = []
    for j, g in enumerate(groups):
        ps = min(max(unit["page_start"], g["page"]), unit["page_end"])
        # end at the last page the sub-clause's OWN blocks reach (covers text that runs onto the next
        # page), never past the clause; a shared page with the next sub-clause is fine.
        pe = min(max(g["last"], ps), unit["page_end"])
        text = _assemble(g["entries"])
        kids.append({"unit_id": f"{unit['unit_id']}.{j + 1}", "kind": "subclause", "marker": g["marker"],
                     "title": _heading_title(g["marker"], text), "page_start": ps, "page_end": pe,
                     "text": text, "flags": ([] if text.strip() else ["empty_text"])})
    unit["children"] = kids
    unit["own_text"] = _assemble(own)


def _derive_section_clauses(unit: dict, entries: list) -> None:
    """For a section-structured GCC, split a SECTION unit into its OWN numbered clauses (1.0, 2.0, ...,
    which restart inside each section) by their markers. Unlike _derive_children this matches BARE
    integers rather than a '{parent}.x' prefix, and skips the section's own heading block. Blocks before
    the first clause become own_text (chapeau); deeper markers (X.Y) stay inside the child's text."""
    own: list = []
    groups: list = []
    for (b, pg, isf, pc) in entries:
        key = normalize_marker(b.marker) or ""
        is_section_head = clean(b.marker).lower().startswith(("section", "part"))
        if re.fullmatch(r"\d+", key) and not is_section_head:  # a bare-integer clause of this section
            groups.append({"marker": clean(b.marker), "page": pg, "last": pg, "entries": []})
        if not groups:
            own.append((b, isf, pc))
        else:
            g = groups[-1]
            g["entries"].append((b, isf, pc))
            g["last"] = pg
    kids: list = []
    for j, g in enumerate(groups):
        ps = min(max(unit["page_start"], g["page"]), unit["page_end"])
        pe = min(max(g["last"], ps), unit["page_end"])
        text = _assemble(g["entries"])
        kids.append({"unit_id": f"{unit['unit_id']}.{j + 1}", "kind": "subclause", "marker": g["marker"],
                     "title": _heading_title(g["marker"], text), "page_start": ps, "page_end": pe,
                     "text": text, "flags": ([] if text.strip() else ["empty_text"])})
    unit["children"] = kids
    unit["own_text"] = _assemble(own)


def _section_num(marker: Optional[str]) -> Optional[str]:
    """The section/part number from a 'SECTION 6 - LABOUR' / 'PART II' marker, else None. Tolerant of a
    title glued onto the marker, which plain normalize_marker would choke on ('6 - labour' != '6')."""
    mm = re.match(r"\s*(?:section|part)\b\s*[-.:]?\s*(?:no\.?\s*)?([0-9]+|[ivxlcdm]+|[a-z])\b",
                  clean(marker), re.I)
    return normalize_marker(mm.group(1)) if mm else None


def _block_starts_unit(unit: dict, block: TextBlockLLM, allow_list_item: bool = False) -> bool:
    """Does this block BEGIN `unit`? Used only to split a page shared by the end of one unit and the
    start of the next (never as the primary assignment - that is page-range driven). Exact marker match,
    or a SECTION/PART heading matched by number, tolerant of a glued-on title or a heading that arrived
    as plain text with no marker."""
    m = unit.get("marker")
    if not m:
        return False
    # A unit begins at a HEADING, never at prose or a recital/preamble list-item that merely shares its
    # number. Rejecting list_item is what stops a preamble's own numbered point '1' from being mistaken
    # for the main clause '1' that restarts on the same page. But a DENSE enumeration page - a proforma's
    # numbered items, an annexure's list of submissions - packs many sibling level-1 units onto ONE page,
    # each printed as a numbered LIST item; there the caller sets allow_list_item so an item block can
    # start its own unit. The preamble->clause transition never sets it (see extract_division_text), so
    # the recital guard is unaffected.
    reject = ("paragraph", "table", "signature", "stamp")
    if not allow_list_item:
        reject += ("list_item",)
    if getattr(block, "kind", None) in reject:
        return False
    bm = normalize_marker(block.marker)
    if bm and bm == normalize_marker(m):
        return True
    us = _section_num(m)
    if us is None:
        return False
    return _section_num(block.marker) == us or _section_num((block.text or "")[:40]) == us


def _reclaim_leading_overflow(units: list, buckets: dict, pages_used: dict) -> None:
    """Move a unit's leading OVERFLOW back into the unit before it - for EVERY adjacent pair, not just
    preamble->clause.

    Every unit owns whole pages [page_start, page_end], but the anchor boundary is page-granular while real
    text is not: a preamble/clause/section routinely wraps PAST a page break - its last sentence, and any
    trailing sub-clauses (X.Y) or recital points, spill onto the TOP of the next unit's first page, ahead of
    that unit's own printed heading. Page-driven bucketing therefore lands that spill in the next unit (e.g.
    clause 2's text begins with clause 1's tail; clause 3's begins with clause 2's chapeau + 2.3/2.4). For
    each adjacent pair, find the block that actually carries the NEXT unit's own heading - via
    _block_starts_unit, so only a real heading (right marker, heading kind, section number tolerated) matches
    and a shared-number recital / list-item / cross-reference never does - and move everything before it back
    into the previous unit.

    Marker-driven and bounded: it never removes the next unit's heading or anything after it, so a unit can't
    be emptied; if the heading is never found (e.g. mis-transcribed), nothing moves and the boundary is left
    as-is for review. Verbatim-safe: it only reassigns existing blocks between units, never rewrites a word.
    Forward iteration chains correctly - a unit that just gave up its own tail can still reclaim the tail the
    unit after it is holding."""
    for idx in range(len(units) - 1):
        u, nxt = units[idx], units[idx + 1]
        if not nxt.get("marker"):
            continue  # next unit has no heading to anchor on (e.g. a preamble) -> nothing to reclaim
        nb = buckets[nxt["unit_id"]]
        k = next((j for j, (b, *_r) in enumerate(nb) if _block_starts_unit(nxt, b)), None)
        if not k:  # None (heading not found) or 0 (unit already starts at its heading) -> nothing to move
            continue
        buckets[u["unit_id"]].extend(nb[:k])
        pages_used[u["unit_id"]].update(p for (_b, p, *_r) in nb[:k])
        buckets[nxt["unit_id"]] = nb[k:]
        pages_used[nxt["unit_id"]] = {p for (_b, p, *_r) in nb[k:]}


def _strip_leading_front_matter(units: list, buckets: dict, pages_used: dict) -> None:
    """Drop a division's opening FRONT MATTER from its first clause.

    A division's opening page can carry matter printed ABOVE the first clause's heading - a letter's
    letterhead, reference/subject lines, addressee and 'Dear Sir'; an agreement's title and recitals.
    When the first clause begins on a LATER page, anchors emit a preamble unit and the recital guard +
    projection handle it. But when the first clause begins on the division's OWN first page (very common
    for letters), _build_units creates NO preamble unit (it only makes one when kept[0].pdf_page > ds), so
    page-driven bucketing lands that front matter inside the first clause. We want the
    clause, not the letterhead - 'we only need from clause 1' - so drop the blocks before the clause's own
    heading. They remain in the raw document_pages record; only the clause's own_text/full_text change.

    Safe by construction, exactly like _reclaim_leading_overflow: it strips ONLY when the clause's own
    heading block is found (list_item tolerated, since a letter's clause 1 is often a numbered prose point
    after non-numbered front matter), so it can never remove the heading or body; if the heading isn't
    found nothing is stripped and the front matter stays as a prefix (noise, not loss). No-op when the
    first unit is a preamble/whole (front matter already separated) or has no heading marker."""
    if not units:
        return
    first = units[0]
    if first.get("kind") in ("preamble", "whole") or not first.get("marker"):
        return
    fb = buckets[first["unit_id"]]
    k = next((j for j, (b, *_r) in enumerate(fb) if _block_starts_unit(first, b, allow_list_item=True)), None)
    if not k:  # None (heading not found) or 0 (already starts at its heading) -> nothing to strip
        return
    buckets[first["unit_id"]] = fb[k:]
    pages_used[first["unit_id"]] = {p for (_b, p, *_r) in fb[k:]}


# A dotted sub-clause is only rescued into its clause when its page is within this many pages of that
# clause's range — the pass exists for a body that spilled onto an ADJACENT page, not a same-numbered
# marker printed far away in the division.
_REDIST_PAGE_MARGIN = 2


def _redistribute_by_marker(div, units, buckets, pages_used) -> None:
    """Marker-driven CORRECTION of the page-driven bucketing — for CLAUSE-level divisions where a clause
    is SHORTER than a page.

    Page-driven assignment (extract_division_text) assumes each clause owns whole pages, which holds for
    long contracts where a clause spans several pages. In a dense/short document several clauses share a
    page and a clause's body routinely lands on a page OWNED by a LATER clause — e.g. clause 9's '9.1'..
    '9.3' print at the top of page 5, which clause 10 owns — so the page owner swallows the body and the
    real clause comes out empty. _reclaim_leading_overflow only recovers this when it can locate the NEXT
    unit's own heading; when that heading read is imperfect the body is lost (both directions: a clause's
    body stuck in the PREVIOUS unit on a shared page fails the same way).

    This pass moves a block whose OWN marker is a DOTTED sub-clause ('x.y') into the main clause 'x' it
    belongs to, when page-bucketing left it in a different clause. It is:
      - safe: routed ONLY on a DOTTED marker ('9.1', '12.8' -> clause 9 / 12), whose parent integer
        unambiguously names a main clause. A BARE integer marker is deliberately NOT routed — in a
        division whose sub-lists RESTART at 1 (bare '1'..'N' printed under each main clause, common in a
        safety guide / annexure), a bare 'N' is a sub-list item of the CURRENT clause, not main clause N,
        so routing it would merge every 'N-th' sub-item across the document into main clause N;
      - verbatim: it only relocates whole blocks, never rewrites a word;
      - scoped: clause-level divisions only (a section division's clause numbers RESTART per section, so
        even a dotted marker would be ambiguous), and it never pulls blocks OUT of a preamble/whole unit.
    A block with no dotted sub-clause marker inherits the current run's target, so a sub-clause's own
    heading line and its continuation paragraphs travel together. Complementary to
    _reclaim_leading_overflow: where that pass already put a block right, nothing moves here."""
    if div.get("leaf_level") != "clause":
        return
    # top-level integer -> clause unit_id, only for clauses whose marker IS a bare integer. Bail on any
    # ambiguity (a non-integer or a repeated integer) so the mapping is never wrong.
    int_to_uid: dict = {}
    for u in units:
        if u.get("kind") != "clause":
            continue
        m = re.fullmatch(r"\(?(\d+)\)?", normalize_marker(u.get("marker")) or "")
        if not m:
            return
        if m.group(1) in int_to_uid:
            return
        int_to_uid[m.group(1)] = u["unit_id"]
    if not int_to_uid:
        return

    uid_span = {u["unit_id"]: (u["page_start"], u["page_end"]) for u in units}

    def _target(block, page: int) -> Optional[str]:
        # DOTTED sub-clause markers ONLY ('9.1', '9.2.3', '12.8' -> main clause 9 / 12). The required dot
        # after the leading integer is what excludes a BARE integer (a restarting sub-list item like '12.
        # Motivate workmen') and a bare body number — neither of which should be pulled into main clause N.
        m = re.fullmatch(r"\(?(\d+)\)?\.\d+(?:\.\d+)*", normalize_marker(getattr(block, "marker", None)) or "")
        if not m:
            return None
        tgt = int_to_uid.get(m.group(1))
        if tgt is None:
            return None
        # LOCALIZED: only rescue a body that spilled onto a page ADJACENT to its clause (the case this
        # pass exists for). If the same number appears far away — a mis-printed '12.8' inside a distant
        # Welding clause, a cross-reference — its page is nowhere near clause 12's range, so leave it put
        # instead of merging it across the document.
        s, e = uid_span[tgt]
        return tgt if (s - _REDIST_PAGE_MARGIN) <= page <= (e + _REDIST_PAGE_MARGIN) else None

    new_buckets: dict = {u["unit_id"]: [] for u in units}
    moved = False
    for u in units:
        uid = u["unit_id"]
        if u.get("kind") != "clause":
            new_buckets[uid] = list(buckets[uid])   # leave preamble/whole exactly as they are
            continue
        cur = uid
        for entry in buckets[uid]:
            tgt = _target(entry[0], entry[1])
            if tgt is not None:
                cur = tgt
            new_buckets[cur].append(entry)
            if cur != uid:
                moved = True
    if not moved:
        return
    for u in units:
        uid = u["unit_id"]
        # Stable sort by page keeps each clause's blocks in physical reading order even when a body was
        # pulled back from a later page than the clause's own heading.
        new_buckets[uid].sort(key=lambda e: e[1])
        buckets[uid] = new_buckets[uid]
        pages_used[uid] = {p for (_b, p, *_r) in new_buckets[uid]}


def extract_division_text(div, pages_text) -> None:
    ds, de = div["start_page"], div["end_page"]
    units = div["units"]
    buckets = {u["unit_id"]: [] for u in units}
    pages_used = {u["unit_id"]: set() for u in units}
    i = 0
    for p in range(ds, de + 1):
        pt = pages_text.get(p)
        if not pt:
            continue
        # PAGE-DRIVEN assignment. Every unit owns pages [page_start, page_end] from stage 2 (contiguous,
        # reliable). Advance to the unit that owns page p purely by page number. Because it never depends
        # on an LLM marker, it can't stall on a missed heading or cascade - each page's blocks always land
        # in the unit whose range covers them, so no section can come out empty when its pages have text.
        while i + 1 < len(units) and p > units[i]["page_end"]:
            i += 1
        # A page that TWO OR MORE sibling units all START on is a dense enumeration (a proforma's numbered
        # items, an annexure's list of submissions) printed as a numbered LIST, not the usual one-heading-
        # per-clause page. There every unit shares the page (page_start == page_end == p), so the page-driven
        # advance above can't move between them and the within-page split is the ONLY separator - it must
        # accept a list_item block as a unit's start, or the first unit swallows the whole page and the rest
        # come out empty. Counting only NON-preamble starts keeps the preamble->clause transition strict
        # (its recitals are list_items too and must stay with the preamble).
        dense = sum(1 for u in units
                    if u["page_start"] == p and u["kind"] not in ("preamble", "whole")) >= 2
        for j, b in enumerate(pt.blocks):
            is_first = j == 0
            # ONLY when the next unit genuinely BEGINS on this same page (a shared boundary page) do we
            # split within the page, at the block that starts it. Off a shared page this never fires.
            while i + 1 < len(units) and units[i + 1]["page_start"] == p:
                # list_item blocks may start a unit only on a dense page AND only once we are past any
                # preamble - so a preamble recital point can never be read as the clause that restarts.
                allow_list = dense and units[i]["kind"] not in ("preamble", "whole")
                if not _block_starts_unit(units[i + 1], b, allow_list_item=allow_list):
                    break
                i += 1
            buckets[units[i]["unit_id"]].append((b, p, is_first, pt.first_block_continues_previous_page))
            pages_used[units[i]["unit_id"]].add(p)
    _reclaim_leading_overflow(units, buckets, pages_used)
    _redistribute_by_marker(div, units, buckets, pages_used)  # dense/short docs: body on a later clause's page
    _strip_leading_front_matter(units, buckets, pages_used)  # letterhead/recitals above a same-page clause 1
    for u in units:
        entries = buckets[u["unit_id"]]
        u["text"] = _assemble([(b, isf, pc) for (b, _pg, isf, pc) in entries])
        pu = sorted(pages_used[u["unit_id"]])
        u["text_pages"] = [pu[0], pu[-1]] if pu else []
        if u["kind"] == "clause" and div.get("leaf_level") == "section":
            _derive_section_clauses(u, entries)        # a section's own numbered clauses (1.0, 2.0, ...)
        elif u["kind"] == "clause":
            _derive_children(u, entries)               # sub-clauses (X.Y) from this read's markers
        else:
            u.setdefault("children", [])
            u["own_text"] = u["text"]
        if not u["text"].strip():
            u.setdefault("flags", []).append("empty_text")


async def run(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, leaves: List[dict],
              pages_out: Optional[dict] = None) -> List[dict]:
    pages_to_read = sorted({p for d in leaves for p in range(d["start_page"], d["end_page"] + 1)})
    page_ctx = _page_contexts(leaves)  # coarse advisory context per page, from the static structure
    _sample = next(iter(page_ctx.values()), "")
    logger.info(f"[staged.extract] per-page STRUCTURAL CONTEXT is ON — hints built for {len(page_ctx)} "
                f"pages | sample: {_sample[:200]}")
    review: List[dict] = []
    page_sem = asyncio.Semaphore(max(4, cfg.concurrency))
    pages_text: dict = {}
    failures: dict = {}  # page -> latest failure reason, for pages with no usable read yet
    done = [0]
    total = len(pages_to_read)

    async def one(p):
        async with page_sem:
            pt, problem = await _read_page(llm, renderer, cfg, p, page_ctx.get(p))
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == total:
            logger.info(f"[staged.extract] pages {done[0]}/{total} read")
        if pt is None:
            # surface WHY (finish_reason=MAX_TOKENS / RECITATION / SAFETY / BLOCKLIST / INVALID_JSON /
            # ERROR:... ) — the reason is otherwise lost (only page numbers survive). This is the FIRST
            # pass; recovery re-reads it next, so it is not yet a hard failure.
            logger.warning(f"[staged.extract] page {p} unreadable on first pass ({problem}) — will retry")
            failures[p] = problem
            return
        if problem == "truncated":
            review.append({"code": "page_truncated", "pages": [p],
                           "message": f"page {p} transcription hit output limit (text may be cut off)"})
        pages_text[p] = pt

    await asyncio.gather(*[one(p) for p in pages_to_read])

    # Recovery: re-read every page the parallel pass could not transcribe, so `failed_pages` -> 0 whenever
    # the failure was recoverable (non-deterministic block, or an overflow that only a split read can fit).
    await _recover_pages(llm, renderer, cfg, failures, page_ctx, pages_text)

    # Only pages still unread after recovery are reported as genuine failures. Log EACH one at error level
    # with its actual reason (RECITATION / SAFETY / BLOCKLIST / MAX_TOKENS / INVALID_JSON / ERROR:...), so a
    # persisted failed page is never a bare number — the cause is always in the logs.
    for p in sorted(failures):
        reason = failures[p]
        logger.error(f"[staged.extract] page {p} UNREADABLE after recovery — {reason}")
        review.append({"code": "page_unreadable", "pages": [p], "reason": reason,
                       "message": f"page {p} could not be transcribed ({reason})"})

    for div in leaves:
        extract_division_text(div, pages_text)
    if pages_out is not None:
        pages_out.update(pages_text)
    failed = total - len(pages_text)
    logger.info(f"[staged.extract] {len(pages_text)}/{total} pages read, {failed} failed")
    if failed:
        from collections import Counter
        breakdown = Counter((r.get("reason") or "?").split(":")[0].strip()
                            for r in review if r.get("code") == "page_unreadable")
        logger.warning(f"[staged.extract] unreadable-page reasons: {dict(breakdown)}")
    return review
