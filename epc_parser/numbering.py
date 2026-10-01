"""Deterministic text utilities. No reading happens here: these functions only compare what the LLM calls returned."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
ATTACHMENT_WORDS = ("annexure", "annex", "appendix", "schedule", "exhibit", "attachment", "form", "format", "enclosure", "part")
NAMED_MARKER_WORDS = ("article", "clause", "section", "chapter", "part", "sub-clause", "para", "paragraph", "rule") + ATTACHMENT_WORDS

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_QUOTES = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"'}


def clean(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).translate(_DASHES).translate(_QUOTES)
    return re.sub(r"\s+", " ", s).strip()


def norm_text(s: str | None) -> str:
    """Comparison form: case-folded, punctuation-light, whitespace-collapsed."""
    s = clean(s).casefold()
    s = re.sub(r"\[illegible\]", " ", s)
    s = re.sub(r"[^\w\s.()-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def similarity(a: str | None, b: str | None, prefix: bool = True) -> float:
    """prefix=True compares the common-length prefix (scan reads stop at the first full stop or colon)."""
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    if not prefix:
        return SequenceMatcher(None, a, b).ratio()
    n = min(len(a), len(b), 120)
    return SequenceMatcher(None, a[:n], b[:n]).ratio()


def roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(c not in ROMAN for c in s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        v = ROMAN[c]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total if 0 < total < 400 else None


def int_to_roman(n: int) -> str:
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
            (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, r in vals:
        while n >= v:
            out, n = out + r, n - v
    return out


def _fix_numeral_ocr(token: str) -> str:
    """Fix l/I/O/S confusions only inside tokens that are mostly digits and dots."""
    if not token or not re.search(r"\d", token):
        return token
    if len(re.sub(r"[\d.]", "", token)) > max(1, len(token) // 3):
        return token
    return token.translate(str.maketrans({"l": "1", "I": "1", "|": "1", "O": "0", "o": "0", "S": "5"}))


def normalize_marker(marker: str | None) -> str | None:
    """Canonical key for a clause marker.

    "14 .1" -> "14.1", "14.1." -> "14.1", "Clause 7" -> "7", "(a)" -> "(a)", "iv)" -> "(iv)",
    "Annexure - III" -> "annexure:3", "Schedule C" -> "schedule:c", "l4.1" -> "14.1".
    """
    m = clean(marker)
    if not m:
        return None
    low = m.casefold().strip(" :-–")
    for w in ATTACHMENT_WORDS:
        mm = re.match(rf"^{w}(?:ure|ures|s)?\b\s*[-.:]?\s*(?:no\.?\s*)?([a-z0-9]+)\b", low)
        if mm and mm.group(1) not in ("of", "for", "to"):
            kind = {"annex": "annexure", "format": "form"}.get(w, w)
            return f"{kind}:{attachment_index(mm.group(1))}"
    for w in ("article", "clause", "section", "chapter", "sub-clause", "para", "paragraph", "rule"):
        mm = re.match(rf"^{w}\s*[-.:]?\s*(?:no\.?\s*)?(.+)$", low)
        if mm:
            low = mm.group(1).strip()
            break
    low = re.sub(r"\s*\.\s*", ".", low).strip(". ")
    paren = re.fullmatch(r"\(?\s*([a-z]{1,4}|\d{1,3})\s*\)", low) or re.fullmatch(r"\(\s*([a-z]{1,4}|\d{1,3})\s*\)?", low)
    if paren:
        return f"({paren.group(1)})"
    if re.fullmatch(r"[a-z]", low):
        return f"({low})"
    fixed = _fix_numeral_ocr(re.sub(r"\s+", "", m.strip(" :-–.)")))
    if re.fullmatch(r"\d+(\.\d+)*", fixed):
        parts = fixed.split(".")
        if len(parts) > 1 and parts[-1] == "0":  # "7.0" is the same level as "7"
            parts = parts[:-1]
        return ".".join(str(int(p)) for p in parts)
    return low or None


def attachment_index(tok: str) -> str:
    tok = tok.lower()
    if tok.isdigit():
        return str(int(tok))
    r = roman_to_int(tok)
    if r is not None and (len(tok) > 1 or tok in ("i", "v", "x")):
        return str(r)
    return tok


def attachment_key(text: str | None) -> tuple[str, str] | None:
    """('annexure','3') for 'ANNEXURE-III: Format of BG', 'Annexure 3', 'annex III' ..."""
    low = clean(text).casefold()
    mm = re.search(r"\b(annexure|annex|appendix|schedule|exhibit|attachment|enclosure|form)\b\s*[-.:]?\s*(?:no\.?\s*)?([a-z0-9]{1,6})\b", low)
    if not mm:
        return None
    kind = "annexure" if mm.group(1) in ("annex",) else mm.group(1)
    idx = attachment_index(mm.group(2))
    if kind == "form" and idx in ("of", "for"):
        return None
    return kind, idx


def depth_hint(marker: str | None) -> str:
    key = normalize_marker(marker)
    if not key:
        return "-"
    if re.fullmatch(r"\d+(\.\d+)*", key):
        return f"dotted-{key.count('.') + 1}"
    if key.startswith("("):
        inner = key[1:-1]
        if inner.isdigit():
            return "paren-number"
        if roman_to_int(inner) is not None and (len(inner) > 1 or inner in ("i", "v", "x")):
            return "roman-or-alpha"
        return "alpha"
    if ":" in key:
        return key.split(":")[0]
    return "named"


LEADING_MARKER_RE = re.compile(
    r"^\s*((?:article|clause|section|chapter|part|annexure|appendix|schedule|form)\s*[-.:]?\s*[\w.]+"
    r"|\(?[a-zA-Z]{1,4}\)|\(?\d{1,3}\)|\d+(?:\s*\.\s*\d+)*\.?|[A-Z]\.)",
    re.IGNORECASE,
)


def extract_leading_marker(line: str | None) -> str | None:
    m = LEADING_MARKER_RE.match(clean(line))
    return m.group(1).strip() if m else None


def parse_label(label: str | None) -> tuple[str, int] | None:
    """('gcc', 14) for 'GCC-14', ('', 3) for 'Page 3 of 112', ('', 7) for '- 7 -', ('roman', 3) for 'iii'."""
    s = clean(label).casefold()
    if not s:
        return None
    mm = re.search(r"page\s*(\d+)", s)
    if mm:
        return "", int(mm.group(1))
    mm = re.fullmatch(r"[-–\s]*([ivxlc]+)[-–\s.]*", s)
    if mm and roman_to_int(mm.group(1)):
        return "roman", roman_to_int(mm.group(1))  # type: ignore[return-value]
    mm = re.fullmatch(r"([a-z][a-z/ ]*?)\s*[-/ ]\s*(\d+)", s)
    if mm:
        return mm.group(1).strip(), int(mm.group(2))
    mm = re.search(r"(\d+)", s)
    if mm:
        return re.sub(r"[\d\s\-–.of]+", "", s)[:10], int(mm.group(1))
    return None


def sibling_gaps(markers: list[str | None], max_gap: int) -> list[tuple[str, str, str]]:
    """Find missing markers in a sibling sequence. Returns (prev_marker, missing_marker, next_marker)."""
    gaps: list[tuple[str, str, str]] = []
    keys = [normalize_marker(m) for m in markers]
    for (pm, pk), (nm, nk) in zip(zip(markers, keys), list(zip(markers, keys))[1:]):
        if not pk or not nk:
            continue
        for missing in _between(pk, nk, max_gap):
            gaps.append((pm or pk, missing, nm or nk))
    return gaps


def _between(a: str, b: str, max_gap: int) -> list[str]:
    if re.fullmatch(r"\d+(\.\d+)*", a) and re.fullmatch(r"\d+(\.\d+)*", b):
        pa, pb = a.split("."), b.split(".")
        if len(pa) != len(pb) or pa[:-1] != pb[:-1]:
            return []
        x, y = int(pa[-1]), int(pb[-1])
        if 1 < y - x <= max_gap + 1:
            return [".".join(pa[:-1] + [str(i)]) for i in range(x + 1, y)]
        return []
    ma, mb = re.fullmatch(r"\(([a-z]+)\)", a), re.fullmatch(r"\(([a-z]+)\)", b)
    if ma and mb:
        ia, ib = ma.group(1), mb.group(1)
        ra, rb = roman_to_int(ia), roman_to_int(ib)
        if ra and rb and (len(ia) > 1 or len(ib) > 1):
            return [f"({int_to_roman(i)})" for i in range(ra + 1, rb)] if 1 < rb - ra <= max_gap + 1 else []
        if len(ia) == 1 and len(ib) == 1:
            x, y = ord(ia), ord(ib)
            return [f"({chr(i)})" for i in range(x + 1, y)] if 1 < y - x <= max_gap + 1 else []
    return []


def shingles(text: str, k: int = 5) -> set[str]:
    words = norm_text(text).split()
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def looks_degenerate(text: str, min_repeats: int = 6) -> bool:
    """Detect the looping failure mode (same line repeated many times)."""
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 8]
    run, prev = 1, None
    for l in lines:
        run = run + 1 if l == prev else 1
        if run >= min_repeats:
            return True
        prev = l
    return False
