"""Date Anchor Engine — a SELF-CONTAINED stage that builds the contract-level named-date registry a
downstream due-date engine anchors obligation offsets against ("60 days from the Letter of Intent date").

Runs AFTER `extract`, so it reads already-OCR'd, recitation-recovered text and reuses the same page
renderer. It is deliberately INDEPENDENT of TOC/structure generation: its own prompts, its own schemas,
its own LLM calls. It NEVER touches the macro/anchors/refine/extract/enrich prompts, so it can never add
responsibility to — or destabilise — the working structure pipeline. Any failure returns an empty
registry; it can never block or fail clause parsing.

Pipeline (see the module's __doc__ sections):
  D0 select      - deterministic. Cover divisions (LOA/LOI/agreement/recitals) + definitions clauses +
                   a proximity sweep (anchor-term ↔ date-literal close) over every page's text.
  D1 read        - LLM. Cover pages read as IMAGES (handwritten/stamped dates); body candidates as TEXT.
                   Every date returned WITH its verbatim source snippet (verbatim-grounded, no hallucination).
  D2 normalise   - deterministic. Validate/cross-check each iso against a re-parse of its snippet; a value
                   the code cannot confirm is downgraded or dropped to null (fail-closed: null beats a guess).
  D3 merge       - deterministic. Group by date_type; detect CONFLICTS (same term, different values); apply
                   division precedence (SCC/agreement/contract-data > GCC).
  D4 resolve     - LLM, only on a real conflict. Hand the competing snippets/pages to a focused resolver
                   ("which value governs, or null") — the "confusion pages" decision.
  D5 assemble    - deterministic. registry {name: iso|None} (every name-variant of a resolved type mapped
                   to its value, to maximise the consumer's match rate) + a rich audit trail.

Output (the contract-level named-date registry):
    {"registry": {name: "YYYY-MM-DD" | null, ...}, "anchors": [ {..full record..} ], "conflicts": [ ... ]}
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import logging
logger = logging.getLogger("epc.staged.dates")
from pydantic import BaseModel, Field

from ..config import Settings as StagedConfig
from ..llm import LLMClient as StagedLLM
from ..numbering import clean
from ..render import PageRenderer

try:
    from dateutil import parser as _dtparser  # cross-check snippet parsing (dayfirst=Indian default)
except Exception:  # pragma: no cover - dateutil ships with the app; fail-open to no cross-check
    _dtparser = None


# --------------------------------------------------------------------------- #
# Schemas (kept local — NOT added to the shared models.py the working stages use)
# --------------------------------------------------------------------------- #
DateType = Literal[
    "EFFECTIVE_DATE", "EXECUTION_DATE", "SIGNING_DATE", "AGREEMENT_DATE",
    "COMMENCEMENT_DATE", "START_DATE", "APPOINTED_DATE",
    "LETTER_OF_AWARD_DATE", "LETTER_OF_INTENT_DATE", "LETTER_OF_ACCEPTANCE_DATE",
    "NOTICE_TO_PROCEED_DATE", "WORK_ORDER_DATE", "PURCHASE_ORDER_DATE",
    "MOBILISATION_DATE", "SITE_HANDOVER_DATE", "POSSESSION_DATE",
    "TAKING_OVER_DATE", "COMPLETION_DATE", "SCHEDULED_COMPLETION_DATE",
    "DEFECTS_LIABILITY_START", "CONTRACT_EXPIRY_DATE", "END_DATE",
    "BID_DATE", "TENDER_DATE", "BID_VALIDITY_DATE", "SIGNATURE_DATE",
    "OTHER_DEFINED_DATE",
]
Confidence = Literal["high", "medium", "low"]


class DateAnchor(BaseModel):
    name: str = Field(description="The named date term EXACTLY as written (e.g. 'Effective Date', 'Letter of "
                                  "Intent', 'Appointed Date', 'Taking-Over Certificate'). Never paraphrased.")
    date_type: DateType = Field(description="Canonical classification; OTHER_DEFINED_DATE if none fits.")
    iso_value: Optional[str] = Field(None, description="Concrete date YYYY-MM-DD ONLY if explicitly stated/printed for "
                                     "this term. If defined/named but NO concrete date is stated, set null. NEVER guess.")
    source_snippet: Optional[str] = Field(None, description="The VERBATIM text (or handwritten/stamped mark) the date "
                                          "was read from, copied exactly. Required whenever iso_value is non-null.")
    is_handwritten: bool = Field(False, description="True if the date was handwritten or stamped rather than printed.")
    confidence: Confidence = Field("medium", description="Your certainty this is the correct value for this term.")


class DateAnchorsResponse(BaseModel):
    date_anchors: List[DateAnchor] = Field(default_factory=list)


PartySide = Literal["CLIENT", "CONTRACTOR"]


class Party(BaseModel):
    side: PartySide = Field(description="Which of the two principal sides: CLIENT (owner / employer / client / "
                            "purchaser / buyer) or CONTRACTOR (contractor / vendor / supplier / service provider / "
                            "seller).")
    role: str = Field(description="The role term the contract uses for this party (e.g. 'Owner', 'Employer', "
                      "'Client', 'Contractor', 'Vendor', 'Supplier').")
    name: str = Field(description="The party's canonical legal name EXACTLY as written (e.g. 'Capacit'e "
                      "Infraprojects Ltd').")
    aliases: List[str] = Field(default_factory=list, description="Defined terms / short forms the contract body uses "
                               "for this party (e.g. 'the Contractor', 'Capacit'e').")


class HeadFacts(BaseModel):
    """The read schema: named dates AND the two principal parties, from the same cover/recital read."""
    date_anchors: List[DateAnchor] = Field(default_factory=list)
    parties: List[Party] = Field(default_factory=list, description="ONLY the two principal parties the contract is "
                                 "made BETWEEN — never the Engineer, guarantor, insurer or any third party.")


class ConflictResolution(BaseModel):
    iso_value: Optional[str] = Field(None, description="The single governing date YYYY-MM-DD, or null if genuinely "
                                     "indeterminate. NEVER invent a value not present in the pages shown.")
    reason: Optional[str] = Field(None, description="Which source governs and why (e.g. 'SCC overrides GCC').")


class DateVerdict(BaseModel):
    id: int = Field(description="The candidate id, echoed back.")
    keep: bool = Field(description="True to keep this anchor on the register, False to drop it.")
    name: Optional[str] = Field(None, description="Corrected canonical term name (short noun phrase), or null to keep "
                                "the candidate's name. Never a sentence, raw date, or reference number.")
    iso_value: Optional[str] = Field(None, description="Corrected value YYYY-MM-DD, or null to keep the candidate's "
                                     "value. Set only if the snippet actually states this date for this term.")
    reason: Optional[str] = Field(None, description="One-line why (e.g. 'sentence fragment', 'value not tied to term').")


class DateJudgeResult(BaseModel):
    verdicts: List[DateVerdict] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Prompts (OWNED by this stage; never merged into any working-stage prompt)
# --------------------------------------------------------------------------- #
_READ_SYSTEM = (
    "You extract the DATE ANCHOR REGISTRY of one Indian EPC / construction contract: every explicitly "
    "NAMED or DEFINED date, so a downstream engine can resolve obligations phrased as 'X days/months from "
    "<that date>'. You are given a slice of the contract (text and/or page images).\n"
    "\n"
    "INCLUDE:\n"
    "  - defined date terms: Effective Date, Execution/Signing Date, Agreement Date, Commencement/Start "
    "Date, Appointed Date, Mobilisation Date, Site Hand-over/Possession Date, Taking-Over Date, "
    "Completion/Scheduled Completion Date, Defects-Liability start, Contract Expiry/End Date, and any "
    "other term the contract defines as a date;\n"
    "  - document-level dates printed or WRITTEN on a cover/letter: the date of the Letter of Award (LOA), "
    "Letter of Intent (LOI), Letter of Acceptance, Notice to Proceed (NTP), Work/Purchase Order, and the "
    "Agreement/signing date. Use the name as written (e.g. 'Letter of Intent').\n"
    "\n"
    "HANDWRITING & STAMPS — READ THEM. On cover letters the operative date is very often HANDWRITTEN or "
    "STAMPED, next to the reference/letter number, on a date line, or beside the signature. Transcribe "
    "these; set is_handwritten=true. If a handwritten date is genuinely illegible, set iso_value=null "
    "(never guess a value).\n"
    "\n"
    "DISAMBIGUATION — a page can carry several dates. Capture the term's OWN date, NOT a date it merely "
    "REFERENCES. On a Letter of Intent, the LOI date is the letter's own date (by its number / date line "
    "or signature); a 'Reference: our meeting dated ...' line is a referenced event, NOT the LOI date — do "
    "not record it as the LOI's date. If both a printed letter date and a separate signature date appear, "
    "record the letter's date as the instrument's date and, if useful, the signature date separately as "
    "SIGNATURE_DATE.\n"
    "\n"
    "NAMING DATED EVENTS STATED IN PROSE — when the text gives a dated event in a sentence rather than as "
    "a labelled term (e.g. 'the Contractor submitted its offer vide letter dated <date>', 'the tender was "
    "floated on <date>', 'addendum issued on <date>'), STILL capture it, but set `name` to the "
    "CANONICAL short term for that event, never the sentence: an offer/bid/tender submitted -> 'Tender "
    "Submission Date'; a tender floated/invited/issued -> 'Tender Notice Date'; a Work Order issued -> "
    "'Work Order'; an addendum/corrigendum issued -> 'Addendum Date'. Put the stated date in iso_value and "
    "the sentence in source_snippet. This keeps a real date out of a messy fragment name.\n"
    "\n"
    "RULES:\n"
    "  - iso_value = the concrete date as YYYY-MM-DD ONLY IF explicitly stated for that term; else null. "
    "NEVER infer, compute, convert across calendars, or guess.\n"
    "  - Indian dates are day-first: '05/03/2024' is 5 March 2024, not 3 May.\n"
    "  - For EVERY date with a non-null iso_value, copy the exact text/mark it came from into "
    "source_snippet. No snippet -> do not emit a value.\n"
    "  - `name` = the DEFINED-TERM the date belongs to, written as the SHORT NOUN PHRASE the contract "
    "uses (e.g. 'Effective Date', 'Letter of Intent', 'Appointed Date', 'Taking-Over Certificate'). It is "
    "NEVER a sentence or clause fragment ('tender floated vide letter dated ...'), NEVER a raw date "
    "('January 06, 2023'), NEVER a reference / document number ('LOI NO: ABC/DEF/123-2024'), and NEVER the "
    "date_type code itself ('SIGNATURE_DATE' -> use 'Signature Date'). A pure DURATION ('Contract Period', "
    "'Defects Liability Period') is NOT a date anchor — skip it unless a concrete calendar date is stated "
    "for it. Classify each into date_type (OTHER_DEFINED_DATE if none fits). Do not duplicate a term.\n"
    "  - Bind a value to a term ONLY when the text actually states that date FOR that term. Do not attach "
    "a nearby date to a term the text does not tie it to.\n"
    "  - If the slice declares no named/defined dates, return an empty date_anchors list.\n"
    "\n"
    "PRINCIPAL PARTIES — also return the TWO principal parties this contract is made BETWEEN (its "
    "signatories): exactly one CLIENT side (owner / employer / client / purchaser / buyer) and one "
    "CONTRACTOR side (contractor / vendor / supplier / service provider / seller). For each, give its "
    "canonical legal name exactly as written, the role term the contract uses, and the defined-term "
    "aliases the body uses for it (e.g. 'the Contractor'). Do NOT include the Engineer, Project Manager, "
    "Architect, guarantor / bank, insurer, consultant, sub-contractor or any third party a clause merely "
    "mentions — obligations run only between the two principals. If this slice does not state the parties, "
    "return an empty parties list."
)

_RESOLVE_SYSTEM = (
    "You are a senior contracts lawyer resolving ONE conflicting date. The pages/snippets below each state "
    "a DIFFERENT value for the SAME named date term. Read them and decide the SINGLE governing value.\n"
    "Precedence, strongest first: an executed amendment or addendum > Special Conditions (SCC) / Particular "
    "Conditions > the Contract Data / Appendix to Tender / a dedicated Schedule of Dates > General "
    "Conditions (GCC) > a passing mention. A concrete value in a data sheet overrides a definition that "
    "leaves the date to be fixed.\n"
    "Return the governing date as YYYY-MM-DD, or null if the pages are genuinely indeterminate. NEVER "
    "invent a date that is not present in the pages shown. Give a one-line reason naming which source governs."
)

_JUDGE_SYSTEM = (
    "You are a senior contracts lawyer CURATING a contract's date-anchor register to gold standard. A "
    "junior analyst produced candidate anchors; each carries the VERBATIM snippet it was read from. For "
    "each candidate, decide keep or drop, and correct the name/value where needed. Quality over volume — "
    "when in doubt, DROP.\n"
    "\n"
    "DROP a candidate when it is NOT a genuine, correctly-bound named date, i.e. when it is:\n"
    "  - a sentence or clause fragment that does NOT carry a real calendar date (if it has a messy name "
    "but a genuine, snippet-supported date, RENAME it instead of dropping — see PRESERVE below);\n"
    "  - a raw date used as the name ('January 06, 2023');\n"
    "  - a reference / document number ('LOI NO: ABC/DEF/123-2024');\n"
    "  - a pure DURATION / period, not a date ('Contract Period', 'Defects Liability Period'), unless a "
    "concrete calendar date is genuinely stated for it;\n"
    "  - a value NOT actually tied to the term by its snippet (a nearby date wrongly attached — e.g. the "
    "contract-agreement/stamp date copied onto 'date of acceptance of this LOA');\n"
    "  - a duplicate of another candidate with a cleaner name (keep the clean one).\n"
    "\n"
    "KEEP a candidate when it is a real named/defined date of THIS contract and the snippet supports its "
    "value (or supports the term with no value -> keep with null). You may set `name` to the canonical "
    "short noun-phrase term (e.g. map 'SIGNATURE_DATE' -> 'Signature Date', an LOI number line -> 'Letter "
    "of Intent'), and set `iso_value` to the correct date ONLY if the snippet states it for this term.\n"
    "\n"
    "PRESERVE GROUNDED VALUES — do NOT lose a real date. If a candidate has a messy or fragment NAME but "
    "carries a genuine calendar date the snippet supports for a real named event (e.g. 'the Contractor "
    "submitted its offer vide letter dated <date>' IS the Tender / Bid Submission Date), KEEP it and RENAME it "
    "to the canonical term — never drop it and throw the date away. When the SAME date appears under a "
    "clean-named candidate with NO value and a messy-named candidate WITH the value, keep the value (put "
    "it on the clean name, or rename the messy one and drop the empty duplicate). Only null or drop a "
    "value that is wrong, unsupported by its snippet, or not a real date anchor.\n"
    "\n"
    "COLLAPSE A SEQUENCE OR VARIANTS TO ONE — think about legal effect, not just wording. When several "
    "candidates are the SAME underlying named date expressed as a progression or as synonyms — an original "
    "vs a 'revised' vs a 'final' / 'amended' version of the same submission, meeting or instrument; 'X "
    "Date' vs 'Date of X'; a draft vs the executed version — KEEP ONLY the single GOVERNING one and DROP "
    "the rest. The governing entry is the one that is operative: a revised or final submission SUPERSEDES "
    "the earlier one, an amended/executed date supersedes a draft — so keep the LAST superseding version, "
    "give it the clean base term (e.g. Tender/Original/Revised/Final Tender Submission -> a single 'Tender "
    "Submission Date'), and set its value to that governing date. Do NOT return the original AND the "
    "revised AND the final as separate anchors. Genuinely DISTINCT dates — different documents, or "
    "different events that merely share a value — stay separate.\n"
    "Account for every candidate id exactly once."
)


# --------------------------------------------------------------------------- #
# D0 - candidate selection (deterministic)
# --------------------------------------------------------------------------- #
_COVER_DIVISIONS = {"RECITALS", "CONTRACT_AGREEMENT", "LETTER_OF_AWARD", "LETTER_OF_INTENT"}
_DEFN_TITLE_HINTS = (
    "definition", "interpretation", "commencement", "effective date", "appointed date",
    "contract data", "appendix to tender", "key dates", "schedule of dates",
    "time for completion", "particular conditions", "milestone",
)

_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
           r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DATE_RE = re.compile("|".join([
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\.?,?\s+\d{{4}}\b",   # 15 March 2024 | 1st April, 2023
    rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",   # March 15, 2024
    r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b",                        # 15/03/2024 | 15-03-24
    r"\b\d{4}-\d{2}-\d{2}\b",                                        # ISO
]), re.IGNORECASE)
_ANCHOR_KW_RE = re.compile(
    r"\b(effective\s+date|execution\s+date|date\s+of\s+execution|signing\s+date|commencement\s+date|"
    r"date\s+of\s+commencement|start\s+date|appointed\s+date|agreement\s+date|date\s+of\s+(?:this\s+)?agreement|"
    r"letter\s+of\s+award|letter\s+of\s+intent|letter\s+of\s+acceptance|notice\s+to\s+proceed|work\s+order|"
    r"purchase\s+order|mobili[sz]ation\s+date|site\s+hand(?:ing)?[\s\-]?over|possession\s+date|taking[\s\-]?over|"
    r"completion\s+date|time\s+for\s+completion|defects?\s+liability|contract\s+data|appendix\s+to\s+tender|"
    r"key\s+dates|schedule\s+of\s+dates|with\s+effect\s+from|w\.?e\.?f\.?)\b"
    r"|\b(?:LOA|LOI|NTP)\b"
    r"|\b[A-Z][A-Za-z ]{0,28}?\s+Date[\"'\s]*\bmeans\b",
    re.IGNORECASE,
)


def _dated_anchor_hits(text: str, near: int) -> int:
    """Count anchor-term ↔ date-literal pairs within `near` chars (either order). 0 = not a candidate."""
    terms = [m.start() for m in _ANCHOR_KW_RE.finditer(text)]
    if not terms:
        return 0
    dates = [m.start() for m in _DATE_RE.finditer(text)]
    if not dates:
        return 0
    return sum(1 for t in terms for d in dates if abs(t - d) <= near)


def _unit_text(u: dict) -> str:
    return (u.get("text") or "").strip()


def _select(leaves: List[dict], cfg: StagedConfig) -> Tuple[List[dict], List[dict]]:
    """Return (cover_divisions, text_units).

    cover_divisions: date-bearing cover/head divisions (read as IMAGES).
    text_units: definitions/interpretation/contract-data clauses + proximity-sweep hits (read as TEXT),
    excluding any unit already covered by a cover division. Each text unit is annotated with a label."""
    near = getattr(cfg, "dates_near", 100)
    covers: List[dict] = []
    cover_pages: set = set()
    for div in leaves:
        if (div.get("division_type") or "").upper() in _COVER_DIVISIONS:
            covers.append(div)
            for p in range(div.get("start_page") or 1, (div.get("end_page") or div.get("start_page") or 1) + 1):
                cover_pages.add(p)

    text_units: List[dict] = []
    seen: set = set()
    for div in leaves:
        dtitle = clean(div.get("title"))
        for u in div.get("units", []):
            text = _unit_text(u)
            if not text:
                continue
            ps = u.get("page_start") or 1
            if ps in cover_pages:            # already read as an image via its cover division
                continue
            uid = u.get("unit_id") or (div.get("division_id"), ps, u.get("marker"))
            if uid in seen:
                continue
            title_blob = ((u.get("title") or "") + " " + (u.get("marker") or "") + " " + dtitle).lower()
            hint = any(h in title_blob for h in _DEFN_TITLE_HINTS)
            hits = _dated_anchor_hits(text, near)
            if hint or hits:
                seen.add(uid)
                label = " / ".join(p for p in (dtitle, clean(u.get("marker")), clean(u.get("title"))) if p)
                text_units.append({"label": label or f"page {ps}", "page": ps, "text": text,
                                   "score": hits + (2 if hint else 0)})
    text_units.sort(key=lambda t: (-t["score"], t["page"]))
    return covers, text_units


# --------------------------------------------------------------------------- #
# D1 - read (LLM): cover pages as images, body candidates as text
# --------------------------------------------------------------------------- #
async def _read_cover(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, div: dict) -> List[dict]:
    ds = div.get("start_page") or 1
    de = div.get("end_page") or ds
    cap = getattr(cfg, "dates_cover_max_pages", 6)
    pages = list(range(ds, min(de, ds + cap - 1) + 1))
    try:
        imgs = await asyncio.gather(*[renderer.page(p, getattr(cfg, "anchor_dpi", 200)) for p in pages])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[staged.dates] cover render failed div {div.get('division_id')} ({type(e).__name__})")
        return []
    parts: list = [f"Document: {clean(div.get('title'))[:120]} ({div.get('division_type')}). Read the "
                   f"page image(s) below and extract every named/defined date, including handwritten and "
                   f"stamped dates."]
    for p, img in zip(pages, imgs):
        parts.append(f"\n[Physical PDF Page {p}]")
        parts.append(img)
    res = await llm.call("dates:cover_read", role="reasoner", system=_READ_SYSTEM, parts=parts, schema=HeadFacts,
                         thinking="low", media_resolution=getattr(cfg, "anchor_media", "medium"),
                         max_output_tokens=4096)
    if not res.ok:
        logger.warning(f"[staged.dates] cover read unusable div {div.get('division_id')} ({res.finish_reason})")
        return [], []
    dtype = div.get("division_type")
    rows = [_row(a, pages[0], dtype) for a in res.parsed.date_anchors]
    parties = [_party_row(p, dtype) for p in res.parsed.parties]
    return rows, parties


async def _read_text(llm: StagedLLM, cfg: StagedConfig, units: List[dict]) -> tuple:
    if not units:
        return [], []
    budget = getattr(cfg, "dates_max_chars", 60000)
    blocks: List[str] = []
    total = 0
    page_of: List[int] = []
    for u in units:
        block = f"===== {u['label']} [page {u['page']}] =====\n{u['text']}"
        if total + len(block) > budget:
            break
        blocks.append(block)
        page_of.append(u["page"])
        total += len(block)
    body = "\n\n".join(blocks)
    res = await llm.call("dates:text_read", role="reasoner", system=_READ_SYSTEM,
                         parts=[f"Extract every named/defined date (and the two principal parties, if stated) "
                                f"from the contract text below. Each block is headed with its clause label and "
                                f"[page N].\n\n{body}"],
                         schema=HeadFacts, thinking="minimal", max_output_tokens=8192)
    if not res.ok:
        logger.warning(f"[staged.dates] text read unusable ({res.finish_reason})")
        return [], []
    fallback_page = page_of[0] if page_of else None
    rows = [_row(a, fallback_page, None) for a in res.parsed.date_anchors]
    parties = [_party_row(p, None) for p in res.parsed.parties]
    return rows, parties


def _row(a: DateAnchor, page: Optional[int], division_type: Optional[str]) -> dict:
    return {"name": clean(a.name), "date_type": a.date_type, "iso_value": a.iso_value,
            "source_snippet": a.source_snippet, "is_handwritten": bool(a.is_handwritten),
            "confidence": a.confidence, "page": page, "division_type": division_type}


def _party_row(p: Party, division_type: Optional[str]) -> dict:
    return {"side": p.side, "role": clean(p.role), "name": clean(p.name),
            "aliases": [clean(a) for a in (p.aliases or []) if clean(a)],
            "division_type": division_type}


# --------------------------------------------------------------------------- #
# D2 - normalise + validate (deterministic, fail-closed)
# --------------------------------------------------------------------------- #
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_iso(v: Optional[str]) -> Optional[str]:
    if not v or not _ISO_RE.match(v.strip()):
        return None
    try:
        d = datetime.strptime(v.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None
    return d.isoformat() if 1990 <= d.year <= 2100 else None


def _parse_snippet(snippet: Optional[str]) -> Optional[str]:
    """Independently parse a date out of the verbatim snippet (day-first), for cross-checking the LLM iso."""
    if not snippet or _dtparser is None:
        return None
    m = _DATE_RE.search(snippet)
    if not m:
        return None
    try:
        d = _dtparser.parse(m.group(0), dayfirst=True, fuzzy=True).date()
        return d.isoformat() if 1990 <= d.year <= 2100 else None
    except (ValueError, OverflowError, TypeError):
        return None


def _normalise(rows: List[dict]) -> List[dict]:
    """Validate each iso; cross-check against a re-parse of its snippet. A value the code cannot confirm
    is dropped to null (term kept). Confidence is downgraded on disagreement (fail-closed)."""
    out: List[dict] = []
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        iso = _valid_iso(r.get("iso_value"))
        if iso is not None:
            snip_iso = _parse_snippet(r.get("source_snippet"))
            if not r.get("source_snippet"):
                iso, r["confidence"] = None, "low"        # unglounded value -> drop (fail-closed)
            elif snip_iso and snip_iso != iso:
                # LLM iso and a re-parse of its own snippet disagree (classic DD/MM swap or OCR slip):
                # trust the deterministic snippet parse, flag low confidence for review.
                iso, r["confidence"] = snip_iso, "low"
        r["iso_value"] = iso
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# D2.5 - deterministic name-quality clean (drop obvious non-anchors before the judge)
# --------------------------------------------------------------------------- #
_REF_CODE_RE = re.compile(r"[/:]|\b\d{3,}\b")                      # a code / doc-number / path in the name
_ENUM_TO_LABEL = {t: t.replace("_", " ").title() for t in DateType.__args__}   # SIGNATURE_DATE -> "Signature Date"


_DURATION_RE = re.compile(r"\b(period|duration)$", re.IGNORECASE)   # a span, not a calendar date anchor


def _bad_name(name: str) -> bool:
    """A name that is obviously NOT a defined-term date anchor — a raw date, a reference/number line, a
    long sentence fragment, or a pure duration/period. Dropped deterministically so the LLM judge only
    sees plausible candidates."""
    n = (name or "").strip()
    if not n:
        return True
    if _DATE_RE.search(n) and len(_DATE_RE.sub("", n).strip(" ,.-")) <= 2:
        return True                                                # the name is essentially just a date
    if _REF_CODE_RE.search(n):
        return True
    if _DURATION_RE.search(n):
        return True                                                # 'Defects Liability Period', 'Contract Period'
    if len(n.split()) > 6:
        return True
    return False


def _clean(rows: List[dict]) -> List[dict]:
    """Relabel a bare date_type used as a name, then drop obvious non-anchors. What survives is plausible
    enough to spend LLM judgment on (the judge then removes wrongly-bound values and fragment terms)."""
    out: List[dict] = []
    for r in rows:
        name = (r.get("name") or "").strip()
        if name in _ENUM_TO_LABEL:                                 # 'SIGNATURE_DATE' as the name -> human label
            r["name"] = _ENUM_TO_LABEL[name]
            name = r["name"]
        if _bad_name(name):
            continue
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# D3 - merge + conflict detect (deterministic)
# --------------------------------------------------------------------------- #
_PRECEDENCE = {"CONTRACT_AGREEMENT": 3, "SCC": 3, "LETTER_OF_AWARD": 2, "LETTER_OF_INTENT": 2, "GCC": 1}


def _merge(rows: List[dict]) -> Tuple[Dict[str, dict], List[dict]]:
    """Group by date_type. Returns (resolved_by_type, conflicts).
    resolved_by_type[dtype] = {"iso": value|None, "names": {...}, "note": ...}; conflicts carry the
    competing populated candidates for D4."""
    by_type: Dict[str, List[dict]] = {}
    for r in rows:
        by_type.setdefault(r["date_type"], []).append(r)

    resolved: Dict[str, dict] = {}
    conflicts: List[dict] = []
    for dtype, group in by_type.items():
        names = {r["name"] for r in group if r["name"]}
        populated = [r for r in group if r["iso_value"]]
        distinct = sorted({r["iso_value"] for r in populated})
        if not distinct:
            resolved[dtype] = {"iso": None, "names": names, "note": "defined; no concrete date stated"}
            continue
        if len(distinct) == 1:
            resolved[dtype] = {"iso": distinct[0], "names": names, "note": "single value"}
            continue
        # Conflict: try deterministic precedence by division before asking the LLM.
        best = max(populated, key=lambda r: _PRECEDENCE.get((r.get("division_type") or "").upper(), 0))
        best_rank = _PRECEDENCE.get((best.get("division_type") or "").upper(), 0)
        contenders = [r for r in populated
                      if _PRECEDENCE.get((r.get("division_type") or "").upper(), 0) == best_rank]
        if best_rank > 0 and len({r["iso_value"] for r in contenders}) == 1:
            resolved[dtype] = {"iso": best["iso_value"], "names": names,
                               "note": f"precedence: {best.get('division_type')} governs"}
        else:
            resolved[dtype] = {"iso": distinct[0], "names": names, "note": "UNRESOLVED conflict (provisional)"}
            conflicts.append({"date_type": dtype, "names": names, "candidates": populated})
    return resolved, conflicts


# --------------------------------------------------------------------------- #
# D4 - conflict resolution (LLM, only on a real conflict)
# --------------------------------------------------------------------------- #
async def _resolve(llm: StagedLLM, cfg: StagedConfig, conflict: dict) -> Optional[dict]:
    dtype = conflict["date_type"]
    term = sorted(conflict["names"])[0] if conflict["names"] else dtype
    lines = []
    for c in conflict["candidates"]:
        lines.append(f"- value {c['iso_value']} | source: {c.get('division_type') or '?'} p{c.get('page')} "
                     f"| text: \"{(c.get('source_snippet') or '')[:200]}\"")
    prompt = (f"Named date term in conflict: '{term}' ({dtype}). Competing values:\n"
              + "\n".join(lines) + "\n\nDecide the single governing value (YYYY-MM-DD) or null.")
    try:
        res = await llm.call("dates:resolve", role="reasoner", system=_RESOLVE_SYSTEM, parts=[prompt],
                             schema=ConflictResolution, thinking="low", max_output_tokens=512)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[staged.dates] conflict resolve errored for {dtype} ({type(e).__name__})")
        return None
    if not res.ok:
        return None
    iso = _valid_iso(res.parsed.iso_value)
    # Only accept a value that is actually one of the competing candidates (never a fabricated third date).
    allowed = {c["iso_value"] for c in conflict["candidates"]}
    if iso is not None and iso not in allowed:
        logger.warning(f"[staged.dates] resolver returned out-of-set date for {dtype}; dropping to null")
        iso = None
    return {"iso": iso, "note": f"conflict resolved: {clean(res.parsed.reason) or 'llm decision'}"}


# --------------------------------------------------------------------------- #
# D6 - LLM curation (date judge): keep / drop / fix over grounded candidates
# --------------------------------------------------------------------------- #
async def _judge(llm: StagedLLM, cfg: StagedConfig, rows: List[dict]) -> List[dict]:
    """Strict LLM curation over the candidates (each shown WITH its verbatim snippet), so it can drop a
    term whose value the snippet does not actually support (the wrongly-bound / 'extra' anchors a domain
    expert flags) and fragment/duration terms the deterministic clean let through. Defensive: on
    disabled / empty / failure, returns the rows unchanged (they are already deterministically cleaned)."""
    if not getattr(cfg, "dates_judge_enabled", True) or not rows:
        return rows
    view = [{"id": i, "name": r.get("name"), "date_type": r.get("date_type"),
             "iso_value": r.get("iso_value"), "division_type": r.get("division_type"),
             "source_snippet": (r.get("source_snippet") or "")[:300]} for i, r in enumerate(rows)]
    try:
        res = await llm.call(
            "dates:judge", role="reasoner", system=_JUDGE_SYSTEM,
            parts=["Curate these date-anchor candidates to gold standard. Return one verdict per id.\n\n"
                   + json.dumps(view, ensure_ascii=False)],
            schema=DateJudgeResult, thinking="medium", max_output_tokens=16384)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[staged.dates] judge errored ({type(e).__name__}) -> keeping cleaned candidates")
        return rows
    if not res.ok:
        logger.warning(f"[staged.dates] judge unusable ({res.finish_reason}) -> keeping cleaned candidates")
        return rows
    verdicts = {v.id: v for v in res.parsed.verdicts}
    kept: List[dict] = []
    for i, r in enumerate(rows):
        v = verdicts.get(i)
        if v is None:
            kept.append(r)                                 # fail-safe: unruled candidate survives
            continue
        if not v.keep:
            continue
        if v.name and v.name.strip():
            r["name"] = v.name.strip()
        if v.iso_value:
            iso = _valid_iso(v.iso_value)
            if iso:
                r["iso_value"] = iso
        kept.append(r)
    logger.info(f"[staged.dates] judge: {len(rows)} candidate(s) -> {len(kept)} kept")
    return kept


# --------------------------------------------------------------------------- #
# D5 - assemble + audit
# --------------------------------------------------------------------------- #
def _canon_name(name: str) -> str:
    """Case/format-insensitive key so 'LETTER OF AWARD', 'Letter of Award' and 'Letter of Award (LOA)'
    are ONE entry: lower-cased, parentheticals dropped, punctuation flattened."""
    s = re.sub(r"\(.*?\)", " ", (name or "").lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _display_name(names: List[str]) -> str:
    """The cleanest spelling among case/format variants: prefer not-ALL-CAPS and no parentheses, then the
    shortest; Title-case a remaining ALL-CAPS name ('CONTRACT AGREEMENT' -> 'Contract Agreement')."""
    def rank(n: str):
        lower_words = sum(1 for w in n.split() if w[:1].islower())   # prefer proper capitalization
        return (n.isupper(), ("(" in n or ")" in n), lower_words, len(n))
    best = sorted((n.strip() for n in names if n and n.strip()), key=rank)[0]
    if best.isupper() and len(best) > 5:
        best = best.title()
    return best


def _assemble(rows: List[dict], resolved: Dict[str, dict]) -> dict:
    # Collapse case/format variants of the SAME name into one registry key (populated value wins over
    # null) so the registry shows one clean entry per named date, not 'LOA' three times.
    groups: Dict[str, dict] = {}
    for info in resolved.values():
        for name in info["names"]:
            key = _canon_name(name)
            if not key:
                continue
            g = groups.setdefault(key, {"names": set(), "iso": None})
            g["names"].add(name.strip())
            if g["iso"] is None and info["iso"] is not None:
                g["iso"] = info["iso"]
    registry: Dict[str, Optional[str]] = {}
    for g in groups.values():
        registry[_display_name(list(g["names"]))] = g["iso"]
    anchors = [{"name": r["name"], "date_type": r["date_type"], "iso_value": r["iso_value"],
                "source_snippet": r.get("source_snippet"), "is_handwritten": r.get("is_handwritten", False),
                "confidence": r.get("confidence", "medium"), "page": r.get("page"),
                "division_type": r.get("division_type")} for r in rows]
    return {"registry": registry, "anchors": anchors}


_PARTY_PRIORITY = {"CONTRACT_AGREEMENT": 3, "RECITALS": 3, "LETTER_OF_AWARD": 2, "LETTER_OF_INTENT": 2}


def _pick_parties(candidates: List[List[dict]]) -> List[dict]:
    """From the party lists the reads returned, pick the single best TWO-party set (one CLIENT, one
    CONTRACTOR). Prefers the most authoritative source (contract agreement / recitals) and the most
    complete set. Returns at most two {side, role, name, aliases}."""
    best: List[dict] = []
    best_score = -1
    for parties in candidates:
        named = [p for p in parties if p.get("name")]
        if not named:
            continue
        prio = max((_PARTY_PRIORITY.get((p.get("division_type") or "").upper(), 0) for p in named), default=0)
        sides = {p.get("side") for p in named}
        completeness = 2 if {"CLIENT", "CONTRACTOR"} <= sides else len(named)
        score = prio * 10 + completeness
        if score > best_score:
            best_score = score
            best = named
    out: Dict[str, dict] = {}
    for p in best:
        s = p.get("side")
        if s in ("CLIENT", "CONTRACTOR") and s not in out:
            out[s] = {"side": s, "role": p.get("role"), "name": p.get("name"), "aliases": p.get("aliases") or []}
    result = [out[s] for s in ("CLIENT", "CONTRACTOR") if s in out]
    return result or best[:2]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def run(llm: StagedLLM, renderer: PageRenderer, cfg: StagedConfig, leaves: List[dict]) -> dict:
    """Build the date-anchor registry. Returns {"registry": {name: iso|None}, "anchors": [...],
    "conflicts": [...]}. Empty payload on disabled / no candidates / any failure — never raises."""
    empty = {"registry": {}, "anchors": [], "conflicts": [], "parties": []}
    if not getattr(cfg, "dates_enabled", True):
        return empty
    try:
        covers, text_units = _select(leaves, cfg)
        if not covers and not text_units:
            return empty

        # D1 read (covers as images, body as text) — concurrent. Each read returns (date_rows, parties).
        sem = asyncio.Semaphore(max(2, getattr(cfg, "concurrency", 8)))

        async def _cover(div):
            async with sem:
                return await _read_cover(llm, renderer, cfg, div)

        cover_results = await asyncio.gather(*[_cover(d) for d in covers]) if covers else []
        text_rows, text_parties = await _read_text(llm, cfg, text_units)
        rows = [r for (rws, _p) in cover_results for r in rws] + text_rows
        parties = _pick_parties([prt for (_r, prt) in cover_results] + [text_parties])

        # D2 normalise/validate -> D2.5 deterministic clean -> D6 LLM curation -> D3 merge/conflict.
        rows = _normalise(rows)
        rows = _clean(rows)
        rows = await _judge(llm, cfg, rows)
        resolved, conflicts = _merge(rows)

        # D4 resolve real conflicts (targeted, only when they exist).
        conflict_audit: List[dict] = []
        if conflicts:
            outcomes = await asyncio.gather(*[_resolve(llm, cfg, c) for c in conflicts])
            for c, outcome in zip(conflicts, outcomes):
                if outcome is not None:
                    resolved[c["date_type"]]["iso"] = outcome["iso"]
                    resolved[c["date_type"]]["note"] = outcome["note"]
                conflict_audit.append({"date_type": c["date_type"],
                                       "candidates": [x["iso_value"] for x in c["candidates"]],
                                       "resolved": resolved[c["date_type"]]["iso"],
                                       "note": resolved[c["date_type"]]["note"]})

        # D5 assemble.
        payload = _assemble(rows, resolved)
        payload["conflicts"] = conflict_audit
        payload["parties"] = parties
        pop = sum(1 for v in payload["registry"].values() if v)
        logger.info(f"[staged.dates] registry: {len(payload['registry'])} named date(s), {pop} populated, "
                    f"{len(parties)} principal part(y/ies), {len(covers)} cover div(s), "
                    f"{len(text_units)} text unit(s), {len(conflict_audit)} conflict(s)")
        return payload
    except Exception as e:  # noqa: BLE001 — never let the date engine break clause parsing
        logger.warning(f"[staged.dates] date engine skipped ({type(e).__name__}): {str(e)[:150]}")
        return empty
