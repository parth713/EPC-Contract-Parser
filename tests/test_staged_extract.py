"""Unit tests for staged/extract.py within-page text distribution — the deterministic split that
assigns a verbatim read's blocks to level-1 units. LLM-free, so exercised directly with synthetic
blocks.

Regression focus: a DENSE enumeration page (an annexure's list of submissions, a bank-guarantee
proforma's numbered items) packs many level-1 units onto ONE page, each printed as a numbered LIST
item. The page-driven advance can't move between units that all share the page, so the within-page
split must accept a list_item block as a unit's start - otherwise the first unit swallows the whole
page and the rest come out empty (the observed bug). The preamble->clause transition must stay strict
so a recital point can never be mistaken for the operative clause that restarts numbering.
"""
from epc_parser.staged.extract import extract_division_text
from epc_parser.staged.models import PageTextLLM, TextBlockLLM
from epc_parser.staged.runner import _to_clause_dicts


def _blk(kind, marker, text):
    return TextBlockLLM(kind=kind, marker=marker, text=text)


def _page(*blocks, continues=False):
    return PageTextLLM(first_block_continues_previous_page=continues, blocks=list(blocks))


def _units_on_one_page(did, page, markers):
    """Units as anchors._build_units emits them when several siblings all begin on the same page:
    every unit gets page_start == page_end == page."""
    return [{"unit_id": f"{did}-U{n:02d}", "kind": "clause", "marker": m, "title": m,
             "page_start": page, "page_end": page, "flags": ["shares_start_page"], "children": []}
            for n, m in enumerate(markers, 1)]


def test_dense_list_page_splits_across_all_units():
    """Annexure 'List of Submissions': one page, 4 list_item units. Each must get its OWN text; before
    the fix items 2-4 came out empty because list_item blocks never started a unit."""
    units = _units_on_one_page("D1", 26, ["1", "2", "3", "4"])
    div = {"division_id": "D1", "leaf_level": "clause", "start_page": 26, "end_page": 26, "units": units}
    pages = {26: _page(
        _blk("paragraph", None, "ANNEXURE - IV LIST OF SUBMISSIONS TO BE MADE"),
        _blk("list_item", "1", "1. Quality Policy"),
        _blk("list_item", "2", "2. Organization Profile"),
        _blk("list_item", "3", "3. Organization Chart"),
        _blk("list_item", "4", "4. Construction schedule"),
    )}
    extract_division_text(div, pages)
    texts = [u["text"] for u in units]
    assert all(t.strip() for t in texts), f"some units empty: {texts}"
    assert "Quality Policy" in texts[0]           # the page banner folds into the first unit's text
    assert texts[1] == "2. Organization Profile"
    assert texts[2] == "3. Organization Chart"
    assert texts[3] == "4. Construction schedule"
    assert not any("empty_text" in u.get("flags", []) for u in units)


def test_dense_proforma_items_do_not_pile_into_first_unit():
    """Bank-guarantee proforma: numbered items 6)-9) on one page. Item 6 must NOT contain 7-9's text."""
    units = _units_on_one_page("D2", 16, ["6", "7", "8", "9"])
    div = {"division_id": "D2", "leaf_level": "clause", "start_page": 16, "end_page": 16, "units": units}
    pages = {16: _page(
        _blk("list_item", "6", "6) Any demand for payment shall be in writing."),
        _blk("list_item", "7", "7) This Deed shall come into force with immediate effect."),
        _blk("list_item", "8", "8) Any changes to the terms require consent."),
        _blk("list_item", "9", "9) Between the Bank and the Client this is binding."),
    )}
    extract_division_text(div, pages)
    assert units[0]["text"] == "6) Any demand for payment shall be in writing."
    assert units[1]["text"].startswith("7)")
    assert units[2]["text"].startswith("8)")
    assert units[3]["text"].startswith("9)")
    # nothing from later items leaked into the first
    assert "This Deed shall come into force" not in units[0]["text"]


def test_preamble_recital_not_mistaken_for_restarting_clause():
    """Regression guard: a preamble whose recital point '1' (list_item) overflows onto the page where
    operative clause '1' restarts. The recital must stay with the preamble; only the real clause-kind
    heading may start clause 1. Only ONE non-preamble unit begins on the transition page, so the page
    is NOT dense and list_item tolerance stays OFF."""
    units = [
        {"unit_id": "D3-U01", "kind": "preamble", "marker": None, "title": "Preamble",
         "page_start": 1, "page_end": 2, "flags": [], "children": []},
        {"unit_id": "D3-U02", "kind": "clause", "marker": "1", "title": "Definitions",
         "page_start": 2, "page_end": 3, "flags": [], "children": []},
        {"unit_id": "D3-U03", "kind": "clause", "marker": "2", "title": "Scope",
         "page_start": 3, "page_end": 3, "flags": [], "children": []},
    ]
    div = {"division_id": "D3", "leaf_level": "clause", "start_page": 1, "end_page": 3, "units": units}
    pages = {
        1: _page(_blk("paragraph", None, "WHEREAS the parties have agreed as follows:")),
        2: _page(
            _blk("list_item", "1", "1. the Employer wishes to engage the Contractor"),  # recital overflow
            _blk("clause", "1", "1. DEFINITIONS In this Contract the following terms apply."),
        ),
        3: _page(_blk("clause", "2", "2. SCOPE The Contractor shall perform the Works.")),
    }
    extract_division_text(div, pages)
    preamble, c1, c2 = units
    assert "the Employer wishes to engage" in preamble["text"], "recital point wrongly left the preamble"
    assert c1["text"].startswith("1. DEFINITIONS")
    assert "the Employer wishes to engage" not in c1["text"], "recital leaked into the restarting clause"
    assert c2["text"].startswith("2. SCOPE")


def test_ordinary_two_clauses_per_page_still_split_by_heading():
    """Non-dense page: clause 4 ends and clause 5 begins on the same page, both real clause headings.
    Unaffected by the fix - the heading match already handled this."""
    units = [
        {"unit_id": "D4-U01", "kind": "clause", "marker": "4", "title": "Notice",
         "page_start": 7, "page_end": 8, "flags": [], "children": []},
        {"unit_id": "D4-U02", "kind": "clause", "marker": "5", "title": "Term",
         "page_start": 8, "page_end": 8, "flags": [], "children": []},
    ]
    div = {"division_id": "D4", "leaf_level": "clause", "start_page": 7, "end_page": 8, "units": units}
    pages = {
        7: _page(_blk("clause", "4", "4. NOTICE All notices shall be in writing.")),
        8: _page(
            _blk("paragraph", None, "and delivered to the registered address."),  # clause 4 tail
            _blk("clause", "5", "5. TERM This Contract runs for three years."),
        ),
    }
    extract_division_text(div, pages)
    assert "delivered to the registered address" in units[0]["text"]
    assert units[1]["text"].startswith("5. TERM")
    assert "TERM" not in units[0]["text"]


def test_signatures_and_stamps_dropped_from_clause_text():
    """We want clauses, not a gold-standard OCR of every seal. Party signatures, stamps
    and the running signature/initial space in a page footer must NOT land in a clause's text - they stay
    only in the raw document_pages record."""
    units = [{"unit_id": "D5-U01", "kind": "clause", "marker": "1", "title": "Payment",
              "page_start": 1, "page_end": 1, "flags": [], "children": []}]
    div = {"division_id": "D5", "leaf_level": "clause", "start_page": 1, "end_page": 1, "units": units}
    pages = {1: _page(
        _blk("clause", "1", "1. PAYMENT The Employer shall pay within 30 days."),
        _blk("signature", None, "For and on behalf of the Contractor\nAuthorised Signatory"),
        _blk("stamp", None, "[Round seal: ACME ENGINEERING PVT LTD]"),
        _blk("signature", None, "Witness: ____________"),
    )}
    extract_division_text(div, pages)
    assert units[0]["text"] == "1. PAYMENT The Employer shall pay within 30 days."
    assert "Signatory" not in units[0]["text"]
    assert "seal" not in units[0]["text"].lower()
    assert "Witness" not in units[0]["text"]


def test_letter_front_matter_stripped_when_clause1_shares_first_page():
    """A Letter of Intent whose clause 1 begins on the division's OWN first page: anchors create NO
    preamble unit (only made when clause 1 starts on a later page), so the letterhead / reference /
    addressee / 'Dear Sir' front matter lands inside clause 1. It must be dropped so the clause starts at
    '1.' - 'we only need from clause 1'. The front matter stays in the raw document_pages record."""
    units = [
        {"unit_id": "L1-U01", "kind": "clause", "marker": "1", "title": "Intent",
         "page_start": 1, "page_end": 1, "flags": [], "children": []},
        {"unit_id": "L1-U02", "kind": "clause", "marker": "2", "title": "Scope",
         "page_start": 1, "page_end": 1, "flags": ["shares_start_page"], "children": []},
    ]
    div = {"division_id": "L1", "leaf_level": "clause", "start_page": 1, "end_page": 1, "units": units}
    pages = {1: _page(
        _blk("paragraph", None, "SAIFEE BURHANI Upliftment Trust  Regd. E-25618 (Mumbai)"),
        _blk("paragraph", None, "LOI NO: SBUT/TEC/LOI/SC04/CIL-CSW/189-2022  December 24, 2022"),
        _blk("heading", None, "LETTER OF INTENT"),
        _blk("paragraph", None, "To, M/s. Capacit'e Infraprojects Ltd. ... Chembur, Mumbai - 400071"),
        _blk("paragraph", None, "Subject: Letter of Intent (LOI) for core and shell works."),
        _blk("paragraph", None, "Dear Sir,"),
        _blk("clause", "1", "1. We, Saifee Burhani Upliftment Trust, are pleased to inform you..."),
        _blk("clause", "2", "2. The Contractor shall carry out the Works as per the terms."),
    )}
    extract_division_text(div, pages)
    assert units[0]["text"].startswith("1. We, Saifee Burhani")
    for junk in ("SAIFEE BURHANI Upliftment Trust", "LOI NO", "LETTER OF INTENT", "Dear Sir", "Capacit"):
        assert junk not in units[0]["text"], f"front matter leaked into clause 1: {junk!r}"
    assert units[1]["text"].startswith("2.")


def test_first_clause_at_top_of_page_is_not_stripped():
    """Guard: when the first clause's heading IS the first block (no front matter), nothing is stripped."""
    units = [{"unit_id": "L2-U01", "kind": "clause", "marker": "1", "title": "Definitions",
              "page_start": 1, "page_end": 1, "flags": [], "children": []}]
    div = {"division_id": "L2", "leaf_level": "clause", "start_page": 1, "end_page": 1, "units": units}
    pages = {1: _page(_blk("clause", "1", "1. DEFINITIONS In this Contract the following apply."))}
    extract_division_text(div, pages)
    assert units[0]["text"] == "1. DEFINITIONS In this Contract the following apply."


def test_front_matter_kept_when_clause1_heading_not_found():
    """Safety: if clause 1's heading can't be matched (mis-transcribed), the front matter stays as a
    prefix rather than the clause coming out empty - never lose content."""
    units = [{"unit_id": "L3-U01", "kind": "clause", "marker": "1", "title": "Intent",
              "page_start": 1, "page_end": 1, "flags": [], "children": []}]
    div = {"division_id": "L3", "leaf_level": "clause", "start_page": 1, "end_page": 1, "units": units}
    pages = {1: _page(
        _blk("paragraph", None, "Dear Sir,"),
        _blk("paragraph", None, "We are pleased to inform you..."),  # clause 1 heading not marked
    )}
    extract_division_text(div, pages)
    assert "We are pleased to inform you" in units[0]["text"]  # content preserved, not dropped


def test_letter_closing_signoff_dropped_from_last_clause():
    """Back matter: a letter's closing (courtesy close, 'Yours faithfully', and the office-address / CIN /
    website footer) trails the last clause. Labelled as a signature/closing block by the reader, it must be
    dropped so the last clause keeps only its own terms; it remains in the raw document_pages record."""
    units = [{"unit_id": "L4-U01", "kind": "clause", "marker": "4", "title": "NSC Packages",
              "page_start": 1, "page_end": 1, "flags": [], "children": []}]
    div = {"division_id": "L4", "leaf_level": "clause", "start_page": 1, "end_page": 1, "units": units}
    pages = {1: _page(
        _blk("clause", "4", "4. NSC Packages: All terms & conditions for NSC packages shall be on back to "
                            "back basis with the main contract."),
        _blk("signature", None, "We trust you will find the above in line with your requirement."),
        _blk("signature", None, "Thanking you,\nYours faithfully,"),
        _blk("signature", None, "Mumbai (Head Office): E-605-607, Shrikant Chambers ... Email: info@x.in"),
        _blk("signature", None, "NCR | Bangalore  CIN : L45400MH2012PLC234318  www.example.in"),
    )}
    extract_division_text(div, pages)
    assert units[0]["text"] == ("4. NSC Packages: All terms & conditions for NSC packages shall be on back "
                                "to back basis with the main contract.")
    for junk in ("Yours faithfully", "Thanking you", "Head Office", "CIN", "www."):
        assert junk not in units[0]["text"], f"letter closing leaked into last clause: {junk!r}"


def test_preamble_unit_excluded_from_clause_projection():
    """A preamble/recitals unit is boilerplate before the first real clause - it must not surface as a
    clause row. The operative clauses that follow it must."""
    leaves = [{
        "division_id": "D6", "title": "Contract Agreement",
        "units": [
            {"kind": "preamble", "marker": None, "title": "Preamble", "page_start": 1, "page_end": 1,
             "text": "THIS AGREEMENT is made... WHEREAS the parties have agreed as follows:"},
            {"kind": "clause", "marker": "1", "title": "Definitions", "page_start": 2, "page_end": 2,
             "text": "1. DEFINITIONS In this Contract the following terms apply."},
            {"kind": "clause", "marker": "2", "title": "Scope", "page_start": 2, "page_end": 2,
             "text": "2. SCOPE The Contractor shall perform the Works."},
        ],
    }]
    out = _to_clause_dicts(leaves)
    titles = [c["clause_title"] for c in out]
    assert "Preamble" not in titles
    assert not any("WHEREAS" in c["clause_content"] for c in out)
    assert titles == ["Definitions", "Scope"]


def test_whole_front_matter_dropped_but_deep_whole_kept():
    """A 'whole' unit is an entire division with no clause structure. A FRONT-matter whole (stamp paper /
    cover / divider in the first few pages) is dropped from the clause projection; a 'whole' division that
    begins DEEPER in the bundle is KEPT, since it more likely holds real (if unstructured) content
    (_WHOLE_CLAUSE_MAX_PAGE). Both are still stored in full by the DB/file outputs."""
    leaves = [
        {"division_id": "A", "title": "Stamp Paper", "units": [
            {"kind": "whole", "marker": None, "title": "Stamp Paper",
             "page_start": 1, "page_end": 1, "text": "NON JUDICIAL e-STAMP ..."},
        ]},
        {"division_id": "B", "title": "Letter of Intent", "units": [
            {"kind": "clause", "marker": "1", "title": "Intent", "page_start": 3, "page_end": 3,
             "text": "1. We, Saifee Burhani Upliftment Trust, are pleased to inform you..."},
        ]},
        {"division_id": "C", "title": "Bill of Quantities Attached as Annexure C", "units": [
            {"kind": "whole", "marker": None, "title": "Bill of Quantities Attached as Annexure C",
             "page_start": 40, "page_end": 40, "text": "BILL OF QUANTITIES ATTACHED AS ANNEXURE C"},
        ]},
    ]
    out = _to_clause_dicts(leaves)
    titles = [c["clause_title"] for c in out]
    # front-matter stamp-paper 'whole' dropped; the real clause and the DEEP 'whole' both kept
    assert titles == ["Intent", "Bill of Quantities Attached as Annexure C"], titles
    assert not any("STAMP" in c["clause_content"].upper() for c in out)
