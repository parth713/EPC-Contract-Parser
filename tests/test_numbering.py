from epc_parser.numbering import (attachment_key, extract_leading_marker, looks_degenerate, normalize_marker,
                                  parse_label, sibling_gaps)


def test_markers():
    assert normalize_marker("14 .1") == "14.1"
    assert normalize_marker("l4.1") == "14.1"
    assert normalize_marker("7.0") == "7"
    assert normalize_marker("Clause 7") == "7"
    assert normalize_marker("iv)") == "(iv)"
    assert normalize_marker("Annexure - III") == "annexure:3"
    assert normalize_marker("Annex II") == "annexure:2"


def test_gaps():
    assert sibling_gaps(["7.1", "7.2", "7.4"], 4) == [("7.2", "7.3", "7.4")]
    assert sibling_gaps(["(a)", "(b)", "(d)"], 4) == [("(b)", "(c)", "(d)")]
    assert sibling_gaps(["(i)", "(ii)", "(iv)"], 4) == [("(ii)", "(iii)", "(iv)")]
    assert sibling_gaps(["7.1", "7.9"], 4) == []  # large jump: renumbering, not missed clauses
    assert sibling_gaps(["7.2", "8.1"], 4) == []


def test_labels_and_refs():
    assert parse_label("Page 3 of 112") == ("", 3)
    assert parse_label("GCC-14") == ("gcc", 14)
    assert parse_label("iii") == ("roman", 3)
    assert attachment_key("ANNEXURE-III: Format of BG") == ("annexure", "3")
    assert attachment_key("Form of Agreement") is None
    assert extract_leading_marker("(iv) the Contractor") == "(iv)"


def test_degenerate():
    assert looks_degenerate("\n".join(["| 1 | Cement | bag |"] * 8))
    assert not looks_degenerate("a line here\nanother line here")
