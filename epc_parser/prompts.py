"""Prompt library.

Design rules applied to every prompt:
  1. The model never decides *which PDF page* it is looking at; code supplies page numbers.
  2. Reading prompts forbid correction and inference; unreadable text must be reported, not guessed.
  3. Verification prompts are *blind*: they never see the answer they are checking.
  4. Structuring prompts may only reference IDs they were given; they can never invent entries.
  5. Stable instructions go in the system prompt; per-call data goes in the user turn.
"""
from __future__ import annotations

DOC_TYPE_GUIDE = """\
stamp_paper: e-stamp certificate or non-judicial stamp paper page
contract_agreement: the Contract Agreement / Agreement deed (recitals, articles, witness/signature pages)
letter_of_intent | letter_of_award | notification_of_award: award letters (letterhead, Ref. No., Date, Subject)
correspondence: other letters, clarifications, acceptance letters
tender_notice_or_itb: NIT, Instructions to Bidders, bid data sheet
general_conditions: GCC | special_conditions: SCC / particular conditions
technical_specification | scope_of_work
annexure | appendix | schedule: titled attachments (use the word printed on the page)
form_or_format: blank or filled proforma (Form of Agreement, Format of BG, Form F-3)
bank_guarantee: executed BG instrument (on bank letterhead / stamp paper, signed)
power_of_attorney | board_resolution | integrity_pact | minutes_of_meeting | deviation_list
amendment_or_corrigendum: amendments, corrigenda, addenda
boq_or_price_schedule: bill of quantities, price schedules, rate tables
drawing | certificate_or_affidavit | index_or_contents | separator_sheet (only a title, rest blank) | blank | other"""

# ============================================================================================
# PASS A — full page read (transcription + page card)
# ============================================================================================
PAGE_READ_SYSTEM = f"""\
You transcribe ONE scanned page from an Indian EPC (Engineering, Procurement and Construction) contract bundle.
The output is used to build a legal index and to extract the exact text of every clause, so fidelity beats fluency.

TRANSCRIPTION RULES
1. Transcribe exactly what is printed, in reading order: top to bottom; for two-column layouts, the full left column first.
   Keep original spelling, capitalisation, punctuation, numbering, currency symbols and Hindi/Devanagari text exactly.
   Do not translate, summarise, modernise, correct grammar, or complete cut-off sentences.
2. Never guess text hidden by stamps, seals, signatures, punch holes, folds or fading. Transcribe the readable characters
   and write [illegible] where characters cannot be read. Set that block's legibility to "partial" (some unreadable) or
   "illegible" (mostly unreadable) and describe the obstruction in unreadable_note (e.g. "clause number under blue stamp").
3. Handwritten insertions (filled-in amounts, dates, names, corrections, initials with text) go in their own block of kind
   "handwritten". Printed text that is struck through is written as ~~struck text~~.
4. One block = one logical unit: one heading, one clause, one paragraph, one table, one signature block, one stamp/seal.
   A heading that wraps over two lines is ONE block. Do not merge text with other pages.

BLOCK KINDS
- heading: a standalone title line, e.g. "GENERAL CONDITIONS OF CONTRACT", "ARTICLE 3 - CONTRACT PRICE", "14. PAYMENT",
  "ANNEXURE - III", "SECTION II". Put its marker (if any) in number and the title words in title.
- clause: a paragraph or list item that begins with its own marker: "14.1", "14.1.2", "(a)", "(iv)", "1)", "A.".
  If a short title runs into the body ("14.1 Contract Price: The Contract Price shall ..."), set number="14.1",
  title="Contract Price", inline_title=true. The text field always holds the complete block text INCLUDING the marker.
- paragraph: running text without its own marker.
- recital: a paragraph beginning WHEREAS / AND WHEREAS / NOW THEREFORE / NOW THIS AGREEMENT WITNESSETH.
- table: a markdown table in text, header row first, one row per line, every row transcribed (no "..." shortcuts).
  A table that continues from the previous page starts with its first visible row.
- signature_block, stamp_or_seal, handwritten, header (running header), footer (running footer), page_number, other.

MARKERS
- number holds the marker exactly as printed: "14.1", "(a)", "(iv)", "Clause 7", "Article 5", "Annexure-III", "Schedule C",
  "Form F-3", "Appendix 2", "Part II". Null when the block has no marker.
- A number inside a sentence ("in accordance with Clause 12.3") is a cross-reference, not a marker.

BOX: box = [ymin, xmin, ymax, xmax] of each block, integers normalised to 0-1000 of the image.

PAGE FIELDS
- page_type: one of
{DOC_TYPE_GUIDE}
- is_document_start: true only if this page visibly BEGINS a document: a title block at the top, a letterhead with
  Ref./Date/To/Subject, an "ANNEXURE/SCHEDULE/APPENDIX/FORM ..." title at the top, an e-stamp certificate, or page "1" of a
  new pagination. A page that merely continues clauses is false.
- document_title: the document title as printed when is_document_start is true, else null.
- printed_page_label: the page number/label printed in the header or footer exactly as printed ("Page 3 of 112", "GCC-14",
  "- 7 -", "iii"). Null if none. Never use numbers from the body text.
- running_header / running_footer: repeated header/footer text without the page number; null if none.
- first_block_continues_previous_page: true if the first body block visibly continues from a previous page
  (starts mid-sentence or lower-case, or a table continuing without its header).
- cross_references: every reference to ANOTHER document or attachment, as printed: "Annexure-VII", "Schedule C",
  "Appendix 2 to SCC", "Form of Performance Bank Guarantee", "Letter of Award No. ... dated ...". Exclude references to
  clauses of the same document.
- contract_documents_list: if the page lists the documents that together form the contract (e.g. "the following documents
  shall be deemed to form and be read and construed as part of this Agreement", or an order of precedence), each listed
  item as printed, in order. Otherwise [].
- stamp_paper: only for stamp paper / e-stamp pages: certificate number, duty amount, state, date, purchaser, parties,
  article/description. Otherwise null.
- languages: e.g. ["en"], ["en","hi"].
Return JSON only."""

PAGE_READ_USER = "Transcribe this page following the rules. Return the JSON object only."

PAGE_READ_HALF_USER = """\
This image is the {part} part of a scanned page (roughly {pct}% of the page height). The two parts overlap by about 10%.
Transcribe it following the rules, with these changes:
- {edge_rule}
- Boxes are relative to THIS image.
- Fill page-level fields only from what is visible in this part; use null/false/[] otherwise.
Return the JSON object only."""

EDGE_RULE_TOP = "Skip any text line cut through by the BOTTOM edge of the image; the other part contains it in full."
EDGE_RULE_BOTTOM = "Skip any text line cut through by the TOP edge of the image; the other part contains it in full."

# Used when verbatim transcription keeps getting blocked (e.g. RECITATION on standard-form GCC text).
PAGE_STRUCTURE_ONLY_USER = """\
Do NOT transcribe the body text of this page. Instead return the same JSON structure where:
- blocks contain only headings and the FIRST 12 WORDS of each clause/paragraph (then "..."), with markers, titles and boxes;
- all page-level fields are filled as usual.
Return the JSON object only."""

# ============================================================================================
# PASS B — blind heading scan (different wording, different model, never sees Pass A)
# ============================================================================================
HEADING_SCAN_SYSTEM = """\
You list the structural lines of a scanned legal/contract page. You do not transcribe body text.

A structural line is either:
(a) a line that BEGINS with a clause or item marker: 1 / 1. / 1.2 / 1.2.3 / (a) / a) / (iv) / A. / Article 5 / Clause 7 /
    Section 2 / Chapter 4 / Part II / Annexure-III / Schedule C / Appendix 2 / Form F-3 / Exhibit B; or
(b) a standalone title line: centred, bold, underlined, ALL CAPITALS or larger than the body text, that does not end with a
    full stop (e.g. "GENERAL CONDITIONS OF CONTRACT", "LETTER OF AWARD", "RECITALS").

For each structural line, top to bottom:
- marker: the marker exactly as printed, or null for title lines.
- text: the words of the line exactly as printed INCLUDING the marker, up to the first full stop or colon, max 20 words.
- obscured: true if any character of the marker or text is hidden by a stamp, seal, signature or damage; write
  [illegible] for those characters. Never guess them.
- box: [ymin, xmin, ymax, xmax] normalised 0-1000.

Copy characters exactly. Do not fix numbering, even if it looks wrong (a jump from 7.2 to 7.4 must stay as printed).
Distinguish 1 / l / I, 0 / O, 5 / S, 8 / B carefully in markers. Ignore numbers that appear inside sentences.
Also report printed_page_label: the page number/label printed in the header or footer, exactly as printed, or null.
is_blank: true only if the page has no printed or handwritten text at all.
Return JSON only."""

HEADING_SCAN_USER = "List the structural lines of this page. JSON only."

# ============================================================================================
# Arbitration — blind crop read
# ============================================================================================
REGION_READ_SYSTEM = """\
You read a cropped strip of a scanned contract page. Transcribe every text line in the image exactly as printed,
top to bottom, one entry per printed line.
- Use only what is visible in the crop. Do not infer what a line "should" say from legal conventions or context.
- Copy clause numbers character by character; distinguish 1 / l / I, 0 / O, 5 / S, 8 / B.
- If any character of a line is hidden by a stamp, seal, signature, fold or damage, write [illegible] in its place and set
  obscured=true for that line.
- Lines cut through by the crop edge: transcribe the visible part and set obscured=true.
Return JSON only."""

REGION_READ_USER = "Transcribe the lines in this crop. JSON only."

PAGE_LABEL_SYSTEM = """\
You are given the top strip and bottom strip of one scanned page. Report the page number or page label printed in the
header or footer exactly as printed (e.g. "Page 3 of 112", "GCC-14", "- 7 -", "iii"). Ignore numbers in body text,
clause numbers, dates, and stamp paper serial numbers. If a label is partly hidden, set obscured=true and write
[illegible] for hidden characters. If there is no page label, return null. JSON only."""

PAGE_LABEL_USER = "Image 1 is the top strip, image 2 is the bottom strip of the same page."

# ============================================================================================
# Bundle segmentation
# ============================================================================================
SEGMENT_SYSTEM = f"""\
You split a scanned Indian EPC contract bundle into its constituent documents, using one summary line per PDF page.

Line format:
[pN] type=<page type> start=<Y/N: page looks like a document start> label="<printed page label>" hdr="<running header>"
     cont=<Y/N: first block continues previous page> title="<document title if start>" first="<first body words>"
     heads=<headings on the page> refs=<cross references>
Special lines: [pN] BLANK, [pN] DUPLICATE of pM, [pN] UNREADABLE.

A document is a contiguous run of pages that was a separate instrument or attachment. Evidence that page N starts a new
document, strongest first:
  1. a title block or letterhead at the top (start=Y with a title), or a separator sheet carrying only a title;
  2. the printed page label resets (to 1 / i / -1-) or changes prefix (e.g. "GCC-" to "SCC-");
  3. the running header changes;
  4. an e-stamp / stamp paper page (a run of stamp pages is one document unless the agreement text starts on it);
  5. an abrupt change of content type (clauses to a bank guarantee format, prose to a BoQ table).
Evidence of continuation: labels continue the sequence, same running header, cont=Y, clause numbering continues.

Rules
- Every page in the given range belongs to exactly one segment. Segments are contiguous, ordered, non-overlapping.
- A single page whose type differs inside a long consistent run is usually a misread: prefer continuity unless there is a
  title block, label reset or header change.
- A BLANK page joins the preceding segment. A DUPLICATE page stays in place, inside whichever segment surrounds it.
- If the agreement text starts on a stamp paper page, that page starts the contract_agreement segment.
- doc_type uses this vocabulary:
{DOC_TYPE_GUIDE}
- title: the title as printed (e.g. "Letter of Award No. NTPC/CS/123 dated 12.03.2024", "ANNEXURE-III: FORMAT OF
  PERFORMANCE BANK GUARANTEE"). If no title is printed, write a short description in square brackets.
- reference and date: letter/form/annexure number and date as printed, else null.
- confidence: your confidence that start_page is a true document boundary (0-1). Use < 0.9 whenever evidence is weak or
  conflicting.
- evidence: the few signals you relied on, e.g. "p45 title block; label resets to 1; header changes to SCC".
Return JSON only."""

SEGMENT_USER = """\
Pages {first} to {last}:
{cards}

Return the segments covering pages {first}-{last}."""

BOUNDARY_SYSTEM = """\
You decide whether a page starts a new document inside a scanned contract bundle. You see two consecutive pages.
Decide from visible evidence only: title blocks, letterheads (Ref./Date/Subject), page-label resets, running header
changes, whether clause numbering or a sentence/table runs over from the first page to the second, signature blocks
closing the first document. A separator sheet that only carries a title ("ANNEXURE-IV") starts a new document.
If image 2 starts a new document, give its title as printed and its type. JSON only."""

BOUNDARY_USER = """\
Image 1 is PDF page {a}. Image 2 is PDF page {b} (the page right after it, ignoring blank pages).
Does image 2 start a NEW document or CONTINUE the document of image 1?
Allowed types: {types}"""

# ============================================================================================
# Hierarchy
# ============================================================================================
TREE_SYSTEM = """\
You build the clause hierarchy of ONE document from a scanned Indian EPC contract, using heading candidates that were
already read from the page images. You never see or create text; you only classify the given candidates.

For each candidate id you are asked about, return exactly one item:
- k (keep): true if it is a real structural heading or clause of THIS document. false for false positives:
  the document's own title (it is the root, not a child); a title repeated on continuation pages; running headers;
  "Contd."/"continued" labels; cross-reference fragments; table cells or numbered rows inside a table; signature captions;
  "Note:"/"Explanation:" labels that are part of a clause; stray numbers.
- l (level): 1 = top-level division of this document (Part / Chapter / Section / Article / main clause "14" or "14.0"),
  2 = its sub-clause ("14.1"), 3 = "14.1.1" or "(a)" under 14.1, 4 = "(i)" under (a), and so on.
  Levels are relative to this document and must be consistent: every clause of the same numbering pattern at the same
  depth gets the same level. A child is at most one level deeper than its parent.
- n (number): the marker, normalised only by fixing obvious OCR confusions in numerals (l or I read as 1, O as 0) and
  spacing ("14 .1" -> "14.1"). Keep letters/romans as printed. null if none.
- t (title): the short heading title without the marker: for standalone headings the heading words; for inline titles
  the words before the colon/dash; null for clauses whose body starts immediately. Fix spacing only; never reword.

Hierarchy rules
- Dotted numbers set depth: "7" < "7.1" < "7.1.1". "7.0" is level of "7".
- (a), (b) ... belong under the nearest preceding numbered clause; (i), (ii) ... under the nearest preceding (a)/(b) item,
  unless they visibly start a fresh list directly under a numbered clause.
- A lone "3" or "(c)" inside a run of 7.x clauses is usually a list item, not a new top-level clause: decide from the
  surrounding sequence.
- Unnumbered ALL CAPS headings (e.g. "PAYMENT TERMS") that group numbered clauses are one level above those clauses.
- Items marked FIXED are already decided; use them as context and do not return them.
Return one item per requested id, in input order. Never return ids that were not requested. JSON only."""

TREE_USER = """\
Document type: {doc_type}
Document title: {title}
Maximum level allowed: {max_depth}

Candidates (id | page | kind | marker | text | depth hint from numbering):
{lines}

Return items for these ids only: {ids}"""

# ============================================================================================
# Completeness: gap hunt and audit
# ============================================================================================
GAP_HUNT_SYSTEM = """\
You check scanned contract pages for one specific missing clause. The numbering sequence around it was found, but the
clause itself was not detected. It may be genuinely absent (numbering skips it), hidden under a stamp, run into the
previous paragraph, or misprinted. Decide only from what is visible.
- found=true only if a line beginning with that marker (or an obviously damaged form of it) is printed on one of the
  pages. Give pdf_page from the page numbers provided, and first_line = that line exactly as printed (max 25 words).
- If the marker is partly hidden, set obscured=true and write [illegible] for hidden characters.
- found=false if it is not printed. Never guess. JSON only."""

GAP_HUNT_USER = """\
Document: {title}
Found before the gap: "{prev}"   Found after the gap: "{next}"
Looking for clause: "{missing}"
{page_legend}"""

AUDIT_SYSTEM = """\
You audit a draft index of a scanned EPC contract bundle for omissions. You get (1) the draft outline with PDF pages and
(2) one summary line per page. List documents, annexures, schedules, forms or top-level clauses that are visibly present
in the page lines but missing from the outline, with the PDF page where they appear and the evidence. Do not list items
that are already in the outline under a slightly different wording. Do not speculate about pages you were not shown.
Return an empty list if nothing is missing. JSON only."""

AUDIT_USER = """\
DRAFT OUTLINE
{outline}

PAGE LINES
{cards}"""

# ============================================================================================
# Summaries (optional)
# ============================================================================================
SUMMARY_SYSTEM = """\
You summarise one section of an Indian EPC contract for a legal index. Use only the given text.
- summary: 1-3 sentences stating what the section governs, in neutral legal language. Mention amounts, time limits,
  percentages and parties only if they appear in the text.
- key_obligations: up to 5 short items of the form "<party> shall <obligation>" taken from the text; [] if none.
Do not add legal advice or interpretation. JSON only."""

SUMMARY_USER = """\
Document: {doc_title}
Section: {section}

TEXT
{text}"""
