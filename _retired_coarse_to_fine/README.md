# Retired: standalone coarse-to-fine stage scripts

These six modules were the original **coarse-to-fine** TOC/extraction lineage
(`build_toc` = `stage1_macro → stage1_refine → stage2_anchors → stage2_gaphunt`, plus
`stage3_extract`). They were file-based (each stage read/wrote intermediate JSON in an output dir) and
run as separate `python -m epc_parser.build_toc` / `python -m epc_parser.stage3_extract` commands.

They have been **superseded by `epc_parser/staged/`** — an in-memory, resynced version of the same
lineage (`macro → refine → anchors → gaphunt → extract`) plus the net-new `dates` (contract date-anchor
registry + principal parties) and `enrich` (clause naming / summary / priority / risk) passes, and
optional SQL persistence (`epc_parser/db/`). Run it with:

```bash
python -m epc_parser.staged contract.pdf -o out/ [--db-url sqlite:///epc.db]
```

Nothing in the live package imports these files. They are kept here only for reference and are safe to
delete. The separate **flagship** pipeline (`python -m epc_parser` → `pipeline.py`, with
`stage_pages/segment/tree/text/checks` and A/B blind reads + arbitration) is unaffected and still lives
in `epc_parser/`.
