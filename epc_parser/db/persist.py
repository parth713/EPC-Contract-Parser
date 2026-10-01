"""Persist a staged-pipeline result into any SQL database.

The connection is a single SQLAlchemy URL (``--db-url`` / ``EPC_DB_URL``). An async driver is chosen
automatically, so the plain forms work:

    sqlite:///epc.db                 -> sqlite+aiosqlite:///epc.db          (default; zero setup)
    postgresql://user:pw@host/db     -> postgresql+asyncpg://...            (needs `asyncpg`)
    mysql://user:pw@host/db          -> mysql+aiomysql://...                (needs `aiomysql`)

Tables are created on first use (CREATE TABLE IF NOT EXISTS semantics via metadata.create_all). Each run
inserts one new document row and its full tree (divisions -> clauses -> sub-clauses); the contract-level
date-anchor registry and principal parties are stored as JSON on the document row. Nothing here is
application-specific.
"""
from __future__ import annotations

import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base, Clause, Division, Document, SubClause

log = logging.getLogger("epc.db")

# sync-driver prefix -> async-driver prefix
_ASYNC_DRIVERS = {
    "sqlite:": "sqlite+aiosqlite:",
    "postgresql:": "postgresql+asyncpg:",
    "postgres:": "postgresql+asyncpg:",
    "mysql:": "mysql+aiomysql:",
    "mariadb:": "mariadb+aiomysql:",
}


def to_async_url(db_url: str) -> str:
    """Upgrade a plain SQLAlchemy URL to its async driver, unless one is already specified (`+driver`)."""
    scheme = db_url.split("://", 1)[0]
    if "+" in scheme:            # already an explicit driver, e.g. postgresql+asyncpg
        return db_url
    for prefix, repl in _ASYNC_DRIVERS.items():
        if db_url.startswith(prefix):
            return repl + db_url[len(prefix):]
    return db_url


_NUL = re.compile("\x00")


def _clean(s):
    """Strip NUL bytes before persisting. PostgreSQL text/varchar/JSON cannot store a NUL (\\x00), and OCR
    of a damaged scan can emit them; SQLite tolerates them but truncates C-strings — so drop them for every
    backend. Leaves all other text untouched."""
    if isinstance(s, str):
        return _NUL.sub("", s)
    return s


def _display(u: dict, div_title: str) -> str:
    """Display clause number: "<marker> (<division title>)", matching runner._display_number so the DB and
    the file outputs agree. Imported lazily to avoid a hard dependency cycle."""
    from ..staged.runner import _display_number, _pretty
    return _display_number(u.get("marker"), _pretty(div_title))


async def _insert(session: AsyncSession, result: dict) -> int:
    doc = Document(
        source=_clean(result.get("source")),
        source_sha256=result.get("source_sha256"),
        page_count=result.get("page_count") or 0,
        status=result.get("status"),
        notes=_clean(result.get("notes")),
        pipeline=(result.get("extraction_metadata", {}) or {}).get("pipeline", "staged"),
        date_anchors=result.get("date_anchors") or {},
        parties=result.get("parties") or [],
        stats=(result.get("extraction_metadata") or {}),
        cost=result.get("cost") or {},
        review_queue=result.get("review") or [],
    )
    session.add(doc)
    await session.flush()   # assign doc.id

    for dseq, d in enumerate(result.get("leaves", [])):
        div = Division(
            document_id=doc.id,
            division_key=d.get("division_id"),
            division_type=d.get("division_type"),
            title=_clean(d.get("title")),
            running_header=_clean(d.get("running_header")),
            start_page=d.get("start_page"),
            end_page=d.get("end_page"),
            seq=dseq,
            flags=d.get("flags") or [],
        )
        session.add(div)
        await session.flush()   # assign div.id

        for cseq, u in enumerate(d.get("units", [])):
            clause = Clause(
                document_id=doc.id,
                division_id=div.id,
                unit_id=u.get("unit_id"),
                kind=u.get("kind"),
                marker=_clean(u.get("marker")),
                clause_number=_clean(_display(u, d.get("title") or "")),
                title=_clean(u.get("title")),
                description=_clean(u.get("clause_description") or ""),
                text=_clean(u.get("text") or ""),
                priority=u.get("priority"),
                risk_level=u.get("risk_level"),
                clause_type=u.get("clause_type"),
                page_start=u.get("page_start"),
                page_end=u.get("page_end"),
                seq=cseq,
            )
            session.add(clause)
            await session.flush()
            for sseq, c in enumerate(u.get("children") or []):
                session.add(SubClause(
                    clause_id=clause.id,
                    marker=_clean(c.get("marker")),
                    title=_clean(c.get("title")),
                    text=_clean(c.get("text") or ""),
                    seq=sseq,
                ))
    return doc.id


async def persist(result: dict, db_url: str, *, echo: bool = False) -> dict:
    """Create the schema if needed and insert `result` as one new document tree. Returns a small summary
    ({document_id, divisions, clauses, db_url})."""
    url = to_async_url(db_url)
    engine = create_async_engine(url, echo=echo)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        async with maker() as session:
            doc_id = await _insert(session, result)
            await session.commit()
        divisions = len(result.get("leaves", []))
        clauses = sum(len(d.get("units", [])) for d in result.get("leaves", []))
        log.info("[epc.db] persisted document %s: %d divisions, %d clauses -> %s",
                 doc_id, divisions, clauses, url.split("://", 1)[0])
        return {"document_id": doc_id, "divisions": divisions, "clauses": clauses, "db_url": url}
    finally:
        await engine.dispose()


def persist_sync(result: dict, db_url: str, *, echo: bool = False) -> dict:
    """Blocking convenience wrapper for non-async callers."""
    import asyncio
    return asyncio.run(persist(result, db_url, echo=echo))
