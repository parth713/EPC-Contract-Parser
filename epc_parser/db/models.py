"""SQLAlchemy models for persisting a parsed EPC contract to any SQL database.

A parsed document -> divisions -> clauses -> sub-clauses schema, deliberately STANDALONE: no external
ids or foreign keys, and portable JSON columns (not Postgres-only JSONB) so the same models
run on SQLite, PostgreSQL and MySQL unchanged. The contract-level date-anchor registry and principal
parties are stored as JSON on the document row.

"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "epc_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str | None] = mapped_column(String(1024))
    source_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str | None] = mapped_column(String(32))
    notes: Mapped[str | None] = mapped_column(Text)
    pipeline: Mapped[str | None] = mapped_column(String(32), default="staged")
    date_anchors: Mapped[dict | None] = mapped_column(JSON)
    parties: Mapped[list | None] = mapped_column(JSON)
    stats: Mapped[dict | None] = mapped_column(JSON)
    cost: Mapped[dict | None] = mapped_column(JSON)
    review_queue: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    divisions: Mapped[list["Division"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Division.seq")
    clauses: Mapped[list["Clause"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Clause.seq")


class Division(Base):
    __tablename__ = "epc_divisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("epc_documents.id", ondelete="CASCADE"), index=True)
    division_key: Mapped[str | None] = mapped_column(String(32))   # e.g. "DIV-01"
    division_type: Mapped[str | None] = mapped_column(String(64))
    title: Mapped[str | None] = mapped_column(Text)
    running_header: Mapped[str | None] = mapped_column(Text)
    start_page: Mapped[int | None] = mapped_column(Integer)
    end_page: Mapped[int | None] = mapped_column(Integer)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    flags: Mapped[list | None] = mapped_column(JSON)

    document: Mapped["Document"] = relationship(back_populates="divisions")
    clauses: Mapped[list["Clause"]] = relationship(
        back_populates="division", order_by="Clause.seq")


class Clause(Base):
    __tablename__ = "epc_clauses"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("epc_documents.id", ondelete="CASCADE"), index=True)
    division_id: Mapped[int | None] = mapped_column(
        ForeignKey("epc_divisions.id", ondelete="CASCADE"), index=True)
    unit_id: Mapped[str | None] = mapped_column(String(32))
    kind: Mapped[str | None] = mapped_column(String(32))
    marker: Mapped[str | None] = mapped_column(String(128))
    clause_number: Mapped[str | None] = mapped_column(Text)   # display form, e.g. "5 (Termination)"
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str | None] = mapped_column(Text)            # own text (sub-clauses excluded)
    priority: Mapped[str | None] = mapped_column(String(16))
    risk_level: Mapped[str | None] = mapped_column(String(16))
    clause_type: Mapped[str | None] = mapped_column(String(64))
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    seq: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped["Document"] = relationship(back_populates="clauses")
    division: Mapped["Division"] = relationship(back_populates="clauses")
    subclauses: Mapped[list["SubClause"]] = relationship(
        back_populates="clause", cascade="all, delete-orphan", order_by="SubClause.seq")


class SubClause(Base):
    __tablename__ = "epc_subclauses"

    id: Mapped[int] = mapped_column(primary_key=True)
    clause_id: Mapped[int] = mapped_column(ForeignKey("epc_clauses.id", ondelete="CASCADE"), index=True)
    marker: Mapped[str | None] = mapped_column(String(128))
    title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str | None] = mapped_column(Text)
    seq: Mapped[int] = mapped_column(Integer, default=0)

    clause: Mapped["Clause"] = relationship(back_populates="subclauses")
