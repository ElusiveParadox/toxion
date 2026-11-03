
from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

STATUS_NEW, STATUS_CLAIMED, STATUS_TRAINED, STATUS_REJECTED = "new", "claimed", "trained", "rejected"


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class Base(DeclarativeBase):
    pass


class Molecule(Base):
    __tablename__ = "molecules"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_smiles: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    pubchem_cid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=STATUS_NEW, index=True)
    ad_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    pred_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class NameAlias(Base):
    __tablename__ = "name_aliases"
    name_normalized: Mapped[str] = mapped_column(String(512), primary_key=True)
    canonical_smiles: Mapped[str] = mapped_column(Text, nullable=False)
    pubchem_cid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RetrainRun(Base):
    __tablename__ = "retrain_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    n_molecules: Mapped[int] = mapped_column(Integer, default=0)
    decision: Mapped[str] = mapped_column(String(16), default="aborted")
    val_auc_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_auc_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    ece_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    ece_after: Mapped[float | None] = mapped_column(Float, nullable=True)
