
from __future__ import annotations

import datetime
import re

from sqlalchemy import create_engine, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from toxion.db.models import (STATUS_CLAIMED, STATUS_NEW, STATUS_REJECTED, STATUS_TRAINED, Base, Molecule,
                              NameAlias, RetrainRun)


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())[:512]


def normalize_url(url: str) -> str:
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


class MoleculeStore:
    def __init__(self, url: str):
        url = normalize_url(url)
        kwargs: dict = {"pool_pre_ping": True}
        if url.startswith("postgresql"):
            kwargs["connect_args"] = {"connect_timeout": 15}
        self.engine = create_engine(url, **kwargs)
        Base.metadata.create_all(self.engine)

    def get_molecule(self, canonical_smiles: str) -> Molecule | None:
        with Session(self.engine, expire_on_commit=False) as s:
            return s.scalar(select(Molecule).where(Molecule.canonical_smiles == canonical_smiles))

    def add_molecule(self, canonical_smiles: str, first_name: str, pubchem_cid: int | None = None,
                     ad_similarity: float | None = None, pred_json: str | None = None) -> tuple[Molecule, bool]:
        with Session(self.engine, expire_on_commit=False) as s:
            existing = s.scalar(select(Molecule).where(Molecule.canonical_smiles == canonical_smiles))
            if existing is not None:
                return existing, False
            mol = Molecule(canonical_smiles=canonical_smiles, first_name=first_name, pubchem_cid=pubchem_cid,
                           ad_similarity=ad_similarity, pred_json=pred_json, status=STATUS_NEW)
            s.add(mol)
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                return s.scalar(select(Molecule).where(Molecule.canonical_smiles == canonical_smiles)), False
            return mol, True

    def add_alias(self, name: str, canonical_smiles: str, pubchem_cid: int | None = None) -> None:
        key = normalize_name(name)
        with Session(self.engine) as s:
            if s.get(NameAlias, key) is None:
                s.add(NameAlias(name_normalized=key, canonical_smiles=canonical_smiles, pubchem_cid=pubchem_cid))
                try:
                    s.commit()
                except IntegrityError:
                    s.rollback()

    def lookup_alias(self, name: str) -> NameAlias | None:
        with Session(self.engine, expire_on_commit=False) as s:
            return s.get(NameAlias, normalize_name(name))

    def count_new(self) -> int:
        with Session(self.engine) as s:
            return int(s.scalar(select(func.count()).select_from(Molecule).where(Molecule.status == STATUS_NEW)))

    def claim_batch(self, n: int) -> list[Molecule]:
        with Session(self.engine, expire_on_commit=False) as s:
            rows = list(s.scalars(
                select(Molecule).where(Molecule.status == STATUS_NEW).order_by(Molecule.id)
                .limit(n).with_for_update(skip_locked=True)
            ))
            if len(rows) < n:
                s.rollback()
                return []
            for r in rows:
                r.status = STATUS_CLAIMED
            s.commit()
            return rows

    def set_status(self, ids: list[int], status: str) -> None:
        if not ids:
            return
        with Session(self.engine) as s:
            s.execute(update(Molecule).where(Molecule.id.in_(ids)).values(status=status))
            s.commit()

    def release(self, ids: list[int]) -> None:
        self.set_status(ids, STATUS_NEW)

    def trained_smiles(self, limit: int) -> list[str]:
        if limit <= 0:
            return []
        with Session(self.engine) as s:
            return list(s.scalars(select(Molecule.canonical_smiles).where(Molecule.status == STATUS_TRAINED)
                                  .order_by(Molecule.id.desc()).limit(limit)))

    def record_run(self, **fields) -> None:
        with Session(self.engine) as s:
            s.add(RetrainRun(finished_at=datetime.datetime.now(datetime.timezone.utc), **fields))
            s.commit()

    def status_counts(self) -> dict[str, int]:
        with Session(self.engine) as s:
            rows = s.execute(select(Molecule.status, func.count()).group_by(Molecule.status)).all()
        return {k: int(v) for k, v in rows}


__all__ = ["MoleculeStore", "normalize_name", "normalize_url", "STATUS_TRAINED", "STATUS_REJECTED"]
