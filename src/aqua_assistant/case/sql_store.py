"""SQL-backed case store -- same create/get/save interface as
`store.py::CaseStore`, so `engine.py` (which only depends on that narrow
interface) needs no changes to use this instead.

Persists into `kb/models.py`'s `CaseORM`/`ObservationORM` tables, which
already existed for exactly this purpose but were unused until now. Uses
its own SQLAlchemy engine/session, independent of the KB's own (the KB is
always loaded fresh from YAML fixtures; cases are the only thing meant to
persist across restarts) -- same `create_engine` pattern `kb/loader.py`
already uses.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from aqua_assistant.kb.models import Base, CaseFeedbackORM, CaseORM, ObservationORM

from .models import Case, Observation


def _normalize_db_url(db_url: str) -> str:
    """Managed Postgres providers (Neon, Supabase, Render itself) hand out
    plain `postgres://`/`postgresql://` URLs, which SQLAlchemy resolves to
    the psycopg2 dialect by default -- but this project depends on
    psycopg3 (`psycopg[binary]`, the `postgres` extra), so that default
    guess fails at connect time. Rewriting to the explicit
    `postgresql+psycopg://` dialect makes a copy-pasted provider URL work
    without the user having to know or edit that detail."""
    if db_url.startswith("postgres://"):
        return "postgresql+psycopg://" + db_url[len("postgres://"):]
    if db_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + db_url[len("postgresql://"):]
    return db_url


_OWN_TABLES = [CaseORM.__table__, ObservationORM.__table__, CaseFeedbackORM.__table__]


def _add_missing_columns(engine) -> None:
    """`create_all()` only creates tables that don't exist yet -- it never
    alters one that's already there, so a column added to a model after
    the table was first created on a live database (e.g. Case.concern_id,
    added after this store's first deploy) silently never appears there,
    and every insert referencing it fails. Since every column this
    project has added so far is a plain nullable one, a straightforward
    `ADD COLUMN` closes that gap without a full migration tool.
    ponytail: nullable ADD COLUMN only -- no renames, drops, type changes,
    or NOT NULL columns; reach for a real migration tool (e.g. Alembic)
    if a future change needs any of those."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in _OWN_TABLES:
            if table.name not in existing_tables:
                continue
            existing_columns = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing_columns:
                    ddl_type = column.type.compile(dialect=engine.dialect)
                    conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl_type}"))


class SqlCaseStore:
    def __init__(self, db_url: str) -> None:
        self._engine = create_engine(_normalize_db_url(db_url))
        # Scoped to just these 3 tables, not Base.metadata as a whole:
        # this database is the persistent case store, not a KB copy --
        # creating the KB's own tables here would leave them empty
        # forever (the KB is always loaded fresh from YAML elsewhere).
        Base.metadata.create_all(self._engine, tables=_OWN_TABLES)
        _add_missing_columns(self._engine)

    def create(self, entry_context: str | None = None) -> Case:
        case = Case(entry_context=entry_context)
        now = datetime.now(timezone.utc).isoformat()
        with Session(self._engine) as session:
            session.add(
                CaseORM(id=case.id, created_at=now, updated_at=now, status=case.status, entry_context=entry_context)
            )
            session.commit()
        return case

    def get(self, case_id: str) -> Case:
        with Session(self._engine) as session:
            row = session.get(CaseORM, case_id)
            if row is None:
                raise KeyError(case_id)
            obs_rows = (
                session.query(ObservationORM)
                .filter(ObservationORM.case_id == case_id)
                .order_by(ObservationORM.id)
                .all()
            )
            observations = [
                Observation(
                    evidence_id=o.evidence_id,
                    observed_state=o.observed_state,
                    confidence=o.confidence,
                    raw_value=o.raw_value,
                    source=o.source,
                    question_id=o.question_id,
                    answered_at=datetime.fromisoformat(o.answered_at),
                    superseded=o.superseded,
                )
                for o in obs_rows
            ]
            return Case(
                id=row.id,
                status=row.status,
                observations=observations,
                entry_context=row.entry_context,
                concern_id=row.concern_id,
            )

    def save(self, case: Case) -> None:
        """Persists whatever is new (`case.observations` grows by append
        only) and re-syncs `superseded` on rows already persisted: a
        re-answer flips an earlier same-evidence row's flag in-memory
        (see `Case.add_observation`), and that flip must reach the DB
        too, not just newly-appended rows."""
        with Session(self._engine) as session:
            existing_rows = (
                session.query(ObservationORM)
                .filter(ObservationORM.case_id == case.id)
                .order_by(ObservationORM.id)
                .all()
            )
            for row, obs in zip(existing_rows, case.observations):
                if row.superseded != obs.superseded:
                    row.superseded = obs.superseded

            for obs in case.observations[len(existing_rows):]:
                session.add(
                    ObservationORM(
                        case_id=case.id,
                        evidence_id=obs.evidence_id,
                        observed_state=obs.observed_state,
                        raw_value=obs.raw_value,
                        confidence=obs.confidence,
                        source=obs.source,
                        question_id=obs.question_id,
                        answered_at=obs.answered_at.isoformat(),
                        superseded=obs.superseded,
                    )
                )

            row = session.get(CaseORM, case.id)
            row.status = case.status
            row.concern_id = case.concern_id
            row.updated_at = datetime.now(timezone.utc).isoformat()
            session.commit()

    def save_feedback(self, case_id: str, helpful: bool, comment: str | None = None) -> None:
        with Session(self._engine) as session:
            session.add(
                CaseFeedbackORM(
                    case_id=case_id,
                    helpful=helpful,
                    comment=comment,
                    submitted_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            session.commit()

    def all_feedback(self) -> list[CaseFeedbackORM]:
        with Session(self._engine) as session:
            return session.query(CaseFeedbackORM).all()

    def all_case_ids(self) -> list[str]:
        with Session(self._engine) as session:
            return [row.id for row in session.query(CaseORM.id).all()]
