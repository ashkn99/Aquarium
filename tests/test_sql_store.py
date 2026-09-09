"""SqlCaseStore must round-trip a case identically to the in-memory
CaseStore for the same sequence of create/save/get calls -- it's a
drop-in replacement behind the same interface engine.py depends on."""
from __future__ import annotations

from sqlalchemy import event

from aqua_assistant.case.models import Observation
from aqua_assistant.case.sql_store import SqlCaseStore, _normalize_db_url


def test_normalize_rewrites_provider_urls_to_the_psycopg3_dialect():
    assert _normalize_db_url("postgres://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    assert _normalize_db_url("postgresql://u:p@host/db") == "postgresql+psycopg://u:p@host/db"


def test_normalize_leaves_sqlite_and_explicit_dialects_untouched():
    assert _normalize_db_url("sqlite:///cases.db") == "sqlite:///cases.db"
    assert _normalize_db_url("postgresql+psycopg://u:p@host/db") == "postgresql+psycopg://u:p@host/db"


def test_save_works_with_foreign_keys_enforced_and_no_kb_tables_populated(tmp_path):
    """Regression test for a real production bug: SQLite doesn't enforce
    foreign keys by default, so every other test in this file (and the
    manual local smoke test before deploying) silently passed even though
    observations.evidence_id/question_id used to carry a FK into the
    KB's evidence/questions tables -- tables this store never populates
    (the KB is always loaded fresh from YAML elsewhere; this database
    only ever holds cases/observations/case_feedback). Postgres enforces
    foreign keys unconditionally, so it broke on first deploy. This test
    explicitly turns SQLite's enforcement on (matching kb/loader.py's own
    pattern) to catch this class of bug locally instead of only in
    production."""
    db_path = tmp_path / "fk_enforced.db"
    store = SqlCaseStore(f"sqlite:///{db_path}")

    @event.listens_for(store._engine, "connect")
    def _enable_fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    case = store.create(entry_context="fish")
    case.add_observation(Observation(evidence_id="gasping", observed_state="true", question_id="ask_gasping"))
    store.save(case)  # must not raise a foreign-key violation

    fetched = store.get(case.id)
    assert fetched.observations[0].evidence_id == "gasping"


def _store(tmp_path):
    return SqlCaseStore(f"sqlite:///{tmp_path / 'test_cases.db'}")


def test_create_and_get_round_trip(tmp_path):
    store = _store(tmp_path)
    case = store.create(entry_context="fish")
    fetched = store.get(case.id)
    assert fetched.id == case.id
    assert fetched.entry_context == "fish"
    assert fetched.status == "open"
    assert fetched.observations == []
    assert fetched.concern_id is None


def test_concern_id_persists_once_set_after_creation(tmp_path):
    store = _store(tmp_path)
    case = store.create(entry_context="fish")
    assert store.get(case.id).concern_id is None

    case.concern_id = "fish_mortality"
    store.save(case)

    assert store.get(case.id).concern_id == "fish_mortality"


def test_save_persists_new_observations(tmp_path):
    store = _store(tmp_path)
    case = store.create()
    case.add_observation(Observation(evidence_id="gasping", observed_state="true", confidence=0.9))
    store.save(case)

    fetched = store.get(case.id)
    assert len(fetched.observations) == 1
    assert fetched.observations[0].evidence_id == "gasping"
    assert fetched.observations[0].observed_state == "true"
    assert fetched.observations[0].superseded is False


def test_reanswer_supersedes_the_earlier_observation_after_reload(tmp_path):
    store = _store(tmp_path)
    case = store.create()
    case.add_observation(Observation(evidence_id="gasping", observed_state="true"))
    store.save(case)

    reloaded = store.get(case.id)
    reloaded.add_observation(Observation(evidence_id="gasping", observed_state="false"))
    store.save(reloaded)

    final = store.get(case.id)
    assert len(final.observations) == 2
    assert final.answered_evidence_ids() == {"gasping"}
    active = final.active_observations()
    assert len(active) == 1
    assert active[0].observed_state == "false"
    superseded = [o for o in final.observations if o.superseded]
    assert len(superseded) == 1
    assert superseded[0].observed_state == "true"


def test_get_unknown_case_raises_key_error(tmp_path):
    store = _store(tmp_path)
    try:
        store.get("does-not-exist")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_feedback_round_trip(tmp_path):
    store = _store(tmp_path)
    case = store.create()
    store.save_feedback(case.id, helpful=True, comment="worked great")
    store.save_feedback(case.id, helpful=False)

    all_feedback = store.all_feedback()
    assert len(all_feedback) == 2
    helpful_flags = {f.helpful for f in all_feedback}
    assert helpful_flags == {True, False}


def test_all_case_ids_lists_every_created_case(tmp_path):
    store = _store(tmp_path)
    a = store.create()
    b = store.create()
    assert set(store.all_case_ids()) == {a.id, b.id}
