"""build_report() must summarize whatever is actually in the case store,
recomputed live through the engine -- never a stale/denormalized copy."""
from __future__ import annotations

from aqua_assistant.analytics.report import build_report
from aqua_assistant.case.sql_store import SqlCaseStore
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.kb.loader import load_knowledge_base


def _store(tmp_path):
    return SqlCaseStore(f"sqlite:///{tmp_path / 'report_test.db'}")


def test_empty_store_produces_zeroed_summary(tmp_path):
    kb = load_knowledge_base()
    store = _store(tmp_path)
    engine = AquariumInferenceEngine(kb, store=store)

    report = build_report(store, engine)
    assert report["cases"] == []
    assert report["summary"]["num_cases"] == 0


def test_report_reflects_a_real_completed_case(tmp_path):
    kb = load_knowledge_base()
    store = _store(tmp_path)
    engine = AquariumInferenceEngine(kb, store=store)

    case_id = engine.start_case(entry_context="fish")
    engine.answer(case_id, "white_spots", state="true")
    store.save_feedback(case_id, helpful=True, comment="spot on")

    report = build_report(store, engine)
    assert report["summary"]["num_cases"] == 1
    case_row = report["cases"][0]
    assert case_row["num_questions"] == 1
    assert case_row["top_problem"] == "ich"
    assert case_row["should_stop"] is True
    assert report["summary"]["num_feedback"] == 1
    assert report["summary"]["feedback_helpful_rate"] == 100.0


def test_case_with_no_answers_is_excluded(tmp_path):
    kb = load_knowledge_base()
    store = _store(tmp_path)
    engine = AquariumInferenceEngine(kb, store=store)
    engine.start_case(entry_context="fish")  # started, never answered anything

    report = build_report(store, engine)
    assert report["cases"] == []


def test_safety_alert_turn_is_detected(tmp_path):
    kb = load_knowledge_base()
    store = _store(tmp_path)
    engine = AquariumInferenceEngine(kb, store=store)

    case_id = engine.start_case(entry_context="unsure")
    engine.answer(case_id, "ammonia_ppm", raw_value=3.0)

    report = build_report(store, engine)
    assert report["cases"][0]["safety_alert_turn"] == 1
    assert report["summary"]["cases_with_safety_alert"] == 1
