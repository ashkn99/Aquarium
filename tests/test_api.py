"""End-to-end tests for the HTTP API (api/app.py) -- mirrors
test_end_to_end_case.py's flow, but driven through HTTP instead of
calling the engine directly, since this layer is purely delivery on top
of the same engine."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from .conftest import neutral_default


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api_test.db'}")
    from aqua_assistant.api.app import _case_creation_log, create_app

    _case_creation_log.clear()  # module-level rate-limit state; isolate each test
    return TestClient(create_app())


def _drive_to_completion(client, entry_context=None, max_turns=70):
    # 70, not 40: a fully-neutral answer script carries no strong signal
    # (same as the "ambiguous" scenarios in the v2 behavioral evaluation),
    # so the engine legitimately runs most of the KB before a weak stop --
    # a known, already-reported characteristic, not a bug in this API layer.
    from aqua_assistant.kb.loader import load_knowledge_base

    kb = load_knowledge_base()
    case_id = client.post("/api/cases", json={"entry_context": entry_context}).json()["case_id"]
    status = client.get(f"/api/cases/{case_id}/status").json()
    for _ in range(max_turns):
        if status["should_stop"]:
            return case_id, status
        q = status["next_question"]
        kind, val = neutral_default(kb, q["evidence_id"])
        payload = {"question_id": q["question_id"], "evidence_id": q["evidence_id"]}
        payload["raw_value" if kind == "raw_value" else "state"] = val
        status = client.post(f"/api/cases/{case_id}/answers", json=payload).json()
    raise AssertionError("case did not converge within max_turns")


def test_list_entry_contexts(client):
    resp = client.get("/api/entry-contexts")
    assert resp.status_code == 200
    ids = {ec["id"] for ec in resp.json()}
    assert ids == {"fish", "water", "plants", "aquarium_environment", "unsure"}


def test_start_case_and_get_status(client):
    resp = client.post("/api/cases", json={"entry_context": "fish"})
    assert resp.status_code == 200
    case_id = resp.json()["case_id"]

    status = client.get(f"/api/cases/{case_id}/status").json()
    assert status["case_id"] == case_id
    assert status["should_stop"] is False
    assert status["next_question"] is not None
    assert status["disclaimer"]


def test_unknown_entry_context_returns_400(client):
    resp = client.post("/api/cases", json={"entry_context": "not_a_real_context"})
    assert resp.status_code == 400


def test_unknown_case_returns_404(client):
    resp = client.get("/api/cases/does-not-exist/status")
    assert resp.status_code == 404


def test_answering_updates_status_and_eventually_stops(client):
    case_id, status = _drive_to_completion(client, entry_context="fish")
    assert status["should_stop"] is True
    assert status["next_question"] is None
    assert status["ranked_candidates"]
    assert status["recommendations"] is not None


def test_ich_case_converges_fast_with_recommendations(client):
    case_id = client.post("/api/cases", json={"entry_context": "fish"}).json()["case_id"]
    status = client.get(f"/api/cases/{case_id}/status").json()
    q = status["next_question"]
    # Answer whatever the first question is with a neutral default, then
    # directly report white_spots -- confirms multi-turn state persists
    # correctly across HTTP calls (the case store round-trips properly).
    from aqua_assistant.kb.loader import load_knowledge_base

    kb = load_knowledge_base()
    if q["evidence_id"] != "white_spots":
        kind, val = neutral_default(kb, q["evidence_id"])
        payload = {"question_id": q["question_id"], "evidence_id": q["evidence_id"]}
        payload["raw_value" if kind == "raw_value" else "state"] = val
        status = client.post(f"/api/cases/{case_id}/answers", json=payload).json()

    resp = client.post(
        f"/api/cases/{case_id}/answers",
        json={"question_id": "ask_white_spots", "evidence_id": "white_spots", "state": "true"},
    )
    status = resp.json()
    assert status["should_stop"] is True
    assert status["ranked_candidates"][0]["problem_id"] == "ich"
    assert status["recommendations"]["problem_id"] == "ich"


def test_safety_alert_surfaces_through_the_api(client):
    case_id = client.post("/api/cases", json={"entry_context": "unsure"}).json()["case_id"]
    resp = client.post(
        f"/api/cases/{case_id}/answers",
        json={"question_id": "ask_ammonia_ppm", "evidence_id": "ammonia_ppm", "raw_value": 3.0},
    )
    status = resp.json()
    assert status["safety_alerts"]
    assert status["next_question"]["evidence_id"] == "gasping"


def test_feedback_round_trip_through_api(client):
    case_id, status = _drive_to_completion(client, entry_context="fish")
    resp = client.post(f"/api/cases/{case_id}/feedback", json={"helpful": True, "comment": "nice"})
    assert resp.status_code == 204


def test_rate_limit_blocks_excessive_case_creation(client, monkeypatch):
    import aqua_assistant.api.app as app_module

    monkeypatch.setattr(app_module, "_RATE_LIMIT_MAX_CASES_PER_HOUR", 2)
    app_module._case_creation_log.clear()

    for _ in range(2):
        assert client.post("/api/cases", json={}).status_code == 200
    resp = client.post("/api/cases", json={})
    assert resp.status_code == 429
