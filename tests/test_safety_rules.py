from __future__ import annotations

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.safety.rules import evaluate_safety


@pytest.fixture(scope="module")
def seed_kb():
    return load_knowledge_base()


def obs(evidence_id: str, state: str) -> Observation:
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=1.0)


def test_no_rule_fires_with_no_observations(seed_kb):
    assert evaluate_safety(seed_kb, []) == []


def test_critical_ammonia_fires_and_forces_a_question(seed_kb):
    alerts = evaluate_safety(seed_kb, [obs("ammonia_ppm", "critical")])
    assert len(alerts) == 1
    assert alerts[0].rule_id == "critical_ammonia"
    assert alerts[0].severity == "urgent"
    assert alerts[0].forced_question_id == "ask_gasping"


def test_elevated_ammonia_does_not_fire_the_critical_rule(seed_kb):
    assert evaluate_safety(seed_kb, [obs("ammonia_ppm", "elevated")]) == []


def test_urgent_rule_is_ranked_first_when_urgent_and_caution_both_fire(seed_kb):
    # gasping=true fires the caution-level respiratory_distress_general rule;
    # critical ammonia also fires the urgent-level critical_ammonia rule.
    # The urgent one must be ranked first (it drives the forced question).
    alerts = evaluate_safety(seed_kb, [obs("gasping", "true"), obs("ammonia_ppm", "critical")])
    assert [a.severity for a in alerts] == ["urgent", "caution"]
    assert alerts[0].rule_id == "critical_ammonia"


def test_caution_rule_fires_alone_on_gasping_without_confirmed_cause(seed_kb):
    alerts = evaluate_safety(seed_kb, [obs("gasping", "true")])
    assert len(alerts) == 1
    assert alerts[0].rule_id == "respiratory_distress_general"
    assert alerts[0].severity == "caution"


def test_superseded_observations_do_not_trigger_rules(seed_kb):
    critical = obs("ammonia_ppm", "critical")
    critical.superseded = True
    assert evaluate_safety(seed_kb, [critical]) == []


def test_two_simultaneous_urgent_rules_are_both_reported(seed_kb):
    """Regression test: evaluate_safety used to return only the single
    highest-severity firing rule, so a second, equally urgent condition
    (e.g. critical nitrite alongside critical ammonia) was silently
    dropped from the user's view -- a real "safety rule bypassed" bug.
    Both must now be surfaced, even though only the top one's
    forced_question_id is honored by the engine."""
    alerts = evaluate_safety(seed_kb, [obs("ammonia_ppm", "critical"), obs("nitrite_ppm", "critical")])
    rule_ids = {a.rule_id for a in alerts}
    assert rule_ids == {"critical_ammonia", "critical_nitrite"}
    assert all(a.severity == "urgent" for a in alerts)
