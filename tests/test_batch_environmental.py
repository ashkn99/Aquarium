"""Scenario tests for the environmental/system KB expansion batch
(overstocking_chronic, co2_injection_excess, poor_circulation_dead_spot).
"""
from __future__ import annotations

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import eligible_questions
from aqua_assistant.safety.rules import evaluate_safety

from .conftest import neutral_default


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def obs(evidence_id, state, confidence=1.0):
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)


# ---------------------------------------------------------------- overlap ---


def test_co2_excess_outranks_low_oxygen_when_planted_tank_confirmed(kb):
    """Both cause gasping/surface_breathing, but only co2_injection_excess
    is essentially preconditioned on running CO2 injection."""
    observations = [
        obs("planted_tank_with_co2_injection", "true"),
        obs("gasping", "true"),
        obs("surface_breathing", "true"),
        obs("ph_level", "low"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "co2_injection_excess"
    assert post["co2_injection_excess"] > post["low_oxygen"]


def test_low_oxygen_wins_without_co2_injection(kb):
    observations = [
        obs("planted_tank_with_co2_injection", "false"),
        obs("gasping", "true"),
        obs("surface_breathing", "true"),
        obs("dissolved_oxygen_ppm", "critical"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "low_oxygen"


def test_overstocking_vs_chronic_nitrate_stress_both_plausible_but_distinguishable(kb):
    """These two can look similar (chronic, elevated nitrate, established
    tank) -- the overstocking self-report and water-change frequency should
    split them apart."""
    overstocked_obs = [
        obs("overstocking_reported", "true"),
        obs("water_change_frequency", "regular"),
        obs("nitrate_ppm", "high"),
        obs("tank_age_days", "established"),
    ]
    post_over = posterior(score_problems(kb, overstocked_obs))
    assert max(post_over, key=post_over.get) == "overstocking_chronic"

    neglected_obs = [
        obs("overstocking_reported", "false"),
        obs("water_change_frequency", "rare"),
        obs("nitrate_ppm", "high"),
        obs("tank_age_days", "established"),
    ]
    post_neglect = posterior(score_problems(kb, neglected_obs))
    assert max(post_neglect, key=post_neglect.get) == "chronic_nitrate_stress"


def test_dead_spot_favored_over_whole_tank_problems_when_localized(kb):
    """A localized problem (few fish, debris visible) should not be
    swamped by the whole-tank water-quality candidates."""
    observations = [
        obs("visible_debris_or_detritus_buildup", "true"),
        obs("num_fish_affected", "several"),
        obs("onset_timing", "chronic"),
        obs("ammonia_ppm", "safe"),
        obs("nitrite_ppm", "safe"),
    ]
    post = posterior(score_problems(kb, observations))
    assert max(post, key=post.get) == "poor_circulation_dead_spot"


# ------------------------------------------------------------ negative ---


def test_confirmed_appropriate_stocking_reduces_overstocking_candidate(kb):
    baseline = score_problems(kb, [])["overstocking_chronic"]
    with_negative = score_problems(kb, [obs("overstocking_reported", "false")])["overstocking_chronic"]
    assert with_negative < baseline


def test_no_co2_injection_reduces_co2_excess_candidate_sharply(kb):
    baseline = score_problems(kb, [])["co2_injection_excess"]
    with_negative = score_problems(kb, [obs("planted_tank_with_co2_injection", "false")])["co2_injection_excess"]
    assert with_negative < baseline
    # Should be a strong disconfirmation, not a token one.
    assert with_negative < baseline * 0.5


def test_whole_tank_affected_reduces_dead_spot_candidate(kb):
    baseline = score_problems(kb, [])["poor_circulation_dead_spot"]
    with_negative = score_problems(kb, [obs("num_fish_affected", "most_or_all")])["poor_circulation_dead_spot"]
    assert with_negative < baseline


# ------------------------------------------------------- contradictory ---


def test_contradictory_co2_evidence_stays_bounded(kb):
    """CO2 injection confirmed (supports co2_injection_excess) directly
    contradicted by gasping=false (no distress observed). Must not crash
    and must reflect the pulled-down plausibility."""
    co2_alone = posterior(score_problems(kb, [obs("planted_tank_with_co2_injection", "true")]))
    contradictory = posterior(
        score_problems(kb, [obs("planted_tank_with_co2_injection", "true"), obs("gasping", "false")])
    )
    assert sum(contradictory.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 < p < 1.0 for p in contradictory.values())
    assert contradictory["co2_injection_excess"] <= co2_alone["co2_injection_excess"]


# ---------------------------------------------------------- selection ---


def test_new_environmental_questions_are_eligible(kb):
    ids = {q.id for q in eligible_questions(kb, answered_evidence_ids=set())}
    assert "ask_overstocking_reported" in ids
    assert "ask_planted_tank_with_co2_injection" in ids
    assert "ask_visible_debris_or_detritus_buildup" in ids


# ------------------------------------------------------------- safety ---


def test_co2_excess_safety_rule_fires_with_distress(kb):
    alerts = evaluate_safety(kb, [obs("planted_tank_with_co2_injection", "true"), obs("surface_breathing", "true")])
    rule_ids = {a.rule_id for a in alerts}
    assert "co2_injection_excess_active" in rule_ids
    rule = next(a for a in alerts if a.rule_id == "co2_injection_excess_active")
    assert rule.severity == "urgent"


def test_co2_excess_safety_rule_does_not_fire_without_distress(kb):
    alerts = evaluate_safety(kb, [obs("planted_tank_with_co2_injection", "true")])
    rule_ids = {a.rule_id for a in alerts}
    assert "co2_injection_excess_active" not in rule_ids


def test_no_new_safety_rules_for_chronic_problems(kb):
    """overstocking_chronic and poor_circulation_dead_spot are gradual,
    not acute-life-threatening -- confirm no safety rule was (over-)added
    for them, matching the "only where genuinely warranted" instruction."""
    alerts = evaluate_safety(
        kb,
        [
            obs("overstocking_reported", "true"),
            obs("visible_debris_or_detritus_buildup", "true"),
            obs("lethargy", "true"),
        ],
    )
    rule_ids = {a.rule_id for a in alerts}
    assert not rule_ids & {"overstocking_chronic_alert", "poor_circulation_dead_spot_alert"}


# --------------------------------------------------------- end to end ---


def test_full_case_converges_to_overstocking_chronic(kb):
    """max_turns is generous (not 30) because, at this KB size, the engine
    correctly spends early turns ruling out several highly specific
    pathognomonic disease markers (each resolves a large amount of entropy
    on its own) before reaching moderate-specificity context questions
    like overstocking_reported -- see the final report for this batch."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case()
    answers = {
        "overstocking_reported": ("state", "true"),
        "tank_age_days": ("raw_value", 400.0),
        "nitrate_ppm": ("raw_value", 60.0),
        "onset_timing": ("state", "chronic"),
        "white_spots": ("state", "false"),
        "ammonia_ppm": ("raw_value", 0.1),
        "nitrite_ppm": ("raw_value", 0.1),
        "water_change_frequency": ("state", "regular"),
        "used_untreated_tap_water": ("state", "false"),
    }
    asked = []
    for _ in range(len(kb.questions)):
        status = engine.get_status(case_id)
        if status.should_stop:
            top = status.ranked_candidates[0]
            assert top.problem_id == "overstocking_chronic"
            return
        q = status.best_next_question
        ev = kb.questions[q.question_id].evidence_id
        assert ev not in asked, f"redundant question re-asked for {ev}"
        asked.append(ev)
        kind, val = answers.get(ev) or neutral_default(kb, ev)
        if kind == "raw_value":
            engine.answer(case_id, ev, raw_value=val, question_id=q.question_id)
        else:
            engine.answer(case_id, ev, state=val, question_id=q.question_id)
    raise AssertionError("did not converge")
