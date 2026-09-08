"""Scenario tests for the water-quality KB expansion batch
(chlorine_chloramine_poisoning, chronic_nitrate_stress,
copper_heavy_metal_toxicity), covering:

  - overlapping symptoms vs. the pre-existing water-quality problems
  - negative/disconfirming evidence
  - contradictory observations
  - question-selection behavior
  - safety behavior
"""
from __future__ import annotations

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.safety.rules import evaluate_safety

from .conftest import neutral_default


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def obs(evidence_id, state, confidence=1.0):
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)


# ---------------------------------------------------------------- overlap ---


def test_chlorine_poisoning_outranks_ammonia_when_water_is_actually_safe(kb):
    """Chlorine poisoning and ammonia toxicity share gasping + red gills +
    whole-tank onset. The tap-water history plus clean ammonia/nitrite
    readings should tip the ranking toward chlorine, not ammonia."""
    observations = [
        obs("used_untreated_tap_water", "true"),
        obs("gasping", "true"),
        obs("red_streaks_or_gills", "true"),
        obs("ammonia_ppm", "safe"),
        obs("nitrite_ppm", "safe"),
        obs("num_fish_affected", "most_or_all"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "chlorine_chloramine_poisoning"
    assert post["chlorine_chloramine_poisoning"] > post["ammonia_toxicity"]


def test_ammonia_still_wins_when_ammonia_is_actually_high(kb):
    """Same symptom picture, but this time ammonia genuinely reads high and
    tap water was properly treated -- ammonia_toxicity should win instead."""
    observations = [
        obs("used_untreated_tap_water", "false"),
        obs("gasping", "true"),
        obs("red_streaks_or_gills", "true"),
        obs("ammonia_ppm", "critical"),
        obs("num_fish_affected", "most_or_all"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "ammonia_toxicity"
    assert post["ammonia_toxicity"] > post["chlorine_chloramine_poisoning"]


def test_new_tank_syndrome_vs_chronic_nitrate_stress_differentiate_on_tank_age(kb):
    """Both can show elevated nitrogen-cycle byproducts, but new_tank_syndrome
    is about an immature cycle (new tank) while chronic_nitrate_stress is
    about long-term neglect (established tank) -- tank age should split them."""
    new_tank_obs = [
        obs("tank_age_days", "new"),
        obs("ammonia_ppm", "elevated"),
        obs("recent_fish_added", "true"),
        obs("water_change_frequency", "regular"),
    ]
    post_new = posterior(score_problems(kb, new_tank_obs))
    assert max(post_new, key=post_new.get) == "new_tank_syndrome"

    established_obs = [
        obs("tank_age_days", "established"),
        obs("nitrate_ppm", "high"),
        obs("water_change_frequency", "rare"),
        obs("onset_timing", "chronic"),
    ]
    post_established = posterior(score_problems(kb, established_obs))
    assert max(post_established, key=post_established.get) == "chronic_nitrate_stress"


# ------------------------------------------------------------ negative ---


def test_regular_water_changes_reduce_chronic_nitrate_stress(kb):
    baseline = score_problems(kb, [])["chronic_nitrate_stress"]
    with_regular = score_problems(kb, [obs("water_change_frequency", "regular")])["chronic_nitrate_stress"]
    assert with_regular < baseline


def test_no_copper_exposure_reduces_copper_toxicity_candidate(kb):
    baseline = score_problems(kb, [])["copper_heavy_metal_toxicity"]
    with_negative = score_problems(kb, [obs("copper_source_exposure", "false")])["copper_heavy_metal_toxicity"]
    assert with_negative < baseline


def test_dechlorinated_water_reduces_chlorine_poisoning_candidate(kb):
    baseline = score_problems(kb, [])["chlorine_chloramine_poisoning"]
    with_negative = score_problems(kb, [obs("used_untreated_tap_water", "false")])["chlorine_chloramine_poisoning"]
    assert with_negative < baseline


# ------------------------------------------------------- contradictory ---


def test_contradictory_nitrate_evidence_does_not_crash_and_stays_bounded(kb):
    """Rare water changes (supports chronic_nitrate_stress) directly
    contradicted by a safe nitrate reading (opposes it). The engine must
    combine these without error and keep the posterior a valid distribution
    -- it should not resolve to nonsense (e.g. > 1 probability) and the
    contradiction should pull the candidate down from what "rare" alone
    would suggest."""
    rare_alone = posterior(score_problems(kb, [obs("water_change_frequency", "rare")]))
    contradictory = posterior(
        score_problems(kb, [obs("water_change_frequency", "rare"), obs("nitrate_ppm", "safe")])
    )
    assert sum(contradictory.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 < p < 1.0 for p in contradictory.values())
    assert contradictory["chronic_nitrate_stress"] < rare_alone["chronic_nitrate_stress"]


# ---------------------------------------------------------- selection ---


def test_tap_water_question_is_eligible_and_can_be_selected(kb):
    from aqua_assistant.questions.selection import eligible_questions

    ids = {q.id for q in eligible_questions(kb, answered_evidence_ids=set())}
    assert "ask_used_untreated_tap_water" in ids
    assert "ask_water_change_frequency" in ids
    assert "ask_copper_source_exposure" in ids


def test_tap_water_question_answered_is_excluded_from_eligibility(kb):
    from aqua_assistant.questions.selection import eligible_questions

    remaining = eligible_questions(kb, answered_evidence_ids={"used_untreated_tap_water"})
    assert all(q.evidence_id != "used_untreated_tap_water" for q in remaining)


# ------------------------------------------------------------- safety ---


def test_severe_chlorine_exposure_rule_fires_on_tap_water_plus_gasping(kb):
    alerts = evaluate_safety(kb, [obs("used_untreated_tap_water", "true"), obs("gasping", "true")])
    rule_ids = {a.rule_id for a in alerts}
    assert "severe_chlorine_exposure" in rule_ids
    severe = next(a for a in alerts if a.rule_id == "severe_chlorine_exposure")
    assert severe.severity == "urgent"


def test_severe_chlorine_exposure_rule_does_not_fire_on_tap_water_alone(kb):
    alerts = evaluate_safety(kb, [obs("used_untreated_tap_water", "true")])
    rule_ids = {a.rule_id for a in alerts}
    assert "severe_chlorine_exposure" not in rule_ids


def test_severe_chlorine_exposure_rule_does_not_fire_on_gasping_alone(kb):
    """gasping alone still fires the generic respiratory_distress_general
    caution rule, but must not fire the chlorine-specific urgent rule
    without the tap-water history."""
    alerts = evaluate_safety(kb, [obs("gasping", "true")])
    rule_ids = {a.rule_id for a in alerts}
    assert "severe_chlorine_exposure" not in rule_ids
    assert "respiratory_distress_general" in rule_ids


# --------------------------------------------------------- end to end ---


def test_full_case_converges_to_chlorine_poisoning(kb):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case()
    answers = {
        "used_untreated_tap_water": ("state", "true"),
        "gasping": ("state", "true"),
        "red_streaks_or_gills": ("state", "true"),
        "ammonia_ppm": ("raw_value", 0.1),
        "nitrite_ppm": ("raw_value", 0.1),
        "white_spots": ("state", "false"),
        "num_fish_affected": ("state", "most_or_all"),
        "recent_water_change_large": ("state", "true"),
    }
    asked = []
    for _ in range(30):
        status = engine.get_status(case_id)
        if status.should_stop:
            top = status.ranked_candidates[0]
            assert top.problem_id == "chlorine_chloramine_poisoning"
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
