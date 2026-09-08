"""Scenario tests for the husbandry/behavior KB expansion batch
(aggression_bullying_stress, overfeeding_poor_diet,
acclimation_transport_stress, inadequate_hiding_or_environment_stress),
plus the general mass_mortality_event safety marker introduced alongside it.
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


def test_aggression_beats_bacterial_infection_on_bite_pattern_and_behavior(kb):
    """Torn fins + observed chasing should point to aggression, not disease,
    even though frayed/damaged fins overlap with bacterial_infection."""
    observations = [
        obs("torn_or_missing_fins_from_biting", "true"),
        obs("chasing_or_nipping_observed", "true"),
        obs("num_fish_affected", "single"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "aggression_bullying_stress"
    assert post["aggression_bullying_stress"] > post["bacterial_infection"]


def test_bacterial_infection_wins_without_aggression_signs(kb):
    observations = [
        obs("torn_or_missing_fins_from_biting", "false"),
        obs("chasing_or_nipping_observed", "false"),
        obs("red_streaks_or_gills", "true"),
        obs("frayed_fins", "true"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "bacterial_infection"


def test_acclimation_stress_vs_new_tank_syndrome_split_on_scope(kb):
    """Both involve a recently added fish, but acclimation_transport_stress
    is about the individual fish while new_tank_syndrome is whole-tank."""
    individual_obs = [
        obs("recent_fish_added", "true"),
        obs("improper_acclimation_reported", "true"),
        obs("num_fish_affected", "single"),
        obs("tank_age_days", "established"),
    ]
    post_individual = posterior(score_problems(kb, individual_obs))
    assert max(post_individual, key=post_individual.get) == "acclimation_transport_stress"

    whole_tank_obs = [
        obs("recent_fish_added", "true"),
        obs("tank_age_days", "new"),
        obs("num_fish_affected", "most_or_all"),
        obs("ammonia_ppm", "elevated"),
    ]
    post_whole = posterior(score_problems(kb, whole_tank_obs))
    assert max(post_whole, key=post_whole.get) == "new_tank_syndrome"


# ------------------------------------------------------------ negative ---


def test_no_recent_fish_added_nearly_rules_out_acclimation_stress(kb):
    baseline = score_problems(kb, [])["acclimation_transport_stress"]
    negative = score_problems(kb, [obs("recent_fish_added", "false")])["acclimation_transport_stress"]
    assert negative < baseline * 0.3


def test_confirmed_no_overfeeding_reduces_overfeeding_candidate(kb):
    baseline = score_problems(kb, [])["overfeeding_poor_diet"]
    negative = score_problems(kb, [obs("overfeeding_reported", "false")])["overfeeding_poor_diet"]
    assert negative < baseline


def test_adequate_hiding_places_reduces_environment_stress_candidate(kb):
    baseline = score_problems(kb, [])["inadequate_hiding_or_environment_stress"]
    negative = score_problems(kb, [obs("insufficient_hiding_places", "false")])["inadequate_hiding_or_environment_stress"]
    assert negative < baseline


# ------------------------------------------------------- contradictory ---


def test_contradictory_overfeeding_evidence_stays_bounded(kb):
    """Self-reported overfeeding (positive) directly contradicted by no
    visible uneaten food (negative). Must combine without error."""
    overfeeding_alone = posterior(score_problems(kb, [obs("overfeeding_reported", "true")]))
    contradictory = posterior(
        score_problems(kb, [obs("overfeeding_reported", "true"), obs("uneaten_food_accumulating", "false")])
    )
    assert sum(contradictory.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 < p < 1.0 for p in contradictory.values())
    assert contradictory["overfeeding_poor_diet"] < overfeeding_alone["overfeeding_poor_diet"]


# ------------------------------------------------- correlation group ---


def test_aggression_signs_group_discounts_correlated_confirmation(kb):
    from aqua_assistant.inference.scoring import observation_contribution

    chasing_contribution = observation_contribution(
        kb, "aggression_bullying_stress", obs("chasing_or_nipping_observed", "true")
    )
    torn_contribution = observation_contribution(
        kb, "aggression_bullying_stress", obs("torn_or_missing_fins_from_biting", "true")
    )
    combined_score = score_problems(
        kb, [obs("chasing_or_nipping_observed", "true"), obs("torn_or_missing_fins_from_biting", "true")]
    )["aggression_bullying_stress"]
    baseline_score = score_problems(kb, [])["aggression_bullying_stress"]
    combined_contribution = combined_score - baseline_score

    assert combined_contribution < chasing_contribution + torn_contribution


# ---------------------------------------------------------- selection ---


def test_new_husbandry_questions_are_eligible(kb):
    ids = {q.id for q in eligible_questions(kb, answered_evidence_ids=set())}
    for qid in [
        "ask_torn_or_missing_fins_from_biting",
        "ask_chasing_or_nipping_observed",
        "ask_overfeeding_reported",
        "ask_uneaten_food_accumulating",
        "ask_improper_acclimation_reported",
        "ask_insufficient_hiding_places",
        "ask_color_fading_or_pale",
        "ask_sudden_multiple_fish_death",
    ]:
        assert qid in ids


# ------------------------------------------------------------- safety ---


def test_mass_mortality_rule_fires_alone(kb):
    alerts = evaluate_safety(kb, [obs("sudden_multiple_fish_death", "true")])
    rule_ids = {a.rule_id for a in alerts}
    assert "mass_mortality_event" in rule_ids
    rule = next(a for a in alerts if a.rule_id == "mass_mortality_event")
    assert rule.severity == "urgent"


def test_mass_mortality_rule_does_not_fire_without_it(kb):
    alerts = evaluate_safety(kb, [obs("lethargy", "true"), obs("appetite_loss", "true")])
    rule_ids = {a.rule_id for a in alerts}
    assert "mass_mortality_event" not in rule_ids


def test_mass_mortality_combines_with_other_firing_rules(kb):
    """Confirms the earlier fix (evaluate_safety returns all firing rules)
    still holds with the new rule added: mass mortality plus critical
    ammonia should both be reported, not just one."""
    alerts = evaluate_safety(kb, [obs("sudden_multiple_fish_death", "true"), obs("ammonia_ppm", "critical")])
    rule_ids = {a.rule_id for a in alerts}
    assert {"mass_mortality_event", "critical_ammonia"} <= rule_ids


def test_husbandry_problems_do_not_trigger_new_safety_rules(kb):
    """None of this batch's 4 problems are acute emergencies -- confirm
    their own strongest evidence combinations don't spuriously fire any
    safety rule."""
    observations = [
        obs("torn_or_missing_fins_from_biting", "true"),
        obs("chasing_or_nipping_observed", "true"),
        obs("overfeeding_reported", "true"),
        obs("uneaten_food_accumulating", "true"),
        obs("improper_acclimation_reported", "true"),
        obs("insufficient_hiding_places", "true"),
        obs("color_fading_or_pale", "true"),
    ]
    alerts = evaluate_safety(kb, observations)
    assert alerts == []


# --------------------------------------------------------- end to end ---


def test_full_case_converges_to_aggression_bullying_stress(kb):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case()
    answers = {
        "chasing_or_nipping_observed": ("state", "true"),
        "torn_or_missing_fins_from_biting": ("state", "true"),
        "num_fish_affected": ("state", "single"),
        "white_spots": ("state", "false"),
        "ammonia_ppm": ("raw_value", 0.0),
        "nitrite_ppm": ("raw_value", 0.0),
        "cottony_fuzzy_growth": ("state", "false"),
        "pinecone_scales": ("state", "false"),
    }
    asked = []
    for _ in range(len(kb.questions)):
        status = engine.get_status(case_id)
        if status.should_stop:
            top = status.ranked_candidates[0]
            assert top.problem_id == "aggression_bullying_stress"
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
