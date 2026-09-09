"""Scenario tests for the plant-health KB expansion batch
(plant_nutrient_deficiency, plant_insufficient_light,
plant_algae_competition, plant_co2_deficiency, plant_melt_transition,
plant_root_or_planting_problem, plant_livestock_damage), addressing the
documented "plants entry context is too thin" limitation from the entry-
context-routing milestone.
"""
from __future__ import annotations

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.inference.scoring import observation_contribution, posterior, score_problems
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import eligible_questions
from aqua_assistant.safety.rules import evaluate_safety

from .conftest import neutral_default


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def obs(evidence_id, state, confidence=1.0):
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)


# ---------------------------------------------------------- differential ---


def test_melt_transition_wins_on_recent_planting_and_melting_leaves(kb):
    observations = [
        obs("old_leaves_melting_rapidly", "true"),
        obs("recently_planted_or_moved", "true"),
        obs("new_growth_pale_or_distorted", "false"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "plant_melt_transition"
    assert post["plant_melt_transition"] > post["plant_nutrient_deficiency"]


def test_nutrient_deficiency_wins_without_melt_or_recent_planting(kb):
    observations = [
        obs("older_leaves_affected_first", "true"),
        obs("fertilization_dosing_reported", "none"),
        obs("recently_planted_or_moved", "false"),
        obs("old_leaves_melting_rapidly", "false"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "plant_nutrient_deficiency"


def test_insufficient_light_vs_algae_competition_split_on_photoperiod_and_algae(kb):
    """Opposite ends of the same lighting axis (too little vs too much),
    split by growth form (leggy vs not) and the presence of algae."""
    low_light_obs = [
        obs("leggy_stretching_growth", "true"),
        obs("low_light_duration_or_intensity", "true"),
        obs("visible_algae_growth", "none"),
    ]
    post_low = posterior(score_problems(kb, low_light_obs))
    assert max(post_low, key=post_low.get) == "plant_insufficient_light"

    algae_obs = [
        obs("visible_algae_growth", "heavy"),
        obs("long_photoperiod_or_intense_lighting", "true"),
        obs("leggy_stretching_growth", "false"),
    ]
    post_algae = posterior(score_problems(kb, algae_obs))
    assert max(post_algae, key=post_algae.get) == "plant_algae_competition"


def test_co2_deficiency_wins_on_instability_signal_not_just_shared_poor_growth(kb):
    """stunted/pale new growth alone is shared with plant_nutrient_deficiency
    -- co2_fluctuation_reported is the evidence that should actually tip
    the differential toward plant_co2_deficiency."""
    observations = [
        obs("planted_tank_with_co2_injection", "true"),
        obs("co2_fluctuation_reported", "true"),
        obs("stunted_or_slow_new_growth", "true"),
        obs("new_growth_pale_or_distorted", "true"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "plant_co2_deficiency"
    assert post["plant_co2_deficiency"] > post["plant_nutrient_deficiency"]


def test_root_or_planting_problem_wins_on_direct_physical_signs(kb):
    observations = [
        obs("plant_floating_or_uprooted", "true"),
        obs("roots_visibly_rotting_or_blackened", "true"),
        obs("num_plants_affected", "single"),
    ]
    post = posterior(score_problems(kb, observations))
    assert max(post, key=post.get) == "plant_root_or_planting_problem"


def test_livestock_damage_wins_on_bite_pattern_not_decline_pattern(kb):
    observations = [
        obs("leaves_shredded_or_bitten_pattern", "true"),
        obs("herbivorous_livestock_present", "true"),
    ]
    post = posterior(score_problems(kb, observations))
    assert max(post, key=post.get) == "plant_livestock_damage"

    # A decline-pattern presentation should NOT be mistaken for livestock
    # damage -- the two are meant to be cleanly separable by pattern.
    decline_obs = [
        obs("leaves_shredded_or_bitten_pattern", "false"),
        obs("older_leaves_affected_first", "true"),
        obs("fertilization_dosing_reported", "none"),
    ]
    post_decline = posterior(score_problems(kb, decline_obs))
    assert max(post_decline, key=post_decline.get) == "plant_nutrient_deficiency"


# ------------------------------------------------------------ negative ---


def test_no_melting_nearly_rules_out_melt_transition(kb):
    baseline = score_problems(kb, [])["plant_melt_transition"]
    negative = score_problems(kb, [obs("old_leaves_melting_rapidly", "false")])["plant_melt_transition"]
    assert negative < baseline * 0.3


def test_not_recently_planted_nearly_rules_out_melt_transition(kb):
    """Melt is definitionally tied to a recent planting/replanting event."""
    baseline = score_problems(kb, [])["plant_melt_transition"]
    negative = score_problems(kb, [obs("recently_planted_or_moved", "false")])["plant_melt_transition"]
    assert negative < baseline * 0.3


def test_confirmed_adequate_light_reduces_insufficient_light_candidate(kb):
    baseline = score_problems(kb, [])["plant_insufficient_light"]
    negative = score_problems(kb, [obs("low_light_duration_or_intensity", "false")])["plant_insufficient_light"]
    assert negative < baseline * 0.5


def test_confirmed_no_algae_sharply_reduces_algae_competition_candidate(kb):
    baseline = score_problems(kb, [])["plant_algae_competition"]
    negative = score_problems(kb, [obs("visible_algae_growth", "none")])["plant_algae_competition"]
    assert negative < baseline * 0.3


def test_no_bite_pattern_reduces_livestock_damage_candidate(kb):
    baseline = score_problems(kb, [])["plant_livestock_damage"]
    negative = score_problems(kb, [obs("leaves_shredded_or_bitten_pattern", "false")])["plant_livestock_damage"]
    assert negative < baseline * 0.5


# ------------------------------------------------------------- contradictory ---


def test_contradictory_algae_evidence_stays_bounded(kb):
    """Heavy algae directly contradicted by a long-running low photoperiod
    report. Must combine without error and stay a valid distribution."""
    algae_alone = posterior(score_problems(kb, [obs("visible_algae_growth", "heavy")]))
    contradictory = posterior(
        score_problems(kb, [obs("visible_algae_growth", "heavy"), obs("long_photoperiod_or_intense_lighting", "false")])
    )
    assert sum(contradictory.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 < p < 1.0 for p in contradictory.values())
    assert contradictory["plant_algae_competition"] < algae_alone["plant_algae_competition"]


# ------------------------------------------------- correlation groups ---


def test_plant_general_decline_group_discounts_correlated_confirmation(kb):
    c1 = observation_contribution(kb, "plant_nutrient_deficiency", obs("leaf_yellowing_or_discoloration", "true"))
    c2 = observation_contribution(kb, "plant_nutrient_deficiency", obs("leaf_browning_or_holes", "true"))
    combined_score = score_problems(
        kb, [obs("leaf_yellowing_or_discoloration", "true"), obs("leaf_browning_or_holes", "true")]
    )["plant_nutrient_deficiency"]
    baseline_score = score_problems(kb, [])["plant_nutrient_deficiency"]
    combined_contribution = combined_score - baseline_score

    assert combined_contribution < c1 + c2


def test_root_planting_signs_group_discounts_correlated_confirmation(kb):
    c1 = observation_contribution(kb, "plant_root_or_planting_problem", obs("plant_floating_or_uprooted", "true"))
    c2 = observation_contribution(
        kb, "plant_root_or_planting_problem", obs("roots_visibly_rotting_or_blackened", "true")
    )
    combined_score = score_problems(
        kb, [obs("plant_floating_or_uprooted", "true"), obs("roots_visibly_rotting_or_blackened", "true")]
    )["plant_root_or_planting_problem"]
    baseline_score = score_problems(kb, [])["plant_root_or_planting_problem"]
    combined_contribution = combined_score - baseline_score

    assert combined_contribution < c1 + c2


# ---------------------------------------------------------- selection ---


def test_new_plant_questions_are_eligible(kb):
    ids = {q.id for q in eligible_questions(kb, answered_evidence_ids=set())}
    for qid in [
        "ask_leaf_yellowing_or_discoloration",
        "ask_leaf_browning_or_holes",
        "ask_stunted_or_slow_new_growth",
        "ask_older_leaves_affected_first",
        "ask_new_growth_pale_or_distorted",
        "ask_leggy_stretching_growth",
        "ask_visible_algae_growth",
        "ask_old_leaves_melting_rapidly",
        "ask_leaves_shredded_or_bitten_pattern",
        "ask_plant_floating_or_uprooted",
        "ask_roots_visibly_rotting_or_blackened",
        "ask_low_light_duration_or_intensity",
        "ask_long_photoperiod_or_intense_lighting",
        "ask_co2_fluctuation_reported",
        "ask_fertilization_dosing_reported",
        "ask_substrate_disturbed_or_shallow_for_roots",
        "ask_herbivorous_livestock_present",
        "ask_recently_planted_or_moved",
        "ask_num_plants_affected",
    ]:
        assert qid in ids


# ------------------------------------------------------------- safety ---


def test_plant_problems_do_not_trigger_new_safety_rules(kb):
    """None of these 7 problems are acute fish-threatening emergencies on
    their own -- confirm their own strongest evidence combinations don't
    spuriously fire any safety rule."""
    observations = [
        obs("old_leaves_melting_rapidly", "true"),
        obs("recently_planted_or_moved", "true"),
        obs("older_leaves_affected_first", "true"),
        obs("leggy_stretching_growth", "true"),
        obs("visible_algae_growth", "heavy"),
        obs("plant_floating_or_uprooted", "true"),
        obs("roots_visibly_rotting_or_blackened", "true"),
        obs("leaves_shredded_or_bitten_pattern", "true"),
        obs("stunted_or_slow_new_growth", "true"),
    ]
    alerts = evaluate_safety(kb, observations)
    assert alerts == []


def test_co2_excess_safety_rule_still_fires_alongside_plant_evidence(kb):
    """Cross-domain check: a plant-focused case that also shows fish
    respiratory distress in a CO2-injected tank must still trigger the
    existing co2_injection_excess_active safety rule -- plant evidence
    must never crowd out or suppress an existing safety concern."""
    observations = [
        obs("planted_tank_with_co2_injection", "true"),
        obs("co2_fluctuation_reported", "true"),
        obs("gasping", "true"),
        obs("stunted_or_slow_new_growth", "true"),
    ]
    alerts = evaluate_safety(kb, observations)
    rule_ids = {a.rule_id for a in alerts}
    assert "co2_injection_excess_active" in rule_ids


# --------------------------------------------------------- end to end ---


def test_full_case_starting_from_plants_context_converges_to_melt_transition(kb):
    """User selects the 'plants' entry context and describes a classic
    post-planting melt -- the engine should develop a plant-focused
    differential and converge there, not wander into unrelated fish
    disease markers."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="plants")
    answers = {
        "recently_planted_or_moved": ("state", "true"),
        "old_leaves_melting_rapidly": ("state", "true"),
        "new_growth_pale_or_distorted": ("state", "false"),
        "num_plants_affected": ("state", "several"),
        "white_spots": ("state", "false"),
        "ammonia_ppm": ("raw_value", 0.0),
        "nitrite_ppm": ("raw_value", 0.0),
        "gasping": ("state", "false"),
    }
    asked = []
    for _ in range(len(kb.questions)):
        status = engine.get_status(case_id)
        if status.should_stop:
            top = status.ranked_candidates[0]
            assert top.problem_id == "plant_melt_transition"
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


def test_full_case_starting_from_plants_context_still_surfaces_water_quality_issue(kb):
    """Milestone requirement: a 'plants' entry context must not wall the
    user off from an aquarium-wide water-quality problem when the
    evidence actually points there -- critical ammonia should still win
    even though the user started from the plants concern area."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="plants")
    engine.answer(case_id, "ammonia_ppm", raw_value=3.0)
    engine.answer(case_id, "gasping", state="true")
    engine.answer(case_id, "num_fish_affected", state="most_or_all")
    status = engine.get_status(case_id)
    assert status.ranked_candidates[0].problem_id == "ammonia_toxicity"
    # And the existing safety rule must still be live regardless of entry context.
    assert any(a.rule_id == "critical_ammonia" for a in status.safety_alerts)
