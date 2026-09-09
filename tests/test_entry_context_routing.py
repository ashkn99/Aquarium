"""Behavioral tests for entry-context routing: the "what are you concerned
about?" soft preference layered on top of information-gain question
selection (questions/selection.py's routing_bonus + KnowledgeBase.entry_contexts).
"""
from __future__ import annotations

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import (
    build_question_explanation,
    eligible_questions,
    routing_bonus,
    select_best_question,
)
from aqua_assistant.safety.rules import evaluate_safety

from .conftest import neutral_default


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def obs(evidence_id, state, confidence=1.0):
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)


def _first_n(kb, entry_context, n):
    """Drive n turns of a fresh case under entry_context (answering
    conftest.neutral_default() throughout, so nothing story-specific ever
    gets revealed), returning the (evidence_id, topic) pairs asked.

    Uses neutral_default() rather than a naive false/states[0]/0.0 default
    -- a raw 0.0 reads as "critical" for dissolved_oxygen_ppm, which used
    to never matter here (that question rarely got asked this early) but
    started fabricating a false low-oxygen crisis once safety-aware
    question prioritization began pulling it forward -- same class of bug
    documented in PROJECT_STATUS.md's neutral-default gotcha."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context=entry_context)
    asked = []
    for _ in range(n):
        status = engine.get_status(case_id)
        if status.should_stop:
            break
        q = status.best_next_question
        ev_id = kb.questions[q.question_id].evidence_id
        asked.append((ev_id, kb.evidence[ev_id].topic))
        kind, val = neutral_default(kb, ev_id)
        if kind == "raw_value":
            engine.answer(case_id, ev_id, raw_value=val, question_id=q.question_id)
        else:
            engine.answer(case_id, ev_id, state=val, question_id=q.question_id)
    return asked


# ============================================================ A ===
def test_fish_water_plants_produce_different_early_sequences(kb):
    fish_seq = [e for e, _ in _first_n(kb, "fish", 5)]
    water_seq = [e for e, _ in _first_n(kb, "water", 5)]
    plants_seq = [e for e, _ in _first_n(kb, "plants", 5)]

    assert fish_seq != water_seq
    assert water_seq != plants_seq
    assert fish_seq != plants_seq


# ============================================================ B ===
def test_entry_context_never_changes_diagnostic_scoring(kb):
    """entry_context is never read by inference/scoring.py at all --
    confirmed directly rather than assumed."""
    observations = [obs("ammonia_ppm", "critical"), obs("gasping", "true"), obs("num_fish_affected", "most_or_all")]
    baseline = posterior(score_problems(kb, observations))
    # score_problems/posterior take no entry_context parameter whatsoever;
    # calling them repeatedly must be perfectly reproducible regardless of
    # what routing context anyone might be using elsewhere.
    for _ in range(5):
        assert posterior(score_problems(kb, observations)) == baseline


def test_entry_context_never_changes_case_status_ranking(kb):
    """End-to-end regression: identical answers through engines started
    under every entry context must produce identical ranked_candidates."""
    answers = [("ammonia_ppm", "raw_value", 3.0), ("gasping", "state", "true"), ("num_fish_affected", "state", "most_or_all")]
    results = {}
    for ctx in ["fish", "water", "plants", "aquarium_environment", "unsure", None]:
        engine = AquariumInferenceEngine(kb)
        case_id = engine.start_case(entry_context=ctx)
        for ev_id, kind, val in answers:
            if kind == "raw_value":
                engine.answer(case_id, ev_id, raw_value=val)
            else:
                engine.answer(case_id, ev_id, state=val)
        status = engine.get_status(case_id)
        results[ctx] = [(c.problem_id, c.probability) for c in status.ranked_candidates]

    baseline = results["fish"]
    for ctx, ranking in results.items():
        assert ranking == baseline, f"entry_context={ctx!r} diagnostic ranking differs from baseline"


# ============================================================ C-F ===
def test_fish_routing_favors_fish_symptom_and_general_topics_early(kb):
    topics = {t for _, t in _first_n(kb, "fish", 4)}
    assert topics & {"fish_symptom", "general_context"}
    assert routing_bonus(kb, "fish", "gasping", num_observations=0) > 0
    assert routing_bonus(kb, "fish", "used_untreated_tap_water", num_observations=0) == 0.0


def test_water_routing_favors_water_topics_early(kb):
    topics = {t for _, t in _first_n(kb, "water", 4)}
    assert topics & {"water_parameter", "water_history", "general_context"}
    assert routing_bonus(kb, "water", "ammonia_ppm", num_observations=0) > 0
    assert routing_bonus(kb, "water", "used_untreated_tap_water", num_observations=0) > 0


def test_plant_routing_favors_its_plant_adjacent_topic_early(kb):
    """Following the plant-health KB expansion, 'plants' now has a real
    dedicated topic (plant_symptom) and should lean on it early, not just
    the one CO2-injection fallback fact from before that milestone."""
    topics = {t for _, t in _first_n(kb, "plants", 4)}
    assert topics & {"plant_symptom", "environment_system", "general_context"}
    assert routing_bonus(kb, "plants", "planted_tank_with_co2_injection", num_observations=0) > 0
    assert routing_bonus(kb, "plants", "leaf_yellowing_or_discoloration", num_observations=0) > 0


def test_plants_context_is_no_longer_thin(kb):
    """Regression guard for the previously-documented 'plants entry context
    is too thin' limitation (a single plant-adjacent evidence item). With
    real plant-health evidence in the KB, the opening should include
    multiple plant_symptom questions, not just a CO2 fallback plus generic
    triage.

    Window widened from 6 to 16 turns for the safety-aware-prioritization
    milestone: turns 1-6 are now the bounded, KB-wide safety-linked-
    evidence pool (sudden_multiple_fish_death, dissolved_oxygen_ppm,
    nitrite_ppm, ammonia_ppm, used_untreated_tap_water,
    planted_tank_with_co2_injection) ahead of *any* context's domain
    questions, plant_symptom included -- see
    test_entry_context_still_differentiates_which_pending_safety_evidence_comes_first
    in test_safety_priority.py for the dedicated check that context still
    differentiates *within* that pool, from turn 1."""
    topics = [t for _, t in _first_n(kb, "plants", 16)]
    assert topics.count("plant_symptom") >= 2


def test_environment_routing_favors_system_topics_early(kb):
    topics = {t for _, t in _first_n(kb, "aquarium_environment", 4)}
    assert topics & {"environment_system", "general_context"}
    assert routing_bonus(kb, "aquarium_environment", "planted_tank_with_co2_injection", num_observations=0) > 0
    assert routing_bonus(kb, "aquarium_environment", "overstocking_reported", num_observations=0) > 0


# ============================================================ G ===
def test_unsure_only_boosts_general_context_never_domain_topics(kb):
    for ev_id in ["gasping", "ammonia_ppm", "planted_tank_with_co2_injection", "used_untreated_tap_water"]:
        assert routing_bonus(kb, "unsure", ev_id, num_observations=0) == 0.0
    assert routing_bonus(kb, "unsure", "num_fish_affected", num_observations=0) > 0.0


def test_unsure_diverges_from_every_domain_context(kb):
    unsure_seq = [e for e, _ in _first_n(kb, "unsure", 4)]
    for ctx in ["fish", "water", "aquarium_environment"]:
        assert unsure_seq != [e for e, _ in _first_n(kb, ctx, 4)], f"unsure matched {ctx}"


# ============================================================ H ===
def test_pivots_away_from_fish_disease_markers_when_acute_water_evidence_dominates(kb):
    """Start with 'fish', then feed strong acute water/environment
    evidence (every fish affected, sudden onset, gasping, filter
    disrupted) -- must be free to pivot toward the water/oxygen picture
    rather than continuing down the fish-disease-marker checklist."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")

    engine.answer(case_id, "num_fish_affected", state="most_or_all")
    engine.answer(case_id, "onset_timing", state="sudden")
    engine.answer(case_id, "gasping", state="true")
    engine.answer(case_id, "filter_disrupted", state="true")

    status = engine.get_status(case_id)
    q = status.best_next_question
    ev_id = kb.questions[q.question_id].evidence_id
    topic = kb.evidence[ev_id].topic
    assert topic != "fish_disease_marker"


def test_pivots_away_from_plant_leaning_when_fish_emergency_evidence_dominates(kb):
    """User selects 'plants' but the evidence indicates an aquarium-wide
    fish emergency -- the system must still be able to move toward the
    water-crisis picture."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="plants")
    engine.answer(case_id, "gasping", state="true")
    engine.answer(case_id, "surface_breathing", state="true")
    engine.answer(case_id, "num_fish_affected", state="most_or_all")
    status = engine.get_status(case_id)
    q = status.best_next_question
    ev_id = kb.questions[q.question_id].evidence_id
    # should be chasing the acute picture (water parameters, oxygen,
    # onset/context), not stuck on the plant-adjacent CO2 question alone
    # forever or reverting to unrelated disease markers.
    assert ev_id in {"ammonia_ppm", "nitrite_ppm", "dissolved_oxygen_ppm", "onset_timing", "planted_tank_with_co2_injection", "filter_disrupted"}


# ============================================================ I ===
@pytest.mark.parametrize("ctx", ["fish", "water", "plants", "aquarium_environment", "unsure"])
def test_safety_forced_question_wins_regardless_of_entry_context(kb, ctx):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context=ctx)
    engine.answer(case_id, "ammonia_ppm", raw_value=3.0)  # critical -> forces ask_gasping
    status = engine.get_status(case_id)
    assert status.safety_alerts
    assert status.best_next_question.question_id == "ask_gasping"


def test_safety_does_not_fire_for_every_normal_session(kb):
    """Sanity check on the flip side of I: routing/triage must not somehow
    make safety fire spuriously in an unremarkable session."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    engine.answer(case_id, "lethargy", state="true")
    status = engine.get_status(case_id)
    assert status.safety_alerts == []


# ============================================================ J ===
def test_routing_does_not_resurrect_answered_questions(kb):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    engine.answer(case_id, "gasping", state="true")
    status = engine.get_status(case_id)
    assert status.best_next_question.question_id != "ask_gasping"

    case = engine.store.get(case_id)
    remaining = eligible_questions(kb, case.answered_evidence_ids())
    assert all(q.evidence_id != "gasping" for q in remaining)


# ============================================================ K ===
def test_information_gain_still_differentiates_within_a_boosted_topic(kb):
    """Two fish_symptom questions get the identical routing bonus under
    'fish' context -- their relative order must still come from real
    information-gain differences, not the routing term alone."""
    post = posterior(score_problems(kb, []))
    gasping_e = build_question_explanation(kb, "ask_gasping", post, [], "fish")
    hiding_e = build_question_explanation(kb, "ask_hiding", post, [], "fish")
    assert gasping_e.routing_bonus == hiding_e.routing_bonus
    assert (gasping_e.final_score > hiding_e.final_score) == (gasping_e.adjusted_value > hiding_e.adjusted_value)


def test_routing_bonus_decays_linearly_to_zero(kb):
    b0 = routing_bonus(kb, "fish", "gasping", num_observations=0)
    b_mid = routing_bonus(kb, "fish", "gasping", num_observations=3)
    b_full = routing_bonus(kb, "fish", "gasping", num_observations=6)
    b_after = routing_bonus(kb, "fish", "gasping", num_observations=10)
    assert b0 > b_mid > b_full == b_after == 0.0
    assert b_mid == pytest.approx(b0 / 2, rel=1e-6)


def test_no_entry_context_behaves_exactly_like_pure_information_gain(kb):
    """entry_context=None must reproduce the pre-routing argmax exactly --
    routing is strictly additive and optional."""
    post = posterior(score_problems(kb, []))
    routed_none = select_best_question(kb, post, [], set(), entry_context=None)
    unrouted = select_best_question(kb, post, [], set())
    assert routed_none.question_id == unrouted.question_id
    assert routed_none.final_score == routed_none.adjusted_value == unrouted.adjusted_value


def test_unknown_entry_context_start_case_raises(kb):
    engine = AquariumInferenceEngine(kb)
    with pytest.raises(ValueError):
        engine.start_case(entry_context="not_a_real_context")


def test_all_five_entry_contexts_are_configured(kb):
    assert set(kb.entry_contexts.keys()) == {"fish", "water", "plants", "aquarium_environment", "unsure"}


# ============================================================ L ===
# dissolved_oxygen_ppm reclassification: was mistagged `water_parameter`,
# now `environment_system` (low oxygen is at least as much an
# aquarium/environment problem as a water-chemistry one -- see
# PROJECT_STATUS.md limitation #2). Routing/metadata-only change.
def test_dissolved_oxygen_reclassified_as_environment_system(kb):
    assert kb.evidence["dissolved_oxygen_ppm"].topic == "environment_system"

    # aquarium_environment must no longer be disadvantaged relative to
    # water for this evidence: it now gets the same environment_system
    # weight as the KB's other environment_system-tagged evidence.
    env_bonus = routing_bonus(kb, "aquarium_environment", "dissolved_oxygen_ppm", num_observations=0)
    co2_bonus = routing_bonus(kb, "aquarium_environment", "planted_tank_with_co2_injection", num_observations=0)
    assert env_bonus > 0
    assert env_bonus == co2_bonus

    # water loses its topic-specific routing push for this evidence (that
    # is the correction) -- but this only affects ranking preference, not
    # eligibility (see next test).
    assert routing_bonus(kb, "water", "dissolved_oxygen_ppm", num_observations=0) == 0.0


def test_water_can_still_reach_dissolved_oxygen_via_information_gain(kb):
    """Losing the routing preference must not make dissolved_oxygen_ppm
    unreachable under the 'water' context -- it should remain an eligible,
    informative question that real information gain can still surface."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="water")
    engine.answer(case_id, "gasping", state="true")
    engine.answer(case_id, "surface_breathing", state="true")

    status = engine.get_status(case_id)
    answered = engine.store.get(case_id).answered_evidence_ids()
    candidates = eligible_questions(kb, answered)
    assert any(q.evidence_id == "dissolved_oxygen_ppm" for q in candidates)

    post = posterior(score_problems(kb, engine.store.get(case_id).active_observations()))
    explanation = build_question_explanation(
        kb, "ask_dissolved_oxygen_ppm", post, engine.store.get(case_id).active_observations(), "water"
    )
    assert explanation.information_gain > 0
    assert status.best_next_question is not None


def test_dissolved_oxygen_topic_change_does_not_affect_scoring(kb):
    """Evidence.topic is a routing-only label -- changing it must not
    alter diagnostic scoring for observations on this evidence item."""
    observations = [obs("dissolved_oxygen_ppm", "critical"), obs("gasping", "true")]
    baseline = posterior(score_problems(kb, observations))
    for _ in range(3):
        assert posterior(score_problems(kb, observations)) == baseline
    ranked = sorted(baseline, key=lambda pid: baseline[pid], reverse=True)
    assert ranked[0] == "low_oxygen"
