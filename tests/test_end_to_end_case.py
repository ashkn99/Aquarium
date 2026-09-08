"""End-to-end acceptance test for the MVP milestone (M6).

Each test below is annotated with the numbered success criterion from the
project brief it verifies:

  1. Start with reasonable priors.
  2. Ask an informative question.
  3. Update candidates after the answer.
  4. Avoid redundant questions.
  5. Select a different question based on new evidence.
  6. Converge toward the correct problem(s).
  7. Stop when additional questions have little value.
  8. Explain why candidates were ranked.
  9. Never let information gain override safety rules.
"""
from __future__ import annotations

import pytest

from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.kb.schema import SafetyCondition, SafetyRule
from aqua_assistant.questions.selection import possible_states

from .conftest import neutral_default

# A full answer script keyed by evidence_id, so whichever question the
# engine dynamically picks (order isn't fixed), a scripted answer is ready.
# This describes a fairly clear-cut ammonia-toxicity case.
AMMONIA_CASE_ANSWERS: dict[str, tuple[str, object]] = {
    "gasping": ("state", "true"),
    "rapid_breathing": ("state", "true"),
    "surface_breathing": ("state", "true"),
    "lethargy": ("state", "true"),
    "appetite_loss": ("state", "true"),
    "hiding": ("state", "false"),
    "white_spots": ("state", "false"),
    "frayed_fins": ("state", "false"),
    "clamped_fins": ("state", "false"),
    "red_streaks_or_gills": ("state", "true"),
    "excess_mucus": ("state", "false"),
    "flashing_scratching": ("state", "false"),
    "num_fish_affected": ("state", "most_or_all"),
    "onset_timing": ("state", "sudden"),
    "ammonia_ppm": ("raw_value", 3.0),
    "nitrite_ppm": ("raw_value", 0.1),
    "nitrate_ppm": ("raw_value", 10.0),
    "ph_level": ("raw_value", 7.0),
    "temperature_f": ("raw_value", 78.0),
    "dissolved_oxygen_ppm": ("raw_value", 6.0),
    "tank_age_days": ("raw_value", 200.0),
    "recent_fish_added": ("state", "false"),
    "recent_water_change_large": ("state", "false"),
    "filter_disrupted": ("state", "true"),
    "ph_swings_reported": ("state", "false"),
    "used_untreated_tap_water": ("state", "false"),
    "water_change_frequency": ("state", "regular"),
    "copper_source_exposure": ("state", "false"),
    "overstocking_reported": ("state", "false"),
    "planted_tank_with_co2_injection": ("state", "false"),
    "visible_debris_or_detritus_buildup": ("state", "false"),
    "cottony_fuzzy_growth": ("state", "false"),
    "gold_rust_dust_coating": ("state", "false"),
    "white_fuzzy_patches_around_mouth": ("state", "false"),
    "abnormal_buoyancy_or_swimming": ("state", "false"),
    "pinecone_scales": ("state", "false"),
    "bloated_body": ("state", "false"),
}


def run_scripted_case(engine: AquariumInferenceEngine, answers: dict, max_turns: int = 30):
    """Drives a case to completion using the answer script, returning the
    list of (turn_status, question_asked) pairs and the final status.

    Falls back to conftest.neutral_default() for any evidence the script
    doesn't cover (e.g. evidence added by a later KB batch) -- an
    unscripted answer must never accidentally read as remarkable/positive
    for this fixed ammonia-toxicity scenario."""
    case_id = engine.start_case()
    turns = []
    asked_evidence_ids: list[str] = []

    for _ in range(max_turns):
        status = engine.get_status(case_id)
        turns.append(status)
        if status.should_stop:
            return case_id, turns, asked_evidence_ids

        q = status.best_next_question
        evidence_id = engine.kb.questions[q.question_id].evidence_id
        assert evidence_id not in asked_evidence_ids, f"redundant question re-asked for {evidence_id}"
        asked_evidence_ids.append(evidence_id)

        kind, value = answers.get(evidence_id) or neutral_default(engine.kb, evidence_id)
        if kind == "raw_value":
            engine.answer(case_id, evidence_id, raw_value=value, question_id=q.question_id)
        else:
            engine.answer(case_id, evidence_id, state=value, question_id=q.question_id)

    raise AssertionError("case did not converge within max_turns")


@pytest.fixture(scope="module")
def seed_kb():
    return load_knowledge_base()


def test_criterion_1_starts_with_reasonable_nondegenerate_priors(seed_kb):
    engine = AquariumInferenceEngine(seed_kb)
    case_id = engine.start_case()
    status = engine.get_status(case_id)

    probs = [c.probability for c in status.ranked_candidates]
    assert len(probs) == len(seed_kb.problems)
    assert all(0.0 < p < 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    # No candidate should dominate before any evidence is seen.
    assert max(probs) < 0.5


def test_criterion_2_first_question_is_informative(seed_kb):
    engine = AquariumInferenceEngine(seed_kb)
    case_id = engine.start_case()
    status = engine.get_status(case_id)
    assert status.best_next_question is not None
    assert status.best_next_question.information_gain > 0.0
    assert status.best_next_question.adjusted_value > 0.0


def test_criterion_3_candidates_update_after_an_answer(seed_kb):
    engine = AquariumInferenceEngine(seed_kb)
    case_id = engine.start_case()
    before = {c.problem_id: c.probability for c in engine.get_status(case_id).ranked_candidates}

    engine.answer(case_id, "ammonia_ppm", raw_value=3.0)
    after = {c.problem_id: c.probability for c in engine.get_status(case_id).ranked_candidates}

    assert before != after
    assert after["ammonia_toxicity"] > before["ammonia_toxicity"]


def test_criterion_4_and_6_full_case_has_no_redundant_questions_and_converges(seed_kb):
    engine = AquariumInferenceEngine(seed_kb)
    case_id, turns, asked = run_scripted_case(engine, AMMONIA_CASE_ANSWERS)

    assert len(asked) == len(set(asked)), "a question's evidence was asked more than once"

    final_status = engine.get_status(case_id)
    top = final_status.ranked_candidates[0]
    assert top.problem_id == "ammonia_toxicity"
    assert top.probability > final_status.ranked_candidates[1].probability


def test_criterion_5_different_evidence_leads_to_a_different_next_question(seed_kb):
    # Two cases diverge on the very first answer; the second question asked
    # should differ between them, proving selection responds to new evidence
    # rather than following a fixed script.
    engine_a = AquariumInferenceEngine(seed_kb)
    case_a = engine_a.start_case()
    first_q = engine_a.get_status(case_a).best_next_question
    first_evidence = engine_a.kb.questions[first_q.question_id].evidence_id

    engine_b = AquariumInferenceEngine(seed_kb)
    case_b = engine_b.start_case()

    # Branch A/B answer with the two most extreme possible states for
    # whichever evidence type got asked first (numeric, boolean, or
    # categorical), so this doesn't assume the first question's data type.
    if first_evidence in seed_kb.evidence_ranges_by_evidence:
        engine_a.answer(case_a, first_evidence, raw_value=1_000_000, question_id=first_q.question_id)
        engine_b.answer(case_b, first_evidence, raw_value=0.0, question_id=first_q.question_id)
    else:
        states = possible_states(seed_kb, first_evidence)
        assert len(states) >= 2
        engine_a.answer(case_a, first_evidence, state=states[0], question_id=first_q.question_id)
        engine_b.answer(case_b, first_evidence, state=states[-1], question_id=first_q.question_id)

    second_q_a = engine_a.get_status(case_a).best_next_question
    second_q_b = engine_b.get_status(case_b).best_next_question

    # The two branches must diverge: either they land on different next
    # questions, or one answer was so diagnostic it converged the case
    # outright (best_next_question=None) while the other didn't -- both
    # outcomes prove selection responded to which evidence was seen.
    if second_q_a is not None and second_q_b is not None:
        assert second_q_a.question_id != second_q_b.question_id
    else:
        assert (second_q_a is None) != (second_q_b is None)


def test_criterion_7_stops_with_a_reason_and_no_further_question(seed_kb):
    engine = AquariumInferenceEngine(seed_kb)
    case_id, turns, asked = run_scripted_case(engine, AMMONIA_CASE_ANSWERS)
    final_status = engine.get_status(case_id)

    assert final_status.should_stop is True
    assert final_status.stop_reason is not None
    assert final_status.best_next_question is None
    # Convergence happened before exhausting every question in the KB.
    assert len(asked) < len(seed_kb.questions)


def test_criterion_8_explains_why_the_top_candidate_was_ranked(seed_kb):
    engine = AquariumInferenceEngine(seed_kb)
    case_id, turns, asked = run_scripted_case(engine, AMMONIA_CASE_ANSWERS)
    final_status = engine.get_status(case_id)

    top = final_status.ranked_candidates[0]
    assert top.problem_id == "ammonia_toxicity"
    assert "ammonia_ppm" in top.supporting_evidence
    # A candidate ruled out by strong contrary evidence should show it.
    ich = next(c for c in final_status.ranked_candidates if c.problem_id == "ich")
    assert "white_spots" in ich.contradicting_evidence


def test_criterion_9_safety_alert_forces_question_over_natural_information_gain(seed_kb):
    """Builds a variant KB where the forced safety question is deliberately
    a poor information-gain choice, to unambiguously prove the override --
    not just that it happens to coincide with the natural pick."""
    kb = load_knowledge_base()
    # ask_ph_swings_reported has middling priority/high-ish effort and,
    # with nothing else answered, is not the natural argmax pick.
    kb.safety_rules = [
        SafetyRule(
            id="test_forced_override",
            description="test-only rule",
            all_of=[SafetyCondition(evidence_id="ammonia_ppm", equals_state="critical")],
            severity="urgent",
            message="test",
            forced_question_id="ask_ph_swings_reported",
        )
    ]
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case()

    # Confirm what unconstrained info-gain selection would have picked, for contrast.
    from aqua_assistant.inference.scoring import posterior, score_problems
    from aqua_assistant.questions.selection import select_best_question

    natural_pick = select_best_question(kb, posterior(score_problems(kb, [])), [], set())
    assert natural_pick.question_id != "ask_ph_swings_reported"

    engine.answer(case_id, "ammonia_ppm", raw_value=3.0)
    status = engine.get_status(case_id)

    assert status.safety_alerts
    assert status.safety_alerts[0].forced_question_id == "ask_ph_swings_reported"
    assert status.best_next_question.question_id == "ask_ph_swings_reported"
