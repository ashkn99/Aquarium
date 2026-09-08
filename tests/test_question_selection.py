from __future__ import annotations

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.questions.selection import (
    eligible_questions,
    expected_information_gain,
    select_best_question,
)


def uniform_posterior(kb):
    return posterior(score_problems(kb, []))


def test_answered_evidence_is_excluded_from_eligibility(synthetic_kb):
    remaining = eligible_questions(synthetic_kb, answered_evidence_ids={"e_discriminator"})
    assert all(q.evidence_id != "e_discriminator" for q in remaining)


def test_highly_discriminative_question_outranks_low_value_question(synthetic_kb):
    post = uniform_posterior(synthetic_kb)
    best = select_best_question(synthetic_kb, post, [], answered_evidence_ids=set())
    assert best.question_id == "q_discriminator"
    # sanity: the deliberately low-priority/high-effort question is not the winner
    assert best.question_id != "q_low_value"


def test_no_eligible_questions_returns_none(synthetic_kb):
    all_evidence_ids = {q.evidence_id for q in synthetic_kb.questions.values()}
    post = uniform_posterior(synthetic_kb)
    result = select_best_question(synthetic_kb, post, [], answered_evidence_ids=all_evidence_ids)
    assert result is None


def test_redundant_group_question_has_lower_gain_after_dominant_item_answered(synthetic_kb):
    post_before = uniform_posterior(synthetic_kb)
    gain_before = expected_information_gain(synthetic_kb, post_before, [], "e_g1")

    already_answered = [Observation(evidence_id="e_g1", observed_state="true", confidence=1.0)]
    post_after = posterior(score_problems(synthetic_kb, already_answered))
    gain_after_for_sibling = expected_information_gain(synthetic_kb, post_after, already_answered, "e_g2")

    assert gain_after_for_sibling < gain_before


def test_explanation_reports_information_gain_breakdown(synthetic_kb):
    post = uniform_posterior(synthetic_kb)
    best = select_best_question(synthetic_kb, post, [], answered_evidence_ids=set())
    assert best.information_gain >= 0.0
    assert best.reliability > 0.0
    assert best.effort_cost > 0.0
    assert best.adjusted_value == best.information_gain * best.reliability * best.priority_weight / best.effort_cost
    assert "p1" in best.distinguishes or "p2" in best.distinguishes
