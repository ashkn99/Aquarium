from __future__ import annotations

import math

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.scoring import observation_contribution, posterior, score_problems


def obs(evidence_id: str, state: str, confidence: float = 1.0) -> Observation:
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)


def test_single_observation_contribution_is_exact_log_lr(synthetic_kb):
    # likelihood_positive=0.8, likelihood_background=0.2 -> LR=4 -> log2(4)=2.0
    contribution = observation_contribution(synthetic_kb, "p1", obs("e_ungrouped", "true"))
    assert math.isclose(contribution, 2.0, rel_tol=1e-9)


def test_confidence_scales_contribution_linearly(synthetic_kb):
    full = observation_contribution(synthetic_kb, "p1", obs("e_ungrouped", "true", confidence=1.0))
    half = observation_contribution(synthetic_kb, "p1", obs("e_ungrouped", "true", confidence=0.5))
    assert math.isclose(half, full / 2, rel_tol=1e-9)


def test_uninformative_evidence_contributes_zero(synthetic_kb):
    # p2 has no problem_evidence row for e_ungrouped at all.
    contribution = observation_contribution(synthetic_kb, "p2", obs("e_ungrouped", "true"))
    assert contribution == 0.0


def test_negative_evidence_reduces_candidate_score(synthetic_kb):
    baseline = score_problems(synthetic_kb, [])["p1"]
    with_negative = score_problems(synthetic_kb, [obs("e_neg", "true")])["p1"]
    # likelihood_positive=0.1 < likelihood_background=0.5 -> LR<1 -> negative contribution.
    assert with_negative < baseline


def test_correlated_group_is_discounted_vs_independent_evidence(synthetic_kb):
    grouped_score = score_problems(synthetic_kb, [obs("e_g1", "true"), obs("e_g2", "true"), obs("e_g3", "true")])["p1"]
    # Each item alone contributes log2(4)=2.0. Group discount is 0.5, so
    # the combined total should be 2*(1 + 0.5 + 0.25) = 3.5, not 6.0.
    assert math.isclose(grouped_score, 3.5, rel_tol=1e-9)

    # Three *independent* (ungrouped) items of equal strength would sum in full.
    # e_ungrouped only has one instance in the synthetic KB, so just check the
    # single-item contribution scales as expected and is less than the naive triple.
    naive_triple = 2.0 * 3
    assert grouped_score < naive_triple


def test_group_discount_ranks_by_strength_not_insertion_order(synthetic_kb):
    # Answering in a different order should give the same total, since the
    # engine sorts by |contribution| before applying rank-based discount.
    forward = score_problems(synthetic_kb, [obs("e_g1", "true"), obs("e_g2", "true"), obs("e_g3", "true")])["p1"]
    reversed_order = score_problems(synthetic_kb, [obs("e_g3", "true"), obs("e_g2", "true"), obs("e_g1", "true")])["p1"]
    assert math.isclose(forward, reversed_order, rel_tol=1e-9)


def test_posterior_is_normalized_probability_distribution(synthetic_kb):
    scores = score_problems(synthetic_kb, [obs("e_discriminator", "true")])
    post = posterior(scores)
    assert math.isclose(sum(post.values()), 1.0, rel_tol=1e-9)
    assert all(0.0 <= p <= 1.0 for p in post.values())


def test_strong_evidence_shifts_posterior_toward_matching_problem(synthetic_kb):
    post_for_p1 = posterior(score_problems(synthetic_kb, [obs("e_discriminator", "true")]))
    post_for_p2 = posterior(score_problems(synthetic_kb, [obs("e_discriminator", "false")]))
    assert post_for_p1["p1"] > post_for_p1["p2"]
    assert post_for_p2["p2"] > post_for_p2["p1"]


def test_posterior_never_reaches_exact_zero_or_one(synthetic_kb):
    # Even very strong evidence must leave room for "cannot rule out" language.
    post = posterior(score_problems(synthetic_kb, [obs("e_discriminator", "true")]))
    assert 0.0 < post["p2"] < 1.0
    assert 0.0 < post["p1"] < 1.0


def test_superseded_observations_are_excluded_from_scoring(synthetic_kb):
    active = obs("e_ungrouped", "true")
    superseded = obs("e_ungrouped", "false")
    superseded.superseded = True
    score_with_only_active = score_problems(synthetic_kb, [active])["p1"]
    score_with_superseded_included = score_problems(synthetic_kb, [superseded, active])["p1"]
    assert math.isclose(score_with_only_active, score_with_superseded_included, rel_tol=1e-9)
