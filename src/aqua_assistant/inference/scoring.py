"""Bayesian-style log-likelihood-ratio scoring. Pure functions, no I/O —
this is the part of the system that must work without an LLM.
"""
from __future__ import annotations

import math

from aqua_assistant.case.models import Observation
from aqua_assistant.kb.knowledge_base import KnowledgeBase

# P(state|problem) and P(state|not-problem) are clamped away from 0/1 so a
# single observation can never mathematically force a score to +/- infinity
# (and a posterior probability to exactly 0 or 1) -- outputs must stay
# "most likely" / "cannot rule out", never absolute certainty.
_EPS = 1e-6


def _clamp(p: float) -> float:
    return min(max(p, _EPS), 1 - _EPS)


def observation_contribution(kb: KnowledgeBase, problem_id: str, obs: Observation) -> float:
    """Signed log2 likelihood-ratio contribution of one observation to one
    problem's score, scaled by the observation's confidence.

    Returns 0.0 if the KB has no likelihood row for this
    (problem, evidence, state) combination -- i.e. this evidence is
    considered uninformative for that problem, not evidence against it.
    """
    pe = kb.likelihood(problem_id, obs.evidence_id, obs.observed_state)
    if pe is None:
        return 0.0
    lr = _clamp(pe.likelihood_positive) / _clamp(pe.likelihood_background)
    contribution = obs.confidence * math.log2(lr)
    if pe.weight_cap is not None:
        contribution = max(-pe.weight_cap, min(pe.weight_cap, contribution))
    return contribution


def _grouped_contribution_total(kb: KnowledgeBase, problem_id: str, observations: list[Observation]) -> float:
    """Sum contributions for one problem, discounting correlated evidence
    within the same evidence_group: e.g. gasping + rapid_breathing +
    surface_breathing are all "respiratory distress" and shouldn't count as
    three independent confirmations. The strongest signal in a group counts
    fully; each additional member is discounted by `correlation_discount`
    raised to its rank. Ungrouped evidence always contributes in full.
    """
    by_group: dict[str | None, list[float]] = {}
    for obs in observations:
        ev = kb.evidence.get(obs.evidence_id)
        group_id = ev.evidence_group_id if ev else None
        by_group.setdefault(group_id, []).append(observation_contribution(kb, problem_id, obs))

    total = 0.0
    for group_id, contributions in by_group.items():
        if group_id is None:
            total += sum(contributions)
            continue
        discount = kb.evidence_groups[group_id].correlation_discount
        contributions.sort(key=abs, reverse=True)
        for rank, contribution in enumerate(contributions):
            total += contribution * (discount**rank)
    return total


def score_problems(
    kb: KnowledgeBase,
    observations: list[Observation],
    problem_ids: list[str] | None = None,
) -> dict[str, float]:
    """Raw per-problem log2-scores: log2(prior_weight) + discounted sum of
    observation contributions. Not yet a probability distribution -- see
    `posterior()`.
    """
    ids = problem_ids if problem_ids is not None else list(kb.problems.keys())
    active = [o for o in observations if not o.superseded]
    scores: dict[str, float] = {}
    for pid in ids:
        prior = kb.problems[pid].prior_weight
        scores[pid] = math.log2(max(prior, _EPS)) + _grouped_contribution_total(kb, pid, active)
    return scores


def posterior(scores: dict[str, float]) -> dict[str, float]:
    """Softmax over log2-scores -> a normalized relative-plausibility
    distribution across candidate problems.

    This is a pragmatic differential-diagnosis-style normalization, not a
    strict single-partition Bayesian posterior (candidate problems aren't
    guaranteed mutually exclusive/exhaustive) -- treat it as a ranking
    tool, not a calibrated probability of ground truth.
    """
    if not scores:
        return {}
    max_score = max(scores.values())
    exponentials = {pid: 2 ** (s - max_score) for pid, s in scores.items()}
    total = sum(exponentials.values())
    return {pid: v / total for pid, v in exponentials.items()}
