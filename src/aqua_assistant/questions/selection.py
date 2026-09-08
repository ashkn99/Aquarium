"""Information-gain-driven question selection.

For each eligible question, simulate every possible answer, measure how
much it would be expected to shrink entropy, then adjust that raw
information gain by reliability / priority / effort before ranking.
Reuses the exact same scoring function as real answers, so redundant or
low-value questions (e.g. a second question in an already-confirmed
correlated-evidence group) fall out with low value naturally, instead of
needing hand-coded "don't ask this again" rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.entropy import entropy
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.kb.knowledge_base import KnowledgeBase


@dataclass
class QuestionExplanation:
    question_id: str
    text: str
    information_gain: float
    reliability: float
    priority_weight: float
    effort_cost: float
    adjusted_value: float
    distinguishes: list[str] = field(default_factory=list)


def eligible_questions(kb: KnowledgeBase, answered_evidence_ids: set[str]):
    return [q for q in kb.questions.values() if q.active and q.evidence_id not in answered_evidence_ids]


def possible_states(kb: KnowledgeBase, evidence_id: str) -> list[str]:
    """All states a user could realistically report for this evidence.

    Must include every state offered by that evidence's question(s)
    (question_answers), not just the states that happen to have a
    problem_evidence likelihood row -- otherwise a state with no likelihood
    data (e.g. "no" for a symptom nobody bothered to write a negative row
    for) silently drops out of the answer simulation in
    expected_information_gain, understating that question's real
    information gain instead of correctly treating the missing state as
    uninformative (contribution 0) like any other unmodeled state.
    """
    ranges = kb.evidence_ranges_by_evidence.get(evidence_id)
    if ranges:
        return [r.range_key for r in ranges]

    states = {pe.evidence_state for pe in kb.problem_evidence if pe.evidence_id == evidence_id}
    for q in kb.questions_by_evidence.get(evidence_id, []):
        states.update(a.evidence_state for a in kb.answers_by_question.get(q.id, []))
    return sorted(states)


def _answer_probability(kb: KnowledgeBase, posterior_dist: dict[str, float], evidence_id: str, state: str) -> float:
    """Marginal P(evidence=state) = Sum_j P(problem_j) * P(state|problem_j),
    falling back to a neutral 0.5 for problems with no likelihood row for
    this evidence (uninformative, so it shouldn't sway the estimate)."""
    total = 0.0
    for pid, p in posterior_dist.items():
        pe = kb.likelihood(pid, evidence_id, state)
        total += p * (pe.likelihood_positive if pe is not None else 0.5)
    return total


def simulate_answer(
    kb: KnowledgeBase,
    observations: list[Observation],
    evidence_id: str,
    state: str,
    confidence: float = 0.9,
) -> dict[str, float]:
    simulated = [*observations, Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)]
    return posterior(score_problems(kb, simulated))


def expected_information_gain(
    kb: KnowledgeBase,
    posterior_dist: dict[str, float],
    observations: list[Observation],
    evidence_id: str,
) -> float:
    current_entropy = entropy(posterior_dist)
    states = possible_states(kb, evidence_id)
    if not states:
        return 0.0

    raw_probs = {state: _answer_probability(kb, posterior_dist, evidence_id, state) for state in states}
    total_prob = sum(raw_probs.values()) or 1.0

    expected_entropy = 0.0
    for state, raw_p in raw_probs.items():
        p = raw_p / total_prob
        if p <= 0:
            continue
        expected_entropy += p * entropy(simulate_answer(kb, observations, evidence_id, state))

    return max(0.0, current_entropy - expected_entropy)


def _distinguishes(kb: KnowledgeBase, posterior_dist: dict[str, float], evidence_id: str, top_n: int = 3) -> list[str]:
    informative_problems = kb.problems_informed_by_evidence.get(evidence_id, set())
    top = sorted(posterior_dist, key=lambda pid: posterior_dist[pid], reverse=True)[:top_n]
    return [pid for pid in top if pid in informative_problems] or top


def build_question_explanation(
    kb: KnowledgeBase,
    question_id: str,
    posterior_dist: dict[str, float],
    observations: list[Observation],
) -> QuestionExplanation:
    q = kb.questions[question_id]
    info_gain = expected_information_gain(kb, posterior_dist, observations, q.evidence_id)
    adjusted = info_gain * q.reliability * q.priority_weight / q.effort_cost
    return QuestionExplanation(
        question_id=q.id,
        text=q.text,
        information_gain=info_gain,
        reliability=q.reliability,
        priority_weight=q.priority_weight,
        effort_cost=q.effort_cost,
        adjusted_value=adjusted,
        distinguishes=_distinguishes(kb, posterior_dist, q.evidence_id),
    )


def select_best_question(
    kb: KnowledgeBase,
    posterior_dist: dict[str, float],
    observations: list[Observation],
    answered_evidence_ids: set[str],
) -> QuestionExplanation | None:
    candidates = eligible_questions(kb, answered_evidence_ids)
    if not candidates:
        return None
    explanations = [build_question_explanation(kb, q.id, posterior_dist, observations) for q in candidates]
    explanations.sort(key=lambda e: (-e.adjusted_value, e.effort_cost, e.question_id))
    return explanations[0]
