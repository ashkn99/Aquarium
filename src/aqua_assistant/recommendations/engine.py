"""Recommendation lookup -- deliberately dumb and deterministic.

Diagnosis (probability) and recommendations (what to do) are kept
separate: this module never computes a score, it only looks up
pre-authored guidance for whichever problems the inference engine ranked
highest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aqua_assistant.kb.knowledge_base import KnowledgeBase

_KIND_TO_FIELD = {
    "confirmation_check": "confirmation_checks",
    "immediate_action": "immediate_actions",
    "follow_up_action": "follow_up_actions",
    "avoid": "avoid",
    "escalation_criterion": "escalation_criteria",
}


@dataclass
class RecommendationBundle:
    problem_id: str
    problem_name: str
    confirmation_checks: list[str] = field(default_factory=list)
    immediate_actions: list[str] = field(default_factory=list)
    follow_up_actions: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    escalation_criteria: list[str] = field(default_factory=list)


def get_recommendations(kb: KnowledgeBase, problem_id: str) -> RecommendationBundle:
    bundle = RecommendationBundle(problem_id=problem_id, problem_name=kb.problems[problem_id].name)
    for rec in kb.recommendations_by_problem.get(problem_id, []):
        field_name = _KIND_TO_FIELD.get(rec.kind)
        if field_name:
            getattr(bundle, field_name).append(rec.text)
    return bundle


def get_recommendations_for_top_candidates(
    kb: KnowledgeBase, ranked_problem_ids: list[str], top_n: int = 3
) -> list[RecommendationBundle]:
    return [get_recommendations(kb, pid) for pid in ranked_problem_ids[:top_n]]
