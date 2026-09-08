"""Safety rules -- deliberately independent of the Bayesian scoring math.

Evaluated fresh every turn against the case's current observations. A
firing rule always surfaces ahead of the ranked candidates, and can pin a
forced_question_id that preempts the normal information-gain argmax for
that turn -- so information gain can never override a safety concern.
"""
from __future__ import annotations

from dataclasses import dataclass

from aqua_assistant.case.models import Observation
from aqua_assistant.kb.knowledge_base import KnowledgeBase
from aqua_assistant.kb.schema import SafetyRule

_SEVERITY_RANK = {"urgent": 2, "caution": 1, "info": 0}


@dataclass
class SafetyAlert:
    rule_id: str
    severity: str
    message: str
    escalation_text: str
    forced_question_id: str | None


def _condition_met(observed: dict[str, str], evidence_id: str, equals_state: str) -> bool:
    return observed.get(evidence_id) == equals_state


def _rule_fires(rule: SafetyRule, observed: dict[str, str]) -> bool:
    if not rule.all_of and not rule.any_of:
        return False
    all_ok = all(_condition_met(observed, c.evidence_id, c.equals_state) for c in rule.all_of)
    any_ok = any(_condition_met(observed, c.evidence_id, c.equals_state) for c in rule.any_of) if rule.any_of else True
    return all_ok and any_ok


def evaluate_safety(kb: KnowledgeBase, observations: list[Observation]) -> list[SafetyAlert]:
    """Returns every currently-firing rule, most severe first (deterministic
    tie-break by rule id). Two simultaneously urgent conditions (e.g.
    critical ammonia AND critical nitrite) must both be surfaced -- silently
    keeping only one would mean a real safety concern goes unreported.
    Only the highest-ranked alert's forced_question_id is honored by the
    engine, so at most one forced question is ever active at a time."""
    observed = {o.evidence_id: o.observed_state for o in observations if not o.superseded}
    firing = [r for r in kb.safety_rules if _rule_fires(r, observed)]
    firing.sort(key=lambda r: (-_SEVERITY_RANK.get(r.severity.value, 0), r.id))
    return [
        SafetyAlert(
            rule_id=r.id,
            severity=r.severity.value,
            message=r.message,
            escalation_text=r.escalation_text,
            forced_question_id=r.forced_question_id,
        )
        for r in firing
    ]
