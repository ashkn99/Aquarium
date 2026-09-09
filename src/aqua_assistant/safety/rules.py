"""Safety rules -- deliberately independent of the Bayesian scoring math.

Evaluated fresh every turn against the case's current observations. A
firing rule always surfaces ahead of the ranked candidates, and can pin a
forced_question_id that preempts the normal information-gain argmax for
that turn -- so information gain can never override a safety concern.

`pending_safety_evidence()` and `screening_evidence_ids()` are the two
additions consumed outside this module (by questions/selection.py):

- `pending_safety_evidence()` identifies unanswered evidence belonging
  to a safety rule that is *suspected* -- not merely unresolved, but
  with at least one of its own conditions already observed matching --
  so selection can prioritize confirming/ruling it out over unrelated
  diagnostic markers. Safety behaves as an interrupt this way: it only
  takes over once something concrete points at it, and lets go the
  moment the rule fires or is ruled out.
- `screening_evidence_ids()` identifies evidence belonging to a rule
  with no `all_of` clause -- structurally incapable of showing partial
  signal, so "suspected" doesn't apply to it at all. Selection consumes
  this shape-based fact for its own, domain-scoped mandatory-screen tier
  (see questions/selection.py::select_best_question) rather than this
  module deciding when it applies -- that decision needs entry_context,
  which this module deliberately knows nothing about.

Both are still purely reads of the same rule definitions -- no new rule
concept, no evidence ids hardcoded, and neither decides whether a rule
fires.
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


def screening_evidence_ids(kb: KnowledgeBase) -> set[str]:
    """Evidence gated by a rule with no `all_of` clause -- the same
    KB-shape-derived definition `_rule_suspected` uses for "must be
    screened directly, not gated on suspicion" (e.g. gasping under
    respiratory_distress_general).

    Exposed so question selection can exclude these from entry-context
    routing's decay clock (see questions/selection.py::routing_bonus):
    they get answered as part of the universal TRIAGE screen before
    orientation has had any real turns to act on, and must not silently
    burn down routing's opening window before it gets to run."""
    return {c.evidence_id for r in kb.safety_rules if not r.all_of for c in r.any_of}


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


def _all_of_ruled_out(rule: SafetyRule, observed: dict[str, str]) -> bool:
    """True once any all_of condition has been answered with a state other
    than the one required -- the rule can never fire from here, regardless
    of what else gets answered."""
    return any(c.evidence_id in observed and observed[c.evidence_id] != c.equals_state for c in rule.all_of)


def _any_of_ruled_out(rule: SafetyRule, observed: dict[str, str]) -> bool:
    """True once every any_of condition has been answered and none matched
    -- the any_of branch is dead, so (combined with all_of) the rule can
    never fire. A rule with no any_of clause is never ruled out this way."""
    if not rule.any_of:
        return False
    if not all(c.evidence_id in observed for c in rule.any_of):
        return False
    return not any(_condition_met(observed, c.evidence_id, c.equals_state) for c in rule.any_of)


def _rule_suspected(rule: SafetyRule, observed: dict[str, str]) -> bool:
    """True once at least one of the rule's own conditions is already
    observed matching -- i.e. something concrete already points toward
    this specific rule. This is what makes safety behave like an
    interrupt rather than a standing checklist: a rule with nothing
    observed yet is merely "not ruled out," not "suspected," and must not
    compete for priority against ordinary diagnostic questions.

    A rule with no `all_of` clause (nothing to anchor a partial-signal
    check on -- e.g. respiratory_distress_general's bare "gasping") is
    therefore never "suspected" by this function
    from a clean case; it's handled separately, as a domain-scoped
    mandatory screen, by questions/selection.py (see
    screening_evidence_ids() below and select_best_question's docstring)
    rather than here -- this function only ever means genuine, evidence-
    backed suspicion."""
    return any(_condition_met(observed, c.evidence_id, c.equals_state) for c in (*rule.all_of, *rule.any_of))


def pending_safety_evidence(kb: KnowledgeBase, observations: list[Observation]) -> set[str]:
    """Evidence ids that still matter to a safety rule which is currently
    *suspected* -- neither already firing (the concern is already
    established and surfaced via evaluate_safety), nor ruled out (a
    contradicted condition means it can never fire from here), nor merely
    theoretical (nothing observed yet has actually pointed at it). Purely
    a *selection* signal: it never changes whether a rule fires, what it
    says, or any diagnostic score -- it only tells question selection
    which unanswered evidence still needs confirming for a live, real
    suspicion, derived entirely from the existing safety_rules
    definitions (no evidence ids hardcoded here).

    A rule drops out of the pending set the moment it fires, is ruled
    out, or (from a fresh case) simply hasn't been indicated by anything
    observed yet -- "the relevant risk is adequately established, ruled
    out, or was never actually suggested" is exactly this condition."""
    observed = {o.evidence_id: o.observed_state for o in observations if not o.superseded}
    pending: set[str] = set()
    for rule in kb.safety_rules:
        if not rule.all_of and not rule.any_of:
            continue
        if _rule_fires(rule, observed):
            continue
        if _all_of_ruled_out(rule, observed) or _any_of_ruled_out(rule, observed):
            continue
        if not _rule_suspected(rule, observed):
            continue
        for cond in (*rule.all_of, *rule.any_of):
            if cond.evidence_id not in observed:
                pending.add(cond.evidence_id)
    return pending
