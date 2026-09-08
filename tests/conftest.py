"""Shared test fixtures.

`synthetic_kb` is a small, hand-built KnowledgeBase (not loaded from YAML)
so scoring/entropy/question-selection tests can assert exact,
hand-computed numbers instead of reasoning about the full seed KB.
"""
from __future__ import annotations

import pytest

from aqua_assistant.kb.knowledge_base import KnowledgeBase
from aqua_assistant.kb.schema import (
    Evidence,
    EvidenceGroup,
    Problem,
    ProblemEvidence,
    Question,
    QuestionAnswer,
)

# Per-parameter "unremarkable" numeric readings, for scenario tests that
# need to answer whatever unscripted numeric question the engine happens
# to pick next. A single constant (e.g. 0.0) is NOT safe here: 0.0 means
# "safe" for ammonia/nitrite but "critical" for dissolved oxygen, so it
# silently injects a fabricated crisis into what's meant to be a neutral
# answer. This caused several false test failures during KB expansion
# before being centralized here -- reuse this in every new batch's tests.
NEUTRAL_NUMERIC_DEFAULTS: dict[str, float] = {
    "ammonia_ppm": 0.0,
    "nitrite_ppm": 0.0,
    "nitrate_ppm": 10.0,
    "ph_level": 7.2,
    "temperature_f": 78.0,
    "dissolved_oxygen_ppm": 6.5,
    "tank_age_days": 150.0,
}


def neutral_default(kb: KnowledgeBase, evidence_id: str) -> tuple[str, object]:
    """('raw_value', x) or ('state', s) -- an unremarkable/negative answer
    for whatever evidence_id an unscripted turn happens to ask about."""
    evidence = kb.evidence[evidence_id]
    if evidence.data_type == "numeric":
        return "raw_value", NEUTRAL_NUMERIC_DEFAULTS.get(evidence_id, 0.0)
    from aqua_assistant.questions.selection import possible_states

    states = possible_states(kb, evidence_id)
    return "state", ("false" if "false" in states else states[0])


@pytest.fixture
def synthetic_kb() -> KnowledgeBase:
    problems = {
        "p1": Problem(id="p1", name="Problem One", prior_weight=1.0),
        "p2": Problem(id="p2", name="Problem Two", prior_weight=1.0),
    }

    evidence = {
        "e_ungrouped": Evidence(id="e_ungrouped", name="Ungrouped", data_type="boolean"),
        "e_neg": Evidence(id="e_neg", name="Negative signal", data_type="boolean"),
        "e_g1": Evidence(id="e_g1", name="Group member 1", data_type="boolean", evidence_group_id="g"),
        "e_g2": Evidence(id="e_g2", name="Group member 2", data_type="boolean", evidence_group_id="g"),
        "e_g3": Evidence(id="e_g3", name="Group member 3", data_type="boolean", evidence_group_id="g"),
        # A maximally discriminative item: true strongly implies p1, false strongly implies p2.
        "e_discriminator": Evidence(id="e_discriminator", name="Discriminator", data_type="boolean"),
    }

    groups = {"g": EvidenceGroup(id="g", name="Correlated group", correlation_discount=0.5)}

    # log2(0.8/0.2) == 2.0 exactly -- chosen so contributions are round numbers.
    problem_evidence = [
        ProblemEvidence(problem_id="p1", evidence_id="e_ungrouped", evidence_state="true", likelihood_positive=0.8, likelihood_background=0.2),
        ProblemEvidence(problem_id="p1", evidence_id="e_ungrouped", evidence_state="false", likelihood_positive=0.2, likelihood_background=0.8),
        ProblemEvidence(problem_id="p1", evidence_id="e_neg", evidence_state="true", likelihood_positive=0.1, likelihood_background=0.5),
        ProblemEvidence(problem_id="p1", evidence_id="e_g1", evidence_state="true", likelihood_positive=0.8, likelihood_background=0.2),
        ProblemEvidence(problem_id="p1", evidence_id="e_g2", evidence_state="true", likelihood_positive=0.8, likelihood_background=0.2),
        ProblemEvidence(problem_id="p1", evidence_id="e_g3", evidence_state="true", likelihood_positive=0.8, likelihood_background=0.2),
        ProblemEvidence(problem_id="p1", evidence_id="e_discriminator", evidence_state="true", likelihood_positive=0.95, likelihood_background=0.05),
        ProblemEvidence(problem_id="p1", evidence_id="e_discriminator", evidence_state="false", likelihood_positive=0.05, likelihood_background=0.95),
        ProblemEvidence(problem_id="p2", evidence_id="e_discriminator", evidence_state="true", likelihood_positive=0.05, likelihood_background=0.95),
        ProblemEvidence(problem_id="p2", evidence_id="e_discriminator", evidence_state="false", likelihood_positive=0.95, likelihood_background=0.05),
    ]

    questions = {
        "q_ungrouped": Question(id="q_ungrouped", text="Ungrouped?", evidence_id="e_ungrouped"),
        "q_neg": Question(id="q_neg", text="Negative signal?", evidence_id="e_neg"),
        "q_g1": Question(id="q_g1", text="Group member 1?", evidence_id="e_g1"),
        "q_g2": Question(id="q_g2", text="Group member 2?", evidence_id="e_g2"),
        "q_g3": Question(id="q_g3", text="Group member 3?", evidence_id="e_g3"),
        "q_discriminator": Question(id="q_discriminator", text="Discriminator?", evidence_id="e_discriminator"),
        "q_low_value": Question(
            id="q_low_value", text="Uninformative?", evidence_id="e_neg", priority_weight=0.1, effort_cost=5.0
        ),
    }

    answers = [
        QuestionAnswer(id="a_disc_true", question_id="q_discriminator", evidence_state="true", label="Yes", order=0),
        QuestionAnswer(id="a_disc_false", question_id="q_discriminator", evidence_state="false", label="No", order=1),
    ]

    return KnowledgeBase(
        problems=problems,
        evidence=evidence,
        evidence_groups=groups,
        evidence_ranges=[],
        problem_evidence=problem_evidence,
        questions=questions,
        question_answers=answers,
        recommendations=[],
        safety_rules=[],
    )
