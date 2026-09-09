"""Read-only, query-optimized view over a loaded knowledge base.

This is what the inference/questions/safety/recommendations modules are
handed — a plain object with precomputed indices, independent of how the
data was loaded (YAML fixtures, SQLite, or in a test, hand-built dicts).
"""
from __future__ import annotations

from functools import cached_property

from .schema import (
    Concern,
    EntryContext,
    Evidence,
    EvidenceGroup,
    EvidenceRange,
    Problem,
    ProblemEvidence,
    Question,
    QuestionAnswer,
    Recommendation,
    SafetyRule,
)


class KnowledgeBase:
    def __init__(
        self,
        problems: dict[str, Problem],
        evidence: dict[str, Evidence],
        evidence_groups: dict[str, EvidenceGroup],
        evidence_ranges: list[EvidenceRange],
        problem_evidence: list[ProblemEvidence],
        questions: dict[str, Question],
        question_answers: list[QuestionAnswer],
        recommendations: list[Recommendation],
        safety_rules: list[SafetyRule],
        entry_contexts: dict[str, EntryContext] | None = None,
        concerns: dict[str, Concern] | None = None,
    ) -> None:
        self.problems = problems
        self.evidence = evidence
        self.evidence_groups = evidence_groups
        self.evidence_ranges = evidence_ranges
        self.problem_evidence = problem_evidence
        self.questions = questions
        self.question_answers = question_answers
        self.recommendations = recommendations
        self.safety_rules = safety_rules
        self.entry_contexts = entry_contexts or {}
        self.concerns = concerns or {}

    @cached_property
    def problem_evidence_index(self) -> dict[tuple[str, str, str], ProblemEvidence]:
        return {(pe.problem_id, pe.evidence_id, pe.evidence_state): pe for pe in self.problem_evidence}

    @cached_property
    def evidence_ranges_by_evidence(self) -> dict[str, list[EvidenceRange]]:
        out: dict[str, list[EvidenceRange]] = {}
        for r in sorted(self.evidence_ranges, key=lambda r: r.order):
            out.setdefault(r.evidence_id, []).append(r)
        return out

    @cached_property
    def answers_by_question(self) -> dict[str, list[QuestionAnswer]]:
        out: dict[str, list[QuestionAnswer]] = {}
        for a in sorted(self.question_answers, key=lambda a: a.order):
            out.setdefault(a.question_id, []).append(a)
        return out

    @cached_property
    def questions_by_evidence(self) -> dict[str, list[Question]]:
        out: dict[str, list[Question]] = {}
        for q in self.questions.values():
            out.setdefault(q.evidence_id, []).append(q)
        return out

    @cached_property
    def recommendations_by_problem(self) -> dict[str, list[Recommendation]]:
        out: dict[str, list[Recommendation]] = {}
        for r in sorted(self.recommendations, key=lambda r: r.order):
            out.setdefault(r.problem_id, []).append(r)
        return out

    @cached_property
    def concerns_by_entry_context(self) -> dict[str, list[Concern]]:
        out: dict[str, list[Concern]] = {}
        for c in self.concerns.values():
            out.setdefault(c.entry_context_id, []).append(c)
        return out

    @cached_property
    def problems_informed_by_evidence(self) -> dict[str, set[str]]:
        """evidence_id -> set of problem_ids that have at least one
        likelihood row for that evidence (i.e. it's informative for them)."""
        out: dict[str, set[str]] = {}
        for pe in self.problem_evidence:
            out.setdefault(pe.evidence_id, set()).add(pe.problem_id)
        return out

    def resolve_state_for_value(self, evidence_id: str, raw_value: float) -> str | None:
        """Bucket a raw numeric reading into its evidence_ranges range_key."""
        for r in self.evidence_ranges_by_evidence.get(evidence_id, []):
            lo_ok = r.min_value is None or (raw_value >= r.min_value if r.min_inclusive else raw_value > r.min_value)
            hi_ok = r.max_value is None or (raw_value <= r.max_value if r.max_inclusive else raw_value < r.max_value)
            if lo_ok and hi_ok:
                return r.range_key
        return None

    def likelihood(self, problem_id: str, evidence_id: str, state: str) -> ProblemEvidence | None:
        return self.problem_evidence_index.get((problem_id, evidence_id, state))
