"""AquariumInferenceEngine -- the single façade that ties scoring, question
selection, and the safety layer into the MVP contract: given structured
observations, return ranked candidates, supporting/contradicting evidence,
current uncertainty, the best next question, and an explanation of why.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aqua_assistant.case.models import Case, Observation
from aqua_assistant.case.store import CaseStore
from aqua_assistant.inference.entropy import normalized_entropy
from aqua_assistant.inference.scoring import observation_contribution, posterior, score_problems
from aqua_assistant.kb.knowledge_base import KnowledgeBase
from aqua_assistant.questions.selection import QuestionExplanation, build_question_explanation, select_best_question
from aqua_assistant.safety.rules import SafetyAlert, evaluate_safety


@dataclass
class CandidateExplanation:
    problem_id: str
    problem_name: str
    probability: float
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)


@dataclass
class CaseStatus:
    case_id: str
    ranked_candidates: list[CandidateExplanation]
    uncertainty: float
    best_next_question: QuestionExplanation | None
    safety_alerts: list[SafetyAlert]
    should_stop: bool
    stop_reason: str | None


class AquariumInferenceEngine:
    def __init__(
        self,
        kb: KnowledgeBase,
        store: CaseStore | None = None,
        top_candidate_threshold: float = 0.6,
        entropy_stop_threshold: float = 0.15,
        info_gain_epsilon: float = 0.02,
    ) -> None:
        self.kb = kb
        self.store = store or CaseStore()
        self.top_candidate_threshold = top_candidate_threshold
        self.entropy_stop_threshold = entropy_stop_threshold
        self.info_gain_epsilon = info_gain_epsilon

    def start_case(self, entry_context: str | None = None) -> str:
        """`entry_context` (e.g. "fish", "water", "unsure" -- see
        kb.entry_contexts for the configured set) is purely a question-
        ordering preference recorded on the case; it is never read by
        scoring/posterior computation, only by question selection."""
        if entry_context is not None and entry_context not in self.kb.entry_contexts:
            raise ValueError(f"unknown entry_context {entry_context!r}")
        return self.store.create(entry_context=entry_context).id

    def answer(
        self,
        case_id: str,
        evidence_id: str,
        state: str | None = None,
        raw_value: float | None = None,
        confidence: float | None = None,
        question_id: str | None = None,
        source: str = "manual",
    ) -> None:
        case = self.store.get(case_id)

        if raw_value is not None and state is None:
            state = self.kb.resolve_state_for_value(evidence_id, raw_value)
            if state is None:
                raise ValueError(f"raw_value {raw_value!r} for {evidence_id!r} doesn't fall in any evidence_range")
        if state is None:
            raise ValueError("must supply either `state` or `raw_value`")

        if confidence is None:
            confidence = self._default_confidence(question_id, state)

        case.add_observation(
            Observation(
                evidence_id=evidence_id,
                observed_state=state,
                confidence=confidence,
                raw_value=raw_value,
                source=source,
                question_id=question_id,
            )
        )
        self.store.save(case)

    def _default_confidence(self, question_id: str | None, state: str) -> float:
        if question_id:
            for a in self.kb.answers_by_question.get(question_id, []):
                if a.evidence_state == state:
                    return a.default_confidence
        return 0.9

    def get_status(self, case_id: str) -> CaseStatus:
        case: Case = self.store.get(case_id)
        observations = case.active_observations()

        scores = score_problems(self.kb, observations)
        post = posterior(scores)
        ranked_ids = sorted(post, key=lambda pid: post[pid], reverse=True)
        candidates = [self._explain_candidate(pid, post[pid], observations) for pid in ranked_ids]
        uncertainty = normalized_entropy(post)

        safety_alerts = evaluate_safety(self.kb, observations)
        # Only the most severe alert's forced_question_id is honored, so at
        # most one forced question is ever active -- but every firing rule
        # is still surfaced in safety_alerts below, none are dropped.
        primary_alert = safety_alerts[0] if safety_alerts else None

        answered = case.answered_evidence_ids()
        forced_pending = bool(
            primary_alert
            and primary_alert.forced_question_id
            and self.kb.questions[primary_alert.forced_question_id].evidence_id not in answered
        )
        if forced_pending:
            # A safety-forced question is a direct override, not a ranked
            # pick -- entry-context routing plays no part in it.
            best_question = build_question_explanation(self.kb, primary_alert.forced_question_id, post, observations)
        else:
            best_question = select_best_question(
                self.kb, post, observations, answered, case.entry_context, self.info_gain_epsilon
            )

        if forced_pending:
            # A pending safety confirmation must never be suppressed by the
            # normal diagnostic stop rule -- information gain (or lack of
            # it) can never override a safety concern.
            should_stop, reason = False, None
        else:
            should_stop, reason = self._check_stop(post, uncertainty, best_question)

        return CaseStatus(
            case_id=case_id,
            ranked_candidates=candidates,
            uncertainty=uncertainty,
            best_next_question=None if should_stop else best_question,
            safety_alerts=safety_alerts,
            should_stop=should_stop,
            stop_reason=reason,
        )

    def _explain_candidate(self, problem_id: str, probability: float, observations: list[Observation]) -> CandidateExplanation:
        supporting, contradicting = [], []
        for obs in observations:
            contribution = observation_contribution(self.kb, problem_id, obs)
            if contribution > 1e-9:
                supporting.append(obs.evidence_id)
            elif contribution < -1e-9:
                contradicting.append(obs.evidence_id)
        return CandidateExplanation(
            problem_id=problem_id,
            problem_name=self.kb.problems[problem_id].name,
            probability=probability,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
        )

    def _check_stop(
        self, post: dict[str, float], uncertainty: float, best_question: QuestionExplanation | None
    ) -> tuple[bool, str | None]:
        if not post:
            return True, "no_candidates"
        if max(post.values()) >= self.top_candidate_threshold:
            return True, "confident_top_candidate"
        if uncertainty <= self.entropy_stop_threshold:
            return True, "low_uncertainty"
        if best_question is None:
            return True, "no_eligible_questions"
        if best_question.adjusted_value < self.info_gain_epsilon:
            # Nothing left is informative enough to ask -- the engine has
            # no choice but to stop either way. But if the posterior is
            # still well above the "confidently converged" bar, this is a
            # weak stop (ran out of questions before reaching a clear
            # answer), not a strong one -- worth reporting honestly rather
            # than under the same label as a real convergence.
            if uncertainty > self.entropy_stop_threshold * 2:
                return True, "diminishing_information_gain_ambiguous"
            return True, "diminishing_information_gain"
        return False, None
