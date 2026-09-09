"""HTTP request/response models for the web API.

Kept separate from kb/schema.py, which is the domain KB schema (problems,
evidence, questions, ...) loaded from YAML fixtures -- these describe the
wire format of the HTTP layer instead.
"""
from __future__ import annotations

from pydantic import BaseModel


class StartCaseRequest(BaseModel):
    entry_context: str | None = None


class StartCaseResponse(BaseModel):
    case_id: str


class ConcernOut(BaseModel):
    id: str
    name: str
    description: str = ""


class EntryContextOut(BaseModel):
    id: str
    name: str
    description: str = ""
    concerns: list[ConcernOut] = []


class SetConcernRequest(BaseModel):
    concern_id: str | None = None


class AnswerOption(BaseModel):
    state: str
    label: str


class NextQuestion(BaseModel):
    question_id: str
    evidence_id: str
    text: str
    data_type: str
    unit: str | None = None
    options: list[AnswerOption] | None = None  # None for numeric evidence


class CandidateOut(BaseModel):
    problem_id: str
    problem_name: str
    probability: float
    supporting_evidence: list[str]
    contradicting_evidence: list[str]


class SafetyAlertOut(BaseModel):
    rule_id: str
    severity: str
    message: str
    escalation_text: str


class RecommendationsOut(BaseModel):
    problem_id: str
    problem_name: str
    confirmation_checks: list[str]
    immediate_actions: list[str]
    follow_up_actions: list[str]
    avoid: list[str]
    escalation_criteria: list[str]


class CaseStatusResponse(BaseModel):
    case_id: str
    should_stop: bool
    stop_reason: str | None
    uncertainty: float
    ranked_candidates: list[CandidateOut]
    safety_alerts: list[SafetyAlertOut]
    next_question: NextQuestion | None
    recommendations: RecommendationsOut | None
    disclaimer: str


class AnswerRequest(BaseModel):
    question_id: str
    evidence_id: str
    state: str | None = None
    raw_value: float | None = None


class FeedbackRequest(BaseModel):
    helpful: bool
    comment: str | None = None
