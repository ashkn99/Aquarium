"""Plain, immutable knowledge-base types.

No I/O and no ORM here on purpose — these are the types the inference,
question-selection, safety, and recommendation engines consume, so those
modules can be tested without a database.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DataType(str, Enum):
    boolean = "boolean"
    categorical = "categorical"
    numeric = "numeric"


class ObservationSource(str, Enum):
    question_answer = "question_answer"
    free_text_llm_extracted = "free_text_llm_extracted"
    manual = "manual"


class Severity(str, Enum):
    info = "info"
    caution = "caution"
    urgent = "urgent"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Problem(Frozen):
    id: str
    name: str
    description: str = ""
    category: str = ""
    prior_weight: float = 1.0
    severity_default: Severity = Severity.info
    active: bool = True
    rationale: str = ""
    source: str = ""


class EvidenceGroup(Frozen):
    id: str
    name: str
    description: str = ""
    correlation_discount: float = Field(0.5, ge=0.0, le=1.0)


class Evidence(Frozen):
    id: str
    name: str
    description: str = ""
    data_type: DataType
    unit: Optional[str] = None
    evidence_group_id: Optional[str] = None
    rationale: str = ""
    active: bool = True


class EvidenceRange(Frozen):
    id: str
    evidence_id: str
    range_key: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_inclusive: bool = True
    max_inclusive: bool = False
    label: str = ""
    order: int = 0


class ProblemEvidence(Frozen):
    problem_id: str
    evidence_id: str
    evidence_state: str
    likelihood_positive: float = Field(gt=0.0, le=1.0)
    likelihood_background: float = Field(gt=0.0, le=1.0)
    weight_cap: Optional[float] = None
    rationale: str = ""
    source: str = ""


class Question(Frozen):
    id: str
    text: str
    evidence_id: str
    priority_weight: float = 1.0
    effort_cost: float = Field(1.0, gt=0.0)
    reliability: float = Field(0.9, ge=0.0, le=1.0)
    active: bool = True
    rationale: str = ""


class QuestionAnswer(Frozen):
    id: str
    question_id: str
    evidence_state: str
    label: str
    order: int = 0
    default_confidence: float = Field(0.9, ge=0.0, le=1.0)


class Recommendation(Frozen):
    problem_id: str
    kind: str  # confirmation_check | immediate_action | follow_up_action | avoid | escalation_criterion
    text: str
    order: int = 0
    rationale: str = ""


class SafetyCondition(Frozen):
    evidence_id: str
    equals_state: str


class SafetyRule(Frozen):
    id: str
    description: str
    all_of: list[SafetyCondition] = Field(default_factory=list)
    any_of: list[SafetyCondition] = Field(default_factory=list)
    severity: Severity = Severity.urgent
    message: str
    forced_question_id: Optional[str] = None
    escalation_text: str = ""
