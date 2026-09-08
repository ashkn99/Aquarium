"""SQLAlchemy ORM models — the relational schema.

Used to persist the knowledge base and exercise real foreign-key
referential integrity when fixtures are loaded (see loader.py). The
inference engine never touches these directly; it consumes the plain
KnowledgeBase snapshot in knowledge_base.py instead.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ProblemORM(Base):
    __tablename__ = "problems"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    description: Mapped[str] = mapped_column(default="")
    category: Mapped[str] = mapped_column(default="")
    prior_weight: Mapped[float] = mapped_column(default=1.0)
    severity_default: Mapped[str] = mapped_column(default="info")
    active: Mapped[bool] = mapped_column(default=True)
    rationale: Mapped[str] = mapped_column(default="")
    source: Mapped[str] = mapped_column(default="")


class EvidenceGroupORM(Base):
    __tablename__ = "evidence_groups"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    description: Mapped[str] = mapped_column(default="")
    correlation_discount: Mapped[float] = mapped_column(default=0.5)


class EvidenceORM(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    description: Mapped[str] = mapped_column(default="")
    data_type: Mapped[str]
    unit: Mapped[Optional[str]] = mapped_column(default=None)
    evidence_group_id: Mapped[Optional[str]] = mapped_column(ForeignKey("evidence_groups.id"), default=None)
    rationale: Mapped[str] = mapped_column(default="")
    active: Mapped[bool] = mapped_column(default=True)


class EvidenceRangeORM(Base):
    __tablename__ = "evidence_ranges"

    id: Mapped[str] = mapped_column(primary_key=True)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id"))
    range_key: Mapped[str]
    min_value: Mapped[Optional[float]] = mapped_column(default=None)
    max_value: Mapped[Optional[float]] = mapped_column(default=None)
    min_inclusive: Mapped[bool] = mapped_column(default=True)
    max_inclusive: Mapped[bool] = mapped_column(default=False)
    label: Mapped[str] = mapped_column(default="")
    order: Mapped[int] = mapped_column(default=0)


class ProblemEvidenceORM(Base):
    __tablename__ = "problem_evidence"
    __table_args__ = (UniqueConstraint("problem_id", "evidence_id", "evidence_state"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id"))
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id"))
    evidence_state: Mapped[str]
    likelihood_positive: Mapped[float]
    likelihood_background: Mapped[float]
    weight_cap: Mapped[Optional[float]] = mapped_column(default=None)
    rationale: Mapped[str] = mapped_column(default="")
    source: Mapped[str] = mapped_column(default="")


class QuestionORM(Base):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(primary_key=True)
    text: Mapped[str]
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id"))
    priority_weight: Mapped[float] = mapped_column(default=1.0)
    effort_cost: Mapped[float] = mapped_column(default=1.0)
    reliability: Mapped[float] = mapped_column(default=0.9)
    active: Mapped[bool] = mapped_column(default=True)
    rationale: Mapped[str] = mapped_column(default="")


class QuestionAnswerORM(Base):
    __tablename__ = "question_answers"

    id: Mapped[str] = mapped_column(primary_key=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"))
    evidence_state: Mapped[str]
    label: Mapped[str]
    order: Mapped[int] = mapped_column(default=0)
    default_confidence: Mapped[float] = mapped_column(default=0.9)


class SpeciesORM(Base):
    __tablename__ = "species"

    id: Mapped[str] = mapped_column(primary_key=True)
    common_name: Mapped[str]
    scientific_name: Mapped[Optional[str]] = mapped_column(default=None)
    notes: Mapped[str] = mapped_column(default="")


class SpeciesParameterORM(Base):
    __tablename__ = "species_parameters"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    species_id: Mapped[str] = mapped_column(ForeignKey("species.id"))
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id"))
    normal_min: Mapped[Optional[float]] = mapped_column(default=None)
    normal_max: Mapped[Optional[float]] = mapped_column(default=None)
    notes: Mapped[str] = mapped_column(default="")


class ProblemRecommendationORM(Base):
    __tablename__ = "problem_recommendations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id"))
    kind: Mapped[str]
    text: Mapped[str]
    order: Mapped[int] = mapped_column(default=0)
    rationale: Mapped[str] = mapped_column(default="")


class SafetyRuleORM(Base):
    __tablename__ = "safety_rules"

    id: Mapped[str] = mapped_column(primary_key=True)
    description: Mapped[str]
    condition_json: Mapped[str]
    severity: Mapped[str] = mapped_column(default="urgent")
    message: Mapped[str]
    forced_question_id: Mapped[Optional[str]] = mapped_column(ForeignKey("questions.id"), default=None)
    escalation_text: Mapped[str] = mapped_column(default="")


class CaseORM(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[str]
    updated_at: Mapped[str]
    status: Mapped[str] = mapped_column(default="open")


class ObservationORM(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"))
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id"))
    observed_state: Mapped[str]
    raw_value: Mapped[Optional[float]] = mapped_column(default=None)
    confidence: Mapped[float] = mapped_column(default=1.0)
    source: Mapped[str] = mapped_column(default="manual")
    question_id: Mapped[Optional[str]] = mapped_column(ForeignKey("questions.id"), default=None)
    answered_at: Mapped[str]
    superseded: Mapped[bool] = mapped_column(default=False)
