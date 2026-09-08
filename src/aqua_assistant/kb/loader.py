"""Load YAML fixtures, validate them, persist into SQLite (exercising real
foreign-key referential integrity), then hand back a plain KnowledgeBase
for the inference engine to consume.

The YAML files under kb/fixtures/ are the source of truth: they're
diffable in git and carry inline `rationale`/`source` fields so
aquarium/fish-health experts can review knowledge changes in a PR without
touching code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from .knowledge_base import KnowledgeBase
from .models import (
    Base,
    EvidenceGroupORM,
    EvidenceORM,
    EvidenceRangeORM,
    ProblemEvidenceORM,
    ProblemORM,
    ProblemRecommendationORM,
    QuestionAnswerORM,
    QuestionORM,
    SafetyRuleORM,
)
from .schema import (
    Evidence,
    EvidenceGroup,
    EvidenceRange,
    Problem,
    ProblemEvidence,
    Question,
    QuestionAnswer,
    Recommendation,
    SafetyCondition,
    SafetyRule,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_yaml(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or []


def _parse_safety_rule(d: dict[str, Any]) -> SafetyRule:
    d = dict(d)
    d["all_of"] = [SafetyCondition(**c) for c in d.get("all_of", [])]
    d["any_of"] = [SafetyCondition(**c) for c in d.get("any_of", [])]
    return SafetyRule(**d)


def load_knowledge_base(fixtures_dir: Path | None = None, db_url: str = "sqlite:///:memory:") -> KnowledgeBase:
    fixtures_dir = fixtures_dir or FIXTURES_DIR

    groups = [EvidenceGroup(**d) for d in _load_yaml(fixtures_dir / "evidence_groups.yaml")]
    evidence = [Evidence(**d) for d in _load_yaml(fixtures_dir / "evidence.yaml")]
    problems = [Problem(**d) for d in _load_yaml(fixtures_dir / "problems.yaml")]
    ranges = [EvidenceRange(**d) for d in _load_yaml(fixtures_dir / "evidence_ranges.yaml")]
    questions = [Question(**d) for d in _load_yaml(fixtures_dir / "questions.yaml")]
    prob_ev = [ProblemEvidence(**d) for d in _load_yaml(fixtures_dir / "problem_evidence.yaml")]
    answers = [QuestionAnswer(**d) for d in _load_yaml(fixtures_dir / "question_answers.yaml")]
    recommendations = [Recommendation(**d) for d in _load_yaml(fixtures_dir / "recommendations.yaml")]
    safety_rules = [_parse_safety_rule(d) for d in _load_yaml(fixtures_dir / "safety_rules.yaml")]

    _persist_and_check(db_url, groups, evidence, problems, ranges, questions, prob_ev, answers, recommendations, safety_rules)

    return KnowledgeBase(
        problems={p.id: p for p in problems},
        evidence={e.id: e for e in evidence},
        evidence_groups={g.id: g for g in groups},
        evidence_ranges=ranges,
        problem_evidence=prob_ev,
        questions={q.id: q for q in questions},
        question_answers=answers,
        recommendations=recommendations,
        safety_rules=safety_rules,
    )


def _persist_and_check(
    db_url: str,
    groups: list[EvidenceGroup],
    evidence: list[Evidence],
    problems: list[Problem],
    ranges: list[EvidenceRange],
    questions: list[Question],
    prob_ev: list[ProblemEvidence],
    answers: list[QuestionAnswer],
    recommendations: list[Recommendation],
    safety_rules: list[SafetyRule],
) -> None:
    """Insert fixtures into SQLite in dependency order with foreign_keys=ON,
    so a broken reference (e.g. evidence pointing at a nonexistent group)
    raises an IntegrityError here rather than silently loading."""
    engine = create_engine(db_url)

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(EvidenceGroupORM(**g.model_dump(mode="json")) for g in groups)
        session.flush()

        session.add_all(EvidenceORM(**e.model_dump(mode="json")) for e in evidence)
        session.flush()

        session.add_all(ProblemORM(**p.model_dump(mode="json")) for p in problems)
        session.flush()

        session.add_all(EvidenceRangeORM(**r.model_dump(mode="json")) for r in ranges)
        session.add_all(QuestionORM(**q.model_dump(mode="json")) for q in questions)
        session.flush()

        session.add_all(ProblemEvidenceORM(**pe.model_dump(mode="json")) for pe in prob_ev)
        session.add_all(QuestionAnswerORM(**a.model_dump(mode="json")) for a in answers)
        session.add_all(ProblemRecommendationORM(**r.model_dump(mode="json")) for r in recommendations)
        session.flush()

        for r in safety_rules:
            session.add(
                SafetyRuleORM(
                    id=r.id,
                    description=r.description,
                    condition_json=json.dumps(
                        {
                            "all_of": [c.model_dump() for c in r.all_of],
                            "any_of": [c.model_dump() for c in r.any_of],
                        }
                    ),
                    severity=r.severity.value,
                    message=r.message,
                    forced_question_id=r.forced_question_id,
                    escalation_text=r.escalation_text,
                )
            )

        session.commit()

    engine.dispose()
