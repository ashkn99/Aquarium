from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sqlalchemy.exc import IntegrityError

from aqua_assistant.kb.loader import load_knowledge_base


def test_seed_kb_loads_without_error():
    kb = load_knowledge_base()
    assert len(kb.problems) >= 8
    assert len(kb.evidence) >= 20
    assert len(kb.questions) >= 20
    assert len(kb.problem_evidence) > 0
    assert len(kb.safety_rules) > 0


def test_seed_kb_every_problem_has_recommendations():
    kb = load_knowledge_base()
    for problem_id in kb.problems:
        assert kb.recommendations_by_problem.get(problem_id), f"{problem_id} has no recommendations"


def test_seed_kb_every_question_evidence_id_exists():
    kb = load_knowledge_base()
    for q in kb.questions.values():
        assert q.evidence_id in kb.evidence


def test_seed_kb_every_problem_evidence_row_references_real_ids():
    kb = load_knowledge_base()
    for pe in kb.problem_evidence:
        assert pe.problem_id in kb.problems
        assert pe.evidence_id in kb.evidence


def test_loader_rejects_broken_foreign_key(tmp_path: Path):
    fixtures_dir = tmp_path
    (fixtures_dir / "evidence_groups.yaml").write_text(yaml.safe_dump([]), encoding="utf-8")
    (fixtures_dir / "problems.yaml").write_text(
        yaml.safe_dump([{"id": "p1", "name": "P1"}]), encoding="utf-8"
    )
    (fixtures_dir / "evidence.yaml").write_text(
        yaml.safe_dump(
            [{"id": "e1", "name": "E1", "data_type": "boolean", "evidence_group_id": "nonexistent_group"}]
        ),
        encoding="utf-8",
    )
    for name in ("evidence_ranges", "questions", "problem_evidence", "question_answers", "recommendations", "safety_rules"):
        (fixtures_dir / f"{name}.yaml").write_text(yaml.safe_dump([]), encoding="utf-8")

    with pytest.raises(IntegrityError):
        load_knowledge_base(fixtures_dir=fixtures_dir)
