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


def test_seed_kb_concerns_reference_real_entry_contexts_and_evidence():
    kb = load_knowledge_base()
    assert len(kb.concerns) > 0
    for concern in kb.concerns.values():
        assert concern.entry_context_id in kb.entry_contexts
        for evidence_id in concern.evidence_ids:
            assert evidence_id in kb.evidence


def test_seed_kb_unsure_context_has_no_concerns():
    kb = load_knowledge_base()
    assert kb.concerns_by_entry_context.get("unsure", []) == []


def test_loader_rejects_concern_with_broken_entry_context_fk(tmp_path: Path):
    fixtures_dir = tmp_path
    (fixtures_dir / "evidence_groups.yaml").write_text(yaml.safe_dump([]), encoding="utf-8")
    (fixtures_dir / "problems.yaml").write_text(yaml.safe_dump([{"id": "p1", "name": "P1"}]), encoding="utf-8")
    (fixtures_dir / "evidence.yaml").write_text(
        yaml.safe_dump([{"id": "e1", "name": "E1", "data_type": "boolean"}]), encoding="utf-8"
    )
    (fixtures_dir / "concerns.yaml").write_text(
        yaml.safe_dump(
            [{"id": "c1", "entry_context_id": "nonexistent_context", "name": "C1", "evidence_ids": ["e1"]}]
        ),
        encoding="utf-8",
    )
    for name in ("evidence_ranges", "questions", "problem_evidence", "question_answers", "recommendations", "safety_rules", "entry_contexts"):
        (fixtures_dir / f"{name}.yaml").write_text(yaml.safe_dump([]), encoding="utf-8")

    with pytest.raises(IntegrityError):
        load_knowledge_base(fixtures_dir=fixtures_dir)


def test_loader_rejects_concern_with_broken_evidence_fk(tmp_path: Path):
    fixtures_dir = tmp_path
    (fixtures_dir / "evidence_groups.yaml").write_text(yaml.safe_dump([]), encoding="utf-8")
    (fixtures_dir / "problems.yaml").write_text(yaml.safe_dump([{"id": "p1", "name": "P1"}]), encoding="utf-8")
    (fixtures_dir / "evidence.yaml").write_text(yaml.safe_dump([]), encoding="utf-8")
    (fixtures_dir / "entry_contexts.yaml").write_text(
        yaml.safe_dump([{"id": "fish", "name": "Fish"}]), encoding="utf-8"
    )
    (fixtures_dir / "concerns.yaml").write_text(
        yaml.safe_dump([{"id": "c1", "entry_context_id": "fish", "name": "C1", "evidence_ids": ["nonexistent_evidence"]}]),
        encoding="utf-8",
    )
    for name in ("evidence_ranges", "questions", "problem_evidence", "question_answers", "recommendations", "safety_rules"):
        (fixtures_dir / f"{name}.yaml").write_text(yaml.safe_dump([]), encoding="utf-8")

    with pytest.raises(IntegrityError):
        load_knowledge_base(fixtures_dir=fixtures_dir)


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
