from __future__ import annotations

import pytest

from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.recommendations.engine import get_recommendations, get_recommendations_for_top_candidates

_BANNED_CERTAINTY_WORDS = ("definitely", "guaranteed", "certainly", "100%", "always fatal", "will die")


@pytest.fixture(scope="module")
def seed_kb():
    return load_knowledge_base()


def test_every_problem_has_all_five_recommendation_kinds(seed_kb):
    for problem_id in seed_kb.problems:
        bundle = get_recommendations(seed_kb, problem_id)
        assert bundle.confirmation_checks, problem_id
        assert bundle.immediate_actions, problem_id
        assert bundle.follow_up_actions, problem_id
        assert bundle.avoid, problem_id
        assert bundle.escalation_criteria, problem_id


def test_recommendation_text_avoids_certainty_language(seed_kb):
    all_text = []
    for problem_id in seed_kb.problems:
        bundle = get_recommendations(seed_kb, problem_id)
        all_text += (
            bundle.confirmation_checks
            + bundle.immediate_actions
            + bundle.follow_up_actions
            + bundle.avoid
            + bundle.escalation_criteria
        )
    lowered = " ".join(all_text).lower()
    for banned in _BANNED_CERTAINTY_WORDS:
        assert banned not in lowered, f"found banned certainty phrase: {banned!r}"


def test_get_recommendations_for_top_candidates_respects_ordering_and_limit(seed_kb):
    ranked = ["ammonia_toxicity", "ich", "low_oxygen", "ph_instability"]
    bundles = get_recommendations_for_top_candidates(seed_kb, ranked, top_n=2)
    assert [b.problem_id for b in bundles] == ["ammonia_toxicity", "ich"]
