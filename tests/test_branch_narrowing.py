"""Tests for TARGETED BRANCH: diagnostic-scope narrowing in question
selection (questions/selection.py::narrowed_problem_ids(), consumed by
select_best_question()).

Branches are `Problem.category`, reused as-is -- no new KB concept. Once
one category holds enough of the posterior, question ranking switches
from competing across the whole KB to competing within that category
(plus any outside problem that hasn't actually been ruled out yet). The
real posterior (returned by the engine, used for safety/stopping) is
never touched by this -- only a second, throwaway posterior used purely
for ranking.
"""
from __future__ import annotations

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import (
    BRANCH_DOMINANCE_THRESHOLD,
    build_question_explanation,
    narrowed_problem_ids,
    select_best_question,
)

from .conftest import neutral_default


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def obs(evidence_id, state, confidence=1.0):
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)


def test_no_narrowing_from_a_fresh_case(kb):
    """Nothing dominates a flat prior over 30 problems across 6 categories
    -- ranking must stay KB-wide."""
    post = posterior(score_problems(kb, []))
    assert narrowed_problem_ids(kb, post) is None


def test_narrows_to_the_dominant_category_once_it_crosses_the_threshold(kb):
    post = {"p1": 0.6, "p2": 0.38, "p3": 0.02}

    class FakeProblem:
        def __init__(self, category):
            self.category = category

    class FakeKB:
        problems = {"p1": FakeProblem("water_quality"), "p2": FakeProblem("water_quality"), "p3": FakeProblem("disease")}

    scope = narrowed_problem_ids(FakeKB(), post)
    assert scope == {"p1", "p2"}  # both water_quality members kept (mass 0.98); p3 (disease, 0.02) below the outsider floor


def test_outsider_above_the_floor_is_kept_below_it_is_dropped(kb):
    """A competitor sitting outside the dominant category is kept only if
    it still holds non-trivial posterior mass -- only the dominant
    category is guaranteed inclusion; anything else earns its place on
    its own remaining probability."""
    post = {"p1": 0.62, "p2": 0.36, "p3": 0.02}

    class FakeProblem:
        def __init__(self, category):
            self.category = category

    class FakeKB:
        problems = {"p1": FakeProblem("water_quality"), "p2": FakeProblem("disease"), "p3": FakeProblem("disease")}

    scope = narrowed_problem_ids(FakeKB(), post)
    assert scope == {"p1", "p2"}  # p2 kept (0.36 >= floor), p3 dropped (0.02 < floor)


def test_scoped_ranking_excludes_a_branch_with_negligible_mass(kb):
    """Once strong ammonia evidence has pushed water_quality to dominate,
    a category with negligible remaining mass (e.g. plant_health, never
    even touched by this scenario) must drop out of the narrowed scope
    -- confirming narrowing actually shrinks the candidate set, not just
    computes it."""
    observations = [
        obs("ammonia_ppm", "critical", confidence=0.9),
        obs("num_fish_affected", "most_or_all"),
        obs("onset_timing", "sudden"),
    ]
    post = posterior(score_problems(kb, observations))
    scope = narrowed_problem_ids(kb, post)
    assert scope is not None
    assert "ammonia_toxicity" in scope
    plant_problems = {pid for pid, p in kb.problems.items() if p.category == "plant_health"}
    assert not (plant_problems & scope), f"plant problems leaked into scope: {plant_problems & scope}"
    assert len(scope) < len(kb.problems)


def test_real_posterior_is_never_mutated_by_narrowing(kb):
    """select_best_question must never hand back a narrowed posterior as
    if it were the real one -- it only affects which question is picked,
    never candidates/probabilities returned elsewhere."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="water")
    engine.answer(case_id, "ammonia_ppm", raw_value=3.0)
    engine.answer(case_id, "num_fish_affected", state="most_or_all")

    status = engine.get_status(case_id)
    full_post = posterior(score_problems(kb, engine.store.get(case_id).active_observations()))
    reported = {c.problem_id: c.probability for c in status.ranked_candidates}
    assert reported == pytest.approx(full_post)
    # every problem in the KB is still represented, not just the narrowed branch
    assert set(reported) == set(kb.problems)


def test_branch_narrowing_does_not_change_answer_for_a_confident_case(kb):
    """A textbook Ich case must still resolve identically -- narrowing is
    purely a ranking-time convenience, never a change to the underlying
    inference outcome."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    engine.answer(case_id, "white_spots", state="true")
    status = engine.get_status(case_id)
    assert status.ranked_candidates[0].problem_id == "ich"
    assert status.should_stop
    assert status.stop_reason == "confident_top_candidate"


def test_no_entry_context_branch_narrowing_still_composes_with_routing(kb):
    """Branch narrowing and entry-context routing are independent knobs --
    confirm narrowed ranking still works with entry_context=None (pure
    information gain within scope, no routing bonus)."""
    observations = [obs("ammonia_ppm", "critical"), obs("num_fish_affected", "most_or_all")]
    post = posterior(score_problems(kb, observations))
    answered = {o.evidence_id for o in observations}
    picked = select_best_question(kb, post, observations, answered, entry_context=None)
    assert picked is not None
    assert picked.adjusted_value > 0


def test_branch_dominance_threshold_is_a_single_documented_constant(kb):
    assert 0.5 < BRANCH_DOMINANCE_THRESHOLD < 1.0
