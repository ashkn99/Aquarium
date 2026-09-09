"""Tests for the concern tier in question selection
(questions/selection.py::concern_pending_evidence, consumed by
select_best_question's tier 3) and concern-seeded branch narrowing
(concern_seeded_problem_ids, tier 4's fallback).

Priority order: forced safety question (engine.py) > safety-suspicion
tier > domain-scoped mandatory screen > concern tier > branch narrowing
(or concern-seeded scope, before real evidence makes a branch dominant)
> full-KB/routing ranking. Concern is evidence-side (which questions to
ask) and, unlike entry-context routing, a hard filter -- not just a
bonus -- so it can actually skip unrelated questions instead of merely
reordering them.
"""
from __future__ import annotations

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import (
    _entry_context_declares_a_domain,
    concern_pending_evidence,
    concern_seeded_problem_ids,
    select_best_question,
)

from .conftest import neutral_default


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def obs(evidence_id, state, confidence=1.0):
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)


# ==================================================== concern_pending_evidence (unit) ===


def test_pending_is_empty_with_no_concern(kb):
    assert concern_pending_evidence(kb, None, set()) == set()


def test_pending_is_empty_for_an_unknown_concern_id(kb):
    assert concern_pending_evidence(kb, "not_a_real_concern", set()) == set()


def test_pending_is_the_full_list_before_anything_is_answered(kb):
    pending = concern_pending_evidence(kb, "fish_mortality", set())
    assert pending == {"sudden_multiple_fish_death", "num_fish_affected", "onset_timing"}


def test_pending_narrows_as_evidence_is_answered(kb):
    pending = concern_pending_evidence(kb, "fish_mortality", {"num_fish_affected"})
    assert pending == {"sudden_multiple_fish_death", "onset_timing"}


def test_pending_is_empty_once_every_item_is_answered(kb):
    pending = concern_pending_evidence(
        kb, "fish_mortality", {"sudden_multiple_fish_death", "num_fish_affected", "onset_timing"}
    )
    assert pending == set()


# ==================================================== behavioral: narrows faster ===


def test_concern_surfaces_evidence_that_would_not_win_on_raw_information_gain_alone(kb):
    """sudden_multiple_fish_death has no safety-suspicion path (its rule
    is a single-condition all_of, see safety/rules.py) and only modest
    raw information gain -- without a concern, a fresh 'fish' case does
    not ask about it early (see test_safety_priority.py's
    test_mass_mortality_not_fast_tracked_without_any_signal). Declaring
    the fish_mortality concern must change that."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    engine.set_concern(case_id, "fish_mortality")

    asked = []
    for _ in range(6):
        status = engine.get_status(case_id)
        if status.should_stop:
            break
        q = status.best_next_question
        ev = kb.questions[q.question_id].evidence_id
        asked.append(ev)
        kind, val = neutral_default(kb, ev)
        if kind == "raw_value":
            engine.answer(case_id, ev, raw_value=val, question_id=q.question_id)
        else:
            engine.answer(case_id, ev, state=val, question_id=q.question_id)

    assert "sudden_multiple_fish_death" in asked[:5]


def test_no_concern_behaves_exactly_like_before(kb):
    """concern_id=None must reproduce the pre-existing selection exactly
    -- concern is additive and optional, never a replacement."""
    post = posterior(score_problems(kb, []))
    with_none = select_best_question(kb, post, [], set(), entry_context="fish", concern_id=None)
    without_param = select_best_question(kb, post, [], set(), entry_context="fish")
    assert with_none.question_id == without_param.question_id
    assert with_none.final_score == without_param.final_score


# ==================================================== tier precedence ===


def test_safety_suspicion_still_wins_over_a_declared_concern(kb):
    """Once gasping reads true, the universal respiratory-distress screen
    (safety/rules.py) still takes priority over an unrelated declared
    concern -- a real emergency signal must never be diluted by a
    pre-declared focus area."""
    observations = [obs("gasping", "true")]
    post = posterior(score_problems(kb, observations))
    answered = {"gasping"}
    picked = select_best_question(kb, post, observations, answered, entry_context="fish", concern_id="fish_mortality")
    assert kb.questions[picked.question_id].evidence_id == "surface_breathing"


def test_concern_tier_wins_over_unrelated_high_information_gain_markers(kb):
    """Past the universal safety screen (gasping/surface_breathing, which
    always wins first from a fresh case -- see safety/rules.py), with the
    fish_appearance concern declared, the pick must come from that
    concern's list even though unrelated fish-disease markers (e.g.
    white_spots) carry much higher raw information gain."""
    observations = [obs("gasping", "false"), obs("surface_breathing", "false")]
    answered = {"gasping", "surface_breathing"}
    post = posterior(score_problems(kb, observations))
    picked = select_best_question(
        kb, post, observations, answered, entry_context="fish", concern_id="fish_appearance"
    )
    concern_evidence = set(kb.concerns["fish_appearance"].evidence_ids)
    assert kb.questions[picked.question_id].evidence_id in concern_evidence


def test_concern_tier_empties_and_falls_through_once_exhausted(kb):
    """Once every item on a concern is answered, selection must fall
    through to normal ranking rather than returning None or erroring."""
    answered = {"lethargy", "appetite_loss", "hiding", "flashing_scratching", "chasing_or_nipping_observed", "color_fading_or_pale"}
    observations = [obs(eid, "false") for eid in answered]
    post = posterior(score_problems(kb, observations))
    picked = select_best_question(kb, post, observations, answered, entry_context="fish", concern_id="fish_behavior")
    assert picked is not None
    assert kb.questions[picked.question_id].evidence_id not in answered


# ==================================================== engine.set_concern ===


def test_set_concern_persists_and_is_reflected_in_status(kb):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="water")
    engine.set_concern(case_id, "water_chemistry")

    # Past the universal safety screen first (see the fish_appearance
    # test above for why) -- answer whatever comes up until we're asked
    # something the concern actually governs.
    concern_evidence = set(kb.concerns["water_chemistry"].evidence_ids)
    for _ in range(5):
        status = engine.get_status(case_id)
        ev = kb.questions[status.best_next_question.question_id].evidence_id
        if ev in concern_evidence:
            return
        kind, val = neutral_default(kb, ev)
        if kind == "raw_value":
            engine.answer(case_id, ev, raw_value=val, question_id=status.best_next_question.question_id)
        else:
            engine.answer(case_id, ev, state=val, question_id=status.best_next_question.question_id)
    raise AssertionError("never reached a water_chemistry question")


def test_set_concern_rejects_unknown_concern_id(kb):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    with pytest.raises(ValueError):
        engine.set_concern(case_id, "not_a_real_concern")


def test_set_concern_rejects_a_concern_from_a_different_entry_context(kb):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    with pytest.raises(ValueError):
        engine.set_concern(case_id, "water_chemistry")


def test_set_concern_to_none_clears_it(kb):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    engine.set_concern(case_id, "fish_mortality")
    engine.set_concern(case_id, None)
    case = engine.store.get(case_id)
    assert case.concern_id is None


# ==================================================== _entry_context_declares_a_domain ===


@pytest.mark.parametrize("ctx,expected", [("fish", True), ("water", True), ("plants", True), ("aquarium_environment", True), ("unsure", False), (None, False), ("not_a_real_context", False)])
def test_entry_context_declares_a_domain(kb, ctx, expected):
    assert _entry_context_declares_a_domain(kb, ctx) is expected


# ==================================================== concern_seeded_problem_ids ===


def test_concern_seeded_scope_is_none_without_a_concern(kb):
    assert concern_seeded_problem_ids(kb, None) is None
    assert concern_seeded_problem_ids(kb, "not_a_real_concern") is None


def test_concern_seeded_scope_narrows_to_problems_the_evidence_actually_informs(kb):
    """plant_damage's evidence (melting, shredded leaves, floating/uprooted,
    rotting roots) only has likelihood rows against a small, specific set
    of plant problems -- not the whole plant_health category, and
    certainly not any fish/water problem."""
    scope = concern_seeded_problem_ids(kb, "plant_damage")
    assert scope == {"plant_livestock_damage", "plant_melt_transition", "plant_root_or_planting_problem"}


def test_concern_seeded_scope_never_includes_unrelated_domain_problems(kb):
    scope = concern_seeded_problem_ids(kb, "water_chemistry")
    assert "ich" not in scope  # a fish/parasite problem, not informed by ammonia/nitrite/nitrate
    assert "plant_root_or_planting_problem" not in scope


# ==================================================== integration: concern narrows ranking ===


def test_declared_concern_beats_the_kb_wide_dominant_marker_for_ranking(kb):
    """white_spots is the KB's own documented, near-pathognomonic,
    globally-dominant marker (PROJECT_STATUS.md) -- it wins turn 1 under
    plain entry-context routing alone, with no concern declared. Once
    plant_damage is declared, concern-seeded branch narrowing (tier 4's
    fallback, since no real branch is dominant yet) restricts ranking to
    the problems that concern's evidence actually informs, and
    white_spots -- uninformative for any of them -- properly loses."""
    post = posterior(score_problems(kb, []))

    without_concern = select_best_question(kb, post, [], set(), entry_context="plants", concern_id=None)
    assert kb.questions[without_concern.question_id].evidence_id == "white_spots"

    with_concern = select_best_question(kb, post, [], set(), entry_context="plants", concern_id="plant_damage")
    picked_evidence = kb.questions[with_concern.question_id].evidence_id
    assert picked_evidence != "white_spots"
    informed = kb.problems_informed_by_evidence.get(picked_evidence, set())
    assert informed & concern_seeded_problem_ids(kb, "plant_damage")
