"""Tests for the concern tier in question selection
(questions/selection.py::concern_pending_evidence, consumed by
select_best_question's tier 2).

Priority order: forced safety question (engine.py) > safety-suspicion
tier > concern tier > branch narrowing > full-KB/routing ranking. Concern
is evidence-side (which questions to ask) and, unlike entry-context
routing, a hard filter -- not just a bonus -- so it can actually skip
unrelated questions instead of merely reordering them.
"""
from __future__ import annotations

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import concern_pending_evidence, select_best_question

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
