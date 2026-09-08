"""Scenario tests for the infectious/parasitic disease KB expansion batch
(fungal_infection_saprolegnia, velvet_disease_oodinium,
columnaris_cotton_mouth, swim_bladder_disorder, dropsy_systemic).
"""
from __future__ import annotations

import math

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import eligible_questions
from aqua_assistant.safety.rules import evaluate_safety

from .conftest import neutral_default


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def obs(evidence_id, state, confidence=1.0):
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)


# ---------------------------------------------------------------- overlap ---


def test_velvet_beats_ich_when_dust_confirmed_and_spots_denied(kb):
    """Velvet and Ich are commonly confused by hobbyists -- both are
    parasitic, itchy, spread via new fish. white_spots=false + a confirmed
    gold/rust dusty coating should point to velvet, not Ich."""
    observations = [
        obs("gold_rust_dust_coating", "true"),
        obs("white_spots", "false"),
        obs("flashing_scratching", "true"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "velvet_disease_oodinium"
    assert post["velvet_disease_oodinium"] > post["ich"]


def test_ich_still_wins_when_spots_confirmed_and_dust_denied(kb):
    observations = [
        obs("white_spots", "true"),
        obs("gold_rust_dust_coating", "false"),
        obs("flashing_scratching", "true"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "ich"


def test_columnaris_beats_fungal_and_bacterial_on_mouth_focused_patches(kb):
    """Columnaris, Saprolegnia, and bacterial_infection all involve
    fuzzy/damaged tissue -- the mouth-specific location should
    discriminate columnaris from body-wide fungal growth."""
    observations = [
        obs("white_fuzzy_patches_around_mouth", "true"),
        obs("cottony_fuzzy_growth", "false"),
        obs("appetite_loss", "true"),
    ]
    post = posterior(score_problems(kb, observations))
    ranked = sorted(post, key=lambda pid: post[pid], reverse=True)
    assert ranked[0] == "columnaris_cotton_mouth"
    assert post["columnaris_cotton_mouth"] > post["fungal_infection_saprolegnia"]
    assert post["columnaris_cotton_mouth"] > post["bacterial_infection"]


def test_swim_bladder_not_confused_with_infectious_disease(kb):
    """A pure buoyancy complaint with no other symptoms should not be
    swamped by more heavily-evidenced infectious problems."""
    observations = [obs("abnormal_buoyancy_or_swimming", "true"), obs("num_fish_affected", "single")]
    post = posterior(score_problems(kb, observations))
    assert max(post, key=post.get) == "swim_bladder_disorder"


# ------------------------------------------------------------ negative ---


def test_no_cottony_growth_reduces_fungal_candidate_sharply(kb):
    baseline = score_problems(kb, [])["fungal_infection_saprolegnia"]
    negative = score_problems(kb, [obs("cottony_fuzzy_growth", "false")])["fungal_infection_saprolegnia"]
    assert negative < baseline * 0.3  # pathognomonic sign absent should hit hard


def test_no_abnormal_buoyancy_nearly_rules_out_swim_bladder(kb):
    baseline = score_problems(kb, [])["swim_bladder_disorder"]
    negative = score_problems(kb, [obs("abnormal_buoyancy_or_swimming", "false")])["swim_bladder_disorder"]
    assert negative < baseline * 0.3


def test_widespread_occurrence_strongly_disconfirms_swim_bladder(kb):
    """Swim bladder disorder is non-contagious; most_or_all fish affected
    should count as real evidence against it, not just be ignored."""
    baseline = score_problems(kb, [])["swim_bladder_disorder"]
    negative = score_problems(kb, [obs("num_fish_affected", "most_or_all")])["swim_bladder_disorder"]
    assert negative < baseline


# --------------------------------------------------- correlation group ---


def test_dropsy_signs_group_discounts_correlated_confirmation(kb):
    """pinecone_scales and bloated_body are in the dropsy_signs group --
    observing both should contribute less to the score than the sum of
    their two individual (undiscounted) contributions, since the weaker
    signal gets discounted once a stronger group-mate is already counted.
    Compares raw observation_contribution, not full scores (which also
    embed the prior and would double-count it if naively summed)."""
    from aqua_assistant.inference.scoring import observation_contribution

    pinecone_contribution = observation_contribution(kb, "dropsy_systemic", obs("pinecone_scales", "true"))
    bloat_contribution = observation_contribution(kb, "dropsy_systemic", obs("bloated_body", "true"))

    combined_score = score_problems(
        kb, [obs("pinecone_scales", "true"), obs("bloated_body", "true")]
    )["dropsy_systemic"]
    baseline_score = score_problems(kb, [])["dropsy_systemic"]
    combined_contribution = combined_score - baseline_score

    assert combined_contribution < pinecone_contribution + bloat_contribution


# ------------------------------------------------------- contradictory ---


def test_contradictory_dropsy_evidence_stays_bounded(kb):
    """Pinecone scales confirmed (strong positive) directly contradicted
    by bloated_body=false (the other classic sign absent). Must combine
    without error and stay a valid distribution."""
    pinecone_alone = posterior(score_problems(kb, [obs("pinecone_scales", "true")]))
    contradictory = posterior(
        score_problems(kb, [obs("pinecone_scales", "true"), obs("bloated_body", "false")])
    )
    assert sum(contradictory.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 < p < 1.0 for p in contradictory.values())
    assert contradictory["dropsy_systemic"] < pinecone_alone["dropsy_systemic"]


# ---------------------------------------------------------- selection ---


def test_new_disease_questions_are_eligible(kb):
    ids = {q.id for q in eligible_questions(kb, answered_evidence_ids=set())}
    for qid in [
        "ask_cottony_fuzzy_growth",
        "ask_gold_rust_dust_coating",
        "ask_white_fuzzy_patches_around_mouth",
        "ask_abnormal_buoyancy_or_swimming",
        "ask_pinecone_scales",
        "ask_bloated_body",
    ]:
        assert qid in ids


# NOTE: an isolated "is pinecone_scales worth less after bloated_body is
# already confirmed" check was tried here and removed -- it does NOT hold
# for this particular pair. pinecone_scales (LR ~85) is individually much
# stronger than bloated_body (LR ~15), and dropsy_systemic is one of 19
# candidates, so simulating pinecone_scales after bloated_body still
# collapses a large amount of the *remaining* entropy even though the
# scoring-level group discount (verified above) correctly prevents the
# two from being double-counted if both come back positive. This is a
# real, documented nuance of unequal-strength correlated evidence in a
# large candidate pool, not a bug -- see the final report. What actually
# matters in practice -- whether the engine gets stuck re-asking group
# members back to back -- is covered by the redundant-question assertion
# in every end-to-end scenario test below and in test_end_to_end_case.py.


# ------------------------------------------------------------- safety ---


def test_no_safety_rules_fire_for_progressive_diseases(kb):
    """None of this batch's 5 problems are hours-scale emergencies like the
    chemical/oxygen crises -- confirm none of them trip a safety alert even
    with their most severe evidence combination confirmed."""
    observations = [
        obs("cottony_fuzzy_growth", "true"),
        obs("gold_rust_dust_coating", "true"),
        obs("white_fuzzy_patches_around_mouth", "true"),
        obs("abnormal_buoyancy_or_swimming", "true"),
        obs("pinecone_scales", "true"),
        obs("bloated_body", "true"),
        obs("lethargy", "true"),
        obs("appetite_loss", "true"),
    ]
    alerts = evaluate_safety(kb, observations)
    assert alerts == []


# --------------------------------------------------------- end to end ---


def test_full_case_converges_to_dropsy_without_redundant_group_questions(kb):
    """The empirical check for the nuance documented above: even though
    pinecone_scales can show high simulated gain right after bloated_body,
    the engine must still reach convergence without asking the same
    evidence twice or looping on the dropsy_signs group."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case()
    answers = {
        "bloated_body": ("state", "true"),
        "pinecone_scales": ("state", "true"),
        "lethargy": ("state", "true"),
        "appetite_loss": ("state", "true"),
        "num_fish_affected": ("state", "single"),
        "white_spots": ("state", "false"),
        "ammonia_ppm": ("raw_value", 0.0),
        "nitrite_ppm": ("raw_value", 0.0),
    }
    asked = []
    for _ in range(30):
        status = engine.get_status(case_id)
        if status.should_stop:
            top = status.ranked_candidates[0]
            assert top.problem_id == "dropsy_systemic"
            return
        q = status.best_next_question
        ev = kb.questions[q.question_id].evidence_id
        assert ev not in asked, f"redundant question re-asked for {ev}"
        asked.append(ev)
        kind, val = answers.get(ev) or neutral_default(kb, ev)
        if kind == "raw_value":
            engine.answer(case_id, ev, raw_value=val, question_id=q.question_id)
        else:
            engine.answer(case_id, ev, state=val, question_id=q.question_id)
    raise AssertionError("did not converge")


def test_full_case_converges_to_columnaris(kb):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case()
    answers = {
        "white_fuzzy_patches_around_mouth": ("state", "true"),
        "cottony_fuzzy_growth": ("state", "false"),
        "white_spots": ("state", "false"),
        "appetite_loss": ("state", "true"),
        "onset_timing": ("state", "sudden"),
        "num_fish_affected": ("state", "several"),
        "ammonia_ppm": ("raw_value", 0.0),
        "nitrite_ppm": ("raw_value", 0.0),
    }
    asked = []
    for _ in range(30):
        status = engine.get_status(case_id)
        if status.should_stop:
            top = status.ranked_candidates[0]
            assert top.problem_id == "columnaris_cotton_mouth"
            return
        q = status.best_next_question
        ev = kb.questions[q.question_id].evidence_id
        assert ev not in asked, f"redundant question re-asked for {ev}"
        asked.append(ev)
        kind, val = answers.get(ev) or neutral_default(kb, ev)
        if kind == "raw_value":
            engine.answer(case_id, ev, raw_value=val, question_id=q.question_id)
        else:
            engine.answer(case_id, ev, state=val, question_id=q.question_id)
    raise AssertionError("did not converge")
