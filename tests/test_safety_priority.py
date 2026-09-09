"""Tests for TRIAGE: safety as an interrupt in question selection
(questions/selection.py::select_best_question's tier 1, backed by
safety/rules.py::pending_safety_evidence()).

Priority order, most to least urgent:
  1. forced safety question (engine.py, pre-existing, untouched here)
  2. unanswered evidence belonging to a *suspected* safety rule -- one
     with at least one condition already observed matching, OR (a
     structural exception -- see safety/rules.py::_rule_suspected) a
     rule with no `all_of` clause at all, which has no possible partial
     precursor and so must be screened directly. In this KB that
     exception is exactly `respiratory_distress_general` (gasping /
     surface_breathing) -- a small, bounded, universal opening screen,
     not the whole safety-linked evidence pool.
  3. branch-narrowed or KB-wide information gain + entry-context ranking

An earlier version of this milestone made *every* unresolved safety rule
pending unconditionally from turn 1, regardless of any actual signal --
that fixed the motivating regression (gasping buried at question ~24-30
in a contradictory safe-ammonia/alarming-gasping case) but made every
clean, no-real-danger case measurably longer (it always burned through
the ~8-item safety-linked pool first). This version instead treats safety
as an interrupt: a rule only takes priority once it's actually suspected,
so clean cases never touch it, while the structurally-blind screening
rule still guarantees the original regression can't recur.
"""
from __future__ import annotations

import pytest

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import eligible_questions, routing_bonus, select_best_question
from aqua_assistant.safety.rules import evaluate_safety, pending_safety_evidence, screening_evidence_ids

from .conftest import neutral_default


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def obs(evidence_id, state, confidence=1.0):
    return Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)


def _drive(engine, case_id, answer_script, max_turns=40):
    """Answer whatever question comes next, using answer_script where
    given and conftest.neutral_default() otherwise. Returns the ordered
    list of evidence_ids asked."""
    asked = []
    for _ in range(max_turns):
        status = engine.get_status(case_id)
        if status.should_stop:
            return asked, status
        q = status.best_next_question
        ev = engine.kb.questions[q.question_id].evidence_id
        asked.append(ev)
        kind, val = answer_script.get(ev) or neutral_default(engine.kb, ev)
        if kind == "raw_value":
            engine.answer(case_id, ev, raw_value=val, question_id=q.question_id)
        else:
            engine.answer(case_id, ev, state=val, question_id=q.question_id)
    return asked, engine.get_status(case_id)


# ==================================================== pending_safety_evidence (unit) ===


def test_pending_at_fresh_case_is_bounded_to_the_screening_only_rule(kb):
    """At the very start of a case, nothing is known -- so only a rule
    with no `all_of` clause (structurally impossible to partially
    suspect, see safety/rules.py::_rule_suspected) can be pending.
    Every other rule in this KB has an `all_of` clause and stays out of
    the tier until something concrete actually points at it -- this is
    the fix for the previous version's unconditional "every rule pending
    from turn 1" behavior."""
    pending = pending_safety_evidence(kb, [])
    all_rule_evidence = {c.evidence_id for r in kb.safety_rules for c in (*r.all_of, *r.any_of)}
    assert pending == screening_evidence_ids(kb) == {"gasping", "surface_breathing"}
    assert pending < all_rule_evidence  # a small screen, not the whole pool


def test_dissolved_oxygen_can_never_be_pending_since_its_rule_is_single_condition(kb):
    """critical_dissolved_oxygen has exactly one all_of condition and no
    any_of -- _rule_suspected can only return True for it once
    dissolved_oxygen_ppm itself already reads "critical", at which point
    the rule is already firing (not pending) and dissolved_oxygen_ppm is
    already answered anyway. So it can never appear in the pending set,
    at any point in a case -- it only ever gets asked via normal
    information-gain/branch ranking, exactly like an ordinary diagnostic
    question, which is the intended behavior for a rule with no compound
    structure to suspect."""
    assert "dissolved_oxygen_ppm" not in pending_safety_evidence(kb, [])
    assert "dissolved_oxygen_ppm" not in pending_safety_evidence(kb, [obs("gasping", "true")])
    assert "dissolved_oxygen_ppm" not in pending_safety_evidence(kb, [obs("sudden_multiple_fish_death", "true")])


def test_pending_narrows_once_a_compound_rule_is_suspected_and_drops_when_ruled_out(kb):
    """severe_chlorine_exposure and co2_injection_excess_active both have
    an any_of arm (gasping/surface_breathing) shared with
    respiratory_distress_general, plus their own distinct all_of
    precondition. Observing gasping=true suspects both compound rules
    (their any_of is satisfied), pulling their still-unanswered all_of
    evidence into the pending set alongside the always-pending screen's
    remaining item -- but ruling one compound rule out must not affect
    the other."""
    pending = pending_safety_evidence(kb, [obs("gasping", "true")])
    assert pending == {"surface_breathing", "used_untreated_tap_water", "planted_tank_with_co2_injection"}

    pending2 = pending_safety_evidence(kb, [obs("gasping", "true"), obs("used_untreated_tap_water", "false")])
    assert "used_untreated_tap_water" not in pending2  # severe_chlorine_exposure ruled out
    assert "planted_tank_with_co2_injection" in pending2  # co2 rule: unaffected, still suspected


def test_pending_drops_evidence_once_rule_is_firing(kb):
    """Once critical_ammonia is already firing, ammonia_ppm is resolved
    (answered) so it's naturally out of "pending" -- the alert itself,
    not further questioning, carries the concern from here."""
    pending = pending_safety_evidence(kb, [obs("ammonia_ppm", "critical")])
    assert "ammonia_ppm" not in pending


def test_pending_any_of_ruled_out_only_once_every_branch_is_answered_false(kb):
    """respiratory_distress_general (any_of gasping/surface_breathing) stays
    live as long as either is unanswered -- answering just one false must
    not remove the other from pending."""
    pending_one = pending_safety_evidence(kb, [obs("gasping", "false")])
    assert "surface_breathing" in pending_one

    pending_both = pending_safety_evidence(kb, [obs("gasping", "false"), obs("surface_breathing", "false")])
    assert "gasping" not in pending_both
    assert "surface_breathing" not in pending_both


def test_pending_all_of_and_any_of_combination(kb):
    """co2_injection_excess_active (all_of planted_tank_with_co2_injection,
    any_of gasping/surface_breathing): ruled out once the all_of condition
    is contradicted, even if gasping/surface_breathing are untouched."""
    pending = pending_safety_evidence(kb, [obs("planted_tank_with_co2_injection", "false")])
    # The rule itself is ruled out, but gasping/surface_breathing remain
    # pending via the still-live respiratory_distress_general rule.
    assert "gasping" in pending
    assert "surface_breathing" in pending


# ==================================================== A: gasping regression ===


def test_gasping_surfaces_immediately_in_the_contradictory_scenario(kb):
    """The motivating regression: safe ammonia/nitrite but gasping + red
    streaks used to bury gasping at question ~24-30. It must now be
    reachable essentially immediately."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    asked, status = _drive(
        engine,
        case_id,
        {
            "ammonia_ppm": ("raw_value", 0.0),
            "nitrite_ppm": ("raw_value", 0.0),
            "gasping": ("state", "true"),
            "red_streaks_or_gills": ("state", "true"),
        },
        max_turns=5,
    )
    assert "gasping" in asked
    assert asked.index("gasping") <= 2, f"gasping asked at position {asked.index('gasping') + 1}, expected <= 3"


def test_gasping_alone_triggers_safety_alert_within_first_two_turns(kb):
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    for turn in range(1, 4):
        status = engine.get_status(case_id)
        if status.safety_alerts:
            assert turn <= 2, f"safety alert first appeared at turn {turn}"
            assert any(a.rule_id == "respiratory_distress_general" for a in status.safety_alerts)
            return
        q = status.best_next_question
        ev = kb.questions[q.question_id].evidence_id
        state = "true" if ev == "gasping" else neutral_default(kb, ev)[1]
        if ev == "gasping":
            engine.answer(case_id, ev, state="true", question_id=q.question_id)
        else:
            kind, val = neutral_default(kb, ev)
            if kind == "raw_value":
                engine.answer(case_id, ev, raw_value=val, question_id=q.question_id)
            else:
                engine.answer(case_id, ev, state=val, question_id=q.question_id)
    raise AssertionError("gasping was never asked within 3 turns")


# ==================================================== B/C/D: other safety evidence ===


def test_dissolved_oxygen_not_fast_tracked_without_any_signal(kb):
    """dissolved_oxygen_ppm gates critical_dissolved_oxygen, a single-
    condition all_of-only rule -- it structurally can never be
    "suspected" (see test_dissolved_oxygen_can_never_be_pending... in
    test_safety_priority's unit tests), so from a totally clean, no-signal
    fresh case it is NOT force-prioritized. This is the intended TRIAGE
    behavior: no meaningful suspicion means normal ranking, not a forced
    detour through generic safety evidence."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="unsure")
    asked, _ = _drive(engine, case_id, {}, max_turns=8)
    assert "dissolved_oxygen_ppm" not in asked[:8]


def test_co2_evidence_fast_tracked_once_respiratory_distress_is_confirmed(kb):
    """planted_tank_with_co2_injection gates co2_injection_excess_active, a
    compound rule (all_of + any_of respiratory distress). It only becomes
    suspected -- and thus fast-tracked -- once the shared any_of arm
    (gasping/surface_breathing, part of the universal screen) is actually
    observed true, not merely because the case exists."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="aquarium_environment")
    asked, _ = _drive(engine, case_id, {}, max_turns=8)
    assert "planted_tank_with_co2_injection" not in asked[:8]  # gasping/surface_breathing read false -> never suspected

    engine2 = AquariumInferenceEngine(kb)
    case_id2 = engine2.start_case(entry_context="aquarium_environment")
    asked2, _ = _drive(engine2, case_id2, {"gasping": ("state", "true")}, max_turns=8)
    assert "planted_tank_with_co2_injection" in asked2[:3]  # now suspected -> fast-tracked


def test_co2_overdose_scenario_safety_fires_early_relative_to_session(kb):
    """Previous baseline (PROJECT_STATUS.md, pre-plant-KB): this scenario's
    safety alert "didn't fire until the very last question of 14". Once
    gasping reads true (turn 1, part of the universal screen),
    co2_injection_excess_active's own precondition
    (planted_tank_with_co2_injection) is immediately suspected and
    fast-tracked -- confirming the CO2-specific alert well before a
    13-14 question session would otherwise end."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="aquarium_environment")
    answer_script = {
        "planted_tank_with_co2_injection": ("state", "true"),
        "gasping": ("state", "true"),
        "surface_breathing": ("state", "true"),
    }
    for turn in range(1, 9):
        status = engine.get_status(case_id)
        rule_ids = {a.rule_id for a in status.safety_alerts}
        if "co2_injection_excess_active" in rule_ids:
            assert turn <= 8, f"CO2 safety alert first appeared at turn {turn}"
            return
        if status.should_stop:
            break
        q = status.best_next_question
        ev = kb.questions[q.question_id].evidence_id
        kind, val = answer_script.get(ev) or neutral_default(kb, ev)
        if kind == "raw_value":
            engine.answer(case_id, ev, raw_value=val, question_id=q.question_id)
        else:
            engine.answer(case_id, ev, state=val, question_id=q.question_id)
    raise AssertionError("CO2 safety alert never fired within 8 turns")


def test_mass_mortality_not_fast_tracked_without_any_signal(kb):
    """sudden_multiple_fish_death gates mass_mortality_event, another
    single-condition all_of-only rule -- like dissolved_oxygen_ppm, it
    can never be "suspected" and so is never force-prioritized from a
    clean, no-signal case; but its safety alert still fires the instant
    it's actually observed true, regardless of ranking."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="unsure")
    asked, _ = _drive(engine, case_id, {}, max_turns=8)
    assert "sudden_multiple_fish_death" not in asked[:8]

    engine2 = AquariumInferenceEngine(kb)
    case_id2 = engine2.start_case(entry_context="unsure")
    engine2.answer(case_id2, "sudden_multiple_fish_death", state="true")
    status = engine2.get_status(case_id2)
    assert any(a.rule_id == "mass_mortality_event" for a in status.safety_alerts)


def test_critical_ammonia_evidence_handled_appropriately(kb):
    """ammonia_ppm gates critical_ammonia -- another single-condition
    all_of-only rule, so it's not force-prioritized without signal either
    -- but once it reads critical, the pre-existing forced-question
    mechanism (tier 0 in engine.py, unaffected by this milestone) still
    takes over immediately."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="plants")
    asked, _ = _drive(engine, case_id, {}, max_turns=8)
    assert "ammonia_ppm" not in asked[:8]

    engine2 = AquariumInferenceEngine(kb)
    case_id2 = engine2.start_case(entry_context="plants")
    engine2.answer(case_id2, "ammonia_ppm", raw_value=3.0)
    status = engine2.get_status(case_id2)
    assert status.safety_alerts
    assert status.best_next_question.question_id == "ask_gasping"


# ==================================================== forced question untouched ===


def test_already_firing_forced_safety_question_still_wins_over_safety_tier(kb):
    """Tier 0 (a firing rule's forced_question_id, decided in engine.py)
    must still preempt tier 1 (the suspicion-gated pending-safety tier)
    -- e.g. once critical_ammonia forces ask_gasping, the still-pending
    universal screen (gasping/surface_breathing itself) must not compete
    with it, even though gasping is literally the forced pick's own
    evidence."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="unsure")
    engine.answer(case_id, "ammonia_ppm", raw_value=3.0)
    status = engine.get_status(case_id)
    assert status.safety_alerts
    assert status.best_next_question.question_id == "ask_gasping"
    # Confirm the universal screen is indeed still pending -- the forced
    # question is winning despite competition, not by default.
    observations = engine.store.get(case_id).active_observations()
    pending = pending_safety_evidence(kb, observations)
    assert pending == {"gasping", "surface_breathing"}


@pytest.mark.parametrize("ctx", ["fish", "water", "plants", "aquarium_environment", "unsure"])
def test_safety_forced_question_still_wins_regardless_of_entry_context(kb, ctx):
    """Pre-existing guarantee (test_entry_context_routing.py has the
    original of this) re-checked here as part of this milestone's own
    suite: forced safety questions must be entirely unaffected by the new
    pending-safety tier."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context=ctx)
    engine.answer(case_id, "ammonia_ppm", raw_value=3.0)
    status = engine.get_status(case_id)
    assert status.safety_alerts
    assert status.best_next_question.question_id == "ask_gasping"


# ==================================================== not a generic checklist ===


def test_safety_priority_does_not_block_fast_convergence_on_strong_evidence(kb):
    """A textbook Ich case (a near-pathognomonic marker straight to a
    confident top candidate) must still stop in one question -- safety
    priority must never force a detour through the safety pool once the
    case is already decided by the existing, untouched stop rule."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    engine.answer(case_id, "white_spots", state="true")
    status = engine.get_status(case_id)
    assert status.should_stop
    assert status.stop_reason == "confident_top_candidate"
    assert status.ranked_candidates[0].problem_id == "ich"


def test_safety_priority_pool_is_bounded_not_unbounded(kb):
    """The pending-safety pool can only ever contain evidence ids actually
    referenced by a safety_rule -- confirm that ceiling directly, so a
    future rule addition can't silently turn this into an ever-growing
    checklist without a visible KB change."""
    all_rule_evidence = {c.evidence_id for r in kb.safety_rules for c in (*r.all_of, *r.any_of)}
    pending = pending_safety_evidence(kb, [])
    assert pending <= all_rule_evidence
    assert len(all_rule_evidence) < 10  # today's KB: 8 -- a sanity ceiling, not a magic behavior threshold


def test_routing_bonus_math_is_unaffected_by_safety_priority(kb):
    """routing_bonus() itself is untouched code (see questions/selection.py
    -- the safety tier is a selection-time filter applied *before*
    ranking, never a change to how a bonus is computed), so it must keep
    producing the same context-specific, topic-specific values regardless
    of whether any safety rule is pending. Checked directly rather than
    through a full scripted case: an end-to-end "sequences must diverge"
    check turns out not to be a meaningful one for *this* KB even before
    this milestone -- a handful of near-pathognomonic markers
    (white_spots, cottony_fuzzy_growth, old_leaves_melting_rapidly...)
    dominate on raw information gain alone regardless of routing bonus,
    which is exactly the KB's pre-existing, documented characteristic
    (see PROJECT_STATUS.md limitation on white_spots) -- not something
    the safety tier introduces or should be blamed for reproducing."""
    assert routing_bonus(kb, "plants", "leaf_yellowing_or_discoloration", num_observations=0) > 0
    assert routing_bonus(kb, "fish", "leaf_yellowing_or_discoloration", num_observations=0) == 0.0
    assert routing_bonus(kb, "water", "used_untreated_tap_water", num_observations=0) > 0
    assert routing_bonus(kb, "plants", "used_untreated_tap_water", num_observations=0) == 0.0


def test_every_context_opens_on_the_same_universal_screen(kb):
    """From a fresh case, every entry context picks the same first
    question: the universal screen (gasping, surface_breathing) has only
    two members and both carry the same fish_symptom routing bonus within
    any one context, so raw information gain alone decides between them,
    with no room for entry-context routing to differentiate. This is the
    intended TRIAGE shape -- a fixed, tiny opening screen before
    orientation gets to steer anything -- not a regression in routing."""
    def first_pick(entry_context):
        engine = AquariumInferenceEngine(kb)
        case_id = engine.start_case(entry_context=entry_context)
        status = engine.get_status(case_id)
        return kb.questions[status.best_next_question.question_id].evidence_id

    picks = {ctx: first_pick(ctx) for ctx in ["fish", "water", "plants", "aquarium_environment", "unsure"]}
    assert set(picks.values()) == {"gasping"}


def test_entry_context_differentiates_again_right_after_the_screen(kb):
    """Once the two-item universal screen is answered (both false, no
    suspicion raised) and the single globally-dominant white_spots
    question (KB's own documented near-pathognomonic outlier, unrelated
    to this milestone -- see PROJECT_STATUS.md) is also out of the way,
    entry-context routing takes back over -- confirming the screen
    doesn't cannibalize routing's whole decay window (see
    questions/selection.py::screening_evidence_ids usage in
    build_question_explanation)."""
    def fourth_pick(entry_context):
        engine = AquariumInferenceEngine(kb)
        case_id = engine.start_case(entry_context=entry_context)
        for _ in range(3):
            status = engine.get_status(case_id)
            q = status.best_next_question
            ev = kb.questions[q.question_id].evidence_id
            engine.answer(case_id, ev, state="false", question_id=q.question_id)
        status = engine.get_status(case_id)
        return kb.questions[status.best_next_question.question_id].evidence_id

    picks = {ctx: fourth_pick(ctx) for ctx in ["fish", "water", "plants", "aquarium_environment", "unsure"]}
    assert len(set(picks.values())) > 1, f"every context still picked the same question post-screen: {picks}"


# ==================================================== determinism ===


def test_selection_is_deterministic_across_repeated_calls(kb):
    post_and_obs_cases = [
        [],
        [obs("gasping", "true")],
        [obs("ammonia_ppm", "safe"), obs("red_streaks_or_gills", "true")],
    ]
    for observations in post_and_obs_cases:
        from aqua_assistant.inference.scoring import posterior, score_problems

        post = posterior(score_problems(kb, observations))
        answered = {o.evidence_id for o in observations}
        picks = {
            select_best_question(kb, post, observations, answered, entry_context="fish").question_id
            for _ in range(5)
        }
        assert len(picks) == 1


def test_selection_deterministic_with_fresh_kb_reload(kb):
    """Reload the KB independently (fresh dict/set construction order) and
    confirm the same scenario still picks the same question -- guards
    against any accidental reliance on Python set iteration order."""
    kb_reloaded = load_knowledge_base()
    engine_a = AquariumInferenceEngine(kb)
    engine_b = AquariumInferenceEngine(kb_reloaded)
    case_a = engine_a.start_case(entry_context="fish")
    case_b = engine_b.start_case(entry_context="fish")
    q_a = engine_a.get_status(case_a).best_next_question.question_id
    q_b = engine_b.get_status(case_b).best_next_question.question_id
    assert q_a == q_b


# ==================================================== no-regression spot checks ===


def test_previously_regressed_plant_melt_scenario_still_converges_correctly(kb):
    """Guards the exact bug found while building this milestone: an
    unguarded safety tier could force-select an already-uninformative
    safety question (adjusted_value below info_gain_epsilon) and trigger
    a premature diminishing_information_gain stop, even though a far more
    informative non-safety question (old_leaves_melting_rapidly) was
    sitting right there. The info_gain_epsilon guard on the safety tier
    (see select_best_question) fixes this."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="plants")
    answers = {
        "recently_planted_or_moved": ("state", "true"),
        "old_leaves_melting_rapidly": ("state", "true"),
        "new_growth_pale_or_distorted": ("state", "false"),
        "num_plants_affected": ("state", "several"),
        "white_spots": ("state", "false"),
        "ammonia_ppm": ("raw_value", 0.0),
        "nitrite_ppm": ("raw_value", 0.0),
        "gasping": ("state", "false"),
        "surface_breathing": ("state", "false"),
    }
    asked, status = _drive(engine, case_id, answers)
    assert status.should_stop
    assert status.ranked_candidates[0].problem_id == "plant_melt_transition"
    assert status.ranked_candidates[0].probability > 0.5


def test_safety_does_not_fire_for_every_normal_session(kb):
    """Sanity check carried over from the entry-context-routing milestone:
    the new tier must not somehow make safety fire spuriously."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    engine.answer(case_id, "lethargy", state="true")
    status = engine.get_status(case_id)
    assert status.safety_alerts == []
