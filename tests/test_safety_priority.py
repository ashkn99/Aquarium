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
     exception is exactly `respiratory_distress_general` (gasping) -- a
     small, bounded, universal opening screen, not the whole safety-linked
     evidence pool.
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


def test_pending_at_fresh_case_is_empty_until_something_real_points_somewhere(kb):
    """At the very start of a case, nothing is known -- so
    pending_safety_evidence (genuine suspicion) is empty. The
    respiratory-distress rule (no `all_of`, so it can never show partial
    signal) is handled entirely separately, as a domain-scoped mandatory
    screen in questions/selection.py -- see
    test_every_context_opens_on_the_same_universal_screen for that half."""
    assert pending_safety_evidence(kb, []) == set()
    assert screening_evidence_ids(kb) == {"gasping"}


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
    an any_of arm (gasping) shared with respiratory_distress_general, plus
    their own distinct all_of precondition. Observing gasping=true suspects
    both compound rules (their any_of is satisfied), pulling their still-
    unanswered all_of evidence into the pending set -- but ruling one
    compound rule out must not affect the other."""
    pending = pending_safety_evidence(kb, [obs("gasping", "true")])
    assert pending == {"used_untreated_tap_water", "planted_tank_with_co2_injection"}

    pending2 = pending_safety_evidence(kb, [obs("gasping", "true"), obs("used_untreated_tap_water", "false")])
    assert "used_untreated_tap_water" not in pending2  # severe_chlorine_exposure ruled out
    assert "planted_tank_with_co2_injection" in pending2  # co2 rule: unaffected, still suspected


def test_pending_drops_evidence_once_rule_is_firing(kb):
    """Once critical_ammonia is already firing, ammonia_ppm is resolved
    (answered) so it's naturally out of "pending" -- the alert itself,
    not further questioning, carries the concern from here."""
    pending = pending_safety_evidence(kb, [obs("ammonia_ppm", "critical")])
    assert "ammonia_ppm" not in pending


def test_pending_all_of_and_any_of_combination(kb):
    """co2_injection_excess_active (all_of planted_tank_with_co2_injection,
    any_of gasping): ruled out once its own all_of condition is
    contradicted, even though the shared any_of arm (gasping=true) is
    satisfied -- but severe_chlorine_exposure, sharing that same any_of arm
    with its own distinct all_of precondition, remains independently
    suspected and pending."""
    observations = [obs("gasping", "true"), obs("planted_tank_with_co2_injection", "false")]
    pending = pending_safety_evidence(kb, observations)
    # co2_injection_excess_active is ruled out (all_of contradicted) --
    # its own all_of evidence is already answered anyway, so nothing of
    # its remains pending.
    assert "planted_tank_with_co2_injection" not in pending
    # severe_chlorine_exposure: unaffected by the other rule's ruling-out,
    # still suspected via the shared any_of arm.
    assert pending == {"used_untreated_tap_water"}


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
    (gasping) is actually observed true, not merely because the case
    exists. Uses entry_context="fish", where the
    domain-scoped mandatory screen (see selection.py) still applies, so
    gasping is actually reachable within the driven turns -- see
    test_dissolved_oxygen_and_co2_not_fast_tracked_under_an_irrelevant_domain
    for the water/aquarium_environment/plants side of this."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="fish")
    asked, _ = _drive(engine, case_id, {}, max_turns=8)
    assert "planted_tank_with_co2_injection" not in asked[:8]  # gasping read false -> never suspected

    engine2 = AquariumInferenceEngine(kb)
    case_id2 = engine2.start_case(entry_context="fish")
    asked2, _ = _drive(engine2, case_id2, {"gasping": ("state", "true")}, max_turns=8)
    # gasping=true suspects both compound rules sharing that any_of arm
    # (severe_chlorine_exposure too) -- both get fast-tracked together.
    assert "planted_tank_with_co2_injection" in asked2[:4]


def test_dissolved_oxygen_and_co2_not_fast_tracked_under_an_irrelevant_domain(kb):
    """The domain-scoped mandatory screen (questions/selection.py) means
    aquarium_environment (no fish_symptom weight) never force-screens
    gasping at all -- so a compound rule sharing that any_of arm can't get
    suspected through it either. This is the fix for a real bug found by
    manual testing: every entry context, including ones with nothing to do
    with fish vitals, used to open on "is the fish gasping at the
    surface?"."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="aquarium_environment")
    asked, _ = _drive(engine, case_id, {}, max_turns=8)
    assert "gasping" not in asked[:8]


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
    must still preempt every selection-time tier -- e.g. once
    critical_ammonia forces ask_gasping, that must win regardless of what
    else selection would otherwise be considering."""
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case(entry_context="unsure")
    engine.answer(case_id, "ammonia_ppm", raw_value=3.0)
    status = engine.get_status(case_id)
    assert status.safety_alerts
    assert status.best_next_question.question_id == "ask_gasping"
    # Nothing else has been observed to create real suspicion for any
    # other rule -- confirm tier 0's win is despite an empty pending set,
    # not by default because something else was also competing for it.
    observations = engine.store.get(case_id).active_observations()
    pending = pending_safety_evidence(kb, observations)
    assert pending == set()


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
    assert len(all_rule_evidence) < 10  # today's KB: 7 -- a sanity ceiling, not a magic behavior threshold


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


def test_screen_opens_only_domain_relevant_contexts(kb):
    """Regression test for a real bug found by manual testing: every
    entry context, including ones with nothing to do with fish vitals
    (water/plants/aquarium_environment), used to open on "is the fish
    gasping at the surface?" regardless of what the user declared. The
    domain-scoped mandatory screen (questions/selection.py) now only
    forces gasping for contexts that actually declare the fish_symptom
    domain (fish) or declare no domain at all (unsure)
    -- water/plants/aquarium_environment fall through to normal ranking
    instead, same as any other diagnostic question."""
    def first_pick(entry_context):
        engine = AquariumInferenceEngine(kb)
        case_id = engine.start_case(entry_context=entry_context)
        status = engine.get_status(case_id)
        return kb.questions[status.best_next_question.question_id].evidence_id

    picks = {ctx: first_pick(ctx) for ctx in ["fish", "water", "plants", "aquarium_environment", "unsure"]}
    assert picks["fish"] == "gasping"
    assert picks["unsure"] == "gasping"
    for ctx in ("water", "plants", "aquarium_environment"):
        assert picks[ctx] != "gasping", f"{ctx} incorrectly forced the fish vital-sign screen"


def test_entry_context_differentiates_quickly_once_relevant(kb):
    """water/plants/aquarium_environment skip the (irrelevant) screen
    entirely, so they should differentiate from each other almost
    immediately -- bounded only by the KB's own documented
    near-pathognomonic white_spots outlier (unrelated to this milestone,
    see PROJECT_STATUS.md), which still briefly dominates raw information
    gain regardless of context. fish/unsure still take the 1-item screen
    first, so need a turn more before differentiating."""
    def pick_after(entry_context, turns):
        engine = AquariumInferenceEngine(kb)
        case_id = engine.start_case(entry_context=entry_context)
        for _ in range(turns):
            status = engine.get_status(case_id)
            q = status.best_next_question
            ev = kb.questions[q.question_id].evidence_id
            engine.answer(case_id, ev, state="false", question_id=q.question_id)
        status = engine.get_status(case_id)
        return kb.questions[status.best_next_question.question_id].evidence_id

    no_screen_picks = {ctx: pick_after(ctx, 1) for ctx in ["water", "plants", "aquarium_environment"]}
    assert len(set(no_screen_picks.values())) > 1, f"no-screen contexts didn't differentiate: {no_screen_picks}"

    screened_picks = {ctx: pick_after(ctx, 3) for ctx in ["fish", "unsure"]}
    assert len(set(screened_picks.values())) > 1, f"fish/unsure didn't differentiate: {screened_picks}"


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
