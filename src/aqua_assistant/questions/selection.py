"""Information-gain-driven question selection, with an optional soft
entry-context routing preference layered on top.

For each eligible question, simulate every possible answer, measure how
much it would be expected to shrink entropy, then adjust that raw
information gain by reliability / priority / effort before ranking.
Reuses the exact same scoring function as real answers, so redundant or
low-value questions (e.g. a second question in an already-confirmed
correlated-evidence group) fall out with low value naturally, instead of
needing hand-coded "don't ask this again" rules.

Entry-context routing (see `routing_bonus`) never touches this diagnostic
math -- it only nudges which *question* gets asked next, by adding a
small, decaying bonus to a question's ranking score based on its
evidence's `topic`. It is computed here, in the selection layer, and is
never passed into inference/scoring.py.

`select_best_question` also applies a safety-priority *tier*, ahead of
that ranking: unanswered evidence belonging to a *suspected* safety rule
(see safety/rules.py::pending_safety_evidence()) is preferred over
unrelated diagnostic markers, however high their raw information gain.
This is a selection-time filter, not another additive bonus -- it narrows
which questions are even ranked, rather than nudging their score -- and
it is independent of, and composes cleanly with, entry-context routing.

Below that tier, `select_best_question` also applies a concern tier:
unanswered evidence on the user's chosen `Concern` (a second, more
specific choice after entry_context -- see kb.concerns / kb/schema.py)
is preferred the same way, via concern_pending_evidence(). Unlike
entry-context routing, this is a filter too, not a bonus -- it can
actually skip unrelated questions rather than just reordering them.

Below that, `select_best_question` also applies branch narrowing
(diagnostic-scope narrowing): once one `Problem.category` has come to
dominate the posterior, information gain for ranking purposes is
recomputed within that category (plus any outside problem that still
holds a non-trivial share of the posterior) instead of across the whole
KB. See `narrowed_problem_ids()`. This prevents ~30 problems competing
equally on turn 1 from drowning out the handful of questions that would
actually distinguish within the area the evidence already points to. The
real posterior (returned to callers, used for safety/stopping) is never
touched by this -- narrowing only affects which *question* looks best,
via a second, throwaway posterior computed with the existing
`score_problems(..., problem_ids=...)` subsetting support.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aqua_assistant.case.models import Observation
from aqua_assistant.inference.entropy import entropy
from aqua_assistant.inference.scoring import posterior, score_problems
from aqua_assistant.kb.knowledge_base import KnowledgeBase
from aqua_assistant.safety.rules import pending_safety_evidence, screening_evidence_ids

# How many answered observations it takes for entry-context routing to
# fully decay to zero influence. Linear decay: full weight at 0 answered
# observations, zero weight at/after this many. Simple and explainable by
# design -- the routing preference is meant to shape the *opening* of the
# conversation, not to persist as a standing bias once real evidence is
# accumulating and information gain has plenty to work with.
ROUTING_DECAY_TURNS = 6

# Aggregate posterior mass a single Problem.category needs to hold before
# branch narrowing kicks in for question ranking. Deliberately just past
# a majority -- a category "clearly leading" (not merely "biggest of
# several small slices") is what should stop the whole KB from competing
# on every turn.
BRANCH_DOMINANCE_THRESHOLD = 0.55

# A problem outside the dominant branch stays in the narrowed candidate
# set as long as it holds at least this much posterior mass, so a
# genuine competitor is never silently dropped just for sitting in a
# different category.
BRANCH_OUTSIDER_FLOOR = 0.03


@dataclass
class QuestionExplanation:
    question_id: str
    text: str
    information_gain: float
    reliability: float
    priority_weight: float
    effort_cost: float
    adjusted_value: float
    routing_bonus: float = 0.0
    final_score: float = 0.0
    distinguishes: list[str] = field(default_factory=list)


def eligible_questions(kb: KnowledgeBase, answered_evidence_ids: set[str]):
    return [q for q in kb.questions.values() if q.active and q.evidence_id not in answered_evidence_ids]


def possible_states(kb: KnowledgeBase, evidence_id: str) -> list[str]:
    """All states a user could realistically report for this evidence.

    Must include every state offered by that evidence's question(s)
    (question_answers), not just the states that happen to have a
    problem_evidence likelihood row -- otherwise a state with no likelihood
    data (e.g. "no" for a symptom nobody bothered to write a negative row
    for) silently drops out of the answer simulation in
    expected_information_gain, understating that question's real
    information gain instead of correctly treating the missing state as
    uninformative (contribution 0) like any other unmodeled state.
    """
    ranges = kb.evidence_ranges_by_evidence.get(evidence_id)
    if ranges:
        return [r.range_key for r in ranges]

    states = {pe.evidence_state for pe in kb.problem_evidence if pe.evidence_id == evidence_id}
    for q in kb.questions_by_evidence.get(evidence_id, []):
        states.update(a.evidence_state for a in kb.answers_by_question.get(q.id, []))
    return sorted(states)


def _answer_probability(kb: KnowledgeBase, posterior_dist: dict[str, float], evidence_id: str, state: str) -> float:
    """Marginal P(evidence=state) = Sum_j P(problem_j) * P(state|problem_j),
    falling back to a neutral 0.5 for problems with no likelihood row for
    this evidence (uninformative, so it shouldn't sway the estimate)."""
    total = 0.0
    for pid, p in posterior_dist.items():
        pe = kb.likelihood(pid, evidence_id, state)
        total += p * (pe.likelihood_positive if pe is not None else 0.5)
    return total


def simulate_answer(
    kb: KnowledgeBase,
    observations: list[Observation],
    evidence_id: str,
    state: str,
    confidence: float = 0.9,
    problem_ids: list[str] | None = None,
) -> dict[str, float]:
    simulated = [*observations, Observation(evidence_id=evidence_id, observed_state=state, confidence=confidence)]
    return posterior(score_problems(kb, simulated, problem_ids))


def expected_information_gain(
    kb: KnowledgeBase,
    posterior_dist: dict[str, float],
    observations: list[Observation],
    evidence_id: str,
    problem_ids: list[str] | None = None,
) -> float:
    """`problem_ids`, when given, must match the universe `posterior_dist`
    was itself computed over (e.g. a branch-narrowed posterior) -- the
    before/after entropy comparison is only meaningful when both sides are
    taken over the same set of candidate problems."""
    current_entropy = entropy(posterior_dist)
    states = possible_states(kb, evidence_id)
    if not states:
        return 0.0

    raw_probs = {state: _answer_probability(kb, posterior_dist, evidence_id, state) for state in states}
    total_prob = sum(raw_probs.values()) or 1.0

    expected_entropy = 0.0
    for state, raw_p in raw_probs.items():
        p = raw_p / total_prob
        if p <= 0:
            continue
        expected_entropy += p * entropy(simulate_answer(kb, observations, evidence_id, state, problem_ids=problem_ids))

    return max(0.0, current_entropy - expected_entropy)


def routing_bonus(
    kb: KnowledgeBase,
    entry_context: str | None,
    evidence_id: str,
    num_observations: int,
    decay_turns: int = ROUTING_DECAY_TURNS,
) -> float:
    """Soft routing preference for one evidence item under the given entry
    context, already decayed by how many observations exist so far.

    `weight` comes straight from entry_contexts.yaml's topic_weights for
    this evidence's topic (0.0 if the topic isn't listed there, e.g.
    fish_disease_marker under every context -- never boosted, only ever
    left to compete purely on its own information gain). Decays linearly
    to 0 by `decay_turns` answered observations, so this only shapes the
    opening of the conversation and cannot become a standing bias once
    real evidence is accumulating.
    """
    if not entry_context:
        return 0.0
    context = kb.entry_contexts.get(entry_context)
    if context is None:
        return 0.0
    topic = kb.evidence[evidence_id].topic
    weight = context.topic_weights.get(topic, 0.0)
    if weight == 0.0:
        return 0.0
    decay = max(0.0, 1.0 - num_observations / decay_turns) if decay_turns > 0 else 0.0
    return weight * decay


def _distinguishes(kb: KnowledgeBase, posterior_dist: dict[str, float], evidence_id: str, top_n: int = 3) -> list[str]:
    informative_problems = kb.problems_informed_by_evidence.get(evidence_id, set())
    top = sorted(posterior_dist, key=lambda pid: posterior_dist[pid], reverse=True)[:top_n]
    return [pid for pid in top if pid in informative_problems] or top


def build_question_explanation(
    kb: KnowledgeBase,
    question_id: str,
    posterior_dist: dict[str, float],
    observations: list[Observation],
    entry_context: str | None = None,
    problem_ids: list[str] | None = None,
) -> QuestionExplanation:
    q = kb.questions[question_id]
    info_gain = expected_information_gain(kb, posterior_dist, observations, q.evidence_id, problem_ids)
    adjusted = info_gain * q.reliability * q.priority_weight / q.effort_cost
    # Answers to the universal TRIAGE screen (screening_evidence_ids) don't
    # count against routing's decay clock -- they happen before
    # orientation gets any real turns to act on, so they must not
    # silently burn down its opening window (see screening_evidence_ids).
    screening_ids = screening_evidence_ids(kb)
    routing_observations = sum(1 for o in observations if o.evidence_id not in screening_ids)
    bonus = routing_bonus(kb, entry_context, q.evidence_id, routing_observations)
    return QuestionExplanation(
        question_id=q.id,
        text=q.text,
        information_gain=info_gain,
        reliability=q.reliability,
        priority_weight=q.priority_weight,
        effort_cost=q.effort_cost,
        adjusted_value=adjusted,
        routing_bonus=bonus,
        final_score=adjusted + bonus,
        distinguishes=_distinguishes(kb, posterior_dist, q.evidence_id),
    )


def concern_pending_evidence(kb: KnowledgeBase, concern_id: str | None, answered_evidence_ids: set[str]) -> set[str]:
    """Evidence ids on the user's chosen concern (kb.concerns, set via
    engine.set_concern) that haven't been answered yet. Purely a lookup --
    no scoring/likelihood involved -- consumed by select_best_question as
    a priority tier, the same shape as
    safety.rules.pending_safety_evidence(). Naturally empties once every
    item on the concern is answered, at which point selection falls
    through to branch narrowing / full-KB ranking with no special-casing
    needed."""
    if concern_id is None:
        return set()
    concern = kb.concerns.get(concern_id)
    if concern is None:
        return set()
    return {eid for eid in concern.evidence_ids if eid not in answered_evidence_ids}


def narrowed_problem_ids(kb: KnowledgeBase, posterior_dist: dict[str, float]) -> set[str] | None:
    """The candidate-problem scope for branch-narrowed question ranking, or
    None if no branch is dominant yet (ranking should stay KB-wide).

    Branches are `Problem.category` -- reused as-is, no new KB concept.
    Once one category holds >= BRANCH_DOMINANCE_THRESHOLD of the posterior,
    the scope becomes that category's problems, plus any problem outside
    it that still holds >= BRANCH_OUTSIDER_FLOOR (a real competitor is
    never dropped just for sitting in a different category)."""
    if not posterior_dist:
        return None
    branch_mass: dict[str, float] = {}
    for pid, p in posterior_dist.items():
        branch_mass[kb.problems[pid].category] = branch_mass.get(kb.problems[pid].category, 0.0) + p
    top_branch, top_mass = max(branch_mass.items(), key=lambda kv: kv[1])
    if top_mass < BRANCH_DOMINANCE_THRESHOLD:
        return None
    return {
        pid
        for pid, p in posterior_dist.items()
        if kb.problems[pid].category == top_branch or p >= BRANCH_OUTSIDER_FLOOR
    }


def select_best_question(
    kb: KnowledgeBase,
    posterior_dist: dict[str, float],
    observations: list[Observation],
    answered_evidence_ids: set[str],
    entry_context: str | None = None,
    info_gain_epsilon: float = 0.02,
    concern_id: str | None = None,
) -> QuestionExplanation | None:
    """Ranks eligible questions by final_score = adjusted_value (pure
    information gain, unchanged) + routing_bonus (entry-context preference,
    0.0 when entry_context is None). With no entry_context this is
    identical to plain information-gain ranking -- routing is additive and
    optional, never a replacement for the underlying selector.

    Selection applies three things ahead of that ranking, in order (the
    first tier -- a forced safety question -- is decided upstream in
    engine.py and never reaches this function):

      1. TRIAGE (safety interrupt): unanswered evidence belonging to a
         *suspected* safety rule (safety.rules.pending_safety_evidence())
         -- one with at least one condition already observed matching,
         not merely unresolved -- and still carrying genuine information
         (adjusted_value >= info_gain_epsilon). If any such question is
         eligible, selection is restricted to just those, ranked by the
         same final_score. A rule drops out the moment it fires, is ruled
         out, or its remaining evidence stops teaching us anything --
         at which point this tier is simply empty and normal ranking
         resumes. Evaluated against the full KB-wide posterior/candidate
         set, regardless of branch narrowing below, so a real safety
         concern can never be diluted by diagnostic-scope narrowing.

      2. CONCERN (user-declared focus): unanswered evidence on the
         concern the user picked as a second step after entry_context
         (see kb.concerns / concern_pending_evidence()), same
         adjusted_value >= info_gain_epsilon guard as tier 1. This is
         what lets "my fish" -> "spots on the body" actually skip
         unrelated questions instead of merely nudging their order --
         unlike routing_bonus, this narrows which questions are even
         ranked. It empties the same way tier 1 does: once every item on
         the concern is answered or stops being informative, this tier is
         simply empty and normal ranking resumes below. Also evaluated
         against the full KB-wide posterior, so it composes independently
         of branch narrowing.

      3. TARGETED BRANCH (diagnostic-scope narrowing): if neither tier
         above applies, questions are ranked against a branch-narrowed
         posterior when one is dominant (see narrowed_problem_ids())
         instead of the full KB. This is a second, throwaway posterior
         computed purely for ranking -- the real posterior passed in is
         never mutated or returned narrowed. Early on (no branch dominant
         yet), this is identical to plain KB-wide information gain, which
         is exactly what should rank branch-splitting questions highest
         anyway.

    The `adjusted_value >= info_gain_epsilon` guard on tiers 1 and 2
    exists so neither can ever *force* a question the engine would
    otherwise consider not worth asking at all: `info_gain_epsilon` is
    the same threshold `AquariumInferenceEngine._check_stop()` already
    uses for exactly that judgment (default matches its own default),
    reused here rather than duplicated with a different number.
    """
    candidates = eligible_questions(kb, answered_evidence_ids)
    if not candidates:
        return None

    pending_safety = pending_safety_evidence(kb, observations)
    pending_concern = concern_pending_evidence(kb, concern_id, answered_evidence_ids)
    if pending_safety or pending_concern:
        full_explanations = [
            build_question_explanation(kb, q.id, posterior_dist, observations, entry_context) for q in candidates
        ]
        if pending_safety:
            safety_tier = [
                e
                for e in full_explanations
                if kb.questions[e.question_id].evidence_id in pending_safety and e.adjusted_value >= info_gain_epsilon
            ]
            if safety_tier:
                safety_tier.sort(key=lambda e: (-e.final_score, e.effort_cost, e.question_id))
                return safety_tier[0]
        if pending_concern:
            concern_tier = [
                e
                for e in full_explanations
                if kb.questions[e.question_id].evidence_id in pending_concern and e.adjusted_value >= info_gain_epsilon
            ]
            if concern_tier:
                concern_tier.sort(key=lambda e: (-e.final_score, e.effort_cost, e.question_id))
                return concern_tier[0]

    scope = narrowed_problem_ids(kb, posterior_dist)
    if scope is None:
        scoped_posterior, scope_ids = posterior_dist, None
    else:
        scope_ids = sorted(scope)
        scoped_posterior = posterior(score_problems(kb, observations, scope_ids))

    explanations = [
        build_question_explanation(kb, q.id, scoped_posterior, observations, entry_context, scope_ids)
        for q in candidates
    ]
    explanations.sort(key=lambda e: (-e.final_score, e.effort_cost, e.question_id))
    return explanations[0]
