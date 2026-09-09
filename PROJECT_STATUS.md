# Project Status — Aquarium Troubleshooting Assistant

Last updated: 2026-09-09 (checkpoint after the public-beta deployment +
Question Strategy v2 + concern-tier milestones).
This file exists so a **new Claude Code session with no conversation history**
can pick this project up safely. Read this before touching any code.

## What this project is

An Akinator-style adaptive diagnostic engine for aquarium troubleshooting,
now a **live public beta webapp**: https://aquarium-assistant.onrender.com

Given user-reported symptoms/parameters, it maintains Bayesian
log-likelihood scores over a knowledge base of 30 candidate problems (fish,
water, environment, plant health), picks the next question by expected
information gain layered with entry-context routing + branch narrowing +
a user-declared "concern," and enforces a safety layer that can override
normal question selection. No LLM, no computer vision — pure Python +
YAML-fixture KB + a small FastAPI backend + a plain-JS frontend.

## Current architecture

```
kb/            knowledge base: schema (pydantic) + models (SQLAlchemy ORM,
               FK-enforced in the KB's own throwaway validation DB) +
               loader (YAML -> validated -> SQLite -> plain KnowledgeBase)
               + fixtures/*.yaml (the actual content, including
               concerns.yaml)
case/          runtime case/observation state -- plain dataclasses
               (models.py) + two store implementations behind the same
               create/get/save interface: store.py (in-memory, tests) and
               sql_store.py (persistent, powers the deployed app)
inference/     scoring.py (pure log-likelihood-ratio math) + entropy.py +
               engine.py (AquariumInferenceEngine facade)
questions/     selection.py -- the whole question-priority stack (see
               "Question selection" below)
safety/        rules.py -- independent safety-rule evaluation
recommendations/  engine.py -- dumb lookup from top candidate -> action
               bundle
api/           app.py + schemas.py -- the public FastAPI backend;
               cli.py -- the original bare terminal loop, still works,
               useful for quick local debugging
analytics/     report.py -- replays real logged cases through the engine
               to produce the same aggregate report used to evaluate
               question-strategy changes, but from real usage
frontend/      index.html / app.js / style.css -- plain HTML/CSS/vanilla
               JS, no build step, no framework, served as static files by
               the FastAPI app itself (same origin, no CORS)
```

**Layering is strict and intentional**: `inference/scoring.py` and
`inference/entropy.py` have never been modified since the original MVP.
Nothing outside `questions/selection.py` and `case/*` knows entry_context
or concern_id exist. Nothing outside `safety/rules.py` and
`inference/engine.py`'s `get_status()` knows about forced questions.

## Deployment

- **Live URL**: https://aquarium-assistant.onrender.com (Render free web
  service, auto-deploys on push to `main`, ~30-50s cold start after ~15min
  idle).
- **Database**: Neon free Postgres (`DATABASE_URL` env var on Render).
  Case/observation/feedback data persists there indefinitely; the KB
  itself is always loaded fresh from YAML on every process start and never
  touches this database.
- **GitHub**: https://github.com/ashkn99/Aquarium (public), `main` branch.
- **Deploy runbook**: [DEPLOY.md](DEPLOY.md) — only needed again if setting
  up a *new* environment; the existing one auto-deploys.
- **`render.yaml`** at repo root is the Render Blueprint (build/start
  command, free plan) — Render reads it automatically.

## Completed milestones (chronological)

1–7. MVP engine, stabilization, KB expansion to 23 then 30 problems,
   behavioral evaluation, entry-context routing, `dissolved_oxygen_ppm`
   retagging, plant-health KB expansion. (See git log `0b90687`..`3d6e993`
   for detail; this file no longer carries their full narrative now that
   several later milestones superseded parts of their content — read the
   commit messages if you need the original reasoning.)
8. **Question Strategy v2** (`2b2b299`) — replaced an unconditional
   "every safety rule pending from turn 1" experiment (which fixed a real
   regression but made every clean case longer) with: TRIAGE as a true
   interrupt (a safety rule only gets priority once actually suspected,
   with one structural exception for rules with no `all_of` clause, which
   can never show partial signal) + TARGETED BRANCH (`Problem.category`
   narrowing once the posterior naturally dominates one category). Full
   before/after 10-scenario behavioral evaluation is in this milestone's
   commit message and this session's transcript, not duplicated here.
9. **Public beta webapp** (`3487e2a`, `406ddac`) — FastAPI backend +
   plain-JS frontend + `SqlCaseStore` (persists into `CaseORM`/
   `ObservationORM`, which existed unused in the schema for exactly this),
   feedback capture, `analytics/report.py`, basic per-IP rate limiting,
   trimmed the entry-context picker to fish/water/plants/unsure.
10. **Deployed to Render + Neon** (`d8a535e`) — hit and fixed two
    production-only bugs invisible in SQLite tests (see "Postgres gotchas"
    below); both now have regression tests.
11. **Concern tier** (`2d365e1`) — a second, more specific choice right
    after entry_context (e.g. "my fish" -> "spots on the body"), backed
    by `kb/fixtures/concerns.yaml` (9 concerns). Unlike entry-context
    routing's soft bonus, this is a hard priority *filter* -- it can
    actually skip unrelated questions.
12. **Domain-scoped mandatory screen + concern-seeded branch narrowing**
    (`06d8cb5`) — fixed a real bug found by manual testing: every case
    opened on "is the fish gasping at the surface?" regardless of
    entry_context, because the structurally-blind safety-screen exception
    from milestone 8 had zero awareness of declared domain. Now scoped by
    topic relevance to entry_context (fish/unsure keep it; water/plants/
    aquarium_environment don't). Also seeds initial branch narrowing from
    the declared concern, closing the gap where concern only narrowed
    *evidence* but not *diagnostic scope*. Design grounded in real triage-
    system research (Infermedica's red-flags API; pet-symptom-checker
    apps) — see this session's transcript for sources.

## Question selection — current full priority stack

In order, from `questions/selection.py::select_best_question` (tier 0 is
decided in `engine.py`, before this function is even called):

0. **Forced safety question** (`engine.py`) — a firing urgent rule with a
   `forced_question_id` completely bypasses everything below.
1. **TRIAGE / safety suspicion** (`safety/rules.py::pending_safety_evidence`)
   — a rule with at least one condition already observed matching. Purely
   evidence-driven, ignores entry_context/concern entirely by design (real
   danger must never be diluted by declared focus).
2. **Domain-scoped mandatory screen** (`safety/rules.py::screening_evidence_ids`,
   filtered in `selection.py` by `_entry_context_declares_a_domain`) — a
   rule with no `all_of` clause (today: `respiratory_distress_general`,
   gasping/surface_breathing) can never show partial signal, so it's
   screened directly instead — but only when its evidence's `topic` is
   relevant to the declared entry_context, or no domain was declared
   (`unsure`/`None`).
3. **Concern** (`concern_pending_evidence`) — unanswered evidence on the
   user's declared `Concern` (kb.concerns), a hard filter.
4. **Branch narrowing** (`narrowed_problem_ids`) — once a `Problem.category`
   holds ≥55% of the posterior, ranking narrows to that category (+ any
   outside problem still holding ≥3%). Before any branch is dominant,
   falls back to **concern-seeded scope** (`concern_seeded_problem_ids` —
   the union of problems the declared concern's evidence actually
   informs) if a concern was declared, else full KB.
5. **Entry-context routing bonus** (`routing_bonus`) — additive, decaying
   soft nudge, composes with whichever tier/scope above is active.

Tiers 1–3 all share the `adjusted_value >= info_gain_epsilon` guard so
none can force a question that's stopped being worth asking. All of this
is selection-time only — `inference/scoring.py`/`entropy.py` never see
entry_context or concern_id, tested directly
(`test_entry_context_never_changes_*`).

## Current test count/result

**238 tests, 238 passing**, confirmed from a fresh `pip install -e .[dev]`.

```bash
python -m pytest -q
```

New test files this checkpoint: `test_sql_store.py`, `test_api.py`,
`test_analytics_report.py`, `test_branch_narrowing.py`,
`test_concern_selection.py`. `test_safety_priority.py` was substantially
rewritten twice (once for Question Strategy v2, once for domain-scoped
screening) — its docstrings explain the current semantics, not history.

## Current KB size

| | count |
|---|---|
| problems | 30 (plant_health 7, water_quality 6, environmental 5, husbandry 5, disease 4, parasite 3) |
| evidence items | 64 (fish_symptom 14, general_context 12, environment_system 11, plant_symptom 11, fish_disease_marker 7, water_parameter 5, water_history 4) |
| evidence groups | 9 |
| problem_evidence likelihood rows | 328 |
| questions | 64 |
| question_answers | 120 |
| recommendations | 159 |
| safety_rules | 7 |
| entry_contexts | 5 (fish, water, plants, aquarium_environment, unsure) |
| **concerns** (new) | 9 (3 each under fish/water/plants; none under aquarium_environment/unsure) |

## Postgres gotchas (hit twice in production, both now guarded)

**SQLite silently permits things Postgres rejects — every test in this
project runs on SQLite, so a class of bug only ever shows up live.**

1. **Foreign keys across independently-populated databases.** The KB's
   own tables (`evidence`, `questions`, ...) only ever get populated in
   `kb/loader.py`'s throwaway validation database — never in the
   persistent case-store database. `ObservationORM.evidence_id`/
   `question_id` used to carry a `ForeignKey` into those KB tables; SQLite
   doesn't enforce FKs by default so this passed every test, but Postgres
   always enforces them, so every real answer 500'd on first deploy. Fixed
   by making those plain strings (validity is already checked at the
   application layer). `tests/test_sql_store.py` has a regression test
   that explicitly turns SQLite FK enforcement on to catch this class of
   bug locally now.
2. **`create_all()` never alters an existing table.** Adding a column to
   `CaseORM` (e.g. `concern_id`) after the live Neon `cases` table already
   existed meant the column silently never appeared there — every insert
   referencing it 500'd. Fixed: `SqlCaseStore` now inspects its own tables
   on startup and `ADD COLUMN`s anything the model declares that's
   missing (`case/sql_store.py::_add_missing_columns`). **Scoped
   deliberately to nullable ADD COLUMN only** — a future rename, drop, or
   NOT NULL column needs a real migration tool (e.g. Alembic) instead.

**Practical implication for future schema changes**: after editing
`CaseORM`/`ObservationORM`/`CaseFeedbackORM`, the self-heal covers you
automatically on next deploy — but if you ever need something more than
adding a nullable column, don't assume it's covered.

## Known behavioral limitations (be honest about these, don't silently "fix")

1. **Ambiguous/contradictory cases still run to ~45-50 questions before a
   weak stop.** Confirmed in the Question Strategy v2 evaluation (see
   commit `2b2b299` message) and not addressed since: branch narrowing
   only engages once a category is inferred dominant, which by definition
   never happens for a genuinely ambiguous case, so `_check_stop`'s
   `diminishing_information_gain` guard keeps finding *something* barely
   above `info_gain_epsilon` among ~64 questions for a long time. Labeled
   honestly as `diminishing_information_gain_ambiguous` in `engine.py`
   rather than hidden, but not fixed. A real fix needs either a bounded
   exploration budget or a genuine multi-hypothesis margin check — a
   separate design decision, not a tuning knob.
2. **`fish_mortality` concern barely narrows anything** — its evidence
   (`sudden_multiple_fish_death`, `num_fish_affected`, `onset_timing`) is
   mostly `general_context`-topic evidence used by nearly every problem in
   the KB, so `concern_seeded_problem_ids("fish_mortality")` returns ~29
   of 30 problems. Not a bug -- that concern is inherently about urgency,
   not differential narrowing -- but don't expect it to behave like
   `plant_damage` (narrows to 3).
3. **No visual design pass on the concern picker yet** — plain buttons,
   same as the entry-context picker, explicitly deferred by the user's own
   request ("for now we should clear out the logic behind" / cards later).
4. **`white_spots` still opens most conversations that lack both a
   relevant declared domain's mandatory screen and a concern** (documented
   since the original entry-context milestone) — it's the KB's single most
   informative raw signal and this is judged correct, not a bug: routing/
   concern narrowing is calibrated to never suppress a genuinely
   exceptional information-gain signal.
5. **KB completeness gaps remain** (long-standing, unresolved): several
   problems lack full negative/disconfirming rows for every symptom;
   `sudden_multiple_fish_death` deliberately has only representative
   (not exhaustive) problem_evidence rows.
6. **`plant_nutrient_deficiency` covers several distinct underlying
   nutrients**, differentiated only by leaf-age pattern, not by specific
   nutrient — deliberate (hobbyist-observable signs can't reliably go
   further), not a bug.

## Important design decisions (and why)

- **Pure Python + SQLite (KB validation) + Postgres (case persistence) +
  YAML fixtures**, no numpy/pandas, no ORM leaking into inference/
  questions/safety/recommendations (`KnowledgeBase` is a plain object).
- **Safety is structurally independent of scoring and of routing/concern**
  — evaluated fresh every turn from raw observations; a firing rule's
  forced question bypasses all ranking with zero special-casing needed
  elsewhere.
- **Concern is a selection-time filter, not a scoring change** — same
  architectural boundary entry_context already established
  (`inference/scoring.py`/`entropy.py` never see it), just enforced as a
  hard filter instead of a soft bonus because a bonus provably couldn't
  overcome this KB's near-pathognomonic-marker crowding.
- **The case-store schema deliberately does NOT share foreign-key
  constraints with the KB schema**, even though both currently live in
  `kb/models.py`'s same `Base`/`DeclarativeBase` for code-organization
  convenience — see "Postgres gotchas" above. If you add a new
  case-store column that logically references KB content (another
  evidence_id, question_id, problem_id, concern_id...), make it a plain
  string, not a `ForeignKey`.
- **No migration tool** — `SqlCaseStore`'s self-heal (nullable
  `ADD COLUMN` only) has covered every schema change so far. Don't reach
  for Alembic preemptively; do reach for it the moment a change needs a
  rename, drop, type change, or NOT NULL column.

## Explicitly NOT implemented (do not add without being asked)

- Species-specific scoring/parameters (`species`/`species_parameters`
  tables exist in the SQL schema but are unused).
- LLM integration / free-text symptom interpretation
  (`Observation.source` has a `free_text_llm_extracted` enum value
  reserved for this, unused). Computer vision.
- Prevalence discounting (weighting rare-disease markers down).
- Visual/card-based UI for the entry-context or concern pickers — plain
  buttons only, by explicit user request to land the logic first.
- Alembic or any other migration tool (see above).
- User accounts, auth, or any PII collection — the public beta is
  deliberately anonymous (no login) with no name/email/contact field
  anywhere, including in feedback.

## Exact next recommended milestone

In priority order:

1. **Let the public beta actually collect usage** — nothing forces this,
   but the whole point of `analytics/report.py` was to eventually run it
   against real tester data instead of scripted scenarios. Nobody has
   done that yet (the Neon DB is empty as of this checkpoint — every case
   created during development/testing was deliberately cleaned up).
2. **Visual pass on the entry-context/concern pickers** (cards with
   graphics) — explicitly deferred, was going to be the very next UX step
   before the "same first question" bug took priority.
3. **The ambiguous-case stopping-logic bottleneck** (limitation #1 above)
   — a real, identified, unresolved architectural gap. Needs a design
   decision (bounded exploration budget vs. margin-based stop), not a
   quick fix.
4. Only after real usage data exists: revisit KB completeness gaps or
   concern/branch-narrowing calibration using *actual* tester behavior
   from `analytics/report.py`, rather than more scripted scenarios.

## What a new Claude Code session needs to know before touching code

- **Read this file first**, then skim `git log --oneline` and the
  docstrings at the top of `inference/engine.py`, `questions/selection.py`,
  and `safety/rules.py` — they carry real design rationale.
- **Run `python -m pytest -q` before and after any change.** 238 passing
  is the known-good baseline captured here.
- **SQLite ≠ Postgres for foreign keys and schema changes** — see
  "Postgres gotchas" above. Any change touching `case/sql_store.py` or
  `kb/models.py`'s Case/Observation/CaseFeedback ORMs should be verified
  against the *real* Neon database before considering it done, not just
  local SQLite tests. The Neon connection string is not stored anywhere
  in this repo (by design) — ask the user for it if you need to verify
  live, and never write it to any file, committed or not.
- **Never pass `entry_context` or `concern_id` into `inference/scoring.py`
  or `inference/entropy.py`.** Load-bearing, regression-tested boundary.
- **The KB fixtures in `src/aqua_assistant/kb/fixtures/*.yaml` are the
  actual source of truth for domain content** — always edit these.
- **Windows `MAX_PATH` gotcha**: scratch scripts run from a deeply-nested
  scratch-workspace path can fail to open via the native `python.exe`
  with a cryptic `[Errno 2] No such file or directory`. Copy to
  `C:\Users\<user>\AppData\Local\Temp\` and run from there if this happens.
- **Test-writing gotcha**: never default unscripted numeric/categorical
  answers to a naive constant or `sorted(states)[0]` — use
  `tests/conftest.py::neutral_default()`. Also: a fully-neutral answer
  script (no real signal) can legitimately take 45-50 turns to converge
  (see limitation #1) — don't assume a low `max_turns` in a test's driving
  loop without checking against that.
- **Render free tier cold-starts** (~30-50s after ~15min idle) — if a
  live-verification script's first request seems to fail/hang, that's
  normal; poll a few times before concluding something's actually broken.
- **This project now has a real API, UI, and deployment** — if asked to
  "wire up the frontend" or similar, there already is one; check
  `frontend/app.js` and `api/app.py` before assuming greenfield work.
