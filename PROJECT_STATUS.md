# Project Status — Aquarium Troubleshooting Assistant

Last updated: 2026-09-08 (checkpoint after the entry-context routing milestone).
This file exists so a **new Claude Code session with no conversation history** can
pick this project up safely. Read this before touching any code.

## What this project is

An Akinator-style adaptive diagnostic engine for aquarium troubleshooting.
Given user-reported symptoms/parameters, it maintains Bayesian log-likelihood
scores over a knowledge base of ~23 candidate problems, picks the next
question by expected information gain (with a soft entry-context routing
preference layered on top), and enforces a safety layer that can override
normal question selection. **No LLM, no UI, no API, no persistence yet** —
this is a pure Python library + a knowledge base, driven today only by
Python calls and a bare CLI (`python -m aqua_assistant.api.cli`).

## Current architecture

```
kb/          knowledge base: schema (pydantic) + models (SQLAlchemy ORM,
             FK-enforced) + loader (YAML -> validated -> SQLite -> plain
             KnowledgeBase) + fixtures/*.yaml (the actual content)
case/        runtime case/observation state -- plain dataclasses, in-memory
             store (case/store.py). No persistence yet.
inference/   scoring.py (pure log-likelihood-ratio math, correlated-evidence
             discounting, softmax posterior) + entropy.py (Shannon entropy) +
             engine.py (AquariumInferenceEngine facade -- the only class
             most callers need)
questions/   selection.py -- information-gain question ranking, PLUS this
             milestone's entry-context routing_bonus()
safety/      rules.py -- independent safety-rule evaluation, can force a
             specific next question, always bypasses routing
recommendations/  engine.py -- dumb lookup from top candidate -> action
             bundle, no probability involved
api/         cli.py -- bare terminal loop for manual testing. Not a real UI.
```

**Layering is strict and intentional**: `inference/scoring.py` and
`inference/entropy.py` have never been modified since the original MVP.
Nothing outside `questions/selection.py` and `case/*` knows entry_context
exists. Nothing outside `safety/rules.py` and `inference/engine.py`'s
`get_status()` knows about forced questions.

## Completed milestones (chronological)

1. **MVP inference engine** — scoring, entropy, question selection, safety,
   recommendations, 8-problem seed KB. 44 tests.
2. **Stabilization pass** — found and fixed 2 real bugs: `possible_states()`
   silently dropped the "no" branch from info-gain simulation for most
   boolean questions (understated their real information gain); safety
   alerts were collapsing to only the single most severe one instead of
   reporting every firing rule. Both fixed, git initialized, first commit.
3. **KB expansion** — audited the original 8 problems for missing
   negative/disconfirming evidence (+40 rows), then added 15 new problems
   in 4 domain batches (water quality, environmental, disease, husbandry).
   97 tests.
4. **Behavioral evaluation** (no code changes) — ran 10 realistic simulated
   conversations through the engine and found the *question ordering* had
   degraded as the KB grew: nearly every conversation opened with the same
   4-9 question disease-marker checklist regardless of the user's actual
   complaint, and safety-relevant questions could get crowded out for many
   turns (CO2-overdose scenario: safety alert didn't fire until the very
   last question of 14).
5. **Entry-context routing** (this checkpoint) — added a soft "what are you
   concerned about?" preference (fish / water / plants / aquarium_environment
   / unsure) that biases early question ordering without ever touching
   diagnostic scoring. 120 tests. **Committed** (`1416599`).

## Entry-context routing — current design

**Concept**: the user optionally names a starting concern area. This is
*never* diagnostic evidence — it only nudges which question gets asked next.

**Mechanism**:
- Every `Evidence` item has a free-text `topic` (`general_context`,
  `fish_symptom`, `fish_disease_marker`, `water_parameter`, `water_history`,
  `environment_system`) — same unenforced-categorical-string convention
  already used by `Problem.category`, no separate registry table.
- `kb/fixtures/entry_contexts.yaml` maps each of the 5 contexts to
  `topic_weights: {topic: weight}`. `general_context` has weight in *every*
  context (including `unsure`) — this is what gives every context a
  sensible "triage-shaped" opening; domain-specific weights on top are what
  differentiate contexts from each other.
- `questions/selection.py::routing_bonus()`:
  `bonus = topic_weight * max(0, 1 - observations_answered / 6)` (linear
  decay, `ROUTING_DECAY_TURNS = 6`, module constant).
- Ranking key changed from `adjusted_value` to
  `final_score = adjusted_value + routing_bonus`. `adjusted_value` itself
  (pure info gain × reliability × priority / effort) is **never modified**.
  `entry_context=None` reproduces the pre-routing selector exactly (tested).
- `Case.entry_context: str | None`, set via
  `engine.start_case(entry_context="fish")` (validated against
  `kb.entry_contexts`, raises `ValueError` on an unknown id).
- Safety-forced questions bypass this entire ranking path (pre-existing
  branch in `engine.py::get_status()`), so safety always wins with zero
  special-casing.
- `_check_stop()` still reads `adjusted_value`, not `final_score` — routing
  cannot affect the stop/continue decision, only question choice.

**Weights are hand-calibrated, not principled.** They were tuned by
inspecting actual `adjusted_value` numbers at turn 1 against this specific
23-problem KB until the behavior looked right (see commit `1416599` message
for the numbers). **They will need re-tuning if the KB grows meaningfully**
— there's no formula, just empirical iteration. See
`tests/test_entry_context_routing.py` for the exact expected behaviors this
calibration must keep satisfying.

## Current test count/result

**120 tests, 120 passing**, confirmed from a completely fresh
`pip install -e .[dev]` (no cached state). Run with:

```bash
python -m pytest -q
```

Test files: `test_scoring.py`, `test_entropy.py`, `test_question_selection.py`,
`test_safety_rules.py`, `test_kb_loader.py`, `test_recommendations.py`,
`test_end_to_end_case.py` (MVP criteria), `test_batch_water_quality.py`,
`test_batch_environmental.py`, `test_batch_disease.py`,
`test_batch_husbandry.py` (KB-expansion batch tests),
`test_entry_context_routing.py` (this milestone).

## Current KB size

| | count |
|---|---|
| problems | 23 (water_quality 6, environmental 5, disease 4, parasite 3, husbandry 5) |
| evidence items | 45 (fish_symptom 14, fish_disease_marker 7, general_context 10, water_parameter 6, water_history 4, environment_system 4) |
| evidence groups (correlated-symptom clusters) | 7 |
| problem_evidence likelihood rows | 259 |
| questions | 45 |
| question_answers (answer options) | 79 |
| recommendations | 124 |
| safety_rules | 7 |
| entry_contexts | 5 (fish, water, plants, aquarium_environment, unsure) |

Original target from the founding plan was ~20-30 problems / 50-100
evidence / 50-100 questions — currently at the low end of that range by
design ("small and explicit" over padding for its own sake).

## Known behavioral limitations (be honest about these, don't silently "fix")

1. **`plants` entry context is thin.** The KB has exactly one plant-adjacent
   evidence item (`planted_tank_with_co2_injection`, tagged
   `environment_system`). There is no dedicated plant-health evidence
   (leaf condition, algae, melting/yellowing, lighting/fertilization). The
   `plants` context currently reduces to "ask about CO2 slightly early,
   then fall back to general triage" — in one replay scenario it produced
   an *identical* final result and question count to `unsure`. Fixing this
   properly means adding real plant-health evidence/problems — out of
   scope until explicitly requested.
2. **`dissolved_oxygen_ppm` is tagged `water_parameter`, not
   `environment_system`.** For a low-oxygen scenario, the `water` context
   reaches the critical reading faster (11 questions) than
   `aquarium_environment` (13 questions) — arguably backwards, since low
   oxygen is at least as much an "environment" problem. `Evidence.topic`
   is a single string; giving an item two topics would need a real schema
   change (list-valued topic or a many-to-many table), not done here.
3. **A single dominant marker (`white_spots`, raw info gain ~1.7) still
   opens almost every conversation regardless of entry context.** This is
   intentional — routing is calibrated to never overwhelm a genuinely
   exceptional information-gain signal — but it means turn 1 is often
   identical across contexts; differentiation shows up from turn 2 onward,
   not turn 1.
4. **Routing measurably speeds up *some* scenarios (CO2 overdose: 14→11
   questions; general water quality: 18→16) but not others** (chronic
   nitrate stress and the deliberately-contradictory safe-ammonia case
   both stayed the same total length, ~17 and ~32-33 questions
   respectively — routing reorders those, it doesn't shorten them). Do not
   assume routing improves every scenario; it only helps when the chosen
   context's topics actually overlap with what turns out to be relevant.
5. **In the contradictory-evidence case, routing surfaced one alarming
   symptom much earlier (`red_streaks_or_gills`: question 21→2) but not
   the specific symptom gating the safety rule (`gasping`: 25→24,
   essentially unchanged).** Improving "safety-relevant evidence surfaces
   fast" further would need routing (or a separate mechanism) to boost
   specifically safety-linked evidence, not just topically-preferred
   evidence — not implemented.
6. **KB completeness gaps remain** (documented in earlier milestones,
   unresolved): several problems lack full negative/disconfirming rows for
   every symptom (only audited where a confident direction was
   defensible); `sudden_multiple_fish_death` deliberately has only
   representative (not exhaustive) problem_evidence rows.

## Important design decisions (and why)

- **Pure Python + SQLite + YAML fixtures**, no numpy/pandas — scale is
  small enough (tens of problems/evidence) that plain Python loops are
  simplest and clearest. YAML fixtures are the source of truth (diffable,
  reviewable by non-engineers); SQLite is populated at load time purely to
  exercise real FK referential-integrity checks, not for runtime querying.
- **`KnowledgeBase` is a plain Python object**, not the ORM — inference/
  questions/safety/recommendations never see SQLAlchemy objects, only
  plain pydantic-derived data. Keeps those modules trivially testable
  without a database.
- **Log-likelihood-ratio scoring with correlated-evidence group discounting**
  (`inference/scoring.py`) — verified never to double-count correlated
  symptoms (e.g. gasping/rapid_breathing/surface_breathing).
- **Safety is structurally independent of scoring and of routing** — it's
  evaluated fresh every turn from raw observations, can force a specific
  next question, and that forced-question branch in `engine.py` completely
  bypasses both information-gain ranking and entry-context routing. This
  is why "safety always overrides routing" required zero special-case code.
- **`_check_stop()` reads only `adjusted_value` (pure info gain), never
  `final_score` (routing-influenced)** — a deliberate choice so routing can
  never make the engine stop too early or too late; it can only reorder
  which question comes next while evidence is still being gathered.
- **Entry-context topics are a single free-text string per evidence item,
  not a list** — kept deliberately simple; the known cost is limitation #2
  above (an item can't cleanly belong to two topics).

## Explicitly NOT implemented (do not add without being asked)

- Species-specific scoring/parameters (`species`/`species_parameters`
  tables exist in the SQL schema but are unused — no fixture data, no
  logic reads them).
- Persistence for cases/observations (in-memory only; `cases`/
  `observations` SQLAlchemy tables exist in the schema for a future API
  but nothing writes to them yet).
- Any real API or frontend/UI. `api/cli.py` is a bare manual-testing loop.
- LLM integration / free-text symptom interpretation. `Observation.source`
  has a `free_text_llm_extracted` enum value reserved for this, unused.
- Computer vision.
- Prevalence discounting (weighting rare-disease markers down because
  they're rare) — flagged as a possible future refinement, not built.
- A "triage" layer as a separate architectural concept — it was folded
  into entry-context routing via the shared `general_context` topic weight
  present in every context (see design section above). If a future
  request asks for "triage" as if it's missing, it isn't missing — it's
  this mechanism.

## Exact next recommended milestone

**Either, in priority order depending on what's wanted:**

1. **Close the `plants` gap** — add a small, explicit set of plant-health
   evidence items (leaf condition, algae, melting/yellowing/browning,
   lighting) and, if warranted, 1-3 plant-health problems, following the
   same audit-first + domain-batch process used for the KB expansion
   milestone. This is the most concrete unfinished thread from this
   checkpoint.
2. **Fix the `dissolved_oxygen_ppm` topic mapping** (limitation #2) — either
   accept a small pragmatic duplication (a second `environment_system`-
   flavored oxygen-adjacent evidence item) or decide list-valued topics
   are worth the schema change. Small, isolated, well-understood fix.
3. **Re-run the same kind of behavioral evaluation** (10 realistic
   scenarios, read-only, no code changes) done twice before in this
   project, specifically probing whether entry-context routing plus the
   current KB still shows any of the 5 previously-fixed/previously-flagged
   issues at a larger KB size, before deciding whether routing needs a
   further mechanism (e.g. explicit safety-linked-evidence boosting).
4. Only after 1-3: persistence, then a real API, are the next
   architectural layers per the original founding plan.

Do not jump straight to API/UI/LLM work without doing at least (1) or (3)
first — the founding instructions and every milestone since have
prioritized inference/KB quality over surface area.

## What a new Claude Code session needs to know before touching code

- **Read this file first**, then skim `git log --oneline` (3 commits as of
  this checkpoint) and the docstrings at the top of `inference/engine.py`,
  `questions/selection.py`, and `safety/rules.py` — they carry real design
  rationale, not just descriptions.
- **Run `python -m pytest -q` before and after any change.** 120 passing is
  the known-good baseline captured here.
- **Never pass `entry_context` into `inference/scoring.py` or
  `inference/entropy.py`.** This boundary is load-bearing and
  regression-tested (`test_entry_context_never_changes_*`). If a future
  change seems to require it, that's a sign the design is being violated,
  not a sign the test is wrong.
- **The Windows dev environment has a `MAX_PATH` gotcha**: scratch/temp
  scripts run from this session's deeply-nested scratch-workspace path can
  fail to open via the native `python.exe` with a cryptic
  `[Errno 2] No such file or directory` even though the file exists (git
  bash's `/tmp` alias hides the real path length from you). If a one-off
  script mysteriously "can't be found," copy it to a short path like
  `C:\Users\<user>\AppData\Local\Temp\` and run it from there.
- **Test-writing gotcha already hit twice**: when scripting an "answer
  whatever question comes next" loop for a scenario probe, do not default
  unscripted numeric answers to a single constant like `0.0` — it means
  "safe" for ammonia/nitrite but "critical" for dissolved oxygen and
  silently fabricates a crisis. Use `tests/conftest.py::neutral_default()`
  (already built for this) instead of reinventing it.
- **The KB fixtures in `src/aqua_assistant/kb/fixtures/*.yaml` are the
  actual source of truth for domain content** — always edit these, never
  try to patch knowledge into Python code.
- **This project has never had an API, UI, or LLM component.** If asked to
  "wire up the frontend" or similar, there isn't one yet to wire up to;
  check with the user about scope before assuming one exists.
