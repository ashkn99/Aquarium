"""Aggregate report over real, logged cases -- the same shape of report
used to evaluate Question Strategy v2 against 10 scripted scenarios
(see the project history), now run against actual tester sessions.

Reads every case straight from the case store (no separate "analytics"
copy of the data): a case's persisted observations already record the
exact question sequence in the order it was asked, and
`engine.get_status()` recomputes the final ranking/stop reason from that
same data the live app used -- so this can never drift from what
actually happened.

Run manually: `python -m aqua_assistant.analytics.report`
"""
from __future__ import annotations

import os
import statistics
from collections import Counter

from aqua_assistant.case.sql_store import SqlCaseStore
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.safety.rules import evaluate_safety


def _first_safety_alert_turn(kb, observations) -> int | None:
    for i in range(1, len(observations) + 1):
        if evaluate_safety(kb, observations[:i]):
            return i
    return None


def build_report(store: SqlCaseStore, engine: AquariumInferenceEngine) -> dict:
    kb = engine.kb
    per_case = []
    for case_id in store.all_case_ids():
        case = store.get(case_id)
        observations = case.active_observations()
        if not observations:
            continue  # abandoned before answering anything -- nothing to learn from
        status = engine.get_status(case_id)
        top = status.ranked_candidates[0] if status.ranked_candidates else None
        per_case.append(
            dict(
                case_id=case_id,
                entry_context=case.entry_context,
                num_questions=len(observations),
                top_problem=top.problem_id if top else None,
                top_probability=top.probability if top else None,
                stop_reason=status.stop_reason,
                should_stop=status.should_stop,
                safety_alert_turn=_first_safety_alert_turn(kb, observations),
            )
        )

    questions = [c["num_questions"] for c in per_case]
    n = len(questions) or 1
    stop_reasons = Counter(c["stop_reason"] for c in per_case)
    feedback = store.all_feedback()
    helpful_count = sum(1 for f in feedback if f.helpful)

    summary = dict(
        num_cases=len(per_case),
        num_completed=sum(1 for c in per_case if c["should_stop"]),
        mean_questions=round(statistics.mean(questions), 2) if questions else 0,
        median_questions=statistics.median(questions) if questions else 0,
        pct_le_5=round(100 * sum(1 for q in questions if q <= 5) / n, 1),
        pct_le_10=round(100 * sum(1 for q in questions if q <= 10) / n, 1),
        pct_le_15=round(100 * sum(1 for q in questions if q <= 15) / n, 1),
        stop_reason_counts=dict(stop_reasons),
        cases_with_safety_alert=sum(1 for c in per_case if c["safety_alert_turn"] is not None),
        safety_alert_turns=[c["safety_alert_turn"] for c in per_case if c["safety_alert_turn"] is not None],
        num_feedback=len(feedback),
        feedback_helpful_rate=round(100 * helpful_count / len(feedback), 1) if feedback else None,
        feedback_comments=[f.comment for f in feedback if f.comment],
    )
    return dict(cases=per_case, summary=summary)


def main() -> None:
    kb = load_knowledge_base()
    db_url = os.environ.get("DATABASE_URL", "sqlite:///cases.db")
    store = SqlCaseStore(db_url)
    engine = AquariumInferenceEngine(kb, store=store)

    report = build_report(store, engine)
    import json

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
