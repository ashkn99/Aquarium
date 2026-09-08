"""Bare terminal question/answer loop for manually smoke-testing the
engine. Not a real API or UI -- run with:

    python -m aqua_assistant.api.cli
"""
from __future__ import annotations

from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.kb.knowledge_base import KnowledgeBase
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import possible_states
from aqua_assistant.recommendations.engine import get_recommendations


def _prompt_for_answer(kb: KnowledgeBase, evidence_id: str, question_text: str) -> tuple[str | None, float | None]:
    evidence = kb.evidence[evidence_id]
    if evidence.data_type == "numeric":
        raw = input(f"{question_text} (numeric{f', {evidence.unit}' if evidence.unit else ''}): ").strip()
        return None, float(raw)

    states = possible_states(kb, evidence_id)
    print(question_text)
    for i, state in enumerate(states):
        print(f"  {i}. {state}")
    choice = int(input("> ").strip())
    return states[choice], None


def main() -> None:
    kb = load_knowledge_base()
    engine = AquariumInferenceEngine(kb)
    case_id = engine.start_case()

    print("Aquarium troubleshooting assistant (MVP CLI). Ctrl+C to quit.\n")

    while True:
        status = engine.get_status(case_id)

        print(f"\nUncertainty: {status.uncertainty:.2f}")
        print("Top candidates:")
        for c in status.ranked_candidates[:5]:
            print(f"  {c.probability:5.1%}  {c.problem_name}")

        for alert in status.safety_alerts:
            print(f"\n[SAFETY - {alert.severity.upper()}] {alert.message}")
            print(f"  -> {alert.escalation_text}")

        if status.should_stop:
            print(f"\nStopping ({status.stop_reason}).")
            top = status.ranked_candidates[0]
            print(f"\nMost likely: {top.problem_name} ({top.probability:.1%})")
            print("(Relative plausibility based on current evidence -- not a certain diagnosis.)")
            bundle = get_recommendations(kb, top.problem_id)
            print("\nConfirmation checks:")
            for t in bundle.confirmation_checks:
                print(f"  - {t}")
            print("Immediate actions:")
            for t in bundle.immediate_actions:
                print(f"  - {t}")
            break

        q = status.best_next_question
        evidence_id = kb.questions[q.question_id].evidence_id
        print(f"\n[why this question] info_gain={q.information_gain:.2f} adjusted_value={q.adjusted_value:.2f}")

        state, raw_value = _prompt_for_answer(kb, evidence_id, q.text)
        engine.answer(case_id, evidence_id, state=state, raw_value=raw_value, question_id=q.question_id)


if __name__ == "__main__":
    main()
