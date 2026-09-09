"""Public web API for the aquarium assistant.

Serves the JSON endpoints the frontend (mounted as static files at `/`)
calls, backed by the same `AquariumInferenceEngine` used everywhere else
in this project -- this module is purely a delivery layer, no inference
logic lives here.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from aqua_assistant.case.sql_store import SqlCaseStore
from aqua_assistant.inference.engine import AquariumInferenceEngine
from aqua_assistant.kb.loader import load_knowledge_base
from aqua_assistant.questions.selection import QuestionExplanation, possible_states
from aqua_assistant.recommendations.engine import get_recommendations

from .schemas import (
    AnswerOption,
    AnswerRequest,
    CandidateOut,
    CaseStatusResponse,
    ConcernOut,
    EntryContextOut,
    FeedbackRequest,
    NextQuestion,
    RecommendationsOut,
    SafetyAlertOut,
    SetConcernRequest,
    StartCaseRequest,
    StartCaseResponse,
)

DISCLAIMER = (
    "This is an informational testing tool, not professional veterinary "
    "advice. For a sick or dying animal, or any doubt about an emergency, "
    "contact an aquatic veterinarian."
)

# ponytail: in-memory, single-process, per-IP counter -- fine for a small
# public beta on one instance; if real abuse shows up, replace with a
# proper rate limiter (e.g. slowapi + Redis) rather than growing this.
_RATE_LIMIT_MAX_CASES_PER_HOUR = 20
_case_creation_log: dict[str, list[float]] = {}


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    hits = [t for t in _case_creation_log.get(ip, []) if now - t < 3600]
    if len(hits) >= _RATE_LIMIT_MAX_CASES_PER_HOUR:
        raise HTTPException(status_code=429, detail="Too many cases started from this address, try again later.")
    hits.append(now)
    _case_creation_log[ip] = hits


def create_app() -> FastAPI:
    kb = load_knowledge_base()
    db_url = os.environ.get("DATABASE_URL", "sqlite:///cases.db")
    store = SqlCaseStore(db_url)
    engine = AquariumInferenceEngine(kb, store=store)

    app = FastAPI(title="Aquarium Assistant")

    def _next_question_out(explanation: QuestionExplanation) -> NextQuestion:
        q = kb.questions[explanation.question_id]
        evidence = kb.evidence[q.evidence_id]
        options = None
        if evidence.data_type != "numeric":
            answers = kb.answers_by_question.get(q.id)
            if answers:
                options = [AnswerOption(state=a.evidence_state, label=a.label) for a in answers]
            else:
                options = [AnswerOption(state=s, label=s) for s in possible_states(kb, q.evidence_id)]
        return NextQuestion(
            question_id=q.id,
            evidence_id=q.evidence_id,
            text=q.text,
            data_type=evidence.data_type,
            unit=evidence.unit,
            options=options,
        )

    def _status_response(case_id: str) -> CaseStatusResponse:
        status = engine.get_status(case_id)
        recommendations = None
        if status.should_stop and status.ranked_candidates:
            bundle = get_recommendations(kb, status.ranked_candidates[0].problem_id)
            recommendations = RecommendationsOut(
                problem_id=bundle.problem_id,
                problem_name=bundle.problem_name,
                confirmation_checks=bundle.confirmation_checks,
                immediate_actions=bundle.immediate_actions,
                follow_up_actions=bundle.follow_up_actions,
                avoid=bundle.avoid,
                escalation_criteria=bundle.escalation_criteria,
            )
        return CaseStatusResponse(
            case_id=case_id,
            should_stop=status.should_stop,
            stop_reason=status.stop_reason,
            uncertainty=status.uncertainty,
            ranked_candidates=[
                CandidateOut(
                    problem_id=c.problem_id,
                    problem_name=c.problem_name,
                    probability=c.probability,
                    supporting_evidence=c.supporting_evidence,
                    contradicting_evidence=c.contradicting_evidence,
                )
                for c in status.ranked_candidates
            ],
            safety_alerts=[
                SafetyAlertOut(rule_id=a.rule_id, severity=a.severity, message=a.message, escalation_text=a.escalation_text)
                for a in status.safety_alerts
            ],
            next_question=_next_question_out(status.best_next_question) if status.best_next_question else None,
            recommendations=recommendations,
            disclaimer=DISCLAIMER,
        )

    @app.get("/api/entry-contexts")
    def list_entry_contexts() -> list[EntryContextOut]:
        return [
            EntryContextOut(
                id=ec.id,
                name=ec.name,
                description=ec.description,
                concerns=[
                    ConcernOut(id=c.id, name=c.name, description=c.description)
                    for c in kb.concerns_by_entry_context.get(ec.id, [])
                ],
            )
            for ec in kb.entry_contexts.values()
        ]

    @app.post("/api/cases")
    def start_case(body: StartCaseRequest, request: Request) -> StartCaseResponse:
        _check_rate_limit(request.client.host if request.client else "unknown")
        try:
            case_id = engine.start_case(entry_context=body.entry_context)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StartCaseResponse(case_id=case_id)

    @app.get("/api/cases/{case_id}/status")
    def get_case_status(case_id: str) -> CaseStatusResponse:
        try:
            return _status_response(case_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="case not found") from None

    @app.post("/api/cases/{case_id}/answers")
    def answer_case(case_id: str, body: AnswerRequest) -> CaseStatusResponse:
        try:
            engine.answer(
                case_id,
                body.evidence_id,
                state=body.state,
                raw_value=body.raw_value,
                question_id=body.question_id,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="case not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _status_response(case_id)

    @app.post("/api/cases/{case_id}/concern")
    def set_concern(case_id: str, body: SetConcernRequest) -> CaseStatusResponse:
        try:
            engine.set_concern(case_id, body.concern_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="case not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _status_response(case_id)

    @app.post("/api/cases/{case_id}/feedback", status_code=204)
    def submit_feedback(case_id: str, body: FeedbackRequest) -> None:
        store.save_feedback(case_id, body.helpful, body.comment)

    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()
