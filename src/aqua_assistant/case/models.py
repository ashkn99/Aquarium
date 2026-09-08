"""Runtime case/observation state — plain dataclasses, no persistence.

Kept deliberately independent of the SQLAlchemy `cases`/`observations`
tables in kb/models.py: those define the schema for when this needs to be
persisted (a future API), but the MVP engine only needs an in-memory
representation to run and be tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Observation:
    evidence_id: str
    observed_state: str
    confidence: float = 1.0
    raw_value: float | None = None
    source: str = "manual"
    question_id: str | None = None
    answered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    superseded: bool = False


@dataclass
class Case:
    id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "open"
    observations: list[Observation] = field(default_factory=list)

    def add_observation(self, obs: Observation) -> None:
        """Superseding rather than deleting keeps a correction's audit
        trail: if the user re-answers the same evidence, the old
        observation stays in the log but stops counting toward scoring."""
        for existing in self.observations:
            if existing.evidence_id == obs.evidence_id and not existing.superseded:
                existing.superseded = True
        self.observations.append(obs)

    def active_observations(self) -> list[Observation]:
        return [o for o in self.observations if not o.superseded]

    def answered_evidence_ids(self) -> set[str]:
        return {o.evidence_id for o in self.active_observations()}
