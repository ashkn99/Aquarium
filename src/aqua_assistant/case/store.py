"""In-memory case store.

The engine only depends on this narrow interface (create/get/save), so
swapping in SQLite-backed persistence later (using the `cases` /
`observations` tables in kb/models.py) won't require changing the engine.
"""
from __future__ import annotations

from .models import Case


class CaseStore:
    def __init__(self) -> None:
        self._cases: dict[str, Case] = {}

    def create(self) -> Case:
        case = Case()
        self._cases[case.id] = case
        return case

    def get(self, case_id: str) -> Case:
        return self._cases[case_id]

    def save(self, case: Case) -> None:
        self._cases[case.id] = case
