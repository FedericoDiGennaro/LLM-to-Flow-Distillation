"""Cooperative wall-time cutoffs for crash-resumable allocations."""

from __future__ import annotations

import math
import os
import time


ALLOCATION_DEADLINE_ENV = "ADM_ALLOCATION_DEADLINE_EPOCH"


class AllocationYieldRequired(RuntimeError):
    """The next bounded operation cannot safely finish in this allocation."""

    def __init__(
        self,
        *,
        deadline_epoch: float,
        required_seconds: float,
        remaining_seconds: float,
        operation: str,
        request_hash: str | None = None,
    ) -> None:
        self.deadline_epoch = deadline_epoch
        self.required_seconds = required_seconds
        self.remaining_seconds = remaining_seconds
        self.operation = operation
        self.request_hash = request_hash
        super().__init__(
            "allocation window exhausted before bounded operation: "
            f"operation={operation}, remaining={remaining_seconds:.3f}s, "
            f"required={required_seconds:.3f}s"
        )


def require_allocation_window(
    *,
    required_seconds: float,
    operation: str,
    request_hash: str | None = None,
    now_epoch: float | None = None,
) -> None:
    """Raise before starting work that could cross the registered cutoff."""

    if (
        not isinstance(required_seconds, (int, float))
        or not math.isfinite(required_seconds)
        or required_seconds <= 0
    ):
        raise ValueError("required_seconds must be positive and finite")
    raw_deadline = os.environ.get(ALLOCATION_DEADLINE_ENV)
    if not raw_deadline:
        return
    try:
        deadline_epoch = float(raw_deadline)
    except ValueError as error:
        raise EnvironmentError(
            f"{ALLOCATION_DEADLINE_ENV} must be a Unix epoch"
        ) from error
    if not math.isfinite(deadline_epoch) or deadline_epoch <= 0:
        raise EnvironmentError(
            f"{ALLOCATION_DEADLINE_ENV} must be a positive finite epoch"
        )
    current = time.time() if now_epoch is None else float(now_epoch)
    remaining = deadline_epoch - current
    if remaining < float(required_seconds):
        raise AllocationYieldRequired(
            deadline_epoch=deadline_epoch,
            required_seconds=float(required_seconds),
            remaining_seconds=remaining,
            operation=operation,
            request_hash=request_hash,
        )
