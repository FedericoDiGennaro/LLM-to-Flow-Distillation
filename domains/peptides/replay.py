"""Pure helpers for cumulative replay within one contiguous semantic goal."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def contiguous_same_goal_prior_indices(
    *,
    current_index: int,
    current_goal_id: str,
    current_goal: Mapping[str, Any],
    load_buffer: Callable[[int], Mapping[str, Any]],
) -> tuple[int, ...]:
    """Return the prior suffix belonging to the current semantic-goal segment."""

    if current_index < 0:
        raise ValueError("current_index must be non-negative")
    if not current_goal_id or not isinstance(current_goal, Mapping):
        raise ValueError("current replay segment requires a bound semantic goal")
    selected: list[int] = []
    for index in reversed(range(current_index)):
        buffer = load_buffer(index)
        if (
            buffer.get("training_goal_id") != current_goal_id
            or buffer.get("training_goal") != current_goal
        ):
            break
        selected.append(index)
    return tuple(reversed(selected))


__all__ = ["contiguous_same_goal_prior_indices"]
