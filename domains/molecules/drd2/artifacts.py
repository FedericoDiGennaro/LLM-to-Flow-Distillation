"""Immutable artifact helpers for molecular design campaigns."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from domains.peptides.curriculum_artifacts import sha256_file


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def read_jsonl(
    path: Path, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL object required at {path}:{line_number}")
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"non-empty JSONL required: {path}")
    return rows


def immutable_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"immutable artifact differs: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    immutable_write(
        path,
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def immutable_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    immutable_write(
        path,
        "".join(
            json.dumps(dict(row), allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _model_version_path(*, output_root: Path, requested_model: str) -> Path:
    digest = hashlib.sha256(requested_model.encode("utf-8")).hexdigest()
    return output_root / "model_versions" / "by_requested_model" / f"{digest}.json"


def bind_model_version(
    *,
    output_root: Path,
    role_name: str,
    requested_model: str,
    observed_versions: Sequence[str],
) -> str:
    normalized = {
        str(value).strip() for value in observed_versions if str(value).strip()
    }
    if len(normalized) != 1:
        raise ValueError(
            f"{role_name} response batch must expose exactly one modelVersion"
        )
    observed = next(iter(normalized))
    path = _model_version_path(
        output_root=output_root, requested_model=requested_model
    )
    value = {
        "schema_version": 1,
        "requested_model": requested_model,
        "observed_model_version": observed,
    }
    if path.is_file():
        if read_json(path) != value:
            raise PermissionError(
                f"{role_name} resolved provider modelVersion changed across resume"
            )
    else:
        immutable_json(path, value)
    return observed


def sync_agent_model_version_binding(*, output_root: Path, agent: Any) -> None:
    path = _model_version_path(
        output_root=output_root, requested_model=str(agent.model)
    )
    if not path.is_file():
        return
    value = read_json(path)
    if value.get("requested_model") != agent.model:
        raise PermissionError("requested-model binding changed before role call")
    observer = getattr(agent.client, "observe_model_version", None)
    if not callable(observer):
        raise TypeError("Gemini role client cannot enforce modelVersion binding")
    observer(value.get("observed_model_version"))


def model_version_binding_records(output_root: Path) -> list[dict[str, Any]]:
    root = output_root / "model_versions" / "by_requested_model"
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(root.glob("*.json")):
        value = read_json(path)
        requested = str(value.get("requested_model", ""))
        if (
            value.get("schema_version") != 1
            or path
            != _model_version_path(
                output_root=output_root, requested_model=requested
            )
            or not str(value.get("observed_model_version", "")).strip()
        ):
            raise ValueError("durable requested-model binding is malformed")
        records.append(
            {
                **value,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    return records


__all__ = [
    "_model_version_path",
    "bind_model_version",
    "canonical_json",
    "immutable_json",
    "immutable_jsonl",
    "immutable_write",
    "model_version_binding_records",
    "read_json",
    "read_jsonl",
    "sha256_file",
    "sync_agent_model_version_binding",
]
