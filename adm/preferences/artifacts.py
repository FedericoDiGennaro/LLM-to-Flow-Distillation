"""Immutable, content-addressed artifacts for preference API calls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

from adm.preferences.clients import StructuredRequest, StructuredResponse, canonical_json, value_as_json


ARTIFACT_SCHEMA_VERSION = 1


def _validate_request_hash(request_hash: str) -> None:
    if len(request_hash) != 64:
        raise ValueError("request_hash must be a SHA-256 hexadecimal digest")
    try:
        int(request_hash, 16)
    except ValueError as error:
        raise ValueError("request_hash must be hexadecimal") from error


def _read_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def _write_immutable_json(path: Path, value: Any) -> None:
    """Create one JSON artifact atomically, allowing identical replays only."""
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != serialized:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_text() != serialized:
                raise FileExistsError(f"immutable artifact differs: {path}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class StoredResponse:
    """JSON-normalized response loaded without an SDK or schema class."""

    request_hash: str
    response_id: str | None
    parsed: Any
    usage: Any
    raw_response: Any

    def __post_init__(self) -> None:
        _validate_request_hash(self.request_hash)


class PreferenceArtifactStore(Protocol):
    """Persistence boundary used by elicitation orchestration."""

    def save_request(self, request: StructuredRequest) -> Path:
        """Persist a logical request idempotently and return its path."""

    def save_response(
        self,
        request: StructuredRequest,
        response: StructuredResponse[Any],
    ) -> Path:
        """Persist one response, rejecting conflicts for the same request."""

    def load_response(self, request_hash: str) -> StoredResponse | None:
        """Load a cached response, or return ``None`` for a cache miss."""


class FilePreferenceArtifactStore:
    """Store each request and response under its request SHA-256 digest."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def request_directory(self, request_hash: str) -> Path:
        _validate_request_hash(request_hash)
        return self.root / request_hash

    def save_request(self, request: StructuredRequest) -> Path:
        path = self.request_directory(request.request_hash) / "request.json"
        envelope = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "request_hash": request.request_hash,
            "request": request.as_dict(),
        }
        _write_immutable_json(path, envelope)
        return path

    def save_response(
        self,
        request: StructuredRequest,
        response: StructuredResponse[Any],
    ) -> Path:
        if response.request_hash != request.request_hash:
            raise ValueError("response request hash does not match the request")
        self.save_request(request)
        path = self.request_directory(request.request_hash) / "response.json"
        response_content = {
            "response_id": response.response_id,
            "parsed": value_as_json(response.parsed),
            "usage": value_as_json(response.usage),
            "raw_response": value_as_json(response.raw_response),
        }
        response_content_hash = hashlib.sha256(
            canonical_json(response_content).encode()
        ).hexdigest()
        envelope = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "request_hash": request.request_hash,
            "response_content_hash": response_content_hash,
            **response_content,
        }
        _write_immutable_json(path, envelope)
        return path

    def load_response(self, request_hash: str) -> StoredResponse | None:
        response_path = self.request_directory(request_hash) / "response.json"
        if not response_path.is_file():
            return None
        envelope = _read_json(response_path)
        if envelope.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema in {response_path}")
        if envelope.get("request_hash") != request_hash:
            raise ValueError(f"response request hash mismatch in {response_path}")
        response_content = {
            key: envelope.get(key)
            for key in ("response_id", "parsed", "usage", "raw_response")
        }
        observed_response_hash = hashlib.sha256(
            canonical_json(response_content).encode()
        ).hexdigest()
        if envelope.get("response_content_hash") != observed_response_hash:
            raise ValueError(f"response content hash mismatch in {response_path}")

        request_path = self.request_directory(request_hash) / "request.json"
        if not request_path.is_file():
            raise ValueError(
                f"response has no matching request artifact: {response_path}"
            )
        request_envelope = _read_json(request_path)
        if request_envelope.get("request_hash") != request_hash:
            raise ValueError(f"request hash mismatch in {request_path}")
        observed_hash = hashlib.sha256(
            canonical_json(request_envelope["request"]).encode()
        ).hexdigest()
        if observed_hash != request_hash:
            raise ValueError(f"request content hash mismatch in {request_path}")

        return StoredResponse(
            request_hash=request_hash,
            response_id=response_content["response_id"],
            parsed=response_content["parsed"],
            usage=response_content["usage"],
            raw_response=response_content["raw_response"],
        )
