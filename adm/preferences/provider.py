"""Explicit provider selection; no network or credential reads at import time."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
from typing import Any, Mapping
from urllib.parse import urlsplit

from adm.preferences.gemini_vertex import (
    DEFAULT_BASE_URL, GeminiVertexClient, _read_json, _write_immutable_json,
    validate_api_key,
)

CREDENTIAL_VARIABLES = (
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "LITELLM_API_KEY",
    "API_KEY_FILE",
)


def without_credentials(environment: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in environment.items() if key not in CREDENTIAL_VARIABLES}


def provider_settings() -> dict[str, Any]:
    provider = os.environ.get("ADM_LLM_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        return {"provider": provider, "base_url": DEFAULT_BASE_URL, "model_map": {}}
    if provider != "litellm":
        raise ValueError("ADM_LLM_PROVIDER must be gemini or litellm")
    base = os.environ.get("LITELLM_API_BASE", "").strip().rstrip("/")
    parsed = urlsplit(base)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("LITELLM_API_BASE must be an HTTPS base URL without credentials/query/fragment")
    models = json.loads(os.environ.get("LITELLM_MODEL_MAP", "{}"))
    if not isinstance(models, dict) or any(
        not isinstance(key, str) or not key.strip() or not isinstance(value, str)
        or not value.strip() or value != value.strip() for key, value in models.items()
    ):
        raise ValueError("LITELLM_MODEL_MAP must map configured model names to gateway aliases")
    return {"provider": provider, "base_url": base, "model_map": models}


def provider_key() -> str:
    if provider_settings()["provider"] == "gemini":
        return GeminiVertexClient._environment_key()
    return litellm_key(expected_base_url=provider_settings()["base_url"])


def litellm_key(*, expected_base_url: str | None = None) -> str:
    """Read only the explicitly selected gateway credential, regardless of routing changes."""
    value = os.environ.get("LITELLM_API_KEY")
    filename = os.environ.get("API_KEY_FILE")
    if bool(value) == bool(filename):
        raise EnvironmentError("select exactly one of LITELLM_API_KEY or API_KEY_FILE")
    if filename:
        # Never source/execute a shell file or fall back to another key name.
        lines = [line.strip() for line in Path(filename).read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        assignment = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
        if len(lines) == 1 and not assignment.fullmatch(lines[0]):
            value = lines[0]
        else:
            selected = {}
            for line in lines:
                match = assignment.fullmatch(line)
                if match is None:
                    raise EnvironmentError("credential file must contain literal environment assignments")
                name, contents = match.groups()
                if name not in {"LITELLM_API_KEY", "LITELLM_API_BASE"}:
                    continue
                if name in selected:
                    raise EnvironmentError("duplicate gateway credential/configuration assignment")
                tokens = shlex.split(contents, comments=True)
                if len(tokens) != 1:
                    raise EnvironmentError("invalid gateway credential/configuration assignment")
                selected[name] = tokens[0]
            value = selected.get("LITELLM_API_KEY")
            if "LITELLM_API_BASE" in selected:
                expected = expected_base_url or os.environ.get("LITELLM_API_BASE", "")
                if selected["LITELLM_API_BASE"].rstrip("/") != expected.rstrip("/"):
                    raise EnvironmentError("credential-file gateway URL differs from the selected route")
    try:
        return validate_api_key(value)
    except EnvironmentError:
        raise EnvironmentError("selected provider credential must be nonempty, trimmed printable ASCII") from None


def bind_provider(root: Path, *, legacy_glob: str = "*/attempt_started.json") -> dict[str, Any]:
    """Prevent changing the destination/aliases while reusing existing artifacts."""
    settings = provider_settings()
    path = root / "provider_binding.json"
    if path.exists():
        if _read_json(path) != settings:
            raise PermissionError("provider binding differs; use a separate output directory")
    elif settings["provider"] != "gemini" and any(root.glob(legacy_glob)):
        raise PermissionError("existing unbound provider artifacts cannot be reused with LiteLLM")
    _write_immutable_json(path, settings)
    return settings


def make_structured_client(*, native_client_class=GeminiVertexClient, litellm_client_class=None, **kwargs: Any):
    settings = provider_settings()
    root = kwargs.get("attempt_root")
    if root is not None:
        bind_provider(Path(root))
    if settings["provider"] == "gemini":
        return native_client_class(**kwargs)
    from adm.preferences.litellm_responses import LiteLLMResponsesClient

    # countTokens is a Vertex-only optional fallback, never a hidden gateway call.
    kwargs.pop("count_tokens_on_budget_rejection", None)
    return (litellm_client_class or LiteLLMResponsesClient)(settings=settings, **kwargs)


def observed_provider_model(raw: Mapping[str, Any]) -> Any:
    return raw.get("model") if provider_settings()["provider"] == "litellm" else raw.get("modelVersion")
