#!/usr/bin/env python3
"""Fail-fast registered Gemini model-access probes before GPU work."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

import requests

from adm.campaign import resolve_campaign_registration
from adm.artifacts import bind_model_version
from adm.preferences.clients import StructuredRequest
from adm.preferences.gemini_vertex import GeminiUsageBudget
from adm.preferences.provider import bind_provider, make_structured_client, provider_key, provider_settings


ROLES = ("candidate_judge", "curriculum_controller", "independent_auditor")
ALLOCATION_RE = re.compile(r"[0-9]+")
BASE_URL = "https://aiplatform.googleapis.com/v1/publishers/google/models"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("config must be a JSON object")
    return value


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Gemini preflight receipt differs: {path}")
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
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _git_state(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _validate_static_authorization(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    pool_config_path: Path,
    output_root: Path,
    authorization_path: Path,
    campaign_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    authorization = _read(authorization_path)
    campaign = _read(campaign_path)
    git_state = _git_state(repository_root)
    experiment_id = str(config.get("experiment_id", ""))
    if not experiment_id:
        raise ValueError("preflight config omitted experiment_id")
    if authorization.get("status") != "approved" or campaign.get("status") != "active":
        raise PermissionError(f"{experiment_id} is not approved and active")
    if authorization.get("experiment_id") != experiment_id:
        raise PermissionError("preflight authorization/campaign identity differs")
    registration = resolve_campaign_registration(
        campaign, experiment_id=experiment_id
    )
    template_name = str(config_path.relative_to(repository_root))
    pool_name = str(pool_config_path.relative_to(repository_root))
    expected = {
        "config_template_file": template_name,
        "config_template_file_sha256": _sha256(config_path),
        "pool_config_file": pool_name,
        "pool_config_file_sha256": _sha256(pool_config_path),
        "output_root": str(output_root.resolve()),
        "code_commit": git_state["commit"],
        "candidate_judge": {
            key: config["candidate_judge"][key]
            for key in ("model", "reasoning_effort", "maximum_provider_calls")
        },
        "curriculum_controller": {
            key: config["curriculum_controller"][key]
            for key in ("model", "reasoning_effort", "maximum_provider_calls")
        },
        "independent_auditor": {
            key: config["independent_auditor"][key]
            for key in ("model", "reasoning_effort", "maximum_provider_calls")
        },
        "aggregate_api_budget": dict(config["aggregate_api_budget"]),
        "maximum_preflight_allocations": int(
            config["aggregate_api_budget"]["maximum_preflight_allocations"]
        ),
    }
    if any(authorization.get(name) != value for name, value in expected.items()):
        raise PermissionError("static Gemini preflight authorization differs")
    if git_state["dirty"]:
        raise PermissionError("Gemini preflight requires the authorized clean commit")
    deadline = dt.datetime.fromisoformat(
        str(authorization["deadline_utc"]).replace("Z", "+00:00")
    )
    if dt.datetime.now(dt.timezone.utc) > deadline:
        raise PermissionError(f"{experiment_id} Gemini authorization expired")
    resources = registration.resources
    expected_resources = {
        "config_template": template_name,
        "config_template_sha256": _sha256(config_path),
        "pool_config": pool_name,
        "pool_config_sha256": _sha256(pool_config_path),
    }
    if any(resources.get(name) != value for name, value in expected_resources.items()):
        raise PermissionError("active campaign preflight resources differ")
    budget = campaign.get("budget")
    if (
        not isinstance(budget, Mapping)
        or int(budget.get("max_jobs", 0))
        < int(config["aggregate_api_budget"]["maximum_preflight_allocations"])
        or int(budget.get("max_paid_api_calls", 0))
        < int(config["aggregate_api_budget"]["maximum_paid_calls"])
        or int(budget.get("maximum_prompt_tokens", 0))
        < int(config["aggregate_api_budget"]["maximum_prompt_tokens"])
        or int(budget.get("maximum_output_tokens", 0))
        < int(config["aggregate_api_budget"]["maximum_output_tokens"])
        or float(budget.get("maximum_estimated_api_usd", 0))
        < float(config["aggregate_api_budget"]["maximum_estimated_usd"])
    ):
        raise PermissionError("active campaign budget is smaller than authorization")


def _generation_probe(
    *,
    key: str,
    model: str,
    role_name: str,
    allocation_id: str,
    output_root: Path,
    timeout: float,
    experiment_id: str = "EXP-0083",
    usage_budget: GeminiUsageBudget | None = None,
) -> dict[str, Any]:
    if provider_settings()["provider"] == "litellm":
        return _litellm_generation_probe(
            key=key, model=model, role_name=role_name, allocation_id=allocation_id,
            output_root=output_root, timeout=timeout, experiment_id=experiment_id,
            usage_budget=usage_budget,
        )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Reply briefly with OK."}],
            }
        ],
        "generationConfig": {
            "candidateCount": 1,
            "maxOutputTokens": 128,
            "thinkingConfig": {"thinkingLevel": "low", "includeThoughts": False},
        },
    }
    endpoint = f"{BASE_URL}/{model}:generateContent"
    request_hash = hashlib.sha256(
        _canonical(
            {
                "experiment_id": experiment_id,
                "kind": "allocation_generation_entitlement_probe",
                "allocation_id": allocation_id,
                "role_name": role_name,
                "model": model,
                "body": body,
            }
        ).encode("utf-8")
    ).hexdigest()
    attempt_root = (
        output_root
        / "preflight"
        / "allocations"
        / allocation_id
        / "attempts"
        / hashlib.sha256(model.encode("utf-8")).hexdigest()
    )
    start_path = attempt_root / "attempt_started.json"
    response_path = attempt_root / "provider_response.json"
    body_sha256 = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
    if response_path.is_file():
        envelope = _read(response_path)
        if (
            envelope.get("request_hash") != request_hash
            or envelope.get("provider_request_body_sha256") != body_sha256
        ):
            raise ValueError("stored generation-entitlement probe differs")
    else:
        if start_path.exists():
            raise RuntimeError("generation-entitlement probe completion is ambiguous")
        _immutable_json(
            start_path,
            {
                "schema_version": 1,
                "request_hash": request_hash,
                "endpoint": endpoint,
                "provider_request_body_sha256": body_sha256,
                "provider_request_body": body,
                "budget_projection": {
                    "next_maximum_output_tokens": 128,
                    "kind": "registered_preflight_generation_probe",
                },
                "automatic_retries": 0,
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        response = requests.Session().post(
            endpoint,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-Goog-Api-Key": key,
            },
            json=body,
            timeout=timeout,
        )
        status = int(response.status_code)
        try:
            provider_response = response.json()
        except (ValueError, json.JSONDecodeError):
            provider_response = None
        envelope = {
            "schema_version": 1,
            "request_hash": request_hash,
            "provider_request_body_sha256": body_sha256,
            "http_status": status,
            "provider_response": provider_response,
            "response_text": (
                None if provider_response is not None else str(response.text)
            ),
            "received_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _immutable_json(response_path, envelope)
        response.raise_for_status()
    status = envelope.get("http_status")
    raw = envelope.get("provider_response")
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or not 200 <= status < 300
        or not isinstance(raw, Mapping)
    ):
        raise ValueError("generation-entitlement probe did not return HTTP 2xx JSON")
    version = str(raw.get("modelVersion", "")).strip()
    usage = raw.get("usageMetadata")
    if not version or not isinstance(usage, Mapping):
        raise ValueError("generation-entitlement probe omitted version or usage")
    prompt_tokens = usage.get("promptTokenCount")
    total_tokens = usage.get("totalTokenCount")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 0
        or isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens < prompt_tokens
    ):
        raise ValueError("generation-entitlement probe usage is malformed")
    response_sha256 = hashlib.sha256(_canonical(raw).encode("utf-8")).hexdigest()
    _immutable_json(
        attempt_root / "parsed_response.json",
        {
            "schema_version": 1,
            "request_hash": request_hash,
            "provider_response_sha256": response_sha256,
            "model_version": version,
            "requested_model": model,
            "response_id": str(raw.get("responseId", "preflight")),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "output_tokens": total_tokens - prompt_tokens,
                "total_tokens": total_tokens,
            },
            "validation_kind": "generation_entitlement_metadata_only",
        },
    )
    bind_model_version(
        output_root=output_root,
        role_name=role_name,
        requested_model=model,
        observed_versions=(version,),
    )
    return {
        "schema_version": 1,
        "method": "publishers.models.generateContent",
        "endpoint": endpoint,
        "model": model,
        "http_status": status,
        "generation_requested": True,
        "allocation_id": allocation_id,
        "request_hash": request_hash,
        "model_version": version,
        "prompt_tokens": prompt_tokens,
        "output_tokens": total_tokens - prompt_tokens,
        "credential_value_recorded": False,
    }


def _litellm_generation_probe(
    *, key: str, model: str, role_name: str, allocation_id: str, output_root: Path,
    timeout: float, experiment_id: str, usage_budget: GeminiUsageBudget | None,
) -> dict[str, Any]:
    if usage_budget is None:
        raise PermissionError("LiteLLM preflight requires the registered token/cost budget")
    bind_provider(output_root, legacy_glob="**/attempt_started.json")
    settings = provider_settings()
    request = StructuredRequest(
        model=model, schema_id="generation_entitlement_metadata_only",
        system_prompt="Reply briefly with OK.", user_payload_json="{}",
        max_output_tokens=128, reasoning_effort="low",
        logical_context_json=_canonical({"experiment_id": experiment_id, "allocation_id": allocation_id}),
    )
    client = make_structured_client(
        timeout=timeout, api_key_provider=lambda: key,
        attempt_root=output_root / "preflight" / "gateway_attempts",
        usage_budget=usage_budget,
    )
    body = {"model": settings["model_map"].get(model, model),
            "input": "Reply briefly with OK.", "max_output_tokens": 128,
            "reasoning": {"effort": "low"}, "store": False}
    endpoint = f'{settings["base_url"]}/responses'
    raw, _ = client._dispatch_or_replay(request=request, endpoint=endpoint, body=body, key=key)
    prompt, output = client._usage_counts(raw)
    if raw.get("error") or raw.get("status") not in ("completed", "incomplete"):
        raise ValueError("LiteLLM generation-entitlement probe failed")
    version = client.observe_model_version(raw.get("model"))
    bind_model_version(output_root=output_root, role_name=role_name,
                       requested_model=model, observed_versions=(version,))
    return {
        "schema_version": 1, "method": "responses.create", "endpoint": endpoint,
        "model": model, "http_status": 200, "generation_requested": True,
        "allocation_id": allocation_id, "request_hash": request.request_hash,
        "model_version": version, "prompt_tokens": prompt, "output_tokens": output,
        "model_identity_kind": "gateway_reported_name_not_pinned_revision",
        "credential_value_recorded": False,
    }


def execute(
    config_path: Path,
    output_root: Path,
    *,
    pool_config_path: Path,
    authorization_path: Path,
    campaign_path: Path,
    allocation_id: str | None = None,
) -> dict[str, Any]:
    config = _read(config_path)
    experiment_id = str(config.get("experiment_id", ""))
    if not experiment_id:
        raise ValueError("Gemini fail-fast preflight requires experiment_id")
    key = provider_key()
    _validate_static_authorization(
        config=config,
        config_path=config_path,
        pool_config_path=pool_config_path,
        output_root=output_root,
        authorization_path=authorization_path,
        campaign_path=campaign_path,
    )
    allocation = str(allocation_id or os.environ.get("SLURM_JOB_ID", ""))
    if ALLOCATION_RE.fullmatch(allocation) is None:
        raise EnvironmentError("a numeric Slurm allocation ID is required")
    allocation_root = output_root / "preflight" / "allocations" / allocation
    existing_allocations = (
        [path for path in allocation_root.parent.iterdir() if path.is_dir()]
        if allocation_root.parent.exists()
        else []
    )
    maximum_allocations = int(
        config["aggregate_api_budget"]["maximum_preflight_allocations"]
    )
    if not allocation_root.exists() and len(existing_allocations) >= maximum_allocations:
        raise PermissionError("registered preflight-allocation ceiling exhausted")
    models_by_role = {role: str(config[role]["model"]) for role in ROLES}
    roles_by_model: dict[str, list[str]] = {}
    for role, model in models_by_role.items():
        roles_by_model.setdefault(model, []).append(role)
    results_by_model: dict[str, dict[str, Any]] = {}
    preflight_budget = config["aggregate_api_budget"]
    usage_budget = None
    if provider_settings()["provider"] == "litellm":
        usage_budget = GeminiUsageBudget(
            maximum_prompt_tokens=int(preflight_budget["maximum_preflight_prompt_tokens"]),
            maximum_output_tokens=int(preflight_budget["maximum_preflight_output_tokens"]),
            maximum_estimated_usd=float(preflight_budget["maximum_preflight_estimated_usd"]),
            input_usd_per_million_tokens=float(preflight_budget["preflight_input_usd_per_million_tokens"]),
            output_usd_per_million_tokens=float(preflight_budget["preflight_output_usd_per_million_tokens"]),
        )
    for model, model_roles in sorted(roles_by_model.items()):
        probe_role = model_roles[-1]
        result = _generation_probe(
            key=key,
            model=model,
            role_name=probe_role,
            allocation_id=allocation,
            output_root=output_root,
            timeout=300,
            experiment_id=experiment_id,
            usage_budget=usage_budget,
        )
        results_by_model[model] = result
    role_receipts: dict[str, dict[str, Any]] = {}
    for role, model in models_by_role.items():
        result = results_by_model[model]
        role_path = output_root / "preflight" / f"{role}.json"
        _immutable_json(
            role_path,
            {
                "schema_version": 1,
                "method": result["method"],
                "model": model,
                "http_status": 200,
                "generation_requested": True,
                "model_version": result["model_version"],
                "credential_value_recorded": False,
            },
        )
        role_receipts[role] = {
            "model": model,
            "path": str(role_path.resolve()),
            "sha256": hashlib.sha256(role_path.read_bytes()).hexdigest(),
        }
    aggregate = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "unique_model_count_per_allocation": len(results_by_model),
        "maximum_preflight_allocations": maximum_allocations,
        "role_receipts": role_receipts,
    }
    _immutable_json(output_root / "preflight" / "fail_fast_summary.json", aggregate)
    allocation_summary = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "allocation_id": allocation,
        "generation_entitlement": results_by_model,
    }
    _immutable_json(allocation_root / "summary.json", allocation_summary)
    return {**aggregate, "allocation": allocation_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pool-config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--active-campaign", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            execute(
                args.config.resolve(),
                args.output_root.resolve(),
                pool_config_path=args.pool_config.resolve(),
                authorization_path=args.authorization.resolve(),
                campaign_path=args.active_campaign.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
