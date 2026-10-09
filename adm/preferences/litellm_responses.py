"""LiteLLM Responses transport using the existing durable single-attempt ledger.

The gateway's returned model name is an observed identity, not a guarantee of an
immutable upstream revision. No SDK retries, tools, fallback models or browsing
are enabled by this client.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from adm.preferences.clients import StructuredRequest, StructuredResponse
from adm.preferences.gemini_vertex import (
    GeminiVertexClient, _canonical_json, _read_json, _write_immutable_json,
)
from adm.preferences.provider import litellm_key


def _strict_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _strict_schema(item) for key, item in value.items() if key != "default"}
    if result.get("type") == "object":
        if result.get("additionalProperties") not in (None, False):
            raise ValueError("structured gateway output requires fixed object properties")
        result["additionalProperties"] = False
        result["required"] = list(result.get("properties", {}))
    return result


def request_body(request: StructuredRequest, response_model: type[Any], *, model: str) -> dict[str, Any]:
    schema = _strict_schema(response_model.model_json_schema())
    if schema.get("type") != "object":
        raise ValueError("structured gateway output must be an object")
    body = {
        "model": model,
        "instructions": request.system_prompt,
        "input": [{"role": "user", "content": request.user_payload_json}],
        "max_output_tokens": request.max_output_tokens,
        "store": False,
        "text": {"format": {
            "type": "json_schema", "name": "structured_response",
            "strict": True, "schema": schema,
        }},
    }
    if request.reasoning_effort is not None:
        body["reasoning"] = {"effort": request.reasoning_effort}
    elif request.temperature is not None:
        body["temperature"] = request.temperature
    return body


def usage_counts(raw: Mapping[str, Any]) -> tuple[int, int]:
    usage = raw.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("LiteLLM response omitted usage")
    values = [usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("LiteLLM token counts must be nonnegative integers")
    prompt, output, total = values
    if prompt + output != total:
        raise ValueError("LiteLLM total usage differs from input + output")
    return prompt, output


def parse_response(raw: Mapping[str, Any], response_model: type[Any]) -> dict[str, Any]:
    if raw.get("status") != "completed" or raw.get("error") or raw.get("incomplete_details"):
        raise ValueError("LiteLLM response did not complete successfully")
    prompt, output = usage_counts(raw)
    details = raw["usage"].get("output_tokens_details") or {}
    thoughts = details.get("reasoning_tokens", 0)
    if isinstance(thoughts, bool) or not isinstance(thoughts, int) or not 0 <= thoughts <= output:
        raise ValueError("invalid LiteLLM reasoning-token usage")
    items = raw.get("output")
    if not isinstance(items, list):
        raise ValueError("LiteLLM response omitted output")
    texts, summaries = [], []
    for item in items:
        if item.get("type") == "reasoning":
            summaries.extend(part["text"] for part in item.get("summary", [])
                             if part.get("type") == "summary_text")
            continue
        if item.get("type") != "message" or item.get("role") != "assistant":
            raise ValueError("unexpected tool/non-message LiteLLM output")
        if item.get("status", "completed") != "completed":
            raise ValueError("incomplete LiteLLM message")
        for part in item.get("content", []):
            if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                raise ValueError("LiteLLM refusal or non-text output")
            texts.append(part["text"])
    if not texts:
        raise ValueError("LiteLLM response has no answer text")
    final_text = "".join(texts)
    parsed = response_model.model_validate_json(final_text)
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LiteLLM response omitted model identity")
    return {
        "parsed": parsed, "final_text": final_text, "model_version": model,
        "response_id": raw.get("id"),
        "thought_summary": "\n".join(summaries) or None,
        "thought_summary_available": bool(summaries),
        "usage": {"prompt_tokens": prompt, "output_tokens": output,
                  "completion_tokens": output - thoughts, "thought_tokens": thoughts,
                  "total_tokens": prompt + output},
    }


class LiteLLMResponsesClient(GeminiVertexClient):
    def __init__(self, *, settings: Mapping[str, Any], **kwargs: Any) -> None:
        self.settings = copy.deepcopy(dict(settings))
        super().__init__(base_url=self.settings["base_url"], **kwargs)
        self.api_key_provider = kwargs.get("api_key_provider") or (
            lambda: litellm_key(expected_base_url=self.base_url)
        )

    def _post(self, endpoint: str, **kwargs: Any) -> Any:
        # Do not forward Google credentials or follow redirects to another host.
        key = kwargs["headers"]["X-Goog-Api-Key"]
        kwargs["headers"] = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        return self.session.post(endpoint, allow_redirects=False, **kwargs)

    _usage_counts = staticmethod(usage_counts)

    def _completed_usage(self) -> tuple[int, int]:
        prompt, output = 0, 0
        if self.attempt_root is None:
            return prompt, output
        for path in self.attempt_root.glob("*/attempt_started.json"):
            response_path = path.parent / "provider_response.json"
            if response_path.exists():
                envelope = _read_json(response_path)
                raw = envelope.get("provider_response")
                try:
                    used_prompt, used_output = usage_counts(raw if isinstance(raw, Mapping) else {})
                    prompt += used_prompt
                    output += used_output
                    continue
                except ValueError:
                    pass
            # Unknown completion/billing and error responses consume their
            # reservation. Unlike direct Vertex, gateway errors can be billable.
            reservation = _read_json(path).get("budget_projection")
            if not isinstance(reservation, Mapping):
                if self.usage_budget is not None:
                    raise PermissionError("missing gateway reservation in attempt ledger")
                continue
            prompt += reservation["next_prompt_token_upper_bound"]
            output += reservation["next_maximum_output_tokens"]
        return prompt, output

    def observe_model_version_metadata(self, raw: Mapping[str, Any]) -> str:
        if raw.get("provider_binding") != self.settings:
            raise PermissionError("cached response provider binding differs")
        version = self.observe_model_version(raw.get("model_version"))
        if (raw.get("provider_response") or {}).get("model") != version:
            raise ValueError("LiteLLM wrapper/provider modelVersion mismatch")
        return version

    def parse(self, request: StructuredRequest, response_model: type[Any]) -> StructuredResponse[Any]:
        model = self.settings["model_map"].get(request.model, request.model)
        body = request_body(request, response_model, model=model)
        raw, replayed = self._dispatch_or_replay(
            request=request, endpoint=f"{self.base_url}/responses", body=body,
            key=self.api_key_provider(),
        )
        folder = self.attempt_root / request.request_hash if self.attempt_root else None
        digest = hashlib.sha256(_canonical_json(raw).encode()).hexdigest()
        try:
            normalized = parse_response(raw, response_model)
            self.observe_model_version(normalized["model_version"])
        except Exception as error:
            if folder is not None:
                _write_immutable_json(folder / "parse_failure.json", {
                    "request_hash": request.request_hash, "provider_response_sha256": digest,
                    "exception_type": type(error).__name__,
                })
            raise
        if folder is not None:
            if (folder / "parse_failure.json").exists():
                raise RuntimeError("LiteLLM response has a durable parse-failure marker")
            _write_immutable_json(folder / "parsed_response.json", {
                "schema_version": 1, "request_hash": request.request_hash,
                "provider_response_sha256": digest, "model_version": normalized["model_version"],
                "response_id": normalized["response_id"], "usage": normalized["usage"],
                "model_identity_kind": "gateway_reported_name_not_pinned_revision",
            })
        return StructuredResponse(
            request_hash=request.request_hash, parsed=normalized["parsed"],
            response_id=normalized["response_id"], usage=normalized["usage"],
            raw_response={
                **{key: value for key, value in normalized.items() if key not in ("parsed", "usage")},
                "provider_request_body": body, "provider_response": copy.deepcopy(raw),
                "provider_binding": self.settings,
                "model_identity_kind": "gateway_reported_name_not_pinned_revision",
                "automatic_retries": 0, "provider_response_replayed": replayed,
            },
        )
