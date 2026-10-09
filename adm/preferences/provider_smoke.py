"""Synthetic structured-output smoke: no candidates, generation or training.

Dry-run is the default. --execute explicitly enables at most three provider calls
in one immutable output directory, bounded using the supplied price estimates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from adm.preferences.clients import StructuredRequest
from adm.preferences.gemini_vertex import GeminiUsageBudget, _write_immutable_json
from adm.preferences.provider import make_structured_client, provider_key, provider_settings


class SmokeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["positive", "negative", "abstain"]
    sentinel: Literal["transport-smoke"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calls", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--maximum-usd", type=float, required=True)
    parser.add_argument("--input-usd-per-million", type=float, required=True)
    parser.add_argument("--output-usd-per-million", type=float, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    budget = GeminiUsageBudget(
        maximum_prompt_tokens=20000, maximum_output_tokens=args.calls * args.max_output_tokens,
        maximum_estimated_usd=args.maximum_usd,
        input_usd_per_million_tokens=args.input_usd_per_million,
        output_usd_per_million_tokens=args.output_usd_per_million,
    )
    plan = {"kind": "synthetic_transport_smoke", "provider": provider_settings(),
            "model": args.model, "maximum_calls": args.calls, "budget": vars(budget),
            "max_output_tokens": args.max_output_tokens,
            "candidates_sent": False, "training": False, "automatic_retries": 0}
    if not args.execute:
        print(json.dumps({"status": "dry_run_no_calls", **plan}, indent=2))
        return
    provider_key()
    root = args.output_dir.resolve()
    _write_immutable_json(root / "smoke_plan.json", plan)
    client = make_structured_client(attempt_root=root / "attempts", usage_budget=budget,
                                    timeout=120, requests_per_minute=6)
    rows = []
    for index in range(args.calls):
        label = ("positive", "negative", "abstain")[index]
        request = StructuredRequest.from_payload(
            model=args.model, schema_id="transport_smoke_v1",
            system_prompt="This is a transport test, not a scientific judgment. Copy the supplied label and sentinel exactly into the required JSON.",
            user_payload={"label": label, "sentinel": "transport-smoke"},
            max_output_tokens=args.max_output_tokens, reasoning_effort="low",
            logical_context={"smoke_index": index},
        )
        response = client.parse(request, SmokeAnswer)
        if response.parsed.label != label:
            raise ValueError("smoke label did not match the synthetic input")
        rows.append({"request_hash": request.request_hash, "label": label,
                     "usage": response.usage, "model_version": response.raw_response["model_version"]})
    report = {"status": "passed", "calls": len(rows), "responses": rows,
              "claim": "transport/schema only; not a GPU or full-campaign test"}
    _write_immutable_json(root / "smoke_result.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
