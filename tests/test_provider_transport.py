"""Offline provider regression tests. No real credentials or network required."""
import copy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from pydantic import BaseModel, ConfigDict
import requests

from adm.preferences.clients import StructuredRequest
from adm.preferences.gemini_vertex import GeminiUsageBudget, GeminiVertexClient
from adm.preferences.litellm_responses import LiteLLMResponsesClient, parse_response, request_body
from adm.preferences.provider import (
    bind_provider, make_structured_client, provider_key, provider_settings, without_credentials,
)

ENV = {"ADM_LLM_PROVIDER": "litellm", "LITELLM_API_BASE": "https://gateway.example/v1",
       "LITELLM_API_KEY": "test-key", "LITELLM_MODEL_MAP": '{"configured":"gemini/test"}'}
ROOT = Path(__file__).resolve().parents[1]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def raw_response(value=1):
    return {"id": "resp-test", "model": "resolved-model", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps({"value": value})}]}],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
                      "output_tokens_details": {"reasoning_tokens": 4}}}


def request():
    return StructuredRequest(model="configured", schema_id="test", system_prompt="Return JSON",
                             user_payload_json='{"x":1}', max_output_tokens=128, reasoning_effort="low")


def budget(**changes):
    values = dict(maximum_prompt_tokens=100000, maximum_output_tokens=10000,
                  maximum_estimated_usd=1, input_usd_per_million_tokens=2,
                  output_usd_per_million_tokens=12)
    return GeminiUsageBudget(**{**values, **changes})


def session(raw=None, status=200):
    response = Mock(status_code=status)
    response.json.return_value = raw if raw is not None else raw_response()
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    result = Mock()
    result.post.return_value = response
    return result


class ProviderTests(unittest.TestCase):
    def test_default_preserves_native_client(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(make_structured_client(), GeminiVertexClient)
            self.assertNotIsInstance(make_structured_client(), LiteLLMResponsesClient)

    def test_explicit_key_and_no_google_fallback(self):
        with patch.dict(os.environ, {**ENV, "LITELLM_API_KEY": "", "GEMINI_API_KEY": "wrong"}, clear=True):
            with self.assertRaises(EnvironmentError):
                provider_key()

    def test_explicit_file_is_not_executed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "key.env"
            path.write_text('export LITELLM_API_KEY="test-key"\n')
            with patch.dict(os.environ, {**ENV, "LITELLM_API_KEY": "", "API_KEY_FILE": str(path)}, clear=True):
                self.assertEqual(provider_key(), "test-key")
            with patch.dict(os.environ, {**ENV, "API_KEY_FILE": str(path)}, clear=True):
                with self.assertRaises(EnvironmentError):
                    provider_key()

    def test_destination_validation(self):
        for base in ("http://gateway.example", "https://user:secret@example.com", "https://example.com?k=x", ""):
            with self.subTest(base=base), patch.dict(os.environ, {**ENV, "LITELLM_API_BASE": base}, clear=True):
                with self.assertRaises(ValueError):
                    provider_settings()

    def test_key_file_with_gateway_assignment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "key.env"
            path.write_text('LITELLM_API_KEY="test-key"\nLITELLM_API_BASE="https://gateway.example/v1"\n')
            with patch.dict(os.environ, {**ENV, "LITELLM_API_KEY": "", "API_KEY_FILE": str(path)}, clear=True):
                self.assertEqual(provider_key(), "test-key")
                with patch.dict(os.environ, {"LITELLM_API_BASE": "https://wrong.example"}):
                    with self.assertRaisesRegex(EnvironmentError, "gateway URL differs"):
                        provider_key()

    def test_key_file_does_not_fall_back_or_accept_duplicates(self):
        for contents in ('OPENAI_API_KEY="wrong"\nLITELLM_API_BASE="https://gateway.example/v1"\n',
                         'LITELLM_API_KEY="first"\nLITELLM_API_KEY="second"\n',
                         'LITELLM_API_KEY="first"\necho unsafe\n'):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "key.env"
                path.write_text(contents)
                with patch.dict(os.environ, {**ENV, "LITELLM_API_KEY": "", "API_KEY_FILE": str(path)}, clear=True):
                    with self.assertRaises(EnvironmentError):
                        provider_key()

    def test_gpu_environment_has_no_key_or_key_file(self):
        clean = without_credentials({**ENV, "API_KEY_FILE": "/private/key", "OPENAI_API_KEY": "key",
                                     "GEMINI_API_KEY": "key", "GOOGLE_API_KEY": "key", "PATH": "/bin"})
        self.assertFalse(any("KEY" in key for key in clean))
        self.assertEqual(clean["PATH"], "/bin")

    def test_route_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            root = Path(directory)
            bind_provider(root)
            for key, value in (("ADM_LLM_PROVIDER", "gemini"), ("LITELLM_API_BASE", "https://other.example"),
                               ("LITELLM_MODEL_MAP", '{"configured":"different"}')):
                with patch.dict(os.environ, {key: value}):
                    with self.assertRaises(PermissionError):
                        bind_provider(root)

    def test_cannot_relabel_legacy_attempts(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            path = Path(directory) / "old/attempt_started.json"
            path.parent.mkdir()
            path.write_text("{}")
            with self.assertRaises(PermissionError):
                bind_provider(Path(directory))

    def test_request_preserves_schema_and_reasoning(self):
        body = request_body(request(), Answer, model="gemini/test")
        self.assertEqual(body["model"], "gemini/test")
        self.assertEqual(body["instructions"], request().system_prompt)
        self.assertEqual(body["text"]["format"]["schema"]["required"], ["value"])
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertNotIn("temperature", body)
        self.assertNotIn("tools", body)

    def test_response_counts_reasoning_once(self):
        parsed = parse_response(raw_response(), Answer)
        self.assertEqual(parsed["parsed"].value, 1)
        self.assertEqual(parsed["usage"], {"prompt_tokens": 20, "output_tokens": 10,
                         "total_tokens": 30, "completion_tokens": 6, "thought_tokens": 4})

    def test_malformed_incomplete_refusal_and_usage_rejected(self):
        bad = []
        value = raw_response(); value["status"] = "incomplete"; bad.append(value)
        value = raw_response(); value["usage"]["total_tokens"] = 31; bad.append(value)
        value = raw_response(); value["output"][0]["content"][0] = {"type": "refusal"}; bad.append(value)
        value = raw_response(); value["output"][0]["content"][0]["text"] = "not JSON"; bad.append(value)
        value = raw_response(); value["output"].append({"type": "function_call"}); bad.append(value)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_response(value, Answer)

    def test_dispatch_and_durable_replay(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            wire = session()
            client = make_structured_client(attempt_root=Path(directory), session=wire, usage_budget=budget())
            first = client.parse(request(), Answer)
            second = client.parse(request(), Answer)
            self.assertEqual(first.parsed, second.parsed)
            self.assertTrue(second.raw_response["provider_response_replayed"])
            wire.post.assert_called_once()
            call = wire.post.call_args
            self.assertEqual(call.args[0], "https://gateway.example/v1/responses")
            self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer test-key")
            self.assertNotIn("X-Goog-Api-Key", call.kwargs["headers"])
            self.assertFalse(call.kwargs["allow_redirects"])
            self.assertFalse(any("test-key" in path.read_text() for path in Path(directory).rglob("*.json")))
            client.observe_model_version_metadata(first.raw_response)

    def test_transport_failure_is_not_resent_and_is_reserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            wire = session(); wire.post.side_effect = requests.Timeout("timeout")
            client = make_structured_client(attempt_root=Path(directory), session=wire, usage_budget=budget())
            with self.assertRaises(requests.Timeout):
                client.parse(request(), Answer)
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                client.parse(request(), Answer)
            wire.post.assert_called_once()
            self.assertEqual(client._completed_usage()[1], request().max_output_tokens)

    def test_existing_client_cannot_switch_to_google_credential(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            wire = session()
            client = make_structured_client(attempt_root=Path(directory), session=wire)
            with patch.dict(os.environ, {"ADM_LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "google-secret"}):
                client.parse(request(), Answer)
            self.assertEqual(wire.post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_parse_failure_preserved_without_resend(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            raw = raw_response(); raw["output"][0]["content"][0]["text"] = "bad"
            wire = session(raw)
            client = make_structured_client(attempt_root=Path(directory), session=wire, usage_budget=budget())
            for _ in range(2):
                with self.assertRaises(ValueError):
                    client.parse(request(), Answer)
            wire.post.assert_called_once()
            self.assertTrue((Path(directory) / request().request_hash / "parse_failure.json").is_file())

    def test_budget_stops_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            wire = session()
            client = make_structured_client(attempt_root=Path(directory), session=wire,
                                            usage_budget=budget(maximum_output_tokens=10))
            with self.assertRaises(PermissionError):
                client.parse(request(), Answer)
            wire.post.assert_not_called()

    def test_error_response_consumes_reservation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            wire = session({"error": "rate limit"}, status=429)
            client = make_structured_client(attempt_root=Path(directory), session=wire, usage_budget=budget())
            with self.assertRaises(requests.HTTPError):
                client.parse(request(), Answer)
            self.assertEqual(client._completed_usage()[1], 128)

    def test_cell_penetration_retains_retry_class(self):
        from domains.peptides.cell_penetration.controls import RecoveringLiteLLMClient
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            client = make_structured_client(attempt_root=Path(directory), session=session(),
                                           litellm_client_class=RecoveringLiteLLMClient)
            self.assertIsInstance(client, RecoveringLiteLLMClient)
            self.assertEqual(client.parse(request(), Answer).parsed.value, 1)

    def test_real_role_schemas_can_be_encoded(self):
        modules = ("domains.molecules.gsk3b.teacher", "domains.peptides.teacher",
                   "domains.peptides.cell_penetration.teacher", "domains.peptides.length_policy",
                   "domains.peptides.goals", "domains.peptides.judgments",
                   "domains.peptides.cell_penetration.goals", "domains.peptides.cell_penetration.judgments")
        checked = 0
        for module_name in modules:
            module = importlib.import_module(module_name)
            for name, value in vars(module).items():
                if isinstance(value, type) and issubclass(value, BaseModel) and name.endswith("Response"):
                    with self.subTest(schema=name):
                        request_body(request(), value, model="gemini/test")
                        checked += 1
        self.assertGreaterEqual(checked, 6)

    def test_molecular_role_factory(self):
        from domains.molecules.gsk3b.run import RoleExecutor
        config = json.loads((ROOT / "domains/molecules/gsk3b/configs/dpo.json").read_text())
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            executor = RoleExecutor(role="candidate_judge", config=config, root=Path(directory))
            self.assertIsInstance(executor.client, LiteLLMResponsesClient)

    def test_litellm_preflight_is_not_native_google_call(self):
        from domains.peptides.preflight import _generation_probe
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
            wire = session()
            with patch("adm.preferences.gemini_vertex.requests.Session", return_value=wire):
                result = _generation_probe(key="test-key", model="configured", role_name="candidate_judge",
                    allocation_id="123", output_root=Path(directory), timeout=30, usage_budget=budget())
            self.assertEqual(result["method"], "responses.create")
            self.assertEqual(result["model_version"], "resolved-model")
            self.assertEqual(wire.post.call_args.args[0], "https://gateway.example/v1/responses")

    def test_smoke_default_does_not_read_credentials_or_call(self):
        from adm.preferences.provider_smoke import main
        with patch.dict(os.environ, {**ENV, "LITELLM_API_KEY": ""}, clear=True), \
             patch("adm.preferences.provider_smoke.provider_key") as key, \
             patch("adm.preferences.provider_smoke.make_structured_client") as client, \
             patch("builtins.print"):
            main(["--model", "configured", "--output-dir", "/unused", "--maximum-usd", "1",
                  "--input-usd-per-million", "2", "--output-usd-per-million", "12"])
            key.assert_not_called(); client.assert_not_called()

    def test_cli_prepare_only_needs_no_credentials(self):
        from adm import cli
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True), \
             patch.object(cli, "checkout_root", return_value=ROOT), \
             patch.object(cli, "provider_key") as key, patch.object(cli.subprocess, "run") as run, \
             patch("builtins.print"):
            cli.main(["run", "gsk3b", "--output-dir", directory, "--prepare-only"])
            key.assert_not_called()
            self.assertNotIn("LITELLM_API_KEY", run.call_args.kwargs["env"])

    def test_smoke_three_calls_replay_without_extra_dispatch(self):
        from adm.preferences.provider_smoke import main
        labels = ("positive", "negative", "abstain")
        responses = []
        for label in labels:
            raw = raw_response()
            raw["output"][0]["content"][0]["text"] = json.dumps({"label": label, "sentinel": "transport-smoke"})
            responses.append(session(raw).post.return_value)
        wire = Mock(); wire.post.side_effect = responses
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True), \
             patch("adm.preferences.gemini_vertex.requests.Session", return_value=wire), \
             patch("adm.preferences.gemini_vertex.time.sleep"), patch("builtins.print"):
            args = ["--model", "configured", "--output-dir", directory, "--calls", "3",
                    "--maximum-usd", "1", "--input-usd-per-million", "2",
                    "--output-usd-per-million", "12", "--execute"]
            main(args)
            main(args)
            self.assertEqual(wire.post.call_count, 3)
            self.assertEqual(json.loads((Path(directory) / "smoke_result.json").read_text())["status"], "passed")


if __name__ == "__main__":
    unittest.main()
