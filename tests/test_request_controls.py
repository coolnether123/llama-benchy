"""Offline request-contract tests, including the CLI and complete benchmark suite."""

import asyncio
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from llama_benchy.__main__ import main_async
from llama_benchy.client import CONTEXT_LOAD_USER_MESSAGE, LLMClient
from llama_benchy.config import BenchmarkConfig


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


class FakeCorpus:
    def __init__(self, *args):
        self.tokens = [f"word{i}" for i in range(128)]

    def __len__(self):
        return len(self.tokens)

    def get_tokens(self):
        return self.tokens

    def get_tokenizer(self):
        return FakeTokenizer()


class FakeResponse:
    status = 200

    def __init__(self, session):
        self.session = session
        self.content = self

    async def __aenter__(self):
        self.session.active += 1
        self.session.max_active = max(self.session.max_active, self.session.active)
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *args):
        self.session.active -= 1

    async def json(self):
        return {
            "choices": [{"message": {"content": "Paris"}}],
            "usage": {"prompt_tokens": 10},
        }

    async def read(self):
        return b"{}"

    def __aiter__(self):
        return self.iter_any()

    async def iter_any(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Pa"}, "token_ids": [1, 2]}]},
            {"choices": [{"delta": {"content": "ris"}, "token_ids": [3]}]},
            {"choices": [], "usage": {"prompt_tokens": 8}},
        ]
        for chunk in chunks:
            await asyncio.sleep(0)
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"


class RecordingSession:
    def __init__(self):
        self.posts = []
        self.gets = []
        self.active = 0
        self.max_active = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, url, *, json, headers):
        self.posts.append((url, deepcopy(json), headers.copy()))
        return FakeResponse(self)

    def get(self, url, *, headers):
        self.gets.append((url, headers.copy()))
        return FakeResponse(self)


class RequestControlsTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_controls_reach_every_phase_and_defaults_stay_omitted(self):
        for configured in (False, True):
            with self.subTest(configured=configured):
                argv = [
                    "llama-benchy", "--base-url", "http://router.invalid/v1",
                    "--model", "test/model", "--served-model-name", "served-model",
                    "--pp", "8", "--tg", "3", "--depth", "0", "16",
                    "--runs", "1", "--concurrency", "2",
                    "--enable-prefix-caching", "--latency-mode", "generation",
                    "--format", "json",
                ]
                if configured:
                    argv += [
                        "--api-key", "api-secret-value",
                        "--header", "X-Qwen-Priority: P5",
                        "--header", "X-Qwen-Idle-Only: true",
                        "--header", "X-Qwen-Source: llama-benchy",
                        "--header", "X-Qwen-Job-Type: benchmark",
                        "--header", "X-Qwen-Benchmark-Family: qwen38_quant_study",
                        "--header", "X-Qwen-Benchmark-Name: Qwen 3.8 27B quant study",
                        "--header", "X-Qwen-Benchmark-Run-Id: run-123",
                        "--header", "X-Qwen-Benchmark-Case-Id: case-456",
                        "--header", "X-Secret: header-secret-value",
                        "--temperature", "0", "--seed", "0",
                        "--extra-body", "temperature=0.8,seed=99,ignore_eos=false",
                        "--exact-tg",
                    ]
                session = RecordingSession()
                output = io.StringIO()
                with (
                    TemporaryDirectory() as directory,
                    patch("sys.argv", argv),
                    patch("llama_benchy.__main__.TokenizedCorpus", FakeCorpus),
                    patch("llama_benchy.runner.aiohttp.ClientSession", return_value=session),
                    redirect_stdout(output),
                ):
                    progress_path = Path(directory) / "progress.jsonl"
                    argv += ["--emit-progress", str(progress_path)]
                    await main_async()
                    progress_text = progress_path.read_text()
                    events = [json.loads(line) for line in progress_text.splitlines()]

                # Two warmups, coherence, four latency probes, four standard runs,
                # and eight prefix prefill/decode requests, including per-case warmup.
                self.assertEqual(len(session.posts), 19)
                self.assertEqual(session.max_active, 2)
                for url, payload, headers in session.posts:
                    self.assertEqual(url, "http://router.invalid/v1/chat/completions")
                    self.assertEqual(payload["model"], "served-model")
                    if configured:
                        self.assertEqual(payload["temperature"], 0)
                        self.assertEqual(payload["seed"], 0)
                        self.assertEqual(headers["X-Qwen-Priority"], "P5")
                        self.assertEqual(headers["X-Qwen-Idle-Only"], "true")
                        self.assertEqual(headers["X-Qwen-Source"], "llama-benchy")
                        self.assertEqual(headers["X-Qwen-Job-Type"], "benchmark")
                        self.assertEqual(headers["X-Qwen-Benchmark-Family"], "qwen38_quant_study")
                        self.assertEqual(headers["X-Qwen-Benchmark-Name"], "Qwen 3.8 27B quant study")
                        self.assertEqual(headers["X-Qwen-Benchmark-Run-Id"], "run-123")
                        self.assertEqual(headers["X-Qwen-Benchmark-Case-Id"], "case-456")
                        self.assertEqual(headers["Authorization"], "Bearer api-secret-value")
                        self.assertEqual(headers["X-Secret"], "header-secret-value")
                    else:
                        self.assertNotIn("temperature", payload)
                        self.assertNotIn("seed", payload)
                        self.assertEqual(dict(headers), {"Authorization": "Bearer EMPTY"})

                payloads = [payload for _, payload, _ in session.posts]
                self.assertNotIn("stream", payloads[0])
                self.assertEqual(payloads[1]["messages"][0]["role"], "system")
                self.assertEqual(payloads[2]["max_tokens"], 100)
                for payload in payloads[3:7]:
                    self.assertTrue(payload["stream"])
                    self.assertEqual(payload["max_tokens"], 1)
                for payload in payloads[7:]:
                    self.assertTrue(payload["stream"])
                    self.assertTrue(payload["return_token_ids"])
                    self.assertEqual(payload["stream_options"], {"include_usage": True})
                    self.assertNotIn("cache_prompt", payload)
                    if configured:
                        self.assertEqual(payload["max_tokens"], 3)
                        self.assertEqual(payload["min_tokens"], 3)
                        self.assertTrue(payload["ignore_eos"])
                    else:
                        self.assertNotIn("min_tokens", payload)
                        self.assertNotIn("ignore_eos", payload)
                for prefill_start in (11, 15):
                    for i in range(2):
                        prefill = payloads[prefill_start + i]["messages"]
                        decode = payloads[prefill_start + 2 + i]["messages"]
                        self.assertEqual(prefill[0], decode[0])
                        self.assertEqual(prefill[1]["content"], CONTEXT_LOAD_USER_MESSAGE)
                        self.assertTrue(decode[1]["content"])

                text = output.getvalue()
                self.assertNotIn("api-secret-value", text)
                self.assertNotIn("header-secret-value", text)
                report = json.loads(text[text.index("{"):text.rindex("}") + 1])
                self.assertEqual(len(report["benchmarks"]), 3)
                self.assertNotIn("api-secret-value", progress_text)
                self.assertNotIn("header-secret-value", progress_text)
                self.assertEqual(events[0]["type"], "header")
                self.assertEqual(events[-1]["type"], "bench_complete")
                self.assertEqual(events[-1]["status"], "ok")
                starts = [event for event in events if event["type"] == "request_start"]
                self.assertEqual([event["request_id"] for event in starts], list(range(6)))
                self.assertTrue(all(event["run_index"] == 0 for event in starts))
                for request_id in range(6):
                    self.assertEqual(
                        [event["type"] for event in events if event.get("request_id") == request_id],
                        ["request_start", "request_first_response", "request_first_token",
                         "tokens", "tokens", "request_end"],
                    )

    async def test_ttft_mtp_cache_flag_and_api_latency_headers(self):
        session = RecordingSession()
        client = LLMClient(
            "http://router.invalid/v1", "key", "model",
            request_headers={"X-Qwen-Priority": "P5"}, temperature=0, seed=42,
        )
        result = await client.run_generation(session, "context", "prompt", 3, True)
        self.assertIsNone(result.error)
        self.assertLess(result.first_response_ts, result.first_token_ts)
        self.assertEqual(result.total_tokens, 3)
        self.assertEqual(len(result.token_timestamps), 3)
        self.assertEqual(result.prompt_tokens, 8)
        self.assertFalse(session.posts[0][1]["cache_prompt"])
        self.assertEqual(session.posts[0][1]["temperature"], 0)
        self.assertEqual(session.posts[0][1]["seed"], 42)
        await client.measure_latency(session, "api")
        self.assertEqual(len(session.gets), 3)
        for _, headers in session.gets:
            self.assertEqual(headers["X-Qwen-Priority"], "P5")


class ConfigControlsTests(unittest.TestCase):
    def config_from_args(self, *extra):
        with patch("sys.argv", [
            "llama-benchy", "--base-url", "http://router.invalid/v1",
            "--model", "test/model", *extra,
        ]):
            return BenchmarkConfig.from_args()

    def test_headers_parse_colons_case_insensitively_and_secrets_are_excluded(self):
        config = self.config_from_args(
            "--api-key", "api-secret-value",
            "--header", "X-Secret: old-value",
            "--header", "x-secret: header-secret:value",
            "--header", "authorization: Bearer alternate-secret",
            "--temperature", "0", "--seed", "42",
        )
        self.assertEqual(config.request_headers["x-secret"], "header-secret:value")
        client = LLMClient(
            config.base_url, config.api_key, config.model,
            request_headers=config.request_headers,
        )
        self.assertEqual(client.headers.getall("Authorization"), ["Bearer alternate-secret"])
        self.assertNotIn("api_key", config.model_dump())
        self.assertNotIn("request_headers", config.model_dump())
        for rendered in (repr(config), config.model_dump_json()):
            for secret in ("api-secret-value", "header-secret:value", "alternate-secret"):
                self.assertNotIn(secret, rendered)
        defaults = self.config_from_args()
        self.assertEqual(defaults.request_headers, {})
        self.assertIsNone(defaults.temperature)
        self.assertIsNone(defaults.seed)

    def test_invalid_headers_are_rejected_without_echoing_values(self):
        invalid_headers = ["secret-without-colon", "Bad Name: secret-value", ": secret-value"]
        for codepoint in (*range(32), 127):
            char = chr(codepoint)
            invalid_headers.extend((f"{char}X-Key: secret-value", f"X-Key{char}: secret-value"))
            if char != "\t":
                invalid_headers.append(f"X-Key: secret{char}value")
        for header in invalid_headers:
            with self.subTest(header=header):
                output = io.StringIO()
                with redirect_stderr(output), self.assertRaises(SystemExit) as error:
                    self.config_from_args("--header", header)
                self.assertEqual(error.exception.code, 2)
                self.assertNotIn("secret", output.getvalue())

    def test_header_optional_spaces_empty_values_and_value_tabs(self):
        config = self.config_from_args(
            "--header", " X-Key : one\ttwo ", "--header", "X-Empty:",
        )
        self.assertEqual(config.request_headers, {"x-key": "one\ttwo", "x-empty": ""})

    def test_python_headers_use_case_insensitive_last_value_wins(self):
        custom_headers = {
            "Authorization": "old-secret", "authorization": "new-secret",
            "X-Key": "old", "x-key": "new",
        }
        client = LLMClient(
            "http://router.invalid/v1", "api-key", "model", request_headers=custom_headers,
        )
        models = Mock()
        models.json.return_value = {"data": [{"id": "test/model"}]}
        with patch("llama_benchy.config.requests.get", side_effect=[models, Mock(status_code=200)]) as get:
            BenchmarkConfig._detect_hf_model_from_endpoint(
                "http://router.invalid/v1", "api-key", custom_headers,
            )
        for headers in (client.headers, get.call_args_list[0].kwargs["headers"]):
            self.assertEqual(headers.getall("Authorization"), ["new-secret"])
            self.assertEqual(headers.getall("X-Key"), ["new"])
        self.assertEqual(custom_headers["Authorization"], "old-secret")

    def test_discovery_default_authorization_is_unchanged(self):
        for api_key, expected in (("key", {"Authorization": "Bearer key"}), ("EMPTY", {}), ("", {})):
            with self.subTest(api_key=api_key):
                models = Mock()
                models.json.return_value = {"data": [{"id": "test/model"}]}
                with patch("llama_benchy.config.requests.get", side_effect=[models, Mock(status_code=200)]) as get:
                    BenchmarkConfig._detect_hf_model_from_endpoint("http://router.invalid/v1", api_key)
                self.assertEqual(dict(get.call_args_list[0].kwargs["headers"]), expected)

    def test_discovery_headers_stay_on_endpoint(self):
        models = Mock()
        models.json.return_value = {"data": [{"id": "served", "root": "test/model"}]}
        hf = Mock(status_code=200)
        with patch("llama_benchy.config.requests.get", side_effect=[models, hf]) as get:
            self.assertEqual(
                BenchmarkConfig._detect_hf_model_from_endpoint(
                    "http://router.invalid/v1", "key", {"X-Secret": "endpoint-secret"},
                ),
                ("test/model", "served"),
            )
        self.assertEqual(get.call_args_list[0].kwargs["headers"]["X-Secret"], "endpoint-secret")
        self.assertNotIn("headers", get.call_args_list[1].kwargs)


if __name__ == "__main__":
    unittest.main()
