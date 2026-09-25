import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imprint.http import Registration, chat_request, control_request, make_server
from imprint.recipes import from_files
from imprint.service import Service
from fake_backend import FakeBackend


class FakeService:
    def __init__(self):
        self.calls = []
        self.closed_streams = 0
        self.completed_streams = 0
        self.fail_midway = False

    def inspect(self):
        return {"profiles": [{"name": "memory"}], "active": "memory"}

    def sessions(self):
        return [{"session_id": "session-1", "status": "retained"}]

    def snapshot(self, session, name, scope="committed"):
        self.calls.append(("snapshot", session, name, scope))
        return {"name": name, "token_count": 20}

    def activate(self, name):
        self.calls.append(("activate", name))
        return {"active": name}

    def sleep(self):
        self.calls.append(("sleep",))
        return {"worker_running": False}

    def stream(self, request):
        self.calls.append(("stream", request))
        try:
            yield {
                "type": "start",
                "session_id": "session-1",
                "profile": "memory",
                "cached_tokens": 20,
                "prompt_tokens": 23,
                "artifact_id": "generation-1",
            }
            yield {"type": "delta", "text": "Hello", "token": 1}
            if self.fail_midway:
                raise ValueError("SECRET MEMORY SHOULD NOT BE RETURNED")
            yield {"type": "delta", "text": " world", "token": 2}
            yield {
                "type": "done",
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": 23,
                    "completion_tokens": 2,
                },
            }
            self.completed_streams += 1
        finally:
            self.closed_streams += 1


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.server = make_server(self.service, port=0, token="test-control-token")
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=2
        )
        values = {"Content-Type": "application/json", **(headers or {})}
        payload = json.dumps(body) if body is not None else None
        connection.request(method, path, body=payload, headers=values)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read().decode()
        connection.close()
        return result

    def chat(self, **options):
        return self.request(
            "POST",
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                **options,
            },
        )

    def test_nonstream_is_openai_shaped_and_preserves_cache_metrics(self):
        status, headers, text = self.chat()
        body = json.loads(text)
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["prompt_tokens_details"]["cached_tokens"], 20)
        self.assertEqual(body["usage"]["total_tokens"], 25)
        self.assertEqual(headers["X-Imprint-Session-ID"], "session-1")
        self.assertEqual(headers["X-Imprint-Cache"], "partial")
        self.assertEqual(self.service.closed_streams, 1)
        self.assertEqual(self.service.completed_streams, 1)

    def test_stream_has_one_copy_of_each_delta_and_usage(self):
        status, headers, text = self.chat(
            stream=True, stream_options={"include_usage": True}
        )
        frames = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertEqual(frames[-1], "[DONE]")
        payloads = [json.loads(frame) for frame in frames[:-1]]
        words = [
            choice["delta"].get("content", "")
            for frame in payloads
            for choice in frame["choices"]
        ]
        self.assertEqual("".join(words), "Hello world")
        self.assertEqual(payloads[-1]["usage"]["completion_tokens"], 2)
        self.assertEqual(self.service.completed_streams, 1)

    def test_stream_error_is_reported_without_secret_or_retry(self):
        self.service.fail_midway = True
        status, headers, text = self.chat(stream=True)
        self.assertEqual(status, 200)
        self.assertIn("Generation failed", text)
        self.assertNotIn("SECRET MEMORY", text)
        self.assertEqual(text.count('"content": "Hello"'), 1)
        self.assertEqual(self.service.closed_streams, 1)
        self.assertEqual(len(self.service.calls), 1)

    def test_unsupported_options_are_rejected_before_service_runs(self):
        for options in (
            {"tools": []},
            {"stop": "x"},
            {"max_tokens": True},
            {"max_tokens": 0},
            {"temperature": float("nan")},
            {"top_p": 1.5},
            {"stream": "yes"},
            {"messages": [{"role": "user", "content": []}]},
            {"chat_template_kwargs": {"arbitrary": True}},
        ):
            with self.subTest(options=options):
                self.assertEqual(self.chat(**options)[0], 400)
        self.assertEqual(self.service.calls, [])

    def test_control_requires_token(self):
        self.assertEqual(self.request("POST", "/_imprint/sleep", {})[0], 401)
        self.assertEqual(self.request("GET", "/_imprint/sessions")[0], 401)
        self.assertEqual(self.service.calls, [])
        self.assertEqual(
            self.request(
                "POST",
                "/_imprint/sleep",
                {},
                {"Authorization": "Bearer test-control-token"},
            )[0],
            200,
        )
        self.assertEqual(self.service.calls, [("sleep",)])

    def test_nonascii_control_header_is_rejected_without_crashing(self):
        status, headers, text = self.request(
            "GET", "/_imprint/status", headers={"Authorization": "Bearer café"}
        )
        self.assertEqual(status, 401)

    def test_browser_origins_and_rebound_hosts_are_rejected(self):
        self.assertEqual(
            self.request(
                "GET", "/v1/models", headers={"Origin": "https://example.invalid"}
            )[0],
            403,
        )
        self.assertEqual(
            self.request("GET", "/v1/models", headers={"Host": "example.invalid"})[0],
            403,
        )

    def test_models_are_profile_aliases(self):
        status, headers, text = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in json.loads(text)["data"]], ["memory"])

    def test_registration_and_control_client_work_without_guard_files(self):
        with tempfile.TemporaryDirectory() as root:
            address = f"http://127.0.0.1:{self.server.server_port}"
            with Registration(Path(root), address, "test-control-token"):
                self.assertEqual(
                    os.stat(Path(root) / "control.json").st_mode & 0o777, 0o600
                )
                response = control_request(
                    Path(root),
                    "snapshot",
                    {
                        "session": "session-1",
                        "name": "saved",
                        "scope": "absorbed",
                    },
                )
                self.assertEqual(response["name"], "saved")
                self.assertEqual(
                    self.service.calls[-1],
                    ("snapshot", "session-1", "saved", "absorbed"),
                )
                with self.assertRaisesRegex(RuntimeError, "Another Imprint"):
                    with Registration(Path(root), address, "other-token"):
                        pass
                with self.assertRaisesRegex(RuntimeError, "Another Imprint"):
                    with Registration(Path(root), address, "test-control-token"):
                        pass
                self.assertEqual(
                    [item.name for item in Path(root).iterdir()], ["control.json"]
                )
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_control_client_rejects_nonprivate_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "control.json"
            path.write_text(
                json.dumps({"base_url": "http://127.0.0.1:8460", "token": "secret"})
            )
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "private"):
                control_request(Path(root), "status")

    def test_control_client_rejects_remote_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "control.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "base_url": "http://203.0.113.1:8460",
                        "token": "secret",
                    }
                )
            )
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "loopback"):
                control_request(Path(root), "status")

    def test_max_completion_tokens_alias(self):
        result = chat_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 7,
            }
        )
        self.assertEqual(result["max_tokens"], 7)
        self.assertNotIn("max_completion_tokens", result)

    def test_public_listen_address_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            make_server(self.service, host="0.0.0.0", port=0)


class HttpServiceTests(unittest.TestCase):
    def test_successful_responses_keep_spawned_worker_and_live_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            memory = root / "memory.md"
            memory.write_text("A saved reference")
            service = Service(
                root / "store", backend_factory=FakeBackend, idle_seconds=0
            )
            server = None
            try:
                service.compute(from_files([memory], "memory"), str(model), "memory")
                process = service.worker.process
                server = make_server(service, port=0, token="integration-token")
                thread = threading.Thread(
                    target=server.serve_forever, kwargs={"poll_interval": 0.01}
                )
                thread.start()
                for stream in (False, True):
                    with self.subTest(stream=stream):
                        connection = http.client.HTTPConnection(
                            "127.0.0.1", server.server_port, timeout=5
                        )
                        connection.request(
                            "POST",
                            "/v1/chat/completions",
                            json.dumps(
                                {
                                    "model": "memory",
                                    "messages": [{"role": "user", "content": "hello"}],
                                    "max_tokens": 3,
                                    "stream": stream,
                                }
                            ),
                            {"Content-Type": "application/json"},
                        )
                        response = connection.getresponse()
                        body = response.read().decode()
                        session = response.getheader("X-Imprint-Session-ID")
                        self.assertEqual(response.status, 200, body)
                        connection.close()
                        self.assertTrue(process.is_alive())
                        self.assertEqual(service.worker.process.pid, process.pid)
                        saved = service.snapshot(
                            session, "streamed" if stream else "regular"
                        )
                        self.assertEqual(saved["tail_tokens_computed"], 1)
                        self.assertTrue(process.is_alive())
            finally:
                service.close()
                if server is not None:
                    server.shutdown()
                    server.server_close()
                    thread.join(2)
