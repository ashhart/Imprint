from __future__ import annotations

import contextlib
import fcntl
import hmac
import ipaddress
import json
import math
import os
from pathlib import Path
import secrets
import socket
import stat
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, ProxyHandler
import uuid


MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_TOKENS = 32768


def loopback_host(host: str) -> str:
    if host == "localhost":
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError("Use a numeric loopback address or localhost") from None
    if not address.is_loopback:
        raise ValueError("This release only serves loopback addresses")
    return str(address)


def parse_json(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        result = json.loads(data, object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("Body must be UTF-8 JSON") from None
    if not isinstance(result, dict):
        raise ValueError("Body must be a JSON object")
    return result


def chat_request(body: dict) -> dict:
    allowed = {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "chat_template_kwargs",
    }
    if set(body) - allowed:
        raise ValueError(
            "Unsupported request fields; this release accepts plain-text chat only"
        )
    result = dict(body)
    model = result.get("model", "imprint")
    if not isinstance(model, str) or not model or len(model) > 128:
        raise ValueError("model must be a nonempty profile name")
    result["model"] = model
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Each message must contain role and plain-text content")
        if message["role"] not in ("system", "developer", "user", "assistant"):
            raise ValueError("Unsupported message role")
        if not isinstance(message["content"], str):
            raise ValueError("Only plain-text message content is supported")
    if "max_tokens" in result and "max_completion_tokens" in result:
        raise ValueError("Choose max_tokens or max_completion_tokens, not both")
    maximum = result.pop("max_completion_tokens", result.get("max_tokens", 256))
    if type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT_TOKENS:
        raise ValueError(f"max_tokens must be between 1 and {MAX_OUTPUT_TOKENS}")
    result["max_tokens"] = maximum
    if type(result.get("stream", False)) is not bool:
        raise ValueError("stream must be a boolean")
    options = result.get("stream_options", {})
    if not isinstance(options, dict) or set(options) - {"include_usage"}:
        raise ValueError("Only stream_options.include_usage is supported")
    if type(options.get("include_usage", False)) is not bool:
        raise ValueError("include_usage must be a boolean")
    if options and not result.get("stream", False):
        raise ValueError("stream_options requires streaming")
    for key, minimum, maximum_value in (("temperature", 0, 2), ("top_p", 0, 1)):
        value = result.get(key, 0 if key == "temperature" else 1)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not minimum <= value <= maximum_value
        ):
            raise ValueError(f"{key} is outside the supported range")
    for key, ceiling in (("top_k", 1_000_000), ("seed", 2**32 - 1)):
        if key in result and (
            type(result[key]) is not int or not 0 <= result[key] <= ceiling
        ):
            raise ValueError(
                f"{key} must be a nonnegative integer in the supported range"
            )
    template = result.get("chat_template_kwargs", {})
    if not isinstance(template, dict) or set(template) - {"enable_thinking"}:
        raise ValueError("Only enable_thinking is supported in chat_template_kwargs")
    if "enable_thinking" in template and type(template["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    return result


def usage(event: dict, start: dict, completion_tokens: int) -> dict:
    values = event.get("usage", event)
    prompt = values.get("prompt_tokens", start.get("prompt_tokens", 0))
    completed = values.get("completion_tokens", completion_tokens)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completed,
        "total_tokens": prompt + completed,
        "prompt_tokens_details": {"cached_tokens": start.get("cached_tokens", 0)},
    }


def make_server(
    service, host: str = "127.0.0.1", port: int = 8460, token: str | None = None
):
    host = loopback_host(host)
    control_token = token or secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, format, *args):
            pass

        def send_json(self, status: int, result: dict, headers=None):
            payload = json.dumps(result, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.close_connection = True
            self.wfile.write(payload)

        def fail(self, status: int, message: str):
            self.send_json(
                status, {"error": {"message": message, "type": "imprint_error"}}
            )

        def valid_origin(self):
            if self.headers.get("Origin"):
                self.fail(403, "Browser-origin requests are not supported")
                return False
            authority = self.headers.get("Host", "")
            try:
                parsed = urlsplit("http://" + authority)
                valid = loopback_host(parsed.hostname or "")
                valid = (
                    valid
                    and parsed.port == self.server.server_port
                    and not parsed.username
                )
            except ValueError:
                valid = False
            if not valid:
                self.fail(403, "Host must identify this loopback server")
                return False
            return True

        def authorized(self):
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(
                supplied.encode("utf-8"), ("Bearer " + control_token).encode("utf-8")
            ):
                self.fail(401, "A local control token is required")
                return False
            return True

        def body(self):
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is not supported")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                raise ValueError("One Content-Length header is required")
            try:
                length = int(lengths[0])
            except ValueError:
                raise ValueError("Invalid Content-Length") from None
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("Request body is empty or too large")
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("Incomplete request body")
            return parse_json(data)

        def do_GET(self):
            if not self.valid_origin():
                return
            try:
                if self.path == "/v1/models":
                    state = service.inspect()
                    active = state.get("active")
                    names = [active] if active else []
                    self.send_json(
                        200,
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": name,
                                    "object": "model",
                                    "created": 0,
                                    "owned_by": "imprint",
                                }
                                for name in names
                            ],
                        },
                    )
                elif self.path.startswith("/_imprint/"):
                    if not self.authorized():
                        return
                    if self.path == "/_imprint/sessions":
                        self.send_json(200, {"sessions": service.sessions()})
                    elif self.path == "/_imprint/status":
                        self.send_json(200, service.inspect())
                    else:
                        self.fail(404, "Unknown control endpoint")
                else:
                    self.fail(404, "Unknown endpoint")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.fail(500, "The local service could not complete the request")

        def do_POST(self):
            if not self.valid_origin():
                return
            if self.path.startswith("/_imprint/") and not self.authorized():
                return
            try:
                request = self.body()
                if self.path == "/v1/chat/completions":
                    self.chat(chat_request(request))
                elif self.path == "/_imprint/snapshot":
                    if set(request) - {"session", "name", "scope"}:
                        raise ValueError("Unsupported snapshot fields")
                    if not all(
                        isinstance(request.get(key), str) and request[key]
                        for key in ("session", "name")
                    ):
                        raise ValueError("snapshot requires session and name")
                    scope = request.get("scope", "committed")
                    if scope not in ("committed", "absorbed"):
                        raise ValueError("scope must be committed or absorbed")
                    self.send_json(
                        200,
                        service.snapshot(
                            request["session"], request["name"], scope=scope
                        ),
                    )
                elif self.path == "/_imprint/activate":
                    if set(request) != {"name"} or not isinstance(request["name"], str):
                        raise ValueError("activate requires a profile name")
                    self.send_json(200, service.activate(request["name"]))
                elif self.path == "/_imprint/sleep":
                    if request:
                        raise ValueError("sleep takes no options")
                    self.send_json(200, service.sleep())
                else:
                    self.fail(404, "Unknown endpoint")
            except ValueError:
                self.fail(
                    400,
                    "Invalid or unsupported request; check the documented fields and active profile",
                )
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except Exception:
                self.fail(500, "The local service could not complete the request")

        def chat(self, request):
            iterator = iter(service.stream(request))
            with contextlib.closing(iterator):
                start = next(iterator)
                if start.get("type") != "start":
                    raise RuntimeError("Stream did not begin with metadata")
                cached = start.get("cached_tokens", 0)
                prompt = start.get("prompt_tokens", 0)
                headers = {
                    "X-Imprint-Session-ID": start["session_id"],
                    "X-Imprint-Cache": "miss"
                    if cached == 0
                    else "hit"
                    if cached == prompt
                    else "partial",
                }
                generation = start.get("generation", start.get("artifact_id"))
                if generation:
                    headers["X-Imprint-Profile-Generation"] = generation
                base = {
                    "id": "chatcmpl-" + uuid.uuid4().hex,
                    "created": int(time.time()),
                    "model": request["model"],
                }
                if request.get("stream", False):
                    self.stream_chat(iterator, request, start, base, headers)
                    return
                parts = []
                count = 0
                done = None
                for event in iterator:
                    if done is not None:
                        raise RuntimeError("Unexpected event after completion")
                    if event.get("type") == "delta":
                        parts.append(event.get("text", ""))
                        count += 1
                    elif event.get("type") == "done":
                        done = event
                    else:
                        raise RuntimeError("Unexpected stream event")
                if done is None:
                    raise RuntimeError("Stream ended without completion")
                self.send_json(
                    200,
                    {
                        **base,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "".join(parts),
                                },
                                "finish_reason": done.get("finish_reason", "stop"),
                            }
                        ],
                        "usage": usage(done, start, count),
                    },
                    headers,
                )

        def stream_chat(self, iterator, request, start, base, headers):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            for key, value in headers.items():
                self.send_header(key, str(value))
            self.end_headers()
            self.close_connection = True

            def send(payload):
                text = (
                    payload
                    if isinstance(payload, str)
                    else json.dumps(payload, allow_nan=False)
                )
                self.wfile.write(("data: " + text + "\n\n").encode("utf-8"))
                self.wfile.flush()

            def chunk(delta, finish=None):
                return {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }

            try:
                send(chunk({"role": "assistant", "content": ""}))
                count = 0
                done = None
                for event in iterator:
                    if done is not None:
                        raise RuntimeError("Unexpected event after completion")
                    if event.get("type") == "delta":
                        count += 1
                        send(chunk({"content": event.get("text", "")}))
                    elif event.get("type") == "done":
                        done = event
                    else:
                        raise RuntimeError("Unexpected stream event")
                if done is None:
                    raise RuntimeError("Stream ended without completion")
                send(chunk({}, done.get("finish_reason", "stop")))
                if request.get("stream_options", {}).get("include_usage", False):
                    send(
                        {
                            **base,
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "usage": usage(done, start, count),
                        }
                    )
                send("[DONE]")
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
            except Exception:
                send(
                    {
                        "error": {
                            "message": "Generation failed; no automatic retry was made",
                            "type": "imprint_error",
                        }
                    }
                )
                send("[DONE]")

    class Server(ThreadingHTTPServer):
        daemon_threads = False
        allow_reuse_address = True
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    server = Server((host, port), Handler)
    server.control_token = control_token
    return server


class StoreLease:
    def __init__(self, store_root: Path, busy_message: str):
        self.root = Path(store_root).expanduser()
        self.busy_message = busy_message
        self.fd = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.fd = os.open(self.root, os.O_RDONLY)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self
        except BlockingIOError:
            StoreLease.__exit__(self)
            raise RuntimeError(self.busy_message) from None
        except BaseException:
            StoreLease.__exit__(self)
            raise

    def __exit__(self, *args):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class Registration(StoreLease):
    def __init__(self, store_root: Path, base_url: str, token: str):
        super().__init__(
            store_root, "Another Imprint server or computation owns this store"
        )
        self.path = self.root / "control.json"
        self.token = token
        self.base_url = base_url

    def __enter__(self):
        super().__enter__()
        try:
            fd, temporary = tempfile.mkstemp(prefix=".control-", dir=self.root)
            try:
                with os.fdopen(fd, "w") as target:
                    json.dump(
                        {
                            "version": 1,
                            "base_url": self.base_url,
                            "token": self.token,
                            "pid": os.getpid(),
                        },
                        target,
                    )
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, self.path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary)
            return self
        except BaseException:
            super().__exit__()
            raise

    def __exit__(self, *args):
        try:
            with contextlib.suppress(OSError, ValueError):
                current = json.loads(self.path.read_text())
                if current.get("token") == self.token:
                    self.path.unlink()
        finally:
            super().__exit__(*args)


def serve(service, store_root: Path, host: str = "127.0.0.1", port: int = 8460):
    server = make_server(service, host, port)
    address = server.server_address[0]
    authority = f"[{address}]" if ":" in address else address
    url = f"http://{authority}:{server.server_port}"
    try:
        with Registration(store_root, url, server.control_token):
            try:
                print(f"Imprint listening at {url}/v1", file=sys.stderr, flush=True)
                server.serve_forever(poll_interval=0.2)
            finally:
                service.close()
    finally:
        try:
            service.close()
        finally:
            server.server_close()


def control_request(store_root: Path, action: str, payload: dict | None = None) -> dict:
    path = Path(store_root).expanduser() / "control.json"
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_mode & 0o077
                or info.st_uid != os.getuid()
            ):
                raise ValueError(
                    "Local control configuration must be owned by you and private"
                )
            if info.st_size > 16384:
                raise ValueError("Invalid local control configuration")
            config = parse_json(source.read(16385))
        if config.get("version") != 1:
            raise ValueError("Unsupported local control configuration version")
        endpoint = urlsplit(config["base_url"])
        loopback_host(endpoint.hostname or "")
        if (
            endpoint.scheme != "http"
            or not endpoint.port
            or endpoint.path
            or endpoint.query
            or endpoint.fragment
            or endpoint.username
        ):
            raise ValueError("Invalid local control address")
        token = config["token"]
        if not isinstance(token, str) or not token:
            raise ValueError("Invalid local control token")
    except FileNotFoundError:
        raise RuntimeError(
            "No Imprint server is running for this store; start imprint serve"
        ) from None
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("Invalid local control configuration") from None
    if action not in {"sessions", "status", "snapshot", "activate", "sleep"}:
        raise ValueError("Unknown local control action")
    request = Request(
        config["base_url"] + "/_imprint/" + action,
        data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, newurl):
            raise HTTPError(
                request.full_url, code, "Control redirects are disabled", headers, fp
            )

    try:
        opener = build_opener(ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=600) as response:
            data = response.read(MAX_BODY_BYTES + 1)
            if len(data) > MAX_BODY_BYTES:
                raise ValueError("Control response exceeds the size limit")
            return parse_json(data)
    except HTTPError as error:
        raise RuntimeError(
            f"Imprint control request failed with HTTP {error.code}"
        ) from None
    except (URLError, TimeoutError, ConnectionError):
        raise RuntimeError(
            "The saved Imprint server is unavailable; start imprint serve"
        ) from None
