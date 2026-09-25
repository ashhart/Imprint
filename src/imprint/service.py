import threading
import time
from pathlib import Path

from .learning import learning_mode
from .store import Store
from .worker import Worker, WorkerError as WorkerError


class Service:
    def __init__(
        self,
        store_root,
        model=None,
        name=None,
        learn=False,
        idle_seconds=300,
        backend_factory=None,
    ):
        if idle_seconds < 0:
            raise ValueError("Idle timeout cannot be negative")
        self.store = Store(Path(store_root))
        self.name = name
        self.learn = learning_mode(learn)
        self.model = str(Path(model).expanduser().resolve()) if model else None
        if name and not learn:
            metadata = self.store.profile(name)
            self.model = self.model or metadata["model"]
        if learn and (not name or not model):
            raise ValueError("Learning requires --model and --name")
        self.backend_factory = backend_factory
        self.idle_seconds = idle_seconds
        self.worker = None
        self.lock = threading.RLock()
        self.last_used = time.monotonic()
        self.closed = False
        self.stop_timer = threading.Event()
        self.timer = None
        if idle_seconds:
            self.timer = threading.Thread(target=self._idle_loop, daemon=True)
            self.timer.start()

    def _idle_loop(self):
        while not self.stop_timer.wait(min(self.idle_seconds, 1)):
            with self.lock:
                if (
                    self.worker
                    and time.monotonic() - self.last_used >= self.idle_seconds
                ):
                    self._sleep()

    def _worker(self):
        if self.closed:
            raise ValueError("Service is closed")
        if not self.model:
            raise ValueError("Select a model or saved profile first")
        if self.worker and not self.worker.process.is_alive():
            self.worker.close()
            self.worker = None
        if self.worker is None:
            settings = (str(self.store.root), self.model, self.name, self.learn)
            self.worker = Worker(settings, self.backend_factory)
            try:
                if self.closed:
                    raise ValueError("Service is closed")
                self.worker.ready()
            except BaseException:
                self.worker.close()
                self.worker = None
                raise
        self.last_used = time.monotonic()
        return self.worker

    def compute(self, recipe, model, name):
        with self.lock:
            model = str(Path(model).expanduser().resolve())
            previous = (self.model, self.name, self.learn)
            if self.model != model:
                self._sleep()
                self.model = model
            try:
                result = self._worker().call(
                    "compute", recipe=recipe, model=model, name=name
                )
            except Exception:
                self._sleep()
                self.model, self.name, self.learn = previous
                raise
            self.name = name
            self.last_used = time.monotonic()
            return result

    def snapshot(self, session, name, scope="committed"):
        with self.lock:
            if self.worker is None or not self.worker.process.is_alive():
                raise ValueError("No live session is available; no model was started")
            result = self.worker.call(
                "snapshot", session=session, name=name, scope=scope
            )
            self.last_used = time.monotonic()
            return result

    def inspect(self, name=None):
        with self.lock:
            if name:
                return self.store.profile(name)
            return {
                "profiles": self.store.profiles(),
                "active": self.name,
                "model": self.model,
                "learning_mode": self.learn or "off",
                "worker_running": bool(self.worker and self.worker.process.is_alive()),
                "sessions": self.sessions(),
            }

    def sessions(self):
        with self.lock:
            if self.worker is None or not self.worker.process.is_alive():
                return []
            return self.worker.call("sessions")

    def activate(self, name):
        with self.lock:
            metadata = self.store.profile(name)
            previous = (self.model, self.name, self.learn)
            if metadata["model"] != self.model:
                self._sleep()
                self.model = metadata["model"]
            try:
                result = self._worker().call("activate", name=name)
            except Exception:
                self._sleep()
                self.model, self.name, self.learn = previous
                raise
            self.name = name
            self.learn = False
            self.last_used = time.monotonic()
            return result

    def stream(self, request):
        started = time.perf_counter()
        with self.lock:
            from .runtime import validate_request

            validate_request(request)
            if self.name and request.get("model", self.name) not in {
                self.name,
                "imprint",
            }:
                raise ValueError("Request model must match the active profile")
            first_token_seconds = None
            worker = self._worker()
            events = worker.stream(request)
            try:
                for event in events:
                    if (
                        event["type"] == "delta"
                        and event.get("text")
                        and first_token_seconds is None
                    ):
                        first_token_seconds = time.perf_counter() - started
                    if event["type"] == "done":
                        event["first_token_seconds"] = first_token_seconds
                        event["request_seconds"] = time.perf_counter() - started
                    yield event
            finally:
                events.close()
                self.last_used = time.monotonic()

    def _sleep(self):
        if self.worker:
            self.worker.close()
            self.worker = None
        return {"sleeping": True, "active": self.name, "sessions_evicted": True}

    def sleep(self):
        with self.lock:
            return self._sleep()

    def close(self):
        self.stop_timer.set()
        self.closed = True
        worker = self.worker
        if worker and worker.process.is_alive():
            worker.process.terminate()
        with self.lock:
            self._sleep()
        if self.timer and self.timer is not threading.current_thread():
            self.timer.join(timeout=2)
