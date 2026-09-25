import tempfile
import time
import unittest
from pathlib import Path

from imprint.recipes import from_files
from imprint.service import Service
from fake_backend import FakeBackend


class ServiceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.memory = self.root / "memory.md"
        self.memory.write_text("remember the blue planet")
        self.recipe = from_files([self.memory], "memory")
        self.service = Service(
            self.root / "store", backend_factory=FakeBackend, idle_seconds=0
        )

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    def compute(self):
        return self.service.compute(self.recipe, str(self.model), "memory")

    def request(self):
        return {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 3}


class ServiceTests(ServiceFixture):
    def test_empty_inspection_and_snapshot_do_not_start_worker(self):
        self.assertFalse(self.service.inspect()["worker_running"])
        with self.assertRaisesRegex(ValueError, "No live session"):
            self.service.snapshot("missing", "snapshot")
        self.assertIsNone(self.service.worker)

    def test_compute_sleep_restart_and_restore(self):
        self.compute()
        process = self.service.worker.process
        self.service.sleep()
        self.assertFalse(process.is_alive())
        events = list(self.service.stream(self.request()))
        self.assertGreater(events[0]["cached_tokens"], 0)
        self.assertNotEqual(process.pid, self.service.worker.process.pid)

    def test_snapshot_survives_sleep_without_replaying_prompt(self):
        self.compute()
        session = list(self.service.stream(self.request()))[0]["session_id"]
        saved = self.service.snapshot(session, "conversation")
        self.assertEqual(saved["tail_tokens_computed"], 1)
        self.service.sleep()
        self.assertEqual(
            self.service.inspect("conversation")["artifact_id"], saved["artifact_id"]
        )
        with self.assertRaises(ValueError):
            self.service.snapshot(session, "lost")
        self.assertIsNone(self.service.worker)

    def test_closing_partial_response_exits_only_owned_worker(self):
        self.compute()
        events = self.service.stream(self.request())
        next(events)
        process = self.service.worker.process
        events.close()
        self.assertFalse(process.is_alive())
        self.assertGreater(
            list(self.service.stream(self.request()))[0]["cached_tokens"], 0
        )

    def test_worker_error_does_not_desynchronize_next_request(self):
        self.compute()
        with self.assertRaises(ValueError):
            list(self.service.stream({**self.request(), "tools": []}))
        self.assertGreater(
            list(self.service.stream(self.request()))[0]["cached_tokens"], 0
        )

    def test_idle_timer_unloads_worker_and_keeps_files(self):
        self.service.close()
        self.service = Service(
            self.root / "store", backend_factory=FakeBackend, idle_seconds=0.05
        )
        self.compute()
        process = self.service.worker.process
        deadline = time.monotonic() + 4
        while process.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(process.is_alive())
        self.assertEqual(self.service.inspect("memory")["name"], "memory")

    def test_active_profile_switch_keeps_same_model_process(self):
        self.compute()
        self.service.compute(self.recipe, str(self.model), "second")
        process = self.service.worker.process
        self.service.activate("memory")
        self.assertEqual(self.service.worker.process.pid, process.pid)
        self.assertEqual(
            list(self.service.stream(self.request()))[0]["profile"], "memory"
        )

    def test_closed_service_cannot_restart(self):
        self.compute()
        self.service.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            list(self.service.stream(self.request()))


if __name__ == "__main__":
    unittest.main()


class FailedTransitionTests(ServiceFixture):
    def test_failed_model_switch_restores_previous_configuration(self):
        self.compute()
        other = self.root / "other-model"
        other.mkdir()
        metadata, tokens, path = self.service.store.load("memory")
        metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"name", "artifact_id"}
        }
        metadata["model"] = str(other)
        self.service.store.publish(
            "bad",
            metadata,
            tokens,
            lambda directory: (directory / "state.safetensors").write_bytes(
                path.read_bytes()
            ),
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            self.service.activate("bad")
        self.assertEqual(self.service.model, str(self.model.resolve()))
        self.assertEqual(self.service.name, "memory")
        self.assertIsNone(self.service.worker)
        self.assertGreater(
            list(self.service.stream(self.request()))[0]["cached_tokens"], 0
        )


class ShutdownTests(ServiceFixture):
    def test_close_interrupts_initial_worker_startup(self):
        import threading
        from fake_backend import SlowStartBackend
        from imprint.worker import WorkerError

        self.service.backend_factory = SlowStartBackend
        errors = []

        def compute():
            try:
                self.compute()
            except (ValueError, WorkerError) as error:
                errors.append(error)

        thread = threading.Thread(target=compute)
        thread.start()
        deadline = time.monotonic() + 2
        while self.service.worker is None and time.monotonic() < deadline:
            time.sleep(0.01)
        process = self.service.worker.process
        started = time.monotonic()
        self.service.close()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(process.is_alive())
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(errors)
