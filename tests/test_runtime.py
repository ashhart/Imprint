import tempfile
import unittest
from pathlib import Path

from imprint.recipes import from_files, render_messages
from imprint.runtime import Runtime, validate_request
from fake_backend import FakeBackend


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.memory = self.root / "memory.md"
        self.memory.write_text("Long-lived reference: Î© project.\r\nSecond line.")
        self.recipe = from_files([self.memory], "memory")
        self.runtime = Runtime(
            self.root / "store", self.model, None, False, FakeBackend
        )

    def tearDown(self):
        self.temporary.cleanup()

    def request(self, messages=None, **extra):
        return {
            "messages": messages or [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
            **extra,
        }

    def compute(self):
        return self.runtime.compute(self.recipe, str(self.model), "memory")


class RuntimeTests(RuntimeFixture):
    def test_saved_recipe_continuation_matches_cold_tokens(self):
        profile = self.compute()
        warm = list(self.runtime.stream(self.request()))
        cold_runtime = Runtime(
            self.root / "other", self.model, None, False, FakeBackend
        )
        messages = render_messages(self.recipe) + self.request()["messages"]
        cold = list(cold_runtime.stream(self.request(messages)))
        self.assertEqual(
            [item["token"] for item in warm if item["type"] == "delta"],
            [item["token"] for item in cold if item["type"] == "delta"],
        )
        self.assertEqual(warm[0]["cached_tokens"], profile["token_count"])
        self.assertGreater(warm[0]["cached_tokens"], 0)
        self.assertLess(
            len(self.runtime.backend.calls[1]), len(self.runtime.backend.calls[0])
        )

    def test_full_recipe_request_is_not_prepended_twice(self):
        self.compute()
        messages = render_messages(self.recipe) + self.request()["messages"]
        output = list(self.runtime.stream(self.request(messages)))
        self.assertEqual(
            output[0]["prompt_tokens"], len(self.runtime.backend.tokenize(messages))
        )

    def test_recipe_conflicting_system_is_rejected(self):
        self.compute()
        with self.assertRaisesRegex(ValueError, "second system"):
            list(
                self.runtime.stream(
                    self.request([{"role": "system", "content": "Different"}])
                )
            )

    def test_absorbed_snapshot_never_runs_forward(self):
        output = list(self.runtime.stream(self.request()))
        session = output[0]["session_id"]
        calls = len(self.runtime.backend.calls)
        saved = self.runtime.snapshot(session, "absorbed", "absorbed")
        self.assertEqual(len(self.runtime.backend.calls), calls)
        self.assertEqual(saved["tail_tokens_computed"], 0)
        self.assertEqual(saved["token_count"], output[-1]["total_tokens"] - 1)
        _, tokens, state = self.runtime.store.load("absorbed")
        self.assertEqual(self.runtime.backend.restore(state, len(tokens)), tokens)

    def test_committed_snapshot_absorbs_only_pending_tail_without_sample(self):
        output = list(self.runtime.stream(self.request()))
        session = output[0]["session_id"]
        sampled = self.runtime.backend.samples
        saved = self.runtime.snapshot(session, "committed")
        self.assertEqual(saved["tail_tokens_computed"], 1)
        self.assertEqual(saved["token_count"], output[-1]["total_tokens"])
        self.assertEqual(self.runtime.backend.samples, sampled)
        self.assertEqual(
            self.runtime.snapshot(session, "again")["tail_tokens_computed"], 0
        )

    def test_old_session_is_not_recomputed(self):
        first = list(self.runtime.stream(self.request()))[0]["session_id"]
        list(self.runtime.stream(self.request()))
        calls = len(self.runtime.backend.calls)
        with self.assertRaisesRegex(ValueError, "evicted"):
            self.runtime.snapshot(first, "gone")
        self.assertEqual(len(self.runtime.backend.calls), calls)

    def test_learn_captures_only_stable_prefix_and_reuses_it(self):
        self.runtime.name = "agent"
        self.runtime.learn = True
        messages = [
            {"role": "system", "content": "Shared instruction"},
            {"role": "user", "content": "private first question"},
        ]
        first = list(self.runtime.stream(self.request(messages)))
        second = list(
            self.runtime.stream(
                self.request(
                    messages[:-1] + [{"role": "user", "content": "another question"}]
                )
            )
        )
        self.assertEqual(first[0]["cached_tokens"], 0)
        self.assertGreater(second[0]["cached_tokens"], 0)
        _, tokens, _ = self.runtime.store.load("agent")
        self.assertNotIn("private", bytes(tokens).decode())

    def test_captured_mismatch_uses_cold_prefill(self):
        self.runtime.name = "agent"
        self.runtime.learn = True
        list(
            self.runtime.stream(
                self.request(
                    [
                        {"role": "system", "content": "Shared"},
                        {"role": "user", "content": "hello"},
                    ]
                )
            )
        )
        changed = list(
            self.runtime.stream(
                self.request(
                    [
                        {"role": "system", "content": "Different"},
                        {"role": "user", "content": "hello"},
                    ]
                )
            )
        )
        self.assertEqual(changed[0]["cached_tokens"], 0)

    def test_wrong_model_identity_is_rejected_before_forward(self):
        self.compute()
        self.runtime.backend.identity = {"other": True}
        calls = len(self.runtime.backend.calls)
        with self.assertRaisesRegex(ValueError, "identity"):
            list(self.runtime.stream(self.request()))
        self.assertEqual(len(self.runtime.backend.calls), calls)

    def test_context_capacity_checked_before_forward(self):
        self.runtime.backend.limit = 5
        with self.assertRaisesRegex(ValueError, "Context"):
            self.compute()
        self.assertEqual(self.runtime.backend.calls, [])

    def test_tools_are_explicitly_rejected(self):
        self.recipe["tools"] = [{"type": "function"}]
        with self.assertRaisesRegex(ValueError, "Tool"):
            self.compute()
        self.assertEqual(self.runtime.backend.calls, [])

    def test_validation_rejects_invalid_sampling_and_media(self):
        for extra in (
            {"temperature": float("nan")},
            {"seed": -1},
            {"max_tokens": True},
            {"tools": []},
            {"top_p": 2},
            {"chat_template_kwargs": {"anything": 1}},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_request(self.request(**extra))

    def test_alias_targets_active_profile(self):
        self.compute()
        events = list(self.runtime.stream(self.request(model="imprint")))
        self.assertEqual(events[0]["profile"], "memory")


if __name__ == "__main__":
    unittest.main()


class FailureAndFlushTests(RuntimeFixture):
    def test_finalize_reads_consuming_segment_once(self):
        class BufferedDecoder:
            def __init__(self):
                self.pending = ""

            @property
            def last_segment(self):
                result, self.pending = self.pending, ""
                return result

            def add_token(self, token):
                pass

            def finalize(self):
                self.pending = "tail"

        self.runtime.backend.detokenizer = BufferedDecoder
        events = list(self.runtime.stream(self.request()))
        self.assertEqual(
            "".join(item["text"] for item in events if item["type"] == "delta"), "tail"
        )

    def test_failed_tail_absorption_evicts_damaged_session(self):
        events = list(self.runtime.stream(self.request()))
        session = events[0]["session_id"]

        def failed_advance(tokens, cache):
            cache.extend(tokens)
            raise RuntimeError("Simulated backend failure")

        self.runtime.backend.advance = failed_advance
        with self.assertRaises(RuntimeError):
            self.runtime.snapshot(session, "failed")
        self.assertEqual(self.runtime.sessions(), [])
        with self.assertRaisesRegex(ValueError, "evicted"):
            self.runtime.snapshot(session, "again")

    def test_activate_consumes_preloaded_state_once(self):
        self.compute()
        original = self.runtime.backend.restore
        calls = []

        def counted(path, count):
            calls.append(count)
            return original(path, count)

        self.runtime.backend.restore = counted
        self.runtime.activate("memory")
        self.assertEqual(len(calls), 1)
        first = list(self.runtime.stream(self.request()))
        self.assertEqual(len(calls), 1)
        second = list(self.runtime.stream(self.request()))
        self.assertEqual(len(calls), 2)
        self.assertEqual(first[0]["cached_tokens"], second[0]["cached_tokens"])


class GenerationFailureTests(RuntimeFixture):
    def test_partial_backend_failure_evicts_damaged_session(self):
        original = self.runtime.backend.advance
        calls = []

        def failed_second_advance(tokens, cache):
            result = original(tokens, cache)
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("Simulated failure after cache mutation")
            return result

        self.runtime.backend.advance = failed_second_advance
        events = self.runtime.stream(self.request(max_tokens=2))
        session = next(events)["session_id"]
        next(events)
        with self.assertRaises(RuntimeError):
            next(events)
        self.assertEqual(self.runtime.sessions(), [])
        with self.assertRaisesRegex(ValueError, "evicted"):
            self.runtime.snapshot(session, "unsafe")
