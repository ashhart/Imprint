import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fake_backend import FakeBackend
from imprint.recipes import from_files
from imprint.runtime import Runtime
from imprint.service import Service


def request(
    instruction="Shared instruction", history=None, question="private question"
):
    return {
        "messages": [
            {"role": "system", "content": instruction},
            *(history or []),
            {"role": "user", "content": question},
        ],
        "max_tokens": 3,
    }


HISTORY = [
    {"role": "user", "content": "older question"},
    {"role": "assistant", "content": "older answer"},
]


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.runtime = Runtime(
            self.root / "store", self.model, "agent", "first-turn", FakeBackend
        )

    def run_request(self, **kwargs):
        return list(self.runtime.stream(request(**kwargs)))

    def saved(self):
        metadata, tokens, path = self.runtime.store.load("agent")
        self.assertEqual(self.runtime.backend.restore(path, len(tokens)), tokens)
        return metadata, bytes(tokens).decode()

    def assert_cold_output(self, events, incoming):
        cold = Runtime(self.root / "cold", self.model, None, False, FakeBackend)
        expected = list(cold.stream(incoming))
        self.assertEqual(
            [event["token"] for event in events if event["type"] == "delta"],
            [event["token"] for event in expected if event["type"] == "delta"],
        )

    def test_first_turn_updates_changed_instructions_and_reuses_next_fresh_turn(self):
        self.run_request()
        original, _ = self.saved()
        changed = self.run_request(instruction="Changed instruction")
        updated, text = self.saved()
        self.assertNotEqual(original["artifact_id"], updated["artifact_id"])
        self.assertIn("Changed instruction", text)
        self.assertNotIn("private question", text)
        self.assertEqual(changed[0]["cached_tokens"], 0)
        repeated = self.run_request(instruction="Changed instruction", question="new")
        self.assertEqual(repeated[0]["cached_tokens"], updated["token_count"])
        self.assert_cold_output(changed, request(instruction="Changed instruction"))

    def test_first_turn_never_learns_followup_even_before_first_capture(self):
        followup = self.run_request(history=HISTORY)
        self.assertEqual(self.runtime.store.profiles(), [])
        self.assertEqual(followup[0]["cached_tokens"], 0)
        self.run_request()
        saved, _ = self.saved()
        with patch.object(self.runtime, "publish", side_effect=AssertionError("write")):
            reused = self.run_request(history=HISTORY)
            missed = self.run_request(instruction="Different", history=HISTORY)
        self.assertEqual(reused[0]["cached_tokens"], saved["token_count"])
        self.assertEqual(missed[0]["cached_tokens"], 0)
        self.assertEqual(self.saved()[0]["artifact_id"], saved["artifact_id"])

    def test_first_turn_requires_one_final_user_message_and_leading_instructions(self):
        cases = [
            [{"role": "user", "content": "alone"}],
            request()["messages"] + [{"role": "user", "content": "another"}],
            request()["messages"] + [{"role": "assistant", "content": "prefill"}],
            [{"role": "system", "content": "instructions only"}],
        ]
        for messages in cases:
            with self.subTest(messages=messages):
                list(self.runtime.stream({"messages": messages, "max_tokens": 1}))
                self.assertEqual(self.runtime.store.profiles(), [])

    def test_unchanged_prefix_is_not_republished(self):
        self.run_request()
        with patch.object(self.runtime, "publish", side_effect=AssertionError("write")):
            self.run_request(question="different question")

    def test_continuous_captures_history_but_excludes_current_question(self):
        self.runtime.learn = "continuous"
        self.run_request()
        first, _ = self.saved()
        self.runtime.backend.calls.clear()
        followup = self.run_request(history=HISTORY)
        learned, text = self.saved()
        self.assertGreater(learned["token_count"], first["token_count"])
        self.assertIn("older question", text)
        self.assertIn("older answer", text)
        self.assertNotIn("private question", text)
        self.assertEqual(followup[0]["cached_tokens"], first["token_count"])
        self.assertEqual(
            len(self.runtime.backend.calls[0]),
            learned["token_count"] - first["token_count"],
        )
        self.assert_cold_output(followup, request(history=HISTORY))
        with patch.object(self.runtime, "publish", side_effect=AssertionError("write")):
            repeated = self.run_request(history=HISTORY, question="new question")
        self.assertEqual(repeated[0]["cached_tokens"], learned["token_count"])

    def test_continuous_can_learn_assistant_history_without_system_message(self):
        self.runtime.learn = "continuous"
        incoming = {"messages": HISTORY + request()["messages"][-1:], "max_tokens": 2}
        list(self.runtime.stream(incoming))
        _, text = self.saved()
        self.assertIn("older answer", text)
        self.assertNotIn("private question", text)

    def test_continuous_new_conversation_does_not_reuse_or_slice_old_history(self):
        self.runtime.learn = "continuous"
        self.run_request(history=HISTORY)
        old, _ = self.saved()
        fresh = self.run_request(question="new conversation")
        new, text = self.saved()
        self.assertEqual(fresh[0]["cached_tokens"], 0)
        self.assertLess(new["token_count"], old["token_count"])
        self.assertNotIn("older question", text)
        self.assert_cold_output(fresh, request(question="new conversation"))

    def test_learning_off_preserves_profile_on_changed_request(self):
        self.run_request()
        saved, _ = self.saved()
        self.runtime.learn = False
        self.run_request(instruction="Changed")
        self.assertEqual(self.saved()[0]["artifact_id"], saved["artifact_id"])

    def test_recipe_and_explicit_snapshot_are_not_overwritten_by_learning(self):
        memory = self.root / "memory.md"
        memory.write_text("Fixed reference")
        recipe = from_files([memory])
        self.runtime.compute(recipe, self.model, "agent")
        saved, _ = self.saved()
        with patch.object(self.runtime, "publish", side_effect=AssertionError("write")):
            list(self.runtime.stream({"messages": request()["messages"][-1:]}))
        self.assertEqual(self.saved()[0]["artifact_id"], saved["artifact_id"])
        session = self.runtime.sessions()[0]["session_id"]
        self.runtime.snapshot(session, "agent")
        snapshot, _ = self.saved()
        self.run_request(instruction="Changed")
        self.assertEqual(self.saved()[0]["artifact_id"], snapshot["artifact_id"])

    def test_failed_save_keeps_previous_profile_and_no_exportable_session(self):
        self.run_request()
        saved, _ = self.saved()
        with patch.object(
            self.runtime.backend, "save", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_request(instruction="Changed")
        self.assertEqual(self.runtime.sessions(), [])
        self.assertEqual(self.saved()[0]["artifact_id"], saved["artifact_id"])
        self.assertGreater(self.run_request()[0]["cached_tokens"], 0)

    def test_context_dependent_template_prefix_mismatch_skips_learning(self):
        original = self.runtime.backend.tokenize

        def contextual(messages, options=None):
            return [len(messages[-1]["content"])] + original(messages, options)

        self.runtime.backend.tokenize = contextual
        events = self.run_request()
        self.assertEqual(events[0]["cached_tokens"], 0)
        self.assertEqual(self.runtime.store.profiles(), [])

    def test_probe_rejection_does_not_reject_valid_requests(self):
        original = self.runtime.backend.tokenize

        def restricted(messages, options=None):
            if not messages[-1]["content"].startswith("{"):
                raise ValueError("Template requires structured text")
            return original(messages, options)

        self.runtime.backend.tokenize = restricted
        for mode in ("first-turn", "continuous"):
            with self.subTest(mode=mode):
                self.runtime.learn = mode
                events = self.run_request(question='{"question":"hello"}')
                self.assertEqual(events[-1]["type"], "done")
                self.assertEqual(self.runtime.store.profiles(), [])
        with self.assertRaisesRegex(ValueError, "structured text"):
            self.runtime.prefix(request()["messages"][:-1], {})


class LearningServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()

    def service(self, mode):
        service = Service(
            self.root / "store",
            model=self.model,
            name="agent",
            learn=mode,
            backend_factory=FakeBackend,
            idle_seconds=0,
        )
        self.addCleanup(service.close)
        return service

    def test_modes_survive_sleep_and_restart_and_report_status(self):
        for mode in ("first-turn", "continuous"):
            with self.subTest(mode=mode):
                service = self.service(mode)
                list(service.stream(request()))
                service.sleep()
                self.assertEqual(service.inspect()["learning_mode"], mode)
                updated = request(instruction=f"Changed {mode}")
                if mode == "continuous":
                    updated = request(instruction=f"Changed {mode}", history=HISTORY)
                list(service.stream(updated))
                self.assertEqual(service.learn, mode)
                saved = service.store.profile("agent")
                service.sleep()
                warm = list(service.stream(updated))
                self.assertEqual(warm[0]["cached_tokens"], saved["token_count"])
                service.close()

    def test_explicit_use_disables_learning_after_restart(self):
        service = self.service(True)
        self.assertEqual(service.learn, "first-turn")
        list(service.stream(request()))
        saved = service.store.profile("agent")
        service.activate("agent")
        service.sleep()
        self.assertEqual(service.inspect()["learning_mode"], "off")
        list(service.stream(request(instruction="Changed")))
        self.assertEqual(
            service.store.profile("agent")["artifact_id"], saved["artifact_id"]
        )

    def test_invalid_mode_rejected_before_worker_start(self):
        with self.assertRaisesRegex(ValueError, "Learning mode"):
            self.service("sometimes")
