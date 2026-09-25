import json
import os
import stat
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from imprint.store import Store, StoreError


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "store"
        self.store = Store(self.root)
        self.metadata = {
            "model": "/models/local",
            "identity": {"digest": "test"},
            "mode": "recipe",
        }

    def publish(self, name="workspace", state=b"state", tokens=None, metadata=None):
        return self.store.publish(
            name,
            metadata or self.metadata,
            tokens or [1, 2, 3],
            lambda directory: (directory / "state.safetensors").write_bytes(state),
        )

    def test_roundtrip_and_private_permissions(self):
        published = self.publish()
        restored, tokens, state = self.store.load("workspace")
        self.assertEqual(published, restored)
        self.assertEqual(tokens, [1, 2, 3])
        self.assertEqual(state.read_bytes(), b"state")
        self.assertEqual(self.store.profiles(), [published])
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        for path in state.parent.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(
            stat.S_IMODE(
                (self.store.profile_directory / "workspace.json").stat().st_mode
            ),
            0o600,
        )

    def test_update_retains_immutable_previous_artifact(self):
        first = self.publish()
        second = self.publish(state=b"new state")
        self.assertNotEqual(first["artifact_id"], second["artifact_id"])
        self.assertEqual(
            (
                self.store.artifacts / first["artifact_id"] / "state.safetensors"
            ).read_bytes(),
            b"state",
        )
        self.assertEqual(self.store.load("workspace")[2].read_bytes(), b"new state")

    def test_identical_content_reuses_artifact(self):
        first = self.publish()
        second = self.publish("second")
        self.assertEqual(first["artifact_id"], second["artifact_id"])
        self.assertEqual(len(list(self.store.artifacts.iterdir())), 1)

    def test_failed_writer_preserves_previous_profile(self):
        first = self.publish()

        def fail(directory):
            (directory / "state.safetensors").write_bytes(b"partial")
            raise RuntimeError("interrupted write")

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.store.publish("workspace", self.metadata, [8], fail)
        self.assertEqual(self.store.profile("workspace"), first)
        self.assertFalse(list(self.store.artifacts.glob(".staging-*")))

    def test_failed_pointer_replace_preserves_previous_profile(self):
        first = self.publish()
        with patch("imprint.store.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.publish(state=b"replacement")
        self.assertEqual(self.store.profile("workspace"), first)
        self.assertEqual(len(list(self.store.profile_directory.iterdir())), 1)

    def test_rejects_invalid_names_before_writer(self):
        for name in ("../outside", "a/b", "Caps", "a\n", "", "a" * 65, None, 12):
            with self.subTest(name=name), self.assertRaises(StoreError):
                self.publish(name=name)
        self.assertEqual(list(self.store.artifacts.iterdir()), [])

    def test_rejects_invalid_tokens(self):
        for tokens in ([], [True], [-1], [2**32], [1.5], ["1"], (1, 2)):
            with self.subTest(tokens=tokens), self.assertRaises(StoreError):
                self.store.publish("work", self.metadata, tokens, lambda path: None)

    def test_metadata_token_count_must_match(self):
        for count in (4, True, 3.0):
            with self.subTest(count=count), self.assertRaises(StoreError):
                self.publish(metadata={**self.metadata, "token_count": count})

    def test_metadata_must_be_plain_finite_json(self):
        for bad in (float("nan"), float("inf"), {1: "a"}, Path("x")):
            with self.subTest(bad=bad), self.assertRaises(StoreError):
                self.publish(metadata={**self.metadata, "bad": bad})

    def test_rejects_reserved_metadata_fields(self):
        for key in ("name", "artifact_id"):
            with self.subTest(key=key), self.assertRaises(StoreError):
                self.publish(metadata={**self.metadata, key: "x"})

    def test_payload_checksum_detects_same_size_corruption(self):
        self.publish()
        state = self.store.load("workspace")[2]
        state.write_bytes(b"other")
        with self.assertRaisesRegex(StoreError, "checksum"):
            self.store.load("workspace")

    def test_payload_size_detects_truncation(self):
        self.publish()
        state = self.store.load("workspace")[2]
        state.write_bytes(b"x")
        with self.assertRaisesRegex(StoreError, "size"):
            self.store.load("workspace")

    def test_manifest_checksum_protects_metadata(self):
        self.publish()
        state = self.store.load("workspace")[2]
        manifest = state.parent / "manifest.json"
        content = json.loads(manifest.read_text())
        content["metadata"]["model"] = "/wrong/model"
        manifest.write_text(json.dumps(content))
        with self.assertRaisesRegex(StoreError, "manifest checksum"):
            self.store.load("workspace")

    def test_pointer_rejects_duplicates_nonfinite_and_traversal(self):
        self.publish()
        pointer = self.store.profile_directory / "workspace.json"
        for content in (
            '{"format":"afterglow.profile.v1","artifact_id":"../outside"}',
            '{"format":"afterglow.profile.v1","artifact_id":NaN}',
            '{"format":"a","format":"b","artifact_id":"x"}',
        ):
            with self.subTest(content=content):
                pointer.write_text(content)
                with self.assertRaises(StoreError):
                    self.store.load("workspace")

    def test_symlink_state_rejected_without_chmod_target(self):
        target = self.root / "outside"
        target.write_bytes(b"private")
        target.chmod(0o644)
        with self.assertRaises(StoreError):
            self.store.publish(
                "work",
                self.metadata,
                [1],
                lambda path: (path / "state.safetensors").symlink_to(target),
            )
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_symlink_payload_and_profile_rejected_on_load(self):
        self.publish()
        state = self.store.load("workspace")[2]
        target = self.root / "external"
        target.write_bytes(state.read_bytes())
        state.unlink()
        state.symlink_to(target)
        with self.assertRaises(StoreError):
            self.store.load("workspace")
        pointer = self.store.profile_directory / "workspace.json"
        pointer.unlink()
        pointer.symlink_to(target)
        with self.assertRaises(StoreError):
            self.store.profile("workspace")

    def test_writer_cannot_package_extra_files(self):
        def writer(path):
            (path / "state.safetensors").write_bytes(b"state")
            (path / "weights.safetensors").write_bytes(b"weights")

        with self.assertRaisesRegex(StoreError, "only state"):
            self.store.publish("work", self.metadata, [1], writer)
        self.assertEqual(list(self.store.artifacts.iterdir()), [])

    def test_empty_or_missing_state_rejected(self):
        for writer in (
            lambda path: None,
            lambda path: (path / "state.safetensors").write_bytes(b""),
        ):
            with self.subTest(writer=writer), self.assertRaises(StoreError):
                self.store.publish("work", self.metadata, [1], writer)

    def test_concurrent_publications_are_whole_generations(self):
        def publish(index):
            return self.publish(state=f"state-{index}".encode(), tokens=[index])

        with ThreadPoolExecutor(max_workers=4) as pool:
            generations = list(pool.map(publish, range(8)))
        metadata, tokens, state = self.store.load("workspace")
        self.assertIn(metadata, generations)
        self.assertEqual(state.read_bytes(), f"state-{tokens[0]}".encode())
        self.assertFalse(list(self.store.artifacts.glob(".staging-*")))

    def test_concurrent_identical_publications_deduplicate(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            profiles = list(
                pool.map(lambda index: self.publish(f"work-{index}"), range(8))
            )
        self.assertEqual(len({profile["artifact_id"] for profile in profiles}), 1)
        self.assertEqual(len(self.store.profiles()), 8)

    def test_corrupt_existing_artifact_is_not_republished(self):
        self.publish()
        state = self.store.load("workspace")[2]
        state.write_bytes(b"other")
        with self.assertRaisesRegex(StoreError, "corrupt"):
            self.publish()

    def test_storage_symlink_is_rejected(self):
        alias = Path(self.temporary.name) / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(StoreError):
            Store(alias)

    def test_invalid_mode_and_cyclic_metadata_have_clear_errors(self):
        for mode in ([], {}, None, "unsupported"):
            with self.subTest(mode=mode), self.assertRaises(StoreError):
                self.publish(metadata={**self.metadata, "mode": mode})
        cycle = {}
        cycle["self"] = cycle
        with self.assertRaises(StoreError):
            self.publish(metadata=cycle)

    def test_hardlinked_state_cannot_modify_external_file_permissions(self):
        target = self.root / "external"
        target.write_bytes(b"weights")
        target.chmod(0o644)
        with self.assertRaises(StoreError):
            self.store.publish(
                "work",
                self.metadata,
                [1],
                lambda path: os.link(target, path / "state.safetensors"),
            )
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_replaced_storage_directory_is_rejected(self):
        self.publish()
        original = self.root / "original-artifacts"
        self.store.artifacts.rename(original)
        self.store.artifacts.symlink_to(original, target_is_directory=True)
        with self.assertRaises(StoreError):
            self.store.load("workspace")
