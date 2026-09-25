import json
import tempfile
import unittest
from pathlib import Path

from imprint.recipes import (
    DEFAULT_SYSTEM,
    RecipeError,
    from_files,
    load_recipe,
    render_messages,
)


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.memory = self.root / "memory.md"
        self.content = "  café\r\n雪\n\n"
        self.memory.write_bytes(self.content.encode("utf-8"))
        self.recipe = {
            "format": "afterglow.recipe.v1",
            "name": "work",
            "blocks": [
                {"role": "system", "text": "Be helpful."},
                {"role": "user", "file": "memory.md"},
            ],
            "tools": [],
            "template_options": {},
            "answer_reserve_tokens": 4096,
        }
        self.path = self.root / "recipe.json"

    def write_recipe(self):
        self.path.write_text(json.dumps(self.recipe))
        return load_recipe(self.path)

    def test_load_resolves_relative_file_and_preserves_all_text(self):
        recipe = self.write_recipe()
        self.assertEqual(recipe["blocks"][1], {"role": "user", "text": self.content})
        self.assertEqual(
            render_messages(recipe)[1], {"role": "user", "content": self.content}
        )
        self.assertNotIn("file", recipe["blocks"][1])

    def test_from_files_preserves_order_and_stable_system_message(self):
        second = self.root / "second.txt"
        second.write_text("Second")
        recipe = from_files([self.memory, second], "notes")
        self.assertEqual(recipe["name"], "notes")
        self.assertEqual(
            render_messages(recipe),
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": self.content},
                {"role": "user", "content": "Second"},
            ],
        )

    def test_embedded_text_does_not_read_files(self):
        self.recipe["blocks"] = [{"role": "user", "text": "memory.md"}]
        self.memory.unlink()
        self.assertEqual(
            render_messages(self.write_recipe()),
            [{"role": "user", "content": "memory.md"}],
        )

    def test_tools_and_template_options_are_copied(self):
        tool = {
            "type": "function",
            "function": {"name": "search", "parameters": {"type": "object"}},
        }
        self.recipe["tools"] = [tool]
        self.recipe["template_options"] = {"enable_thinking": False}
        result = self.write_recipe()
        self.assertEqual(result["tools"], [tool])
        self.assertEqual(result["template_options"], {"enable_thinking": False})
        result["tools"][0]["function"]["name"] = "changed"
        self.assertEqual(tool["function"]["name"], "search")

    def test_rejects_duplicate_keys_and_nonfinite_json(self):
        for content in (
            '{"name":"a","name":"b"}',
            '{"number":NaN}',
            '{"number":1e999}',
        ):
            self.path.write_text(content)
            with self.subTest(content=content), self.assertRaises(RecipeError):
                load_recipe(self.path)

    def test_rejects_invalid_block_structure(self):
        for block in (
            {"role": "user", "text": "x", "file": "memory.md"},
            {"role": "user"},
            {"role": "user", "text": []},
            {"role": "tool", "text": "x"},
            {"role": [], "text": "x"},
            {"role": "user", "text": ""},
        ):
            self.recipe["blocks"] = [block]
            with self.subTest(block=block), self.assertRaises(RecipeError):
                self.write_recipe()

    def test_rejects_invalid_top_level_values(self):
        for key, value in (
            ("name", "bad/name"),
            ("blocks", []),
            ("tools", ["bad"]),
            ("template_options", []),
            ("answer_reserve_tokens", True),
            ("answer_reserve_tokens", 0),
            ("format", "future"),
        ):
            original = self.recipe[key]
            self.recipe[key] = value
            with self.subTest(key=key), self.assertRaises(RecipeError):
                self.write_recipe()
            self.recipe[key] = original

    def test_rejects_unknown_top_level_keys(self):
        self.recipe["unexpected"] = True
        with self.assertRaises(RecipeError):
            self.write_recipe()

    def test_rejects_unresolved_file_blocks_at_render_boundary(self):
        with self.assertRaisesRegex(RecipeError, "Resolve file"):
            render_messages(self.recipe)

    def test_rejects_missing_empty_and_non_utf8_memory(self):
        self.memory.unlink()
        for payload in (None, b"", b"\xff"):
            if payload is not None:
                self.memory.write_bytes(payload)
            with self.subTest(payload=payload), self.assertRaises(RecipeError):
                from_files([self.memory])

    def test_requires_at_least_one_file(self):
        with self.assertRaises(RecipeError):
            from_files([])

    def test_only_text_messages_reach_renderer(self):
        self.recipe["blocks"] = [
            {"role": "user", "text": [{"type": "image_url", "image_url": "a"}]}
        ]
        with self.assertRaises(RecipeError):
            self.write_recipe()
