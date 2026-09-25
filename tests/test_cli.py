import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imprint.cli import main, parser
from imprint.http import Registration


class FakeService:
    instances = []
    fail = False

    def __init__(self, root, **kwargs):
        self.root = root
        self.kwargs = kwargs
        self.closed = False
        self.calls = []
        self.instances.append(self)

    def close(self):
        self.closed = True

    def inspect(self, name=None):
        return {"profiles": [], "name": name}

    def compute(self, recipe, model, name):
        self.calls.append((recipe, model, name))
        if self.fail:
            raise RuntimeError("Owned worker failed")
        return {"name": name, "token_count": 23, "worker_running": False}


class CliTests(unittest.TestCase):
    def setUp(self):
        FakeService.instances = []
        FakeService.fail = False
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.memory = self.root / "memory.md"
        self.memory.write_text("Project facts")

    def run_cli(self, args):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main(
                ["--store", str(self.root / "store"), *args],
                service_factory=FakeService,
            )
        return code, output.getvalue(), errors.getvalue()

    def test_compute_files_invokes_service_and_closes_owned_worker(self):
        code, output, errors = self.run_cli(
            [
                "compute",
                "--files",
                str(self.memory),
                "--model",
                str(self.model),
                "--name",
                "work",
                "--json",
            ]
        )
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output)["result"]["token_count"], 23)
        service = FakeService.instances[0]
        self.assertEqual(service.calls[0][0]["blocks"][-1]["text"], "Project facts")
        self.assertEqual(service.calls[0][1:], (str(self.model.resolve()), "work"))
        self.assertTrue(service.closed)

    def test_compute_failure_still_closes_owned_worker(self):
        FakeService.fail = True
        code, output, errors = self.run_cli(
            [
                "compute",
                "--files",
                str(self.memory),
                "--model",
                str(self.model),
                "--name",
                "work",
            ]
        )
        self.assertEqual(code, 7)
        self.assertTrue(FakeService.instances[0].closed)

    def test_standalone_compute_refuses_store_owned_by_server(self):
        with Registration(self.root / "store", "http://127.0.0.1:8460", "test-token"):
            code, output, errors = self.run_cli(
                [
                    "compute",
                    "--files",
                    str(self.memory),
                    "--model",
                    str(self.model),
                    "--name",
                    "work",
                ]
            )
        self.assertEqual(code, 7)
        self.assertIn("compute --session", errors)
        self.assertIn("stop the server", errors)
        self.assertEqual(FakeService.instances, [])

    def test_session_snapshot_uses_control_and_never_starts_service(self):
        with patch(
            "imprint.http.control_request", return_value={"name": "saved"}
        ) as request:
            code, output, errors = self.run_cli(
                [
                    "compute",
                    "--session",
                    "session-1",
                    "--name",
                    "saved",
                    "--scope",
                    "absorbed",
                ]
            )
        self.assertEqual(code, 0, errors)
        self.assertEqual(FakeService.instances, [])
        self.assertEqual(
            request.call_args.args[1:],
            (
                "snapshot",
                {
                    "session": "session-1",
                    "name": "saved",
                    "scope": "absorbed",
                },
            ),
        )

    def test_live_snapshot_rejects_model_override(self):
        code, output, errors = self.run_cli(
            [
                "compute",
                "--session",
                "session-1",
                "--name",
                "saved",
                "--model",
                str(self.model),
            ]
        )
        self.assertEqual(code, 2)
        self.assertEqual(FakeService.instances, [])

    def test_inspect_sessions_uses_control(self):
        with patch(
            "imprint.http.control_request", return_value={"sessions": []}
        ) as request:
            code, output, errors = self.run_cli(["inspect", "--sessions"])
        self.assertEqual(code, 0, errors)
        self.assertEqual(request.call_args.args[1], "sessions")
        self.assertEqual(FakeService.instances, [])

    def test_missing_model_is_rejected_before_worker_creation(self):
        code, output, errors = self.run_cli(
            ["compute", "--files", str(self.memory), "--name", "work"]
        )
        self.assertEqual(code, 2)
        self.assertEqual(FakeService.instances, [])

    def test_no_model_download_for_nonlocal_model(self):
        code, output, errors = self.run_cli(
            [
                "compute",
                "--files",
                str(self.memory),
                "--name",
                "work",
                "--model",
                "not-a-local-model",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("automatic downloads are disabled", errors)
        self.assertEqual(FakeService.instances, [])

    def test_serve_validates_and_always_closes(self):
        with patch("imprint.http.serve") as serve:
            code, output, errors = self.run_cli(
                [
                    "serve",
                    "--name",
                    "agent",
                    "--model",
                    str(self.model),
                    "--learn",
                    "--idle-unload",
                    "2m",
                ]
            )
        self.assertEqual(code, 0, errors)
        service = FakeService.instances[0]
        self.assertEqual(service.kwargs["idle_seconds"], 120)
        self.assertEqual(service.kwargs["learn"], "first-turn")
        self.assertTrue(service.closed)
        self.assertEqual(serve.call_count, 1)

    def test_parser_learning_modes(self):
        cases = (
            ([], False),
            (["--learn"], "first-turn"),
            (["--learn", "first-turn"], "first-turn"),
            (["--learn", "continuous"], "continuous"),
        )
        for options, expected in cases:
            with self.subTest(options=options):
                arguments = parser().parse_args(["serve", "--name", "agent", *options])
                self.assertEqual(arguments.learn, expected)
                if not options:
                    self.assertIs(arguments.learn, False)

    def test_serve_passes_explicit_learning_mode_to_service(self):
        cases = (
            ([], False),
            (["--learn", "first-turn"], "first-turn"),
            (["--learn", "continuous"], "continuous"),
        )
        for options, expected in cases:
            with self.subTest(options=options), patch("imprint.http.serve"):
                code, output, errors = self.run_cli(
                    [
                        "serve",
                        "--name",
                        "agent",
                        "--model",
                        str(self.model),
                        *options,
                    ]
                )
                self.assertEqual(code, 0, errors)
                self.assertEqual(FakeService.instances[-1].kwargs["learn"], expected)
                self.assertTrue(FakeService.instances[-1].closed)

    def test_learning_modes_require_local_model(self):
        for options in (
            ["--learn"],
            ["--learn", "first-turn"],
            ["--learn", "continuous"],
        ):
            with self.subTest(options=options):
                code, output, errors = self.run_cli(
                    ["serve", "--name", "agent", *options]
                )
                self.assertEqual(code, 2)
                self.assertIn("--learn requires a local --model", errors)
                self.assertEqual(FakeService.instances, [])

    def test_parser_rejects_unknown_learning_mode(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            parser().parse_args(["serve", "--name", "agent", "--learn", "everything"])
        self.assertEqual(raised.exception.code, 2)

    def test_serve_rejects_public_binding_before_service_creation(self):
        code, output, errors = self.run_cli(
            ["serve", "--name", "agent", "--host", "0.0.0.0"]
        )
        self.assertEqual(code, 2)
        self.assertEqual(FakeService.instances, [])

    def test_parser_requires_one_input_and_safe_profile_name(self):
        scenarios = [
            ["compute", "--name", "work"],
            ["compute", "--session", "one", "--files", "two", "--name", "work"],
            ["compute", "--session", "one", "--name", "../bad"],
        ]
        for arguments in scenarios:
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    parser().parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)

    def test_global_options_work_before_and_after_subcommand(self):
        before = parser().parse_args(["--json", "--store", "a", "inspect"])
        after = parser().parse_args(["inspect", "--json", "--store", "a"])
        self.assertEqual(vars(before), vars(after))

    def test_help_never_imports_neural_runtime(self):
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from imprint.cli import parser; parser().format_help(); "
            "assert not any(name == 'mlx' or name.startswith('mlx.') or name == 'mlx_lm' "
            "or name.startswith('mlx_lm.') for name in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(ROOT / "src")],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
