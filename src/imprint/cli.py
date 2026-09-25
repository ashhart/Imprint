from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys


DEFAULT_STORE = Path.home() / ".cache" / "imprint"


def duration(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", value)
    if not match:
        raise argparse.ArgumentTypeError(
            "Use seconds or a duration such as 30s, 5m or 1h"
        )
    seconds = float(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("Idle duration must be positive and finite")
    return seconds


def name(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise argparse.ArgumentTypeError(
            "Use 1-64 lowercase letters, digits, underscores or hyphens, starting with a letter"
        )
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="imprint",
        description="Save and restore local model context; this release supports plain text without tool parsing",
    )
    result.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE,
        help="Private local artifact directory",
    )
    result.add_argument(
        "--json", action="store_true", help="Write a versioned JSON result"
    )
    commands = result.add_subparsers(dest="command", required=True)

    def command(label, help):
        child = commands.add_parser(label, help=help)
        child.add_argument(
            "--store",
            type=Path,
            default=argparse.SUPPRESS,
            help="Private local artifact directory",
        )
        child.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Write a versioned JSON result",
        )
        return child

    compute = command(
        "compute", "Save context from files, a recipe or a running session"
    )
    inputs = compute.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--files", nargs="+", type=Path, metavar="FILE")
    inputs.add_argument("--recipe", type=Path)
    inputs.add_argument("--session")
    compute.add_argument("--name", required=True, type=name)
    compute.add_argument("--model", type=Path, help="An existing local model directory")
    compute.add_argument("--engine", choices=["mlx"], default="mlx")
    compute.add_argument(
        "--scope", choices=["committed", "absorbed"], default="committed"
    )

    serve = command("serve", "Run the local OpenAI-compatible endpoint")
    serve.add_argument("--name", required=True, type=name)
    serve.add_argument(
        "--model", type=Path, help="Local model directory for a new profile"
    )
    serve.add_argument("--engine", choices=["mlx"], default="mlx")
    serve.add_argument(
        "--learn",
        nargs="?",
        const="first-turn",
        default=False,
        choices=["first-turn", "continuous"],
        help="Capture fresh-turn instructions by default, or conversation history with continuous",
    )
    serve.add_argument("--idle-unload", type=duration, default=300, metavar="DURATION")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8460)

    inspect = command(
        "inspect", "Inspect saved profiles or a running server's sessions"
    )
    inspect.add_argument("name", nargs="?", type=name)
    inspect.add_argument("--sessions", action="store_true")
    use = command("use", "Activate a saved profile on the running server")
    use.add_argument("name", type=name)
    command("sleep", "Unload the running server's model worker")
    return result


def local_model(path: Path | None) -> Path:
    if path is None:
        raise ValueError("Provide --model with an existing local model directory")
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(
            "Model directory does not exist; automatic downloads are disabled"
        )
    return resolved


def emit(value, machine: bool):
    if machine:
        print(json.dumps({"version": 1, "ok": True, "result": value}, allow_nan=False))
    else:
        print(json.dumps(value, indent=2, allow_nan=False))


def main(argv=None, service_factory=None) -> int:
    argument_parser = parser()
    arguments = argument_parser.parse_args(argv)
    store = arguments.store.expanduser().resolve()
    try:
        if arguments.command == "compute" and arguments.session:
            from .http import control_request

            if arguments.model:
                raise ValueError(
                    "Live snapshots inherit the session's model; omit --model"
                )
            result = control_request(
                store,
                "snapshot",
                {
                    "session": arguments.session,
                    "name": arguments.name,
                    "scope": arguments.scope,
                },
            )
        elif arguments.command == "inspect" and arguments.sessions:
            from .http import control_request

            if arguments.name:
                raise ValueError("Choose a profile name or --sessions")
            result = control_request(store, "sessions")
        elif arguments.command in {"sleep", "use"}:
            from .http import control_request

            action = "activate" if arguments.command == "use" else "sleep"
            result = control_request(
                store, action, {"name": arguments.name} if action == "activate" else {}
            )
        else:
            if service_factory is None:
                from .service import Service

                service_factory = Service
            if arguments.command == "compute":
                from .http import StoreLease
                from .recipes import from_files, load_recipe

                if arguments.scope != "committed":
                    raise ValueError("--scope is only valid with --session")
                model = local_model(arguments.model)
                recipe = (
                    from_files(arguments.files, arguments.name)
                    if arguments.files
                    else load_recipe(arguments.recipe)
                )
                busy = "A running Imprint server or computation owns this store; use compute --session for retained state, or stop the server before standalone computation"
                with StoreLease(store, busy):
                    service = service_factory(store)
                    try:
                        result = service.compute(recipe, str(model), arguments.name)
                    finally:
                        service.close()
            elif arguments.command == "inspect":
                service = service_factory(store)
                try:
                    result = service.inspect(arguments.name)
                finally:
                    service.close()
            elif arguments.command == "serve":
                from .http import loopback_host, serve

                host = loopback_host(arguments.host)
                if not 1 <= arguments.port <= 65535:
                    raise ValueError("port must be between 1 and 65535")
                if arguments.learn and not arguments.model:
                    raise ValueError("--learn requires a local --model")
                model = local_model(arguments.model) if arguments.model else None
                service = service_factory(
                    store,
                    model=model,
                    name=arguments.name,
                    learn=arguments.learn,
                    idle_seconds=arguments.idle_unload,
                )
                try:
                    serve(service, store, host, arguments.port)
                finally:
                    service.close()
                return 0
            else:
                raise ValueError("Unsupported command")
        emit(result, arguments.json)
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        code = (
            2
            if isinstance(error, ValueError)
            else 5
            if isinstance(error, OSError)
            else 7
        )
        if arguments.json:
            print(
                json.dumps(
                    {"version": 1, "ok": False, "error": str(error), "exit_code": code}
                )
            )
        else:
            print(f"imprint: {error}", file=sys.stderr)
        return code
