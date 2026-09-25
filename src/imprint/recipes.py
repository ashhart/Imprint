from __future__ import annotations

from pathlib import Path

from .store import StoreError, _decode, _encode, validate_name


class RecipeError(ValueError):
    pass


DEFAULT_SYSTEM = "Help with the project using the supplied reference material."


def _text(path: Path) -> str:
    try:
        content = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise RecipeError(f"Cannot read UTF-8 memory file {path}: {error}") from error
    if not content:
        raise RecipeError(f"Memory file is empty: {path}")
    return content


def _validate(recipe: dict, base: Path | None = None) -> dict:
    if not isinstance(recipe, dict):
        raise RecipeError("Recipe must be a JSON object")
    allowed = {
        "format",
        "name",
        "blocks",
        "tools",
        "template_options",
        "answer_reserve_tokens",
    }
    if set(recipe) != allowed:
        raise RecipeError(
            "Recipe must contain format, name, blocks, tools, "
            "template_options and answer_reserve_tokens only"
        )
    if recipe["format"] != "afterglow.recipe.v1":
        raise RecipeError("Unsupported recipe format")
    try:
        validate_name(recipe["name"])
        _encode(recipe)
    except StoreError as error:
        raise RecipeError(str(error)) from error
    if not isinstance(recipe["blocks"], list) or not recipe["blocks"]:
        raise RecipeError("Recipe requires at least one message block")
    resolved = []
    for block in recipe["blocks"]:
        if not isinstance(block, dict) or set(block) not in (
            {"role", "text"},
            {"role", "file"},
        ):
            raise RecipeError(
                "Each block requires a role and exactly one of text or file"
            )
        role = block["role"]
        if not isinstance(role, str) or role not in {
            "system",
            "developer",
            "user",
            "assistant",
        }:
            raise RecipeError("Unsupported message role")
        if "file" in block:
            if base is None:
                raise RecipeError(
                    "Resolve file blocks with load_recipe before rendering"
                )
            if not isinstance(block["file"], str) or not block["file"]:
                raise RecipeError("File block requires a nonempty path")
            content = _text(base / Path(block["file"]).expanduser())
        else:
            content = block["text"]
        if not isinstance(content, str) or not content:
            raise RecipeError("Message text must be a nonempty string")
        resolved.append({"role": block["role"], "text": content})
    tools = recipe["tools"]
    if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
        raise RecipeError("Tools must be a list of JSON objects")
    if not isinstance(recipe["template_options"], dict):
        raise RecipeError("Template options must be a JSON object")
    reserve = recipe["answer_reserve_tokens"]
    if type(reserve) is not int or reserve < 1:
        raise RecipeError("answer_reserve_tokens must be a positive integer")
    result = {**recipe, "blocks": resolved}
    return _decode(_encode(result))


def load_recipe(path: Path | str) -> dict:
    path = Path(path).expanduser().absolute()
    try:
        recipe = _decode(path.read_bytes())
    except (OSError, StoreError) as error:
        raise RecipeError(f"Cannot load recipe {path}: {error}") from error
    return _validate(recipe, path.parent)


def from_files(paths: list[Path | str], name: str = "workspace") -> dict:
    paths = list(paths)
    if not paths:
        raise RecipeError("Supply at least one memory file")
    blocks = [{"role": "system", "text": DEFAULT_SYSTEM}]
    blocks.extend(
        {"role": "user", "text": _text(Path(path).expanduser())} for path in paths
    )
    recipe = {
        "format": "afterglow.recipe.v1",
        "name": name,
        "blocks": blocks,
        "tools": [],
        "template_options": {},
        "answer_reserve_tokens": 4096,
    }
    return _validate(recipe)


def render_messages(recipe: dict) -> list[dict]:
    resolved = _validate(recipe)
    return [
        {"role": block["role"], "content": block["text"]}
        for block in resolved["blocks"]
    ]
