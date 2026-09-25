import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


MODEL_SUFFIXES = {".safetensors", ".json", ".model", ".txt", ".jinja", ".tiktoken"}


def local_model(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir() or not (resolved / "config.json").is_file():
        raise ValueError("--model must be an existing local MLX model directory")
    if not list(resolved.glob("*.safetensors")):
        raise ValueError("The local model directory has no safetensors weights")
    return resolved


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(path):
    path = local_model(path)
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix in MODEL_SUFFIXES:
            files[item.relative_to(path).as_posix()] = file_digest(item)
    return {
        "files": files,
        "engine": "mlx",
        "codec": "afterglow.mlx.v1",
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "machine": platform.machine(),
        "os": platform.mac_ver()[0],
        "position_policy": "native-unmodified",
        "prefill_chunk": 512,
    }


def context_limit(config):
    text = config.get("text_config", config)
    limit = text.get("max_position_embeddings", text.get("n_positions"))
    if type(limit) is not int or limit <= 0:
        raise ValueError("Model config must declare a finite context limit")
    return limit


def read_config(path):
    with (Path(path) / "config.json").open() as stream:
        return json.load(stream)
