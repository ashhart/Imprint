from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Callable


class StoreError(ValueError):
    pass


def validate_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) is None
    ):
        raise StoreError(
            "Name must start with a lowercase letter and contain 1–64 "
            "lowercase letters, digits, underscores or hyphens"
        )
    return name


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise StoreError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise StoreError(f"Non-finite JSON number: {value}")


def _validate_json(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StoreError("JSON numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise StoreError("JSON object keys must be strings")
            _validate_json(item)
        return
    raise StoreError(f"Unsupported JSON value: {type(value).__name__}")


def _encode(value: object) -> bytes:
    try:
        _validate_json(value)
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise StoreError(f"Invalid JSON value: {error}") from error


def _decode(data: bytes) -> object:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant
        )
        _validate_json(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as error:
        raise StoreError(f"Invalid JSON: {error}") from error


def _read(path: Path, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_size > maximum:
                raise StoreError(f"Invalid file or oversized JSON: {path.name}")
            data = stream.read(maximum + 1)
            if len(data) > maximum:
                raise StoreError(f"Oversized JSON: {path.name}")
            return data
    except OSError as error:
        raise StoreError(f"Cannot read {path.name}: {error.strerror}") from error


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _directory(path: Path) -> None:
    if path.is_symlink():
        raise StoreError(f"Storage directory cannot be a symbolic link: {path.name}")
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not path.is_dir():
            raise StoreError(f"Storage path is not a directory: {path.name}")
        path.chmod(0o700)
    except OSError as error:
        raise StoreError(
            f"Cannot create storage directory {path}: {error.strerror}"
        ) from error


def _payload(path: Path, sync: bool = False) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size == 0
            ):
                raise StoreError(
                    f"Payload must be a nonempty regular file with no hard links: {path.name}"
                )
            if sync:
                os.fchmod(stream.fileno(), 0o600)
            digest = hashlib.sha256()
            size = 0
            while block := stream.read(1024 * 1024):
                size += len(block)
                digest.update(block)
            after = os.fstat(stream.fileno())
            unchanged = (after.st_size, after.st_mtime_ns) == (
                before.st_size,
                before.st_mtime_ns,
            )
            if size != before.st_size or not unchanged:
                raise StoreError(f"Payload changed while reading: {path.name}")
            if sync:
                os.fsync(stream.fileno())
            return {"name": path.name, "bytes": size, "sha256": digest.hexdigest()}
    except OSError as error:
        raise StoreError(
            f"Cannot read payload {path.name}: {error.strerror}"
        ) from error


def _tokens(tokens: object) -> list[int]:
    if not isinstance(tokens, list) or not tokens:
        raise StoreError("A snapshot requires a nonempty token list")
    if any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in tokens):
        raise StoreError("Tokens must be unsigned 32-bit integers")
    return tokens


class Store:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().absolute()
        self.artifacts = self.root / "artifacts"
        self.profile_directory = self.root / "profiles"
        for directory in (self.root, self.artifacts, self.profile_directory):
            _directory(directory)

    def _check_directories(self) -> None:
        for path in (self.root, self.artifacts, self.profile_directory):
            if path.is_symlink() or not path.is_dir():
                raise StoreError(
                    f"Storage directory changed or disappeared: {path.name}"
                )

    def _manifest(self, artifact_id: str) -> tuple[dict, Path]:
        if (
            not isinstance(artifact_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_id) is None
        ):
            raise StoreError("Invalid artifact identifier")
        directory = self.artifacts / artifact_id
        if directory.is_symlink() or not directory.is_dir():
            raise StoreError("Artifact directory is missing or is a symbolic link")
        manifest = _decode(_read(directory / "manifest.json", 16 * 1024 * 1024))
        if not isinstance(manifest, dict) or set(manifest) != {
            "format",
            "metadata",
            "payloads",
        }:
            raise StoreError("Invalid artifact manifest")
        if manifest["format"] != "afterglow.store.v1":
            raise StoreError("Unsupported artifact format")
        if hashlib.sha256(_encode(manifest)).hexdigest() != artifact_id:
            raise StoreError("Artifact manifest checksum mismatch")
        metadata = manifest["metadata"]
        if (
            not isinstance(metadata, dict)
            or type(metadata.get("token_count")) is not int
            or metadata["token_count"] < 1
        ):
            raise StoreError("Invalid artifact metadata")
        if "name" in metadata or "artifact_id" in metadata:
            raise StoreError("Artifact metadata contains reserved fields")
        payloads = manifest["payloads"]
        if not isinstance(payloads, list) or len(payloads) != 2:
            raise StoreError("Artifact must contain exactly two payloads")
        expected = {"tokens.json", "state.safetensors"}
        for payload in payloads:
            if not isinstance(payload, dict) or set(payload) != {
                "name",
                "bytes",
                "sha256",
            }:
                raise StoreError("Invalid payload descriptor")
            name = payload["name"]
            if not isinstance(name, str) or name not in expected:
                raise StoreError("Invalid or duplicate payload path")
            expected.remove(name)
            if type(payload["bytes"]) is not int or payload["bytes"] < 1:
                raise StoreError("Invalid payload size")
            checksum = payload["sha256"]
            if (
                not isinstance(checksum, str)
                or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            ):
                raise StoreError("Invalid payload checksum")
            try:
                details = (directory / name).lstat()
            except OSError as error:
                raise StoreError(f"Missing payload: {name}") from error
            if not stat.S_ISREG(details.st_mode) or details.st_size != payload["bytes"]:
                raise StoreError(f"Payload size or type mismatch: {name}")
        return manifest, directory

    def _resolve(self, name: str) -> tuple[str, dict, Path]:
        self._check_directories()
        validate_name(name)
        pointer = _decode(_read(self.profile_directory / f"{name}.json", 65536))
        if (
            not isinstance(pointer, dict)
            or set(pointer) != {"format", "artifact_id"}
            or pointer["format"] != "afterglow.profile.v1"
        ):
            raise StoreError("Invalid profile pointer")
        artifact_id = pointer["artifact_id"]
        manifest, directory = self._manifest(artifact_id)
        return artifact_id, manifest, directory

    def profile(self, name: str) -> dict:
        artifact_id, manifest, _ = self._resolve(name)
        return {**manifest["metadata"], "name": name, "artifact_id": artifact_id}

    def profiles(self) -> list[dict]:
        self._check_directories()
        return [
            self.profile(path.stem)
            for path in sorted(self.profile_directory.glob("*.json"))
        ]

    def load(self, name: str) -> tuple[dict, list[int], Path]:
        artifact_id, manifest, directory = self._resolve(name)
        for expected in manifest["payloads"]:
            if _payload(directory / expected["name"]) != expected:
                raise StoreError(f"Payload checksum mismatch: {expected['name']}")
        tokens = _tokens(_decode(_read(directory / "tokens.json", 128 * 1024 * 1024)))
        if len(tokens) != manifest["metadata"]["token_count"]:
            raise StoreError("Token count does not match artifact metadata")
        metadata = {**manifest["metadata"], "name": name, "artifact_id": artifact_id}
        return metadata, tokens, directory / "state.safetensors"

    def publish(
        self,
        name: str,
        metadata: dict,
        tokens: list[int],
        writer: Callable[[Path], None],
    ) -> dict:
        self._check_directories()
        validate_name(name)
        tokens = list(_tokens(tokens))
        if not isinstance(metadata, dict):
            raise StoreError("Artifact metadata must be a JSON object")
        metadata = _decode(_encode(metadata))
        if "name" in metadata or "artifact_id" in metadata:
            raise StoreError("Metadata fields name and artifact_id are reserved")
        mode = metadata.get("mode", "recipe")
        if not isinstance(mode, str) or mode not in {
            "recipe",
            "captured",
            "continuation",
        }:
            raise StoreError("Unsupported request mode")
        count = metadata.get("token_count", len(tokens))
        if type(count) is not int or count != len(tokens):
            raise StoreError("Metadata token_count does not match tokens")
        metadata["token_count"] = len(tokens)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.artifacts))
        pointer_temp = None
        try:
            writer(staging)
            if {path.name for path in staging.iterdir()} != {"state.safetensors"}:
                raise StoreError("State writer must write only state.safetensors")
            state = _payload(staging / "state.safetensors", sync=True)
            token_data = _encode(tokens)
            if len(token_data) > 128 * 1024 * 1024:
                raise StoreError("Token list is too large")
            _write(staging / "tokens.json", token_data)
            manifest = {
                "format": "afterglow.store.v1",
                "metadata": metadata,
                "payloads": [_payload(staging / "tokens.json"), state],
            }
            manifest_data = _encode(manifest)
            if len(manifest_data) > 16 * 1024 * 1024:
                raise StoreError("Artifact metadata is too large")
            artifact_id = hashlib.sha256(manifest_data).hexdigest()
            _write(staging / "manifest.json", manifest_data)
            _sync_directory(staging)
            destination = self.artifacts / artifact_id
            try:
                os.rename(staging, destination)
            except OSError:
                if not destination.exists():
                    raise
                existing, directory = self._manifest(artifact_id)
                valid = all(
                    _payload(directory / item["name"]) == item
                    for item in existing["payloads"]
                )
                if existing != manifest or not valid:
                    raise StoreError("Existing immutable artifact is corrupt")
                shutil.rmtree(staging)
            _sync_directory(self.artifacts)
            pointer = {"format": "afterglow.profile.v1", "artifact_id": artifact_id}
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{name}-", dir=self.profile_directory
            )
            pointer_temp = Path(temporary)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_encode(pointer))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pointer_temp, self.profile_directory / f"{name}.json")
            pointer_temp = None
            _sync_directory(self.profile_directory)
            return {**metadata, "name": name, "artifact_id": artifact_id}
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            if pointer_temp is not None:
                pointer_temp.unlink(missing_ok=True)
