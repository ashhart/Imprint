import math
import time
import uuid
from pathlib import Path

from .learning import learning_context, learning_mode
from .recipes import render_messages
from .store import Store


def common_prefix(sequences):
    size = min(map(len, sequences))
    for index in range(size):
        if any(sequence[index] != sequences[0][index] for sequence in sequences[1:]):
            return sequences[0][:index]
    return sequences[0][:size]


def validate_request(request):
    allowed = {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "chat_template_kwargs",
    }
    if not isinstance(request, dict) or set(request) - allowed:
        raise ValueError("Request contains unsupported fields")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Only role and plain-text content are supported")
        if message["role"] not in {
            "system",
            "developer",
            "user",
            "assistant",
        } or not isinstance(message["content"], str):
            raise ValueError("Unsupported message role or content")
    maximum = request.get("max_tokens", 256)
    if type(maximum) is not int or not 1 <= maximum <= 32768:
        raise ValueError("max_tokens must be an integer from 1 to 32768")
    for key, lower, upper, default in (("temperature", 0, 10, 0), ("top_p", 0, 1, 1)):
        value = request.get(key, default)
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or not lower <= value <= upper
        ):
            raise ValueError(f"Invalid {key}")
    if type(request.get("top_k", 0)) is not int or request.get("top_k", 0) < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if "seed" in request and (
        type(request["seed"]) is not int or not 0 <= request["seed"] < 2**32
    ):
        raise ValueError("seed must be an unsigned 32-bit integer")
    options = request.get("chat_template_kwargs", {})
    if (
        not isinstance(options, dict)
        or set(options) - {"enable_thinking"}
        or any(type(value) is not bool for value in options.values())
    ):
        raise ValueError(
            "Only boolean enable_thinking is supported in chat_template_kwargs"
        )
    return maximum


class Runtime:
    def __init__(self, store_root, model, name, learn, backend_factory=None):
        self.store = Store(Path(store_root))
        self.model_path = str(Path(model).expanduser().resolve())
        self.name = name
        self.learn = learning_mode(learn)
        self.session = None
        self.prepared = None
        if backend_factory is None:
            from .mlx_backend import MLXBackend

            backend_factory = MLXBackend
        self.backend = backend_factory(Path(self.model_path))

    def metadata(self, mode, recipe=None):
        return {
            "model": self.model_path,
            "identity": self.backend.identity,
            "mode": mode,
            "recipe": recipe,
        }

    def prefix(self, messages, options, required=True):
        probes = ("Imprint alpha", "Different zebra", "12345")
        rendered = [
            self.backend.tokenize(
                messages + [{"role": "user", "content": probe}], options
            )
            for probe in probes
        ]
        tokens = common_prefix(rendered)
        if required and not tokens:
            raise ValueError("This template has no reusable prefix for these messages")
        return tokens

    def publish(self, name, cache, tokens, metadata):
        return self.store.publish(
            name,
            metadata,
            tokens,
            lambda directory: self.backend.save(directory / "state.safetensors", cache),
        )

    def compute(self, recipe, model, name):
        if str(Path(model).expanduser().resolve()) != self.model_path:
            raise ValueError("Compute model differs from worker model")
        if recipe.get("tools"):
            raise ValueError("Tool templates are not supported by this release")
        messages = render_messages(recipe)
        tokens = self.prefix(messages, recipe.get("template_options", {}))
        self.session = None
        self.prepared = None
        self.backend.check_capacity(
            len(tokens), recipe.get("answer_reserve_tokens", 1024)
        )
        cache = self.backend.new_cache()
        self.backend.advance(tokens, cache)
        result = self.publish(name, cache, tokens, self.metadata("recipe", recipe))
        self.prepared = (result, tokens, None, cache)
        self.name = name
        return result

    def sessions(self):
        if self.session is None:
            return []
        state = self.session
        return [
            {
                "session_id": state["id"],
                "committed_tokens": len(state["tokens"]),
                "absorbed_tokens": state["absorbed"],
                "pending_tokens": len(state["tokens"]) - state["absorbed"],
            }
        ]

    def snapshot(self, session, name, scope="committed"):
        if scope not in {"committed", "absorbed"}:
            raise ValueError("scope must be committed or absorbed")
        state = self.session
        if state is None or state["id"] != session:
            raise ValueError(
                "Session is unavailable or evicted; no prompt was replayed"
            )
        tail = state["tokens"][state["absorbed"] :]
        computed = 0
        if scope == "committed" and tail:
            try:
                self.backend.advance(tail, state["cache"])
            except Exception:
                self.session = None
                raise
            state["absorbed"] = len(state["tokens"])
            computed = len(tail)
        tokens = state["tokens"][: state["absorbed"]]
        result = self.publish(
            name, state["cache"], tokens, self.metadata("continuation")
        )
        return {**result, "tail_tokens_computed": computed, "session_id": session}

    def activate(self, name):
        metadata, tokens, path = self.store.load(name)
        self.compatible(metadata)
        self.session = None
        self.prepared = None
        self.backend.check_capacity(len(tokens), 1)
        cache = self.backend.restore(path, len(tokens))
        self.prepared = (metadata, tokens, path, cache)
        self.name = name
        self.learn = False
        return {"active": name, "token_count": len(tokens)}

    def compatible(self, metadata):
        if metadata.get("identity") != self.backend.identity:
            raise ValueError(
                "Blob identity differs from the loaded model, tokenizer, runtime or cache codec"
            )

    def prepare(self, request):
        messages = request["messages"]
        options = request.get("chat_template_kwargs", {})
        if self.name and request.get("model", self.name) not in {
            self.name,
            "imprint",
        }:
            raise ValueError("Request model must match the active profile")
        saved = None
        if self.name and any(
            profile["name"] == self.name for profile in self.store.profiles()
        ):
            profile = self.store.profile(self.name)
            if (
                self.prepared
                and profile["artifact_id"] == self.prepared[0]["artifact_id"]
            ):
                saved = self.prepared[:3]
            else:
                self.prepared = None
                saved = self.store.load(self.name)
            metadata, _, _ = saved
            self.compatible(metadata)
            recipe = metadata.get("recipe")
            if metadata.get("mode") == "recipe" and recipe:
                fixed = render_messages(recipe)
                if messages[: len(fixed)] != fixed:
                    if any(
                        message["role"] in {"system", "developer"}
                        for message in messages
                    ):
                        raise ValueError(
                            "Recipe requests cannot supply a second system or developer message"
                        )
                    messages = fixed + messages
                configured = recipe.get("template_options", {})
                if options and options != configured:
                    raise ValueError("Request template options differ from the recipe")
                options = configured
        tokens = self.backend.tokenize(messages, options)
        return tokens, saved, options

    def learning_prefix(self, request, tokens, saved, options):
        if not self.name or (saved and saved[0].get("mode") != "captured"):
            return []
        context = learning_context(request["messages"], self.learn)
        if context is None:
            return []
        try:
            prefix = self.prefix(context, options, required=False)
        except ValueError:
            return []
        if len(prefix) >= len(tokens) or tokens[: len(prefix)] != prefix:
            return []
        return prefix

    def stream(self, request):
        started = time.perf_counter()
        maximum = validate_request(request)
        tokens, saved, options = self.prepare(request)
        self.session = None
        self.backend.check_capacity(len(tokens), maximum)
        cache = None
        reused = 0
        if saved:
            _, prefix, path = saved
            if tokens[: len(prefix)] == prefix and len(tokens) > len(prefix):
                cache = (
                    self.prepared[3]
                    if self.prepared
                    else self.backend.restore(path, len(prefix))
                )
                reused = len(prefix)
        self.prepared = None
        if cache is None:
            cache = self.backend.new_cache()
        consumed = reused
        prefix = self.learning_prefix(request, tokens, saved, options)
        if len(prefix) > consumed:
            self.backend.advance(prefix[consumed:], cache)
            self.publish(self.name, cache, prefix, self.metadata("captured"))
            consumed = len(prefix)
        logits = self.backend.advance(tokens[consumed:], cache)
        state = {
            "id": uuid.uuid4().hex,
            "tokens": list(tokens),
            "absorbed": len(tokens),
            "cache": cache,
        }
        self.session = state
        try:
            yield {
                "type": "start",
                "session_id": state["id"],
                "profile": self.name,
                "artifact_id": saved[0]["artifact_id"] if reused else None,
                "cached_tokens": reused,
                "prompt_tokens": len(tokens),
            }
            decoder = self.backend.detokenizer()
            sampler = self.backend.sampler(request)
            generated = 0
            reason = "length"
            first_token_seconds = None
            for index in range(maximum):
                if index:
                    logits = self.backend.advance(
                        state["tokens"][state["absorbed"] :], cache
                    )
                    state["absorbed"] = len(state["tokens"])
                token = self.backend.sample(logits, sampler)
                if self.backend.is_eos(token):
                    reason = "stop"
                    break
                state["tokens"].append(token)
                generated += 1
                decoder.add_token(token)
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - started
                yield {"type": "delta", "text": decoder.last_segment, "token": token}
            decoder.finalize()
            final_segment = decoder.last_segment
            if final_segment:
                yield {"type": "delta", "text": final_segment, "token": None}
            yield {
                "type": "done",
                "session_id": state["id"],
                "finish_reason": reason,
                "prompt_tokens": len(tokens),
                "completion_tokens": generated,
                "total_tokens": len(tokens) + generated,
                "cached_tokens": reused,
                "generation_seconds": time.perf_counter() - started,
                "first_token_seconds": first_token_seconds,
            }
        except BaseException:
            self.session = None
            raise
