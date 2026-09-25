import json
import platform

from .identity import context_limit, local_model, model_identity, read_config


class MLXBackend:
    def __init__(self, model_path):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("The MLX backend requires Apple silicon macOS")
        self.path = local_model(model_path)
        self.config = read_config(self.path)
        self.limit = context_limit(self.config)
        self.identity = model_identity(self.path)
        if self.identity["mlx"] != "0.31.2" or self.identity["mlx_lm"] != "0.31.3":
            raise ValueError(
                "Install imprint-cache[mlx] to use the supported adapter API versions"
            )
        try:
            import psutil
            import mlx.core as mx
            from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import load
        except ImportError as error:
            raise ValueError("Install the MLX extra: pip install '.[mlx]'") from error
        weights = sum(item.stat().st_size for item in self.path.glob("*.safetensors"))
        if weights + 2 * 1024**3 > psutil.virtual_memory().available:
            raise ValueError(
                "Insufficient available memory to load this model with a 2 GiB reserve"
            )
        self.mx = mx
        self.psutil = psutil
        self.make_prompt_cache = make_prompt_cache
        self.make_sampler = make_sampler
        self.classes = {"KVCache": KVCache, "ArraysCache": ArraysCache}
        self.model, self.tokenizer = load(
            str(self.path),
            tokenizer_config={"trust_remote_code": False, "local_files_only": True},
            trust_remote_code=False,
        )
        if not self.tokenizer.has_chat_template:
            raise ValueError("This release requires a model with a chat template")
        self.new_cache()

    def new_cache(self):
        cache = self.make_prompt_cache(self.model)
        if not cache or any(
            type(layer) not in self.classes.values() for layer in cache
        ):
            raise ValueError(
                "This release supports only exact KVCache and ArraysCache layer types"
            )
        return cache

    def tokenize(self, messages, options=None):
        options = dict(options or {})
        unknown = set(options) - {"enable_thinking"}
        if unknown or any(type(value) is not bool for value in options.values()):
            raise ValueError(
                "Only boolean enable_thinking is supported in template options"
            )
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **options
        )
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int for token in tokens)
        ):
            raise ValueError("The chat template did not produce a nonempty token list")
        return tokens

    def check_capacity(self, token_count, reserve):
        if token_count + reserve > self.limit:
            raise ValueError(
                "Prompt plus answer reserve exceeds the model context limit"
            )
        config = self.config.get("text_config", self.config)
        layers = config.get("num_hidden_layers", config.get("n_layer"))
        heads = config.get("num_key_value_heads", config.get("num_attention_heads"))
        dimension = config.get("head_dim")
        if (
            dimension is None
            and config.get("hidden_size")
            and config.get("num_attention_heads")
        ):
            dimension = config["hidden_size"] // config["num_attention_heads"]
        if not all(
            type(value) is int and value > 0 for value in (layers, heads, dimension)
        ):
            raise ValueError(
                "Model config lacks dimensions needed for memory admission"
            )
        required = 2 * layers * heads * dimension * (token_count + reserve) * 4
        if required + 1024**3 > self.psutil.virtual_memory().available:
            raise ValueError(
                "Insufficient available memory for a conservative context allocation"
            )

    def advance(self, tokens, cache):
        if not tokens:
            raise ValueError("At least one unprocessed token is required")
        logits = None
        for offset in range(0, len(tokens), 512):
            batch = self.mx.array(tokens[offset : offset + 512])[None]
            logits = self.model(batch, cache=cache)[:, -1, :]
            self.mx.eval(logits, *self._arrays(cache))
        return logits

    def _arrays(self, cache):
        arrays = []
        for layer in cache:
            arrays.extend(item for item in layer.state if item is not None)
            for name in ("left_padding", "lengths"):
                item = getattr(layer, name, None)
                if item is not None:
                    arrays.append(item)
        return arrays

    def sampler(self, request):
        if "seed" in request:
            self.mx.random.seed(request["seed"])
        return self.make_sampler(
            temp=request.get("temperature", 0.0),
            top_p=request.get("top_p", 1.0),
            top_k=request.get("top_k", 0),
        )

    def sample(self, logits, sampler):
        logprobs = logits - self.mx.logsumexp(logits, axis=-1, keepdims=True)
        return int(sampler(logprobs).item())

    def detokenizer(self):
        decoder = self.tokenizer.detokenizer
        decoder.reset()
        return decoder

    def is_eos(self, token):
        return token in self.tokenizer.eos_token_ids

    def save(self, path, cache):
        arrays = {}

        def encode(value):
            if value is None or isinstance(value, (str, bool, int, float)):
                return value
            if isinstance(value, (list, tuple)):
                return {
                    "sequence": [encode(item) for item in value],
                    "tuple": isinstance(value, tuple),
                }
            if not isinstance(value, self.mx.array):
                raise ValueError("Unsupported cache state value")
            key = str(len(arrays))
            arrays[key] = value
            return {"tensor": key}

        layers = []
        for layer in cache:
            if type(layer) not in self.classes.values():
                raise ValueError("Unsupported cache layer")
            layers.append(
                {
                    "kind": type(layer).__name__,
                    "state": encode(layer.state),
                    "meta": encode(layer.meta_state),
                    "left_padding": encode(getattr(layer, "left_padding", None)),
                    "lengths": encode(getattr(layer, "lengths", None)),
                }
            )
        self.mx.eval(*arrays.values())
        header = json.dumps(
            {"format": "afterglow.mlx.v1", "layers": layers}, allow_nan=False
        )
        self.mx.save_safetensors(str(path), arrays, {"afterglow": header})

    def restore(self, path, count):
        arrays, metadata = self.mx.load(str(path), return_metadata=True)
        header = json.loads(metadata["afterglow"])
        if header.get("format") != "afterglow.mlx.v1":
            raise ValueError("Unsupported state codec")
        expected = self.new_cache()
        layers = header.get("layers", [])
        if len(layers) != len(expected):
            raise ValueError("Stored cache layer count differs from model")

        def decode(value):
            if isinstance(value, dict):
                if set(value) == {"tensor"}:
                    return arrays[value["tensor"]]
                if set(value) == {"sequence", "tuple"}:
                    sequence = [decode(item) for item in value["sequence"]]
                    return tuple(sequence) if value["tuple"] else sequence
                raise ValueError("Invalid cache state descriptor")
            if value is None or isinstance(value, (str, bool, int, float)):
                return value
            raise ValueError("Invalid cache state descriptor")

        cache = []
        for descriptor, template in zip(layers, expected):
            kind = descriptor["kind"]
            if kind != type(template).__name__ or kind not in self.classes:
                raise ValueError("Stored cache layer type differs from model")
            state = decode(descriptor["state"])
            if kind == "KVCache":
                if not isinstance(state, (list, tuple)) or len(state) != 2:
                    raise ValueError("Invalid attention cache shape")
                keys, values = state
                valid = all(
                    isinstance(item, self.mx.array)
                    and item.ndim == 4
                    and item.shape[0] == 1
                    for item in state
                )
                if not valid or keys.shape[:3] != values.shape[:3]:
                    raise ValueError("Invalid attention cache shape")
            layer = self.classes[kind].from_state(state, decode(descriptor["meta"]))
            if kind == "KVCache" and layer.offset != count:
                raise ValueError("Cache position disagrees with saved token count")
            if kind == "ArraysCache":
                if len(layer.state) != len(template.state):
                    raise ValueError("Recurrent cache state size differs from model")
                layer.left_padding = decode(descriptor["left_padding"])
                layer.lengths = decode(descriptor["lengths"])
            cache.append(layer)
        self.mx.eval(*self._arrays(cache))
        return cache
