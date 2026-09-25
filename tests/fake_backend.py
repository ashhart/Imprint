import json


class Decoder:
    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = chr(token)

    def finalize(self):
        self.last_segment = ""


class FakeBackend:
    def __init__(self, model):
        self.identity = {"model": model.name, "fake": True}
        self.limit = 100000
        self.calls = []
        self.samples = 0

    def new_cache(self):
        return []

    def tokenize(self, messages, options=None):
        text = (
            "".join(f"<{item['role']}>{item['content']}\n" for item in messages)
            + "<assistant>"
        )
        return list(text.encode("utf-8"))

    def check_capacity(self, count, reserve):
        if count + reserve > self.limit:
            raise ValueError("Context limit exceeded")

    def advance(self, tokens, cache):
        if not tokens:
            raise ValueError("Missing suffix")
        self.calls.append(list(tokens))
        cache.extend(tokens)
        return sum(cache) % 26 + ord("a")

    def sample(self, logits, sampler):
        self.samples += 1
        return logits

    def sampler(self, request):
        return None

    def detokenizer(self):
        result = Decoder()
        result.reset()
        return result

    def is_eos(self, token):
        return False

    def save(self, path, cache):
        path.write_text(json.dumps(cache))

    def restore(self, path, count):
        cache = json.loads(path.read_text())
        if len(cache) != count:
            raise ValueError("Position mismatch")
        return cache


class SlowStartBackend(FakeBackend):
    def __init__(self, model):
        import time

        time.sleep(10)
        super().__init__(model)
