import copy
import json
import unittest
from pathlib import Path

from imprint.mlx_backend import MLXBackend


class Array:
    def __init__(self, shape, dtype="float16"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype


class KVCache:
    def __init__(self):
        self.state = []
        self.offset = 0
        self.meta_state = ()

    @classmethod
    def from_state(cls, state, meta):
        layer = cls()
        layer.state = state
        layer.meta_state = meta
        layer.offset = state[0].shape[2]
        return layer


class ArraysCache:
    def __init__(self):
        self.state = [None, None]
        self.meta_state = ()
        self.left_padding = None
        self.lengths = None

    @classmethod
    def from_state(cls, state, meta):
        layer = cls()
        layer.state = state
        layer.meta_state = meta
        return layer


class ArrayIO:
    array = Array

    def __init__(self):
        self.arrays = None
        self.metadata = None
        self.evaluated = []

    def save_safetensors(self, path, arrays, metadata):
        self.arrays = copy.deepcopy(arrays)
        self.metadata = copy.deepcopy(metadata)

    def load(self, path, return_metadata=False):
        return copy.deepcopy(self.arrays), copy.deepcopy(self.metadata)

    def eval(self, *arrays):
        self.evaluated.extend(arrays)


class CodecTests(unittest.TestCase):
    def setUp(self):
        self.backend = MLXBackend.__new__(MLXBackend)
        self.backend.mx = ArrayIO()
        self.backend.classes = {"KVCache": KVCache, "ArraysCache": ArraysCache}
        self.backend.model = object()
        self.backend.make_prompt_cache = lambda model: [KVCache(), ArraysCache()]
        self.kv = KVCache.from_state((Array((1, 2, 7, 4)), Array((1, 2, 7, 4))), ())
        self.recurrent = ArraysCache.from_state(
            [Array((1, 8, 4)), Array((1, 4, 8))], ()
        )
        self.recurrent.left_padding = Array((1,), "int32")
        self.recurrent.lengths = Array((1,), "int32")
        self.path = Path("unused-by-array-io.safetensors")

    def save(self):
        self.backend.save(self.path, [self.kv, self.recurrent])

    def mutate(self, update):
        header = json.loads(self.backend.mx.metadata["afterglow"])
        update(header)
        self.backend.mx.metadata["afterglow"] = json.dumps(header)

    def test_complete_hybrid_state_roundtrip_and_eager_materialization(self):
        self.save()
        self.backend.mx.evaluated.clear()
        layers = self.backend.restore(self.path, 7)
        self.assertEqual(layers[0].offset, 7)
        self.assertIsInstance(layers[0].state, tuple)
        self.assertIsInstance(layers[1].state, list)
        self.assertEqual(layers[1].state[0].shape, (1, 8, 4))
        self.assertEqual(layers[1].lengths.dtype, "int32")
        self.assertEqual(layers[1].left_padding.shape, (1,))
        self.assertEqual(len(self.backend.mx.evaluated), 6)

    def test_cache_positions_must_match_saved_tokens(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "position"):
            self.backend.restore(self.path, 8)

    def test_arbitrary_cache_class_cannot_be_deserialized(self):
        self.save()
        self.mutate(lambda header: header["layers"][0].update(kind="ArbitraryCode"))
        with self.assertRaisesRegex(ValueError, "type"):
            self.backend.restore(self.path, 7)

    def test_recurrent_state_size_is_checked(self):
        self.save()
        self.mutate(lambda header: header["layers"][1]["state"]["sequence"].pop())
        with self.assertRaisesRegex(ValueError, "size"):
            self.backend.restore(self.path, 7)

    def test_layer_count_is_checked(self):
        self.save()
        self.mutate(lambda header: header["layers"].pop())
        with self.assertRaisesRegex(ValueError, "count"):
            self.backend.restore(self.path, 7)

    def test_unknown_codec_is_rejected(self):
        self.save()
        self.mutate(lambda header: header.update(format="future-codec"))
        with self.assertRaisesRegex(ValueError, "codec"):
            self.backend.restore(self.path, 7)

    def test_value_position_must_match_keys(self):
        self.kv.state = (Array((1, 2, 7, 4)), Array((1, 2, 6, 4)))
        self.save()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.backend.restore(self.path, 7)

    def test_batch_size_greater_than_one_is_rejected(self):
        self.kv.state = (Array((2, 2, 7, 4)), Array((2, 2, 7, 4)))
        self.save()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.backend.restore(self.path, 7)


if __name__ == "__main__":
    unittest.main()
