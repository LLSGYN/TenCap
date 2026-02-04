import tempfile
import unittest
from pathlib import Path

import numpy as np

import tencap as tc

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class TestTenCapTorch(unittest.TestCase):
    def test_dump_torch_missing(self) -> None:
        if torch is not None:
            self.skipTest("torch installed")

        with tempfile.TemporaryDirectory() as td:
            tc.setup(Path(td), interval=1, max_captures=1)
            with tc.scope("torch", prefix="p"):
                with self.assertRaises(ModuleNotFoundError):
                    tc.dump_torch(object(), name="x")

    @unittest.skipUnless(torch is not None, "torch not installed")
    def test_dump_torch_roundtrip(self) -> None:
        assert torch is not None  # make type checkers happy

        with tempfile.TemporaryDirectory() as td:
            tc.setup(Path(td), interval=1, max_captures=2)

            t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
            with tc.scope("torch", prefix="p"):
                p = tc.dump_torch(t, name="t")

            self.assertIsNotNone(p)
            self.assertTrue(Path(p).exists())

            loaded = torch.load(p)
            self.assertIsInstance(loaded, torch.Tensor)
            self.assertEqual(getattr(loaded, "device").type, "cpu")
            self.assertEqual(loaded.dtype, t.dtype)
            self.assertTrue(torch.equal(loaded, t.detach().cpu()))

    @unittest.skipUnless(torch is not None, "torch not installed")
    def test_dump_torch_type_error(self) -> None:
        assert torch is not None

        with tempfile.TemporaryDirectory() as td:
            tc.setup(Path(td), interval=1, max_captures=1)
            with tc.scope("torch", prefix="p"):
                with self.assertRaises(TypeError):
                    tc.dump_torch(np.arange(3), name="x")


if __name__ == "__main__":
    unittest.main()

