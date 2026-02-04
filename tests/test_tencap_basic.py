import tempfile
import unittest
from pathlib import Path

import numpy as np

import tencap as tc


class TestTenCapBasic(unittest.TestCase):
    def test_interval_and_max_captures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tc.setup(root, interval=2, max_captures=2)

            x = np.arange(6).reshape(2, 3)
            with tc.scope("train"):
                p0 = tc.dump_np(x, name="x")
            self.assertIsNotNone(p0)
            self.assertTrue(Path(p0).exists())
            self.assertIn("cap_000000", str(p0))

            tc.step()  # step=1

            with tc.scope("train"):
                p1 = tc.dump_np(x, name="x")
            self.assertIsNone(p1)  # interval=2, step=1 skips

            tc.step()  # step=2

            with tc.scope("train"):
                p2 = tc.dump_np(x, name="x")
            self.assertIsNotNone(p2)
            self.assertTrue(Path(p2).exists())
            self.assertIn("cap_000001", str(p2))

            tc.step()  # step=3
            tc.step()  # step=4

            with tc.scope("train"):
                p3 = tc.dump_np(x, name="x")
            self.assertIsNone(p3)  # max_captures=2 reached

    def test_same_step_same_capture_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tc.setup(root, interval=1, max_captures=10)

            x = np.arange(3)
            with tc.scope("t"):
                p0 = tc.dump_np(x, name="v")
                p1 = tc.dump_np(x, name="v")  # collision -> suffix
            self.assertIsNotNone(p0)
            self.assertIsNotNone(p1)
            self.assertNotEqual(Path(p0), Path(p1))
            self.assertTrue(Path(p0).exists())
            self.assertTrue(Path(p1).exists())
            self.assertIn("cap_000000", str(p0))
            self.assertIn("cap_000000", str(p1))

            tc.step()

            with tc.scope("t"):
                p2 = tc.dump_np(x, name="v")
            self.assertIsNotNone(p2)
            self.assertIn("cap_000001", str(p2))

    def test_nested_scope_tag_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tc.setup(root, interval=1, max_captures=1)

            x = np.arange(2)
            with tc.scope("outer"):
                with tc.scope("inner"):
                    p = tc.dump_np(x, name="x")
            self.assertIsNotNone(p)
            self.assertIn("outer", str(p))
            self.assertIn("inner", str(p))


if __name__ == "__main__":
    unittest.main()
