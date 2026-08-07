import unittest

from labs.patch_vllm_event_wait import MARKER, patched_text


SOURCE = """import contextlib

import numpy as np
import torch


class AsyncOutput:
    def __init__(self):
        self.copy_event = torch.cuda.Event(blocking=True)

    def wait(self):
        self.copy_event.synchronize()


class AsyncModelRunnerOutput:
    def __init__(self):
        self.copy_event = torch.cuda.Event(blocking=True)

    def wait(self):
        self.copy_event.synchronize()
"""


class PatchVllmEventWaitTest(unittest.TestCase):
    def test_patch_is_strict_and_replaces_both_wait_sites(self) -> None:
        output = patched_text(SOURCE)
        self.assertIn(MARKER, output)
        self.assertEqual(output.count("self.copy_event = _new_copy_event()"), 2)
        self.assertEqual(output.count("_wait_copy_event(self.copy_event)"), 2)
        self.assertNotIn("torch.cuda.Event(blocking=True)", output)
        with self.assertRaises(ValueError):
            patched_text(output)


if __name__ == "__main__":
    unittest.main()
