import json
import tempfile
import unittest
from pathlib import Path

from labs.summarize_ascend_sync import load_sync_events, summarize


class SummarizeAscendSyncTest(unittest.TestCase):
    def test_loads_complete_sync_events(self) -> None:
        payload = {
            "traceEvents": [
                {
                    "name": "aclrtSynchronizeStreamWithTimeout",
                    "ph": "X",
                    "dur": 2000,
                    "pid": 1,
                    "tid": 2,
                },
                {
                    "name": "aclnnMatmul",
                    "ph": "X",
                    "dur": 3000,
                    "pid": 1,
                    "tid": 2,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples = load_sync_events(path, duration_unit="us")
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].family, "aclrt")
        self.assertEqual(samples[0].duration_us, 2000)

    def test_does_not_sum_nested_api_families(self) -> None:
        payload = {
            "traceEvents": [
                {
                    "name": "aclrtSynchronizeStream",
                    "ph": "X",
                    "dur": 2000,
                },
                {"name": "rtStreamSynchronize", "ph": "X", "dur": 1900},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples = load_sync_events(path, duration_unit="us")
        result = summarize(samples, decode_steps=2, generated_tokens=16)
        self.assertEqual(result["preferred_nonduplicated_family"], "aclrt")
        self.assertEqual(result["selected_sync_host_wall_ms"], 2.0)
        self.assertEqual(
            result["selected_sync_host_wall_ms_per_decode_step"],
            1.0,
        )
        self.assertEqual(
            list(result["selected_sync_host_wall_ms_by_source"].values()),
            [2.0],
        )

    def test_control_can_omit_decode_normalization(self) -> None:
        payload = {
            "traceEvents": [
                {
                    "name": "aclrtSynchronizeDevice",
                    "ph": "X",
                    "dur": 1000,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples = load_sync_events(path, duration_unit="us")
        result = summarize(samples, decode_steps=0, generated_tokens=8)
        self.assertIsNone(result["selected_sync_host_wall_ms_per_decode_step"])


if __name__ == "__main__":
    unittest.main()
