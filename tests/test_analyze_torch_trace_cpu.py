import json
import tempfile
import unittest
from pathlib import Path

from labs.analyze_torch_trace_cpu import CpuEvent, classify, load_events, summarize


class AnalyzeTorchTraceCpuTest(unittest.TestCase):
    def test_nested_events_produce_exclusive_wall_time(self) -> None:
        trace = {
            "traceEvents": [
                {
                    "ph": "M", "name": "thread_name", "pid": 1, "tid": 2,
                    "args": {"name": "EngineCore"},
                },
                {
                    "ph": "X", "name": "scheduler.schedule", "cat": "python_function",
                    "pid": 1, "tid": 2, "ts": 100, "dur": 100,
                },
                {
                    "ph": "X", "name": "cudaEventSynchronize", "cat": "cuda_runtime",
                    "pid": 1, "tid": 2, "ts": 120, "dur": 40,
                },
                {
                    "ph": "X", "name": "gpu kernel", "cat": "kernel",
                    "pid": 0, "tid": 7, "ts": 120, "dur": 20,
                },
                {
                    "ph": "X", "name": "Profiler Step", "cat": "user_annotation",
                    "pid": "Spans", "tid": "PyTorch Profiler", "ts": 100, "dur": 100,
                },
                {
                    "ph": "X", "name": "GPU annotation", "cat": "gpu_user_annotation",
                    "pid": 0, "tid": 7, "ts": 100, "dur": 100,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "trace.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            events, names = load_events(path)
            payload = summarize(events, names, requests=2, decode_steps=4)

        self.assertEqual(payload["events"], 2)
        self.assertEqual(payload["total_self_cpu_ms"], 0.1)
        self.assertEqual(payload["summed_cpu_self_wall_ms"], 0.1)
        self.assertEqual(
            payload["normalization"]["self_cpu_ms_per_request"], 0.05
        )
        self.assertEqual(
            payload["normalization"]["self_cpu_ms_per_decode_step"], 0.025
        )
        rows = {row["category"]: row for row in payload["by_analysis_category"]}
        self.assertEqual(rows["vllm_scheduler"]["self_wall_ms"], 0.06)
        self.assertEqual(rows["cuda_sync_wait_wall"]["self_wall_ms"], 0.04)
        self.assertEqual(
            rows["cuda_sync_wait_wall"]["self_cpu_ms_per_request"], 0.02
        )

    def test_rejects_non_positive_normalization(self) -> None:
        with self.assertRaisesRegex(ValueError, "requests must be positive"):
            summarize([], {}, requests=0)
        with self.assertRaisesRegex(ValueError, "decode_steps must be positive"):
            summarize([], {}, decode_steps=-1)

    def test_classifies_python_lock_wait_separately(self) -> None:
        event = CpuEvent(
            name="<built-in method acquire of _thread.lock object>",
            category="python_function",
            pid=1,
            tid=2,
            start_us=0,
            end_us=10,
            args={},
        )
        self.assertEqual(classify(event), "python_lock_or_wait_wall")


if __name__ == "__main__":
    unittest.main()
