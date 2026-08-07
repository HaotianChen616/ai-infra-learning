import json
import tempfile
import unittest
from pathlib import Path

from labs.analyze_torch_trace_cpu import load_events, summarize


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
            ]
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "trace.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            events, names = load_events(path)
            payload = summarize(events, names)

        self.assertEqual(payload["events"], 2)
        self.assertEqual(payload["summed_cpu_self_wall_ms"], 0.1)
        rows = {row["category"]: row for row in payload["by_analysis_category"]}
        self.assertEqual(rows["vllm_scheduler"]["self_wall_ms"], 0.06)
        self.assertEqual(rows["cuda_sync_wait_wall"]["self_wall_ms"], 0.04)


if __name__ == "__main__":
    unittest.main()
