import json
import tempfile
import unittest
from pathlib import Path

from labs.compare_cpu_profiles import compare, flatten, markdown, metric_sort_key


class CompareCpuProfilesTest(unittest.TestCase):
    def test_flattens_probe_summary(self) -> None:
        payload = {
            "perf_stat": {
                "counters": {"task-clock": 20, "cycles": 12},
                "derived": {"ipc": 1.5},
                "normalization": {
                    "counters_per_request": {"task-clock": 10, "cycles": 6},
                    "counters_per_decode_step": {"instructions": 3},
                },
            },
            "pyspy_all": {
                "total_samples": 10,
                "by_category": [{"category": "numpy", "share_pct": 30}],
            },
            "pyspy_gil": {"total_samples": 4},
            "gil_sample_ratio_proxy_pct": 40,
        }
        metrics = flatten(payload)
        self.assertEqual(metrics["perf_derived.ipc"], 1.5)
        self.assertEqual(metrics["perf_counter_per_request.task-clock"], 10)
        self.assertEqual(metrics["perf_counter_per_decode_step.instructions"], 3)
        self.assertEqual(metrics["pyspy_share.numpy.pct"], 30)
        self.assertEqual(metrics["pyspy_gil.sample_ratio_proxy_pct"], 40)
        names = sorted(metrics, key=metric_sort_key)
        self.assertLess(
            names.index("perf_counter_per_request.task-clock"),
            names.index("perf_counter.task-clock"),
        )

    def test_prioritizes_total_self_cpu_and_warns_on_time_transfer(self) -> None:
        baseline = {
            "total_self_cpu_ms": 100,
            "normalization": {
                "self_cpu_ms_per_request": 10,
                "self_cpu_ms_per_decode_step": 1,
            },
            "by_analysis_category": [
                {
                    "category": "cuda_sync_wait_wall",
                    "self_wall_ms": 40,
                    "self_cpu_ms_per_request": 4,
                    "self_cpu_ms_per_decode_step": 0.4,
                    "share_of_summed_self_wall_pct": 40,
                }
            ],
        }
        moved = {
            "total_self_cpu_ms": 110,
            "normalization": {
                "self_cpu_ms_per_request": 11,
                "self_cpu_ms_per_decode_step": 1.1,
            },
            "by_analysis_category": [
                {
                    "category": "cuda_sync_wait_wall",
                    "self_wall_ms": 20,
                    "self_cpu_ms_per_request": 2,
                    "self_cpu_ms_per_decode_step": 0.2,
                    "share_of_summed_self_wall_pct": 18.18,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline_path = directory / "baseline.json"
            moved_path = directory / "moved.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            moved_path.write_text(json.dumps(moved), encoding="utf-8")
            payload = compare(
                [("baseline", baseline_path), ("moved", moved_path)]
            )

        self.assertIn("did not", payload["warnings"][0])
        self.assertEqual(
            payload["metrics"][0]["metric"],
            "torch_trace.total_self_cpu_ms",
        )
        self.assertTrue(markdown(payload).startswith("## Self CPU transfer warnings"))


if __name__ == "__main__":
    unittest.main()
