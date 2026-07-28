import tempfile
import unittest
from pathlib import Path

from labs.summarize_cuda_sync import (
    ApiSample,
    is_sync_api,
    load_api_samples,
    summarize,
)


class SummarizeCudaSyncTest(unittest.TestCase):
    def test_sync_api_filter(self) -> None:
        for name in (
            "cudaDeviceSynchronize",
            "cudaEventSynchronize",
            "cudaStreamSynchronize",
            "cuCtxSynchronize",
            "cuEventSynchronize_v2",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_sync_api(name))
        for name in ("cudaStreamWaitEvent", "cudaEventRecord", "cudaMemcpyAsync"):
            with self.subTest(name=name):
                self.assertFalse(is_sync_api(name))

    def test_loads_nsys_csv_after_report_preamble(self) -> None:
        content = """Generating SQLite file...
** CUDA API Trace:
"Start (ns)","Duration (ns)","Name","Pid","Tid","Thread Name"
1000,2000,"cudaEventSynchronize",11,22,"worker"
3000,500,"cudaLaunchKernel",11,22,"worker"
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(content, encoding="utf-8")
            samples = load_api_samples(path)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].duration_ns, 2000)
        self.assertEqual(samples[0].thread_name, "worker")

    def test_summary_keeps_host_wall_semantics(self) -> None:
        samples = [
            ApiSample("cudaEventSynchronize", 1_000_000, 0, "1", "2", "worker"),
            ApiSample(
                "cudaEventSynchronize",
                3_000_000,
                2_000_000,
                "1",
                "2",
                "worker",
            ),
            ApiSample("cudaLaunchKernel", 100_000, 6_000_000, "1", "2", "worker"),
        ]
        payload = summarize(samples, decode_steps=2, generated_tokens=16)
        self.assertEqual(payload["sync_samples"], 2)
        self.assertEqual(payload["total_sync_host_wall_ms"], 4.0)
        self.assertEqual(payload["sync_host_wall_ms_per_decode_step"], 2.0)
        self.assertEqual(payload["sync_host_wall_us_per_generated_token"], 250.0)
        self.assertEqual(payload["by_api"][0]["p50_us"], 2000.0)


if __name__ == "__main__":
    unittest.main()
