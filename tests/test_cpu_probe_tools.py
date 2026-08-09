import tempfile
import unittest
from pathlib import Path

from labs.diff_proc_interrupts import diff
from labs.summarize_cpu_probes import summarize


class CpuProbeToolsTest(unittest.TestCase):
    def test_perf_pyspy_and_irq_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            perf = directory / "perf.csv"
            perf.write_text(
                "1000,msec,task-clock\n"
                "2000,,cycles\n1000,,instructions\n"
                "100,,branches\n10,,branch-misses\n"
                "50,,context-switches\n",
                encoding="utf-8",
            )
            all_raw = directory / "all.raw"
            all_raw.write_text(
                "root;numpy.core.multiarray.array 30\n"
                "root;get_num_common_prefix_blocks 20\n",
                encoding="utf-8",
            )
            gil_raw = directory / "gil.raw"
            gil_raw.write_text(
                "root;get_num_common_prefix_blocks 10\n", encoding="utf-8"
            )
            payload = summarize(
                perf, all_raw, gil_raw, requests=2, decode_steps=4
            )
            self.assertEqual(payload["perf_stat"]["derived"]["ipc"], 0.5)
            self.assertEqual(
                payload["perf_stat"]["derived"]["context_switches_per_on_cpu_second"],
                50,
            )
            self.assertEqual(payload["gil_sample_ratio_proxy_pct"], 20)
            normalization = payload["perf_stat"]["normalization"]
            self.assertEqual(
                normalization["counters_per_request"]["task-clock"], 500
            )
            self.assertEqual(
                normalization["counters_per_request"]["cycles"], 1000
            )
            self.assertEqual(
                normalization["counters_per_decode_step"]["instructions"], 250
            )

            before = directory / "interrupts-before.txt"
            after = directory / "interrupts-after.txt"
            before.write_text(
                "           CPU0       CPU1\n 42:          1          2 PCI-MSI nvidia\n",
                encoding="utf-8",
            )
            after.write_text(
                "           CPU0       CPU1\n 42:          4          8 PCI-MSI nvidia\n",
                encoding="utf-8",
            )
            irq = diff(before, after)
            self.assertEqual(irq["rows"][0]["total_delta"], 9)
            self.assertEqual(irq["rows"][0]["per_cpu_delta"], {"CPU0": 3, "CPU1": 6})

    def test_rejects_invalid_probe_normalization(self) -> None:
        with self.assertRaisesRegex(ValueError, "requests must be positive"):
            summarize(None, None, None, requests=0)


if __name__ == "__main__":
    unittest.main()
