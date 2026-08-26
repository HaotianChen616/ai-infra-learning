import tempfile
import unittest
from pathlib import Path

from labs.gds.summarize_gds_results import (
    parse_gdsio_output,
    parse_manifest_runs,
    summarize_runs,
)


SAMPLE_OUTPUT = """
IoType: READ XferType: GPUD Threads: 4 DataSetSize: 4194304/4194304(KiB)
IOSize: 1024(KiB) Throughput: 6.524658 GiB/sec, Avg_Latency: 1197.370995 usecs ops: 799606 total_time 119.679102 secs
\tUser time (seconds): 2.00
\tSystem time (seconds): 3.00
\tPercent of CPU this job got: 4%
"""


class SummarizeGDSResultsTest(unittest.TestCase):
    def test_parse_gdsio_output(self) -> None:
        result = parse_gdsio_output(SAMPLE_OUTPUT)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["io_type"], "READ")
        self.assertEqual(result["xfer_type"], "GPUD")
        self.assertEqual(result["reported_threads"], 4)
        self.assertAlmostEqual(result["throughput_gib_s"], 6.524658)
        self.assertAlmostEqual(result["avg_latency_us"], 1197.370995)
        self.assertEqual(result["cpu_percent"], 4.0)
        self.assertEqual(result["user_time_s"], 2.0)
        self.assertEqual(result["system_time_s"], 3.0)

    def test_parser_uses_final_result(self) -> None:
        second = SAMPLE_OUTPUT.replace("6.524658", "7.25")
        result = parse_gdsio_output(SAMPLE_OUTPUT + second)
        assert result is not None
        self.assertEqual(result["throughput_gib_s"], 7.25)

    def test_parse_manifest_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run.log").write_text(SAMPLE_OUTPUT, encoding="utf-8")
            (root / "runs.tsv").write_text(
                "\t".join(
                    [
                        "run_id",
                        "repetition",
                        "gpu",
                        "operation",
                        "transfer_code",
                        "transfer_name",
                        "io_size",
                        "workers",
                        "dataset_size",
                        "file",
                        "log_file",
                        "exit_code",
                        "command",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "r1",
                        "1",
                        "0",
                        "0",
                        "0",
                        "gds",
                        "1M",
                        "4",
                        "4G",
                        "/mnt/gds/input.bin",
                        "run.log",
                        "0",
                        "gdsio ...",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = parse_manifest_runs(root / "runs.tsv")
            self.assertEqual(parsed[0]["parse_status"], "ok")
            summary = summarize_runs(parsed)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["samples"], 1)
            self.assertAlmostEqual(summary[0]["throughput_gib_s_median"], 6.524658)

    def test_missing_log_is_preserved_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runs.tsv").write_text(
                "run_id\trepetition\tgpu\toperation\ttransfer_code\tio_size\tworkers\tdataset_size\tfile\tlog_file\texit_code\n"
                "r1\t1\t0\t0\t0\t1M\t1\t1G\t/mnt/a\tmissing.log\t1\n",
                encoding="utf-8",
            )
            parsed = parse_manifest_runs(root / "runs.tsv")
            self.assertEqual(parsed[0]["parse_status"], "missing_log")
            self.assertEqual(summarize_runs(parsed), [])


if __name__ == "__main__":
    unittest.main()
