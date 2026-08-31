import json
import tempfile
import unittest
from pathlib import Path

from labs.metax_gds.summarize_mxfio_results import (
    parse_fio_json,
    parse_manifest_runs,
    summarize_runs,
)


SAMPLE_FIO = {
    "fio version": "fio-3.40",
    "jobs": [
        {
            "jobname": "mas",
            "read": {
                "io_bytes": 4 * 1024**3,
                "bw_bytes": 3 * 1024**3,
                "iops": 3072.0,
                "runtime": 10000,
                "total_ios": 30000,
                "clat_ns": {
                    "mean": 250000.0,
                    "percentile": {"99.000000": 900000.0},
                },
            },
            "usr_cpu": 2.5,
            "sys_cpu": 7.5,
        }
    ],
}


class SummarizeMxFioResultsTest(unittest.TestCase):
    def test_parse_fio_json(self) -> None:
        result = parse_fio_json(SAMPLE_FIO)
        self.assertEqual(result["fio_version"], "fio-3.40")
        self.assertAlmostEqual(result["read_gib_s"], 3.0)
        self.assertAlmostEqual(result["read_gib"], 4.0)
        self.assertAlmostEqual(result["latency_mean_us"], 250.0)
        self.assertAlmostEqual(result["latency_p99_us"], 900.0)
        self.assertEqual(result["sys_cpu_percent"], 7.5)

    def test_parse_manifest_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "result.json").write_text(json.dumps(SAMPLE_FIO), encoding="utf-8")
            (root / "runs.tsv").write_text(
                "run_id\trepetition\tmode\tgpu\tio_size\tnumjobs\tregion_size\tfile\tjson_file\texit_code\n"
                "r1\t1\tmas\t0\t1M\t1\t4G\t/mnt/test.bin\tresult.json\t0\n",
                encoding="utf-8",
            )
            parsed = parse_manifest_runs(root / "runs.tsv")
            self.assertEqual(parsed[0]["parse_status"], "ok")
            summary = summarize_runs(parsed)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["mode"], "mas")
            self.assertAlmostEqual(summary[0]["read_gib_s_median"], 3.0)

    def test_invalid_json_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bad.json").write_text("not json", encoding="utf-8")
            (root / "runs.tsv").write_text(
                "run_id\trepetition\tmode\tgpu\tio_size\tnumjobs\tregion_size\tfile\tjson_file\texit_code\n"
                "r1\t1\tmas\t0\t1M\t1\t4G\t/mnt/test.bin\tbad.json\t1\n",
                encoding="utf-8",
            )
            parsed = parse_manifest_runs(root / "runs.tsv")
            self.assertTrue(str(parsed[0]["parse_status"]).startswith("invalid_json:"))
            self.assertEqual(summarize_runs(parsed), [])


if __name__ == "__main__":
    unittest.main()
