import unittest

from labs.h2d_d2h_benchmark import (
    Scenario,
    TransferSample,
    bandwidth_gbps,
    build_scenarios,
    create_argument_parser,
    parse_cpu_affinity,
    parse_size,
    percentile,
    summarize_samples,
)


class H2DD2HBenchmarkTest(unittest.TestCase):
    def test_parse_binary_and_decimal_sizes(self) -> None:
        self.assertEqual(parse_size("4KiB"), 4096)
        self.assertEqual(parse_size("1 MiB"), 1 << 20)
        self.assertEqual(parse_size("0.5GB"), 500_000_000)
        self.assertEqual(parse_size("1024"), 1024)

    def test_parse_size_rejects_invalid_values(self) -> None:
        for value in ("", "0", "-1MiB", "many"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_size(value)

    def test_parse_cpu_affinity(self) -> None:
        self.assertEqual(parse_cpu_affinity("2,4-6,5"), (2, 4, 5, 6))

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.50), 2.5)
        self.assertIsNone(percentile([], 0.95))

    def test_bandwidth_uses_decimal_gigabytes(self) -> None:
        self.assertEqual(bandwidth_gbps(1_000_000_000, 1000.0), 1.0)
        self.assertIsNone(bandwidth_gbps(100, 0))

    def test_build_scenario_matrix(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(
            [
                "--backend",
                "npu",
                "--sizes",
                "4KiB,1MiB",
                "--directions",
                "h2d",
                "--host-memory",
                "pinned",
                "--modes",
                "nonblocking",
                "--sync-policies",
                "each,batch",
            ]
        )
        scenarios = build_scenarios(args)
        self.assertEqual(args.backend, "npu")
        self.assertIsNone(args.device)
        self.assertEqual(len(scenarios), 4)
        self.assertEqual(scenarios[0].name, "h2d/4KiB/pinned/nonblocking/each")

    def test_summary_keeps_host_device_and_pipeline_times_separate(self) -> None:
        scenario = Scenario("h2d", 1_000_000, True, True, "each")
        samples = [
            TransferSample(
                scenario=scenario.name,
                iteration=index,
                direction="h2d",
                size_bytes=1_000_000,
                pinned=True,
                non_blocking=True,
                sync_policy="each",
                cpu_prepare_ms=0.01,
                host_api_ms=host,
                device_copy_ms=device,
                completion_ms=completion,
                pipeline_ms=pipeline,
            )
            for index, (host, device, completion, pipeline) in enumerate(
                [
                    (0.02, 0.10, 0.15, 0.16),
                    (0.04, 0.20, 0.25, 0.26),
                ]
            )
        ]
        summary = summarize_samples([scenario], samples)[0]
        self.assertAlmostEqual(summary["host_api_ms_p50"], 0.03)
        self.assertAlmostEqual(summary["device_copy_ms_p50"], 0.15)
        self.assertAlmostEqual(summary["completion_ms_p50"], 0.20)
        self.assertAlmostEqual(summary["pipeline_ms_p50"], 0.21)


if __name__ == "__main__":
    unittest.main()
