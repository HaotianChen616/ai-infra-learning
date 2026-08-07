import unittest

from labs.compare_cpu_profiles import flatten


class CompareCpuProfilesTest(unittest.TestCase):
    def test_flattens_probe_summary(self) -> None:
        payload = {
            "perf_stat": {"counters": {"cycles": 12}, "derived": {"ipc": 1.5}},
            "pyspy_all": {
                "total_samples": 10,
                "by_category": [{"category": "numpy", "share_pct": 30}],
            },
            "pyspy_gil": {"total_samples": 4},
            "gil_sample_ratio_proxy_pct": 40,
        }
        metrics = flatten(payload)
        self.assertEqual(metrics["perf_derived.ipc"], 1.5)
        self.assertEqual(metrics["pyspy_share.numpy.pct"], 30)
        self.assertEqual(metrics["pyspy_gil.sample_ratio_proxy_pct"], 40)


if __name__ == "__main__":
    unittest.main()
