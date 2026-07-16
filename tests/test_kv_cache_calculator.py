import unittest

from labs.kv_cache_calculator import GIB, KVConfig, theoretical_max_concurrency


class KVCacheCalculatorTest(unittest.TestCase):
    def test_gqa_bf16_example_is_128_kib_per_token(self) -> None:
        config = KVConfig(layers=32, kv_heads=8, head_dim=128, dtype="bf16")
        self.assertEqual(config.bytes_per_token, 128 * 1024)
        self.assertEqual(config.bytes_per_sequence(8192), GIB)

    def test_fp8_halves_bf16_capacity(self) -> None:
        bf16 = KVConfig(layers=32, kv_heads=8, head_dim=128, dtype="bf16")
        fp8 = KVConfig(layers=32, kv_heads=8, head_dim=128, dtype="fp8")
        self.assertEqual(fp8.bytes_per_token, bf16.bytes_per_token / 2)

    def test_tp_capacity_limit(self) -> None:
        config = KVConfig(layers=32, kv_heads=8, head_dim=128, dtype="bf16")
        maximum = theoretical_max_concurrency(
            config,
            context_length=8192,
            tp_size=4,
            capacity_gib_per_device=10,
            usable_fraction=1,
        )
        self.assertEqual(maximum, 40)


if __name__ == "__main__":
    unittest.main()
