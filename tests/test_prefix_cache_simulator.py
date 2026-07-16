import unittest

from labs.prefix_cache_simulator import BlockPrefixCache


class PrefixCacheSimulatorTest(unittest.TestCase):
    def test_exact_block_prefix_is_reused(self) -> None:
        cache = BlockPrefixCache(block_size=4)
        cache.serve("a b c d e f g h question one".split())
        result = cache.serve("a b c d e f g h question two".split())
        self.assertEqual(result.reused_tokens, 8)

    def test_change_in_first_block_causes_miss(self) -> None:
        cache = BlockPrefixCache(block_size=4)
        cache.serve("a b c d e f g h".split())
        result = cache.serve("a b changed d e f g h".split())
        self.assertEqual(result.reused_tokens, 0)


if __name__ == "__main__":
    unittest.main()
