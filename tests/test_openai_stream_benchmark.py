import unittest

from labs.openai_stream_benchmark import extract_content, parse_sse_data, percentile


class OpenAIStreamBenchmarkTest(unittest.TestCase):
    def test_parse_sse_data(self) -> None:
        event = parse_sse_data(
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
        )
        self.assertEqual(extract_content(event), "hello")

    def test_done_event_is_ignored(self) -> None:
        self.assertIsNone(parse_sse_data("data: [DONE]"))

    def test_percentile(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertIsNone(percentile([], 0.5))


if __name__ == "__main__":
    unittest.main()
