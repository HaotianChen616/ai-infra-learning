import unittest

from labs.scheduler_simulator import RequestSpec, default_requests, simulate


class SchedulerSimulatorTest(unittest.TestCase):
    def test_all_requests_complete(self) -> None:
        result = simulate(
            default_requests(),
            token_budget=8,
            prefill_chunk_size=4,
            policy="decode-first",
        )
        self.assertTrue(all(state.done for state in result.states))
        self.assertTrue(all(state.first_token_step is not None for state in result.states))

    def test_decode_first_protects_existing_decode(self) -> None:
        requests = [
            RequestSpec("short", 0, 2, 5),
            RequestSpec("late-long", 2, 20, 2),
        ]
        prefill_first = simulate(requests, 4, 4, "prefill-first")
        decode_first = simulate(requests, 4, 4, "decode-first")
        first_completion_prefill = prefill_first.states[0].completion_step
        first_completion_decode = decode_first.states[0].completion_step
        self.assertLess(first_completion_decode, first_completion_prefill)


if __name__ == "__main__":
    unittest.main()
