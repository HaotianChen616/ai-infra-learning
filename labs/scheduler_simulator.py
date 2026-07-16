#!/usr/bin/env python3
"""A small token-budget scheduler simulator for learning purposes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_step: int
    prompt_tokens: int
    output_tokens: int

    def validate(self) -> None:
        if self.arrival_step < 0:
            raise ValueError("arrival_step cannot be negative")
        if self.prompt_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("prompt_tokens and output_tokens must be positive")


@dataclass
class RequestState:
    spec: RequestSpec
    prefill_remaining: int = field(init=False)
    output_remaining: int = field(init=False)
    prefill_done_step: int | None = None
    first_token_step: int | None = None
    completion_step: int | None = None

    def __post_init__(self) -> None:
        self.spec.validate()
        self.prefill_remaining = self.spec.prompt_tokens
        self.output_remaining = self.spec.output_tokens

    @property
    def done(self) -> bool:
        return self.completion_step is not None

    def can_decode(self, step: int) -> bool:
        return (
            not self.done
            and self.prefill_remaining == 0
            and self.prefill_done_step is not None
            and self.prefill_done_step <= step
            and self.output_remaining > 0
        )


@dataclass(frozen=True)
class StepTrace:
    step: int
    events: tuple[str, ...]
    tokens_used: int


@dataclass
class SimulationResult:
    policy: str
    states: list[RequestState]
    trace: list[StepTrace]

    @property
    def average_ttft(self) -> float:
        values = [
            state.first_token_step - state.spec.arrival_step
            for state in self.states
            if state.first_token_step is not None
        ]
        return mean(values)

    @property
    def average_latency(self) -> float:
        values = [
            state.completion_step - state.spec.arrival_step
            for state in self.states
            if state.completion_step is not None
        ]
        return mean(values)


def default_requests() -> list[RequestSpec]:
    return [
        RequestSpec("chat-a", arrival_step=0, prompt_tokens=6, output_tokens=5),
        RequestSpec("chat-b", arrival_step=1, prompt_tokens=4, output_tokens=4),
        RequestSpec("long-prompt", arrival_step=2, prompt_tokens=20, output_tokens=3),
    ]


def simulate(
    requests: Iterable[RequestSpec],
    token_budget: int,
    prefill_chunk_size: int,
    policy: str,
) -> SimulationResult:
    if token_budget <= 0 or prefill_chunk_size <= 0:
        raise ValueError("token_budget and prefill_chunk_size must be positive")
    if policy not in {"decode-first", "prefill-first"}:
        raise ValueError("policy must be decode-first or prefill-first")

    states = [RequestState(spec) for spec in requests]
    trace: list[StepTrace] = []
    step = 0

    def schedule_decode(budget: int, events: list[str]) -> int:
        for state in states:
            if budget == 0:
                break
            if not state.can_decode(step):
                continue
            state.output_remaining -= 1
            budget -= 1
            if state.first_token_step is None:
                state.first_token_step = step
            events.append(f"decode:{state.spec.request_id}")
            if state.output_remaining == 0:
                state.completion_step = step + 1
                events.append(f"complete:{state.spec.request_id}")
        return budget

    def schedule_prefill(budget: int, events: list[str]) -> int:
        for state in states:
            if budget == 0:
                break
            if state.done or state.spec.arrival_step > step or state.prefill_remaining == 0:
                continue
            tokens = min(prefill_chunk_size, state.prefill_remaining, budget)
            state.prefill_remaining -= tokens
            budget -= tokens
            events.append(f"prefill:{state.spec.request_id}({tokens})")
            if state.prefill_remaining == 0:
                # Decode begins no earlier than the next scheduling step.
                state.prefill_done_step = step + 1
        return budget

    while not all(state.done for state in states):
        if step > 100_000:
            raise RuntimeError("simulation did not converge")

        budget = token_budget
        events: list[str] = []
        if policy == "decode-first":
            budget = schedule_decode(budget, events)
            budget = schedule_prefill(budget, events)
        else:
            budget = schedule_prefill(budget, events)
            budget = schedule_decode(budget, events)

        trace.append(
            StepTrace(
                step=step,
                events=tuple(events) if events else ("idle",),
                tokens_used=token_budget - budget,
            )
        )
        step += 1

    return SimulationResult(policy=policy, states=states, trace=trace)


def print_result(result: SimulationResult, show_trace: bool) -> None:
    print(f"\nPolicy: {result.policy}")
    print("-" * 72)
    if show_trace:
        for item in result.trace:
            print(
                f"step {item.step:>2} | tokens={item.tokens_used:>2} | "
                + ", ".join(item.events)
            )
        print("-" * 72)

    print("request       arrival  first-token  complete  TTFT  latency")
    for state in result.states:
        ttft = state.first_token_step - state.spec.arrival_step
        latency = state.completion_step - state.spec.arrival_step
        print(
            f"{state.spec.request_id:<13} {state.spec.arrival_step:>7} "
            f"{state.first_token_step:>12} {state.completion_step:>9} "
            f"{ttft:>5} {latency:>8}"
        )
    print(f"average TTFT: {result.average_ttft:.2f} steps")
    print(f"average latency: {result.average_latency:.2f} steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-budget", type=int, default=8)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=8,
        help="Max prefill tokens one request can consume per step (default: 8).",
    )
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args()

    for policy in ("prefill-first", "decode-first"):
        result = simulate(
            default_requests(),
            token_budget=args.token_budget,
            prefill_chunk_size=args.prefill_chunk_size,
            policy=policy,
        )
        print_result(result, args.show_trace)

    print(
        "\nThis toy model explains scheduling trade-offs; its steps are not GPU time."
    )


if __name__ == "__main__":
    main()
