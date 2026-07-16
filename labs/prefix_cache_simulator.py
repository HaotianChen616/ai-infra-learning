#!/usr/bin/env python3
"""Demonstrate exact, block-aligned prefix-cache reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def simple_tokenize(text: str) -> list[str]:
    """A deliberately simple tokenizer for this educational simulation."""
    return text.split()


@dataclass(frozen=True)
class CacheResult:
    total_tokens: int
    reused_tokens: int

    @property
    def computed_tokens(self) -> int:
        return self.total_tokens - self.reused_tokens

    @property
    def hit_ratio(self) -> float:
        return self.reused_tokens / self.total_tokens if self.total_tokens else 0.0


class BlockPrefixCache:
    """Store exact token prefixes at fixed block boundaries."""

    def __init__(self, block_size: int = 4) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._prefixes: set[tuple[str, ...]] = set()

    def lookup(self, tokens: list[str]) -> int:
        reused = 0
        for end in range(self.block_size, len(tokens) + 1, self.block_size):
            if tuple(tokens[:end]) not in self._prefixes:
                break
            reused = end
        return reused

    def store(self, tokens: list[str]) -> None:
        for end in range(self.block_size, len(tokens) + 1, self.block_size):
            self._prefixes.add(tuple(tokens[:end]))

    def serve(self, tokens: list[str]) -> CacheResult:
        reused = self.lookup(tokens)
        self.store(tokens)
        return CacheResult(total_tokens=len(tokens), reused_tokens=reused)


def demo_prompts() -> Iterable[tuple[str, str]]:
    shared = (
        "You are a helpful assistant. Use concise answers. "
        "Document: KV cache stores attention states."
    )
    return [
        ("first document query", shared + " Question: What is it?"),
        ("same document, new question", shared + " Question: Why is it useful?"),
        (
            "slightly different template",
            "You are a helpful assistant. Use brief answers. "
            "Document: KV cache stores attention states. Question: What is it?",
        ),
    ]


def main() -> None:
    cache = BlockPrefixCache(block_size=4)
    print("Prefix-cache simulation (block size = 4 toy tokens)")
    print("=" * 62)
    for name, prompt in demo_prompts():
        tokens = simple_tokenize(prompt)
        result = cache.serve(tokens)
        print(f"{name}")
        print(f"  total={result.total_tokens:>2}  reused={result.reused_tokens:>2}  "
              f"computed={result.computed_tokens:>2}  hit={result.hit_ratio:.1%}")

    print(
        "\nThe simulator uses whitespace tokens. Real engines use the model's "
        "tokenizer and include model/config context in cache identity."
    )


if __name__ == "__main__":
    main()
