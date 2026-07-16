#!/usr/bin/env python3
"""Estimate logical KV-cache capacity for a Transformer model.

This is an educational estimator. Real serving engines may add alignment,
allocator, block-table, graph-capture, and model-specific overheads.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


GIB = 1024**3
DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass(frozen=True)
class KVConfig:
    layers: int
    kv_heads: int
    head_dim: int
    dtype: str

    def validate(self) -> None:
        if self.layers <= 0 or self.kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("layers, kv_heads and head_dim must be positive")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")

    @property
    def bytes_per_element(self) -> float:
        self.validate()
        return DTYPE_BYTES[self.dtype]

    @property
    def bytes_per_token(self) -> float:
        # Two tensors are cached at every attention layer: Key and Value.
        return (
            2
            * self.layers
            * self.kv_heads
            * self.head_dim
            * self.bytes_per_element
        )

    def bytes_per_sequence(self, context_length: int) -> float:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        return self.bytes_per_token * context_length

    def total_bytes(self, context_length: int, concurrency: int) -> float:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        return self.bytes_per_sequence(context_length) * concurrency


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    index = 0
    while abs(value) >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:,.2f} {units[index]}"


def theoretical_max_concurrency(
    config: KVConfig,
    context_length: int,
    tp_size: int,
    capacity_gib_per_device: float,
    usable_fraction: float,
) -> int:
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if capacity_gib_per_device <= 0:
        raise ValueError("capacity must be positive")
    if not 0 < usable_fraction <= 1:
        raise ValueError("usable_fraction must be in (0, 1]")

    bytes_per_sequence_per_device = config.bytes_per_sequence(context_length) / tp_size
    usable_bytes_per_device = capacity_gib_per_device * GIB * usable_fraction
    return math.floor(usable_bytes_per_device / bytes_per_sequence_per_device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate Transformer KV-cache capacity."
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--kv-capacity-gib-per-device", type=float)
    parser.add_argument(
        "--usable-fraction",
        type=float,
        default=0.9,
        help="Fraction of the stated KV budget treated as usable (default: 0.9).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tp_size <= 0:
        raise SystemExit("--tp-size must be positive")

    config = KVConfig(
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype=args.dtype,
    )
    total = config.total_bytes(args.context_length, args.concurrency)

    print("KV-cache estimate")
    print("=================")
    print(f"Formula: 2 × {args.layers} layers × {args.kv_heads} KV heads "
          f"× {args.head_dim} head dim × {config.bytes_per_element:g} bytes")
    print(f"Per token:            {format_bytes(config.bytes_per_token)}")
    print(
        f"Per {args.context_length:,}-token sequence: "
        f"{format_bytes(config.bytes_per_sequence(args.context_length))}"
    )
    print(f"Logical total:        {format_bytes(total)}")
    print(f"Approx. per TP shard: {format_bytes(total / args.tp_size)}")

    if args.kv_capacity_gib_per_device is not None:
        maximum = theoretical_max_concurrency(
            config=config,
            context_length=args.context_length,
            tp_size=args.tp_size,
            capacity_gib_per_device=args.kv_capacity_gib_per_device,
            usable_fraction=args.usable_fraction,
        )
        print(
            f"Theoretical max concurrency at {args.usable_fraction:.0%} usable "
            f"capacity: {maximum}"
        )

    print("\nCaveat: real engines add model-specific and allocator overheads.")


if __name__ == "__main__":
    main()
