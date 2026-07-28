#!/usr/bin/env python3
"""Verify that a local model config describes Qwen3 INT8 W8A8.

The expected checkpoint format is LLM Compressor/compressed-tensors with
INT8 weights and INT8 input activations. This checks metadata only; the vLLM
startup log and Nsight kernel trace must still confirm the runtime path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def verify_config(config: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    model_type = _normalized(config.get("model_type"))
    if model_type != "qwen3":
        failures.append(f"model_type must be qwen3, got {model_type or '<missing>'}")

    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return {
            "valid": False,
            "failures": ["quantization_config is missing or is not an object"],
            "warnings": warnings,
        }

    quant_method = _normalized(quantization.get("quant_method"))
    if quant_method != "compressed-tensors":
        failures.append(
            "quant_method must be compressed-tensors, "
            f"got {quant_method or '<missing>'}"
        )

    quant_format = _normalized(quantization.get("format"))
    if quant_format != "int-quantized":
        failures.append(
            f"format must be int-quantized, got {quant_format or '<missing>'}"
        )

    config_groups = quantization.get("config_groups")
    if not isinstance(config_groups, dict) or not config_groups:
        failures.append("quantization_config.config_groups is empty")
        groups: list[tuple[str, dict[str, Any]]] = []
    else:
        groups = [
            (name, group)
            for name, group in config_groups.items()
            if isinstance(group, dict)
        ]

    matching_groups: list[dict[str, Any]] = []
    for name, group in groups:
        weights = group.get("weights")
        activations = group.get("input_activations")
        if not isinstance(weights, dict) or not isinstance(activations, dict):
            continue
        if (
            weights.get("num_bits") == 8
            and _normalized(weights.get("type")) == "int"
            and activations.get("num_bits") == 8
            and _normalized(activations.get("type")) == "int"
        ):
            matching_groups.append(
                {
                    "name": name,
                    "targets": group.get("targets", []),
                    "weight_strategy": weights.get("strategy"),
                    "activation_strategy": activations.get("strategy"),
                    "activation_dynamic": activations.get("dynamic"),
                }
            )

    if not matching_groups:
        failures.append("no config group has INT8 weights and INT8 input activations")

    status = _normalized(quantization.get("quantization_status"))
    if status not in {"compressed", "frozen"}:
        warnings.append(
            f"quantization_status is not compressed/frozen: {status or '<missing>'}"
        )

    ignored = quantization.get("ignore", quantization.get("ignored_layers", []))
    return {
        "valid": not failures,
        "model_type": model_type,
        "torch_dtype": config.get("torch_dtype"),
        "quant_method": quant_method,
        "format": quant_format,
        "quantization_status": status,
        "matching_w8a8_groups": matching_groups,
        "ignored_layers": ignored,
        "failures": failures,
        "warnings": warnings,
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_json", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optionally save the verification result as JSON.",
    )
    return parser


def main() -> None:
    args = create_argument_parser().parse_args()
    try:
        config = json.loads(args.config_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read model config: {error}") from error
    if not isinstance(config, dict):
        raise SystemExit("model config root must be a JSON object")

    result = verify_config(config)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
