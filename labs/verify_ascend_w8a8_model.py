#!/usr/bin/env python3
"""Verify a local Qwen3.6-27B ModelSlim/Ascend W8A8 checkpoint."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable


def _leaf_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _leaf_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _leaf_strings(child)
    elif isinstance(value, str):
        yield value


def verify_model_directory(model_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    config_path = model_dir / "config.json"
    quant_path = model_dir / "quant_model_description.json"

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "valid": False,
            "failures": [f"cannot read config.json: {error}"],
            "warnings": warnings,
        }

    model_type = str(config.get("model_type", "")).lower()
    text_config = config.get("text_config", {})
    text_model_type = (
        str(text_config.get("model_type", "")).lower()
        if isinstance(text_config, dict)
        else ""
    )
    if model_type not in {"qwen3_5", "qwen3.6"}:
        failures.append(f"unexpected model_type: {model_type or '<missing>'}")
    if text_model_type and text_model_type != "qwen3_5_text":
        warnings.append(f"unexpected text model_type: {text_model_type}")

    try:
        quant_description = json.loads(quant_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"cannot read quant_model_description.json: {error}")
        quant_description = {}

    quant_values = collections.Counter(
        value.upper() for value in _leaf_strings(quant_description)
    )
    w8a8_entries = sum(count for name, count in quant_values.items() if "W8A8" in name)
    if w8a8_entries == 0:
        failures.append("quant_model_description.json has no W8A8 entries")

    index_candidates = sorted(model_dir.glob("*safetensors.index.json"))
    weight_candidates = sorted(model_dir.glob("*.safetensors"))
    if not index_candidates:
        warnings.append("no safetensors index file found")
    if not weight_candidates:
        failures.append("no safetensors weight files found")

    return {
        "valid": not failures,
        "model_dir": str(model_dir),
        "model_type": model_type,
        "text_model_type": text_model_type,
        "architectures": config.get("architectures", []),
        "num_hidden_layers": (
            text_config.get("num_hidden_layers")
            if isinstance(text_config, dict)
            else None
        ),
        "w8a8_description_entries": w8a8_entries,
        "quantization_value_counts": dict(quant_values.most_common()),
        "weight_file_count": len(weight_candidates),
        "index_files": [path.name for path in index_candidates],
        "failures": failures,
        "warnings": warnings,
        "notes": [
            "This verifies checkpoint metadata, not the runtime kernel path.",
            "Confirm --quantization ascend and W8A8 loading in the server log.",
        ],
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = create_argument_parser().parse_args()
    result = verify_model_directory(args.model_dir)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
