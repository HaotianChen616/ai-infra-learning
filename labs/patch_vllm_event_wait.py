#!/usr/bin/env python3
"""Reversibly add CUDA Event wait-mode experiments to vLLM 0.26.

The patch targets ``vllm/v1/worker/gpu/async_utils.py`` and creates a sibling
``.cpu-wait-experiment.bak`` before changing it. It is intentionally strict:
source patterns must match vLLM 0.26 and restore never deletes the backup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path


MARKER = "# BEGIN AI-INFRA CUDA EVENT WAIT EXPERIMENT"
BACKUP_SUFFIX = ".cpu-wait-experiment.bak"

IMPORT_NEEDLE = "import contextlib\n\nimport numpy as np"
IMPORT_REPLACEMENT = '''import contextlib
import os
import time

import numpy as np'''

HELPERS = r'''

# BEGIN AI-INFRA CUDA EVENT WAIT EXPERIMENT
# Diagnostic-only modes. Keep the default identical to upstream vLLM.
_CUDA_EVENT_WAIT_MODE = os.getenv("VLLM_CUDA_EVENT_WAIT_MODE", "blocking")
_CUDA_EVENT_HYBRID_SPIN_US = int(
    os.getenv("VLLM_CUDA_EVENT_HYBRID_SPIN_US", "50")
)
if _CUDA_EVENT_WAIT_MODE not in {"blocking", "spin", "python_poll", "hybrid"}:
    raise RuntimeError(
        "VLLM_CUDA_EVENT_WAIT_MODE must be blocking, spin, python_poll, or hybrid"
    )
if _CUDA_EVENT_HYBRID_SPIN_US < 0:
    raise RuntimeError("VLLM_CUDA_EVENT_HYBRID_SPIN_US must be non-negative")


def _new_copy_event() -> torch.cuda.Event:
    # blocking=False makes Event.synchronize use CUDA's active-wait path.
    return torch.cuda.Event(blocking=_CUDA_EVENT_WAIT_MODE != "spin")


def _wait_copy_event(event: torch.cuda.Event) -> None:
    if _CUDA_EVENT_WAIT_MODE in {"blocking", "spin"}:
        event.synchronize()
        return
    if _CUDA_EVENT_WAIT_MODE == "python_poll":
        while not event.query():
            pass
        return

    deadline_ns = time.perf_counter_ns() + _CUDA_EVENT_HYBRID_SPIN_US * 1_000
    while time.perf_counter_ns() < deadline_ns:
        if event.query():
            return
    event.synchronize()
# END AI-INFRA CUDA EVENT WAIT EXPERIMENT
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_target() -> Path:
    spec = importlib.util.find_spec("vllm.v1.worker.gpu.async_utils")
    if spec is None or spec.origin is None:
        raise SystemExit("could not locate vllm.v1.worker.gpu.async_utils")
    return Path(spec.origin).resolve()


def installed_version() -> str | None:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None


def patched_text(original: str) -> str:
    if MARKER in original:
        raise ValueError("target is already patched")
    if original.count(IMPORT_NEEDLE) != 1:
        raise ValueError("vLLM import pattern did not match exactly once")
    if original.count("self.copy_event = torch.cuda.Event(blocking=True)") != 2:
        raise ValueError("expected exactly two blocking copy-event constructors")
    if original.count("self.copy_event.synchronize()") != 2:
        raise ValueError("expected exactly two copy-event synchronize calls")

    output = original.replace(IMPORT_NEEDLE, IMPORT_REPLACEMENT, 1)
    insertion = output.index("\n\nclass AsyncOutput")
    output = output[:insertion] + HELPERS + output[insertion:]
    output = output.replace(
        "self.copy_event = torch.cuda.Event(blocking=True)",
        "self.copy_event = _new_copy_event()",
    )
    output = output.replace(
        "self.copy_event.synchronize()",
        "_wait_copy_event(self.copy_event)",
    )
    return output


def atomic_write(path: Path, content: str) -> None:
    descriptor, raw_temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        shutil.copymode(path, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def apply_patch(target: Path, dry_run: bool) -> None:
    original = target.read_text(encoding="utf-8")
    output = patched_text(original)
    backup = Path(str(target) + BACKUP_SUFFIX)
    print(f"target={target}")
    print(f"original_sha256={sha256(target)}")
    print(f"backup={backup}")
    if dry_run:
        print("dry-run: source patterns match; no file changed")
        return
    if not backup.exists():
        shutil.copy2(target, backup)
    elif MARKER in backup.read_text(encoding="utf-8"):
        raise SystemExit(f"refusing patched backup: {backup}")
    atomic_write(target, output)
    print(f"patched_sha256={sha256(target)}")
    print("restart every vLLM process before running an experiment")


def restore(target: Path, dry_run: bool) -> None:
    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.is_file():
        raise SystemExit(f"backup does not exist: {backup}")
    if MARKER in backup.read_text(encoding="utf-8"):
        raise SystemExit("backup contains the experiment marker; refusing restore")
    print(f"target={target}")
    print(f"backup={backup}")
    if dry_run:
        print("dry-run: restore is available; no file changed")
        return
    shutil.copy2(backup, target)
    print(f"restored_sha256={sha256(target)}")
    print("backup retained; restart every vLLM process")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "apply", "restore", "print-target"))
    parser.add_argument("--target", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    target = args.target.resolve() if args.target else locate_target()
    if args.action == "print-target":
        print(target)
        return
    if not target.is_file():
        raise SystemExit(f"target does not exist: {target}")

    version = installed_version()
    if (
        args.action == "apply"
        and version is not None
        and not version.startswith("0.26")
        and not args.allow_version_mismatch
    ):
        raise SystemExit(
            f"installed vLLM is {version}; expected 0.26.x. "
            "Use --allow-version-mismatch only after reviewing the target source."
        )

    content = target.read_text(encoding="utf-8")
    backup = Path(str(target) + BACKUP_SUFFIX)
    if args.action == "status":
        print(f"version={version or 'unknown'}")
        print(f"target={target}")
        print(f"patched={MARKER in content}")
        print(f"backup={backup if backup.exists() else 'missing'}")
        print(f"sha256={sha256(target)}")
    elif args.action == "apply":
        apply_patch(target, args.dry_run)
    else:
        restore(target, args.dry_run)


if __name__ == "__main__":
    main()
