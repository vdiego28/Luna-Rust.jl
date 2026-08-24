#!/usr/bin/env python3
"""Test whether modal equivalence loss scales with cubature stopping policy."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import struct
import subprocess


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_field(path: pathlib.Path) -> list[complex]:
    raw = path.read_bytes()
    values = struct.unpack(f"={len(raw) // 8}d", raw)
    return [complex(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def relative_error(actual: list[complex], expected: list[complex]) -> float:
    numerator = math.sqrt(sum(abs(a - b) ** 2 for a, b in zip(actual, expected)))
    denominator = math.sqrt(sum(abs(value) ** 2 for value in expected))
    return numerator / denominator if denominator else numerator


def sample(
    fixture: str, size: str, backend: str, rtol: float, maxevals: int,
    destination: pathlib.Path, timeout: int,
) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "JULIA_DEPOT_PATH": "/tmp/luna-julia-depot:/home/diego/.julia",
            "JULIA_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "AMALTHEA_NATIVE_GPU": "off",
            "AMALTHEA_AUDIT_FFTW_MODE": "estimate",
            "AMALTHEA_AUDIT_MODAL_RTOL": str(rtol),
            "AMALTHEA_AUDIT_MODAL_MAXEVALS": str(maxevals),
        }
    )
    proc = subprocess.run(
        [
            "taskset", "-c", "2", "julia", "--startup-file=no", "--project",
            str(HERE / "run_sample.jl"), fixture, size, backend,
            "fixed_rhs", str(destination),
        ],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout,
    )
    if proc.returncode:
        raise RuntimeError(f"{fixture}/{size}/{backend} failed:\n{proc.stderr}")
    return json.loads(destination.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=HERE / "results" / "modal-cubature-diagnostic.json",
    )
    args = parser.parse_args()
    sample_root = args.output.with_suffix("")
    sample_root.mkdir(parents=True, exist_ok=True)
    results = []
    for fixture in ("modal_real_scalar", "modal_real_general_modes"):
        for size in ("small", "medium", "large"):
            for rtol, maxevals in (
                (1e-3, 512), (1e-4, 1024), (1e-5, 4096), (1e-6, 16384),
            ):
                cell = {}
                for backend in ("julia", "rust"):
                    path = sample_root / (
                        f"{fixture}-{size}-rtol{rtol:g}-max{maxevals}-{backend}.json"
                    )
                    cell[backend] = sample(
                        fixture, size, backend, rtol, maxevals, path, args.timeout
                    )
                error = relative_error(
                    read_field(pathlib.Path(cell["rust"]["field_path"])),
                    read_field(pathlib.Path(cell["julia"]["field_path"])),
                )
                results.append(
                    {
                        "fixture": fixture,
                        "size": size,
                        "rtol": rtol,
                        "maxevals": maxevals,
                        "rhs_relative_error": error,
                        "julia_seconds": cell["julia"]["elapsed_seconds"],
                        "rust_seconds": cell["rust"]["elapsed_seconds"],
                    }
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "results": results,
                        },
                        indent=2,
                        sort_keys=True,
                    ) + "\n",
                    encoding="utf-8",
                )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
