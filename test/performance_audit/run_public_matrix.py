#!/usr/bin/env python3
"""Randomized fresh-process matrix for the historical/public full API calls."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import random
import statistics
import struct
import subprocess

from run_matrix import bootstrap_half_width, physical_cpu_set, relative_mad


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_field(path: pathlib.Path) -> list[complex]:
    raw = path.read_bytes()
    values = struct.unpack(f"={len(raw) // 8}d", raw)
    return [complex(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def relative_error(actual: list[complex], expected: list[complex]) -> float:
    if len(actual) != len(expected):
        return math.inf
    numerator = math.sqrt(sum(abs(a - b) ** 2 for a, b in zip(actual, expected)))
    denominator = math.sqrt(sum(abs(value) ** 2 for value in expected))
    return numerator / denominator if denominator else numerator


def run_sample(
    config: str,
    backend: str,
    destination: pathlib.Path,
    affinity: str,
    upstream_project: pathlib.Path,
    timeout: int,
) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "JULIA_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "AMALTHEA_AUDIT_FFTW_MODE": "estimate",
            "AMALTHEA_NATIVE_GPU": "off",
            "AMALTHEA_USE_RUST_CUDA_NATIVE": "0",
            "AMALTHEA_NATIVE_FFTW_WISDOM": "0",
            "MPLCONFIGDIR": "/tmp/matplotlib-audit",
        }
    )
    if backend == "upstream":
        env["JULIA_DEPOT_PATH"] = "/tmp/luna-upstream-depot:/home/diego/.julia"
        command = [
            "taskset", "-c", affinity, "julia", "--startup-file=no",
            f"--project={upstream_project}",
            str(HERE / "run_upstream_public_benchmark.jl"),
            config, str(destination),
        ]
    else:
        env["JULIA_DEPOT_PATH"] = "/tmp/luna-julia-depot:/home/diego/.julia"
        command = [
            "taskset", "-c", affinity, "julia", "--startup-file=no", "--project",
            str(HERE / "run_public_benchmark.jl"),
            config, backend, str(destination),
        ]
    proc = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=timeout,
    )
    if proc.returncode:
        raise RuntimeError(
            f"{backend} sample failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(destination.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=("phase_c_reconstruction", "readme_v103"),
        required=True,
    )
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument("--maximum-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--core", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--upstream-project", type=pathlib.Path,
        default=pathlib.Path("/tmp/amalthea-upstream-0a52ffb"),
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    output = args.output or HERE / "results" / f"public-{args.config}.json"
    sample_root = output.with_suffix("")
    sample_root.mkdir(parents=True, exist_ok=True)
    affinity = physical_cpu_set(args.core, 1)
    rng = random.Random(args.seed)
    backends = ("julia", "rust", "upstream")
    samples: dict[str, list[dict]] = {backend: [] for backend in backends}

    for round_index in range(args.maximum_samples):
        order = list(backends)
        rng.shuffle(order)
        for backend in order:
            destination = sample_root / f"{round_index:02d}-{backend}.json"
            if args.resume and destination.is_file():
                result = json.loads(destination.read_text(encoding="utf-8"))
            else:
                result = run_sample(
                    args.config, backend, destination, affinity,
                    args.upstream_project, args.timeout,
                )
            samples[backend].append(result)
            if len(samples[backend]) > 1:
                prior = pathlib.Path(samples[backend][-2]["field_path"])
                if prior.is_file():
                    prior.unlink()
        if round_index + 1 >= args.minimum_samples:
            if all(
                relative_mad([item["elapsed_seconds"] for item in values]) <= 0.03
                and bootstrap_half_width(
                    [item["elapsed_seconds"] for item in values],
                    random.Random(args.seed + round_index),
                ) <= 0.05
                for values in samples.values()
            ):
                break

    oracle = read_field(pathlib.Path(samples["julia"][-1]["field_path"]))
    summaries = []
    equivalence = []
    for backend in backends:
        values = [item["elapsed_seconds"] for item in samples[backend]]
        summaries.append(
            {
                "backend": backend,
                "samples": len(values),
                "median_seconds": statistics.median(values),
                "relative_mad": relative_mad(values),
                "bootstrap_ci_half_width_relative": bootstrap_half_width(
                    values, random.Random(args.seed)
                ),
                "allocated_bytes_median": statistics.median(
                    item["allocated_bytes"] for item in samples[backend]
                ),
                "peak_rss_bytes_median": statistics.median(
                    item["peak_rss_bytes"] for item in samples[backend]
                ),
            }
        )
        field = read_field(pathlib.Path(samples[backend][-1]["field_path"]))
        error = relative_error(field, oracle)
        equivalence.append(
            {"backend": backend, "relative_error": error, "passed": error < 1e-6}
        )
    document = {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": args.config,
        "seed": args.seed,
        "affinity": affinity,
        "summaries": summaries,
        "equivalence": equivalence,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0 if all(item["passed"] for item in equivalence) else 1


if __name__ == "__main__":
    raise SystemExit(main())
