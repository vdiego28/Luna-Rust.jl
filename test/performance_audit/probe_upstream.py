#!/usr/bin/env python3
"""Probe pinned upstream API availability and numerical equivalence."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import pathlib
import struct
import subprocess
import tomllib


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


def run(
    command: list[str], env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=timeout,
    )


def probe_one(
    fixture: dict, upstream_project: pathlib.Path, output_root: pathlib.Path,
    timeout: int, size: str,
) -> dict:
    fixture_id = fixture["id"]
    env = os.environ.copy()
    env.update(
        {
            "JULIA_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib-audit",
            "AMALTHEA_NATIVE_GPU": "off",
            "AMALTHEA_USE_RUST_CUDA_NATIVE": "0",
            "AMALTHEA_AUDIT_FFTW_MODE": "estimate",
            "AMALTHEA_AUDIT_WARMUPS": "0",
        }
    )
    fork_output = output_root / f"{fixture_id}-fork.json"
    upstream_output = output_root / f"{fixture_id}-upstream.json"
    fork_env = env | {"JULIA_DEPOT_PATH": "/tmp/luna-julia-depot:/home/diego/.julia"}
    upstream_env = env | {
        "JULIA_DEPOT_PATH": "/tmp/luna-upstream-depot:/home/diego/.julia"
    }
    try:
        fork = run(
            [
                "julia", "--startup-file=no", "--project",
                str(HERE / "run_sample.jl"), fixture_id, size, "julia",
                "fixed_solve_raw", str(fork_output),
            ],
            fork_env, timeout,
        )
    except subprocess.TimeoutExpired:
        return {"fixture": fixture_id, "declared": fixture["upstream"],
                "status": "fork_timeout"}
    if fork.returncode:
        return {
            "fixture": fixture_id,
            "declared": fixture["upstream"],
            "status": "fork_fixture_failed",
            "stderr": fork.stderr,
        }
    try:
        upstream = run(
            [
                "julia", "--startup-file=no", f"--project={upstream_project}",
                str(HERE / "run_upstream_sample.jl"), fixture_id, size,
                # Raw state avoids upstream's known dense-output FSAL defect.
                "fixed_solve_raw", str(upstream_output),
            ],
            upstream_env, timeout,
        )
    except subprocess.TimeoutExpired:
        return {"fixture": fixture_id, "declared": fixture["upstream"],
                "status": "upstream_timeout"}
    if upstream.returncode:
        return {
            "fixture": fixture_id,
            "declared": fixture["upstream"],
            "status": "unavailable",
            "stderr": upstream.stderr,
        }
    error = relative_error(
        read_field(pathlib.Path(json.loads(upstream_output.read_text())["field_path"])),
        read_field(pathlib.Path(json.loads(fork_output.read_text())["field_path"])),
    )
    return {
        "fixture": fixture_id,
        "declared": fixture["upstream"],
        "status": "equivalent" if error < 1e-6 else "numerically_different",
        "relative_error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-project", type=pathlib.Path,
        default=pathlib.Path("/tmp/amalthea-upstream-0a52ffb"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--size", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--fixture", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--correctness-gate", type=pathlib.Path,
        help="only probe fixtures that passed this size-matched gate JSON",
    )
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=HERE / "results" / "upstream-probe.json",
    )
    args = parser.parse_args()
    with (HERE / "workloads.toml").open("rb") as inventory_file:
        fixtures = [
            item for item in tomllib.load(inventory_file)["fixture"]
            if item["upstream"] != "unavailable"
        ]
    if args.fixture:
        requested = set(args.fixture)
        fixtures = [item for item in fixtures if item["id"] in requested]
        missing = requested - {item["id"] for item in fixtures}
        if missing:
            raise SystemExit(f"unknown or unavailable fixtures: {sorted(missing)}")
    if args.correctness_gate:
        gate = json.loads(args.correctness_gate.read_text(encoding="utf-8"))
        if gate.get("size") != args.size:
            raise SystemExit("correctness gate has a different size")
        passed = {
            item["fixture"] for item in gate.get("results", [])
            if item.get("passed")
        }
        fixtures = [item for item in fixtures if item["id"] in passed]
    sample_root = args.output.with_suffix("")
    sample_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("size", "small") != args.size:
            raise SystemExit("resume output has a different size")
        selected_ids = {item["id"] for item in fixtures}
        results.extend(
            item for item in previous.get("fixtures", [])
            if item.get("fixture") in selected_ids
        )
    completed = {item["fixture"] for item in results}

    def checkpoint() -> None:
        order = {item["id"]: index for index, item in enumerate(fixtures)}
        document = {
            "schema_version": 1,
            "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "upstream_project": str(args.upstream_project),
            "size": args.size,
            "fixtures_expected": len(fixtures),
            "fixtures_completed": len(results),
            "fixtures": sorted(
                results, key=lambda item: order.get(item["fixture"], len(order))
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                probe_one, item, args.upstream_project, sample_root,
                args.timeout, args.size,
            ): item["id"]
            for item in fixtures if item["id"] not in completed
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            checkpoint()
    checkpoint()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
