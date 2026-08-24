#!/usr/bin/env python3
"""Run every currently implemented audit fixture correctness gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import subprocess
import tomllib


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKER = HERE / "check_fixture.jl"
with (HERE / "workloads.toml").open("rb") as inventory_file:
    FIXTURES = tuple(
        fixture["id"] for fixture in tomllib.load(inventory_file)["fixture"]
    )


def run_one(fixture: str, size: str, timeout: int) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "JULIA_DEPOT_PATH": "/tmp/luna-julia-depot:/home/diego/.julia",
            "JULIA_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "AMALTHEA_NATIVE_GPU": "off",
            "AMALTHEA_USE_RUST_CUDA_NATIVE": "0",
            "AMALTHEA_NATIVE_FFTW_WISDOM": "0",
            "AMALTHEA_AUDIT_FFTW_MODE": "estimate",
        }
    )
    command = [
        "julia",
        "--startup-file=no",
        "--project",
        str(CHECKER),
        fixture,
        size,
    ]
    try:
        proc = subprocess.run(
            command, cwd=ROOT, env=env, text=True, capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "fixture": fixture,
            "passed": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timed out after {timeout} seconds",
        }
    if proc.returncode:
        return {
            "fixture": fixture,
            "passed": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    parsed: dict[str, object] = {"fixture": fixture, "passed": True}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {
            "field_length",
            "single_step_relative_error",
            "single_step_tolerance",
            "fixed_solve_relative_error",
            "fixed_solve_tolerance",
            "feature_effect_relative",
            "feature_effect_tolerance",
        }:
            parsed[key] = float(value)
        elif key in {"size", "backend"}:
            parsed[key] = value
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--fixture", action="append", choices=FIXTURES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=HERE / "results" / "correctness-checkpoint2.json",
    )
    args = parser.parse_args()
    selected = tuple(args.fixture or FIXTURES)

    def checkpoint(results: list[dict]) -> None:
        document = {
            "schema_version": 1,
            "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "size": args.size,
            "fixtures_expected": len(selected),
            "fixtures_completed": len(results),
            "fixtures_passed": sum(bool(result["passed"]) for result in results),
            "results": sorted(results, key=lambda item: selected.index(item["fixture"])),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    results: list[dict] = []
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("size") != args.size:
            raise SystemExit("resume output has a different size")
        results.extend(
            item for item in previous.get("results", [])
            if item.get("fixture") in selected
        )
    completed = {item["fixture"] for item in results if item.get("passed")}
    if args.resume:
        results = [item for item in results if item["fixture"] in completed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, fixture, args.size, args.timeout): fixture
            for fixture in selected if fixture not in completed
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            checkpoint(results)
    checkpoint(results)
    document = json.loads(args.output.read_text(encoding="utf-8"))
    print(args.output)
    return 0 if document["fixtures_passed"] == document["fixtures_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
