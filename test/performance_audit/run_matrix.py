#!/usr/bin/env python3
"""Randomized round-robin orchestrator for correctness-gated audit samples."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import select
import statistics
import struct
import subprocess
import sys
import time
import tomllib


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER = HERE / "run_sample.jl"
SESSION_RUNNER = HERE / "run_sample_session.jl"
UPSTREAM_SESSION_RUNNER = HERE / "run_upstream_sample_session.jl"
with (HERE / "workloads.toml").open("rb") as inventory_file:
    INVENTORY = tuple(tomllib.load(inventory_file)["fixture"])
DEFAULT_FIXTURES = tuple(fixture["id"] for fixture in INVENTORY)
UPSTREAM_AVAILABLE = {
    fixture["id"] for fixture in INVENTORY if fixture["upstream"] != "unavailable"
}


def relative_mad(values: list[float]) -> float:
    center = statistics.median(values)
    if center == 0:
        return 0.0
    return statistics.median(abs(value - center) for value in values) / center


def bootstrap_half_width(values: list[float], rng: random.Random, draws: int = 2000) -> float:
    medians = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)
    ]
    medians.sort()
    low = medians[math.floor(0.025 * (draws - 1))]
    high = medians[math.ceil(0.975 * (draws - 1))]
    center = statistics.median(values)
    return (high - low) / 2 / center if center else 0.0


def cell_converged(values: list[float], minimum_samples: int, seed: int) -> bool:
    if len(values) < minimum_samples:
        return False
    return (
        relative_mad(values) <= 0.03
        and bootstrap_half_width(values, random.Random(seed)) <= 0.05
    )


def read_complex_field(path: pathlib.Path) -> list[complex]:
    raw = path.read_bytes()
    values = struct.unpack(f"={len(raw) // 8}d", raw)
    return [complex(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def relative_error(actual: list[complex], expected: list[complex]) -> float:
    if len(actual) != len(expected):
        return math.inf
    numerator = math.sqrt(sum(abs(a - b) ** 2 for a, b in zip(actual, expected)))
    denominator = math.sqrt(sum(abs(value) ** 2 for value in expected))
    return numerator / denominator if denominator else numerator


def latest_existing_field(
    sample_root: pathlib.Path, fixture: str, backend: str
) -> pathlib.Path:
    candidates = sorted(
        sample_root.glob(f"*-{fixture}-{backend}.json"), reverse=True
    )
    for candidate in candidates:
        result = json.loads(candidate.read_text(encoding="utf-8"))
        field = pathlib.Path(result["field_path"])
        if field.is_file():
            return field
    raise FileNotFoundError(
        f"no retained field for {fixture}/{backend} under {sample_root}"
    )


def physical_cpu_set(start_cpu: int, threads: int) -> str:
    """Choose one logical CPU from each physical core, rotating at start_cpu."""
    topology = subprocess.run(
        ["lscpu", "-p=CPU,CORE,SOCKET,NODE"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    representatives: list[int] = []
    seen: set[tuple[int, int, int]] = set()
    for line in topology:
        if not line or line.startswith("#"):
            continue
        cpu, core, socket, node = map(int, line.split(","))
        identity = (socket, node, core)
        if identity not in seen:
            seen.add(identity)
            representatives.append(cpu)
    if threads > len(representatives):
        raise ValueError(
            f"requested {threads} physical cores, host has {len(representatives)}"
        )
    start_index = representatives.index(start_cpu) if start_cpu in representatives else 0
    selected = (representatives[start_index:] + representatives[:start_index])[:threads]
    return ",".join(map(str, selected))


def sample_environment(threads: int, upstream: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "JULIA_DEPOT_PATH": "/tmp/luna-julia-depot:/home/diego/.julia",
            "JULIA_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "AMALTHEA_NATIVE_GPU": "off",
            "AMALTHEA_USE_RUST_CUDA_NATIVE": "0",
            "AMALTHEA_NATIVE_FFTW_WISDOM": "0",
            "AMALTHEA_AUDIT_FFTW_MODE": "estimate",
        }
    )
    if upstream:
        env["JULIA_DEPOT_PATH"] = "/tmp/luna-upstream-depot:/home/diego/.julia"
        env["MPLCONFIGDIR"] = "/tmp/matplotlib-audit"
    return env


def run_sample(
    fixture: str,
    size: str,
    backend: str,
    measurement: str,
    output: pathlib.Path,
    affinity: str,
    threads: int,
    timeout: int,
    upstream_project: pathlib.Path,
) -> dict:
    env = sample_environment(threads, upstream=backend == "upstream")
    if backend == "upstream":
        command = [
            "taskset", "-c", affinity, "julia", "--startup-file=no",
            f"--project={upstream_project}",
            str(HERE / "run_upstream_sample.jl"), fixture, size,
            measurement, str(output),
        ]
    else:
        command = [
            "taskset", "-c", affinity, "julia", "--startup-file=no", "--project",
            str(RUNNER), fixture, size, backend, measurement, str(output),
        ]
    proc = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=timeout,
    )
    if proc.returncode:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"sample failed: {' '.join(command)}")
    return json.loads(output.read_text(encoding="utf-8"))


class SampleSession:
    """One isolated backend process serving randomized one-sample requests."""

    def __init__(
        self,
        backend: str,
        affinity: str,
        threads: int,
        timeout: int,
        log_root: pathlib.Path,
        upstream_project: pathlib.Path,
    ) -> None:
        self.backend = backend
        self.timeout = timeout
        self.stderr_path = log_root / f"session-{backend}.stderr.log"
        self.protocol_path = log_root / f"session-{backend}.stdout.log"
        self.stderr_file = self.stderr_path.open("a", encoding="utf-8")
        self.protocol_file = self.protocol_path.open("a", encoding="utf-8")
        if backend == "upstream":
            command = [
                "taskset", "-c", affinity, "julia", "--startup-file=no",
                f"--project={upstream_project}", str(UPSTREAM_SESSION_RUNNER),
            ]
        else:
            command = [
                "taskset", "-c", affinity, "julia", "--startup-file=no",
                "--project", str(SESSION_RUNNER), backend,
            ]
        self.proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=sample_environment(threads, upstream=backend == "upstream"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            bufsize=0,
        )
        self.stdout_buffer = b""
        try:
            self._wait_for("__AUDIT_READY__")
        except BaseException:
            self._terminate()
            self.stderr_file.close()
            self.protocol_file.close()
            raise

    def _wait_for(self, marker: str) -> str:
        assert self.proc.stdout is not None
        deadline = time.monotonic() + self.timeout
        while True:
            while b"\n" not in self.stdout_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"{self.backend} sample session exceeded {self.timeout}s"
                    )
                readable, _, _ = select.select(
                    [self.proc.stdout.fileno()], [], [], remaining)
                if not readable:
                    raise TimeoutError(
                        f"{self.backend} sample session exceeded {self.timeout}s"
                    )
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    returncode = self.proc.poll()
                    raise RuntimeError(
                        f"{self.backend} sample session exited with "
                        f"return code {returncode}; see {self.stderr_path}"
                    )
                self.stdout_buffer += chunk
            raw_line, self.stdout_buffer = self.stdout_buffer.split(b"\n", 1)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError(
                    f"invalid UTF-8 from {self.backend} sample session"
                ) from error
            self.protocol_file.write(line + "\n")
            self.protocol_file.flush()
            if line.startswith(marker):
                return line

    def run(
        self,
        fixture: str,
        size: str,
        measurement: str,
        output: pathlib.Path,
    ) -> dict:
        assert self.proc.stdin is not None
        request = "\t".join((fixture, size, measurement, str(output))) + "\n"
        self.proc.stdin.write(request.encode("utf-8"))
        self.proc.stdin.flush()
        response = self._wait_for("__AUDIT_")
        if not response.startswith("__AUDIT_OK__\t"):
            raise RuntimeError(
                f"{self.backend} sample failed for {fixture}; "
                f"see {self.stderr_path}"
            )
        return json.loads(output.read_text(encoding="utf-8"))

    def rss_bytes(self) -> int:
        status = pathlib.Path(f"/proc/{self.proc.pid}/status")
        if not status.is_file():
            return 0
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        return 0

    def _terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        if self.proc.stdout is not None and not self.proc.stdout.closed:
            self.proc.stdout.close()

    def close(self) -> None:
        if (self.proc.poll() is None and self.proc.stdin is not None
                and not self.proc.stdin.closed):
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._terminate()
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        if self.proc.stdout is not None and not self.proc.stdout.closed:
            self.proc.stdout.close()
        self.stderr_file.close()
        self.protocol_file.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="append", choices=DEFAULT_FIXTURES)
    parser.add_argument(
        "--exclude-fixture", action="append", choices=DEFAULT_FIXTURES,
        help="exclude a known measurement-specific correctness failure",
    )
    parser.add_argument("--size", choices=("small", "medium", "large"), default="small")
    parser.add_argument(
        "--measurement",
        choices=("setup", "field_sync", "fixed_rhs", "fixed_step", "fixed_solve", "fixed_solve_raw", "adaptive_solve",
                 "dense_output", "result_copy"),
        default="adaptive_solve",
    )
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument("--maximum-samples", type=int, default=30)
    parser.add_argument("--core", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fresh-process-per-sample", action="store_true",
        help="disable persistent per-backend timing sessions",
    )
    parser.add_argument(
        "--session-rss-limit-gib", type=float, default=6.0,
        help="recycle a persistent backend process above this resident size",
    )
    parser.add_argument("--include-upstream", action="store_true")
    parser.add_argument(
        "--upstream-project", type=pathlib.Path,
        default=pathlib.Path("/tmp/amalthea-upstream-0a52ffb"),
    )
    parser.add_argument(
        "--correctness-gate", type=pathlib.Path,
        help="gate JSON; defaults to results/correctness-SIZE.json",
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.include_upstream and not (args.upstream_project / "Project.toml").is_file():
        raise SystemExit(
            f"missing pinned upstream project {args.upstream_project}; "
            "recreate it with the commands in test/performance_audit/README.md"
        )
    requested_fixtures = args.fixture or list(DEFAULT_FIXTURES)
    gate_path = args.correctness_gate or (
        HERE / "results" / f"correctness-{args.size}.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("size") != args.size:
        raise SystemExit(f"correctness gate {gate_path} has wrong size")
    gate_results = {item["fixture"]: item for item in gate["results"]}
    missing_gate = [item for item in requested_fixtures if item not in gate_results]
    if missing_gate:
        raise SystemExit(f"fixtures missing from correctness gate: {missing_gate}")
    excluded_fixtures = [
        item for item in requested_fixtures if not gate_results[item].get("passed")
    ]
    for item in args.exclude_fixture or ():
        if item in requested_fixtures and item not in excluded_fixtures:
            excluded_fixtures.append(item)
    if args.include_upstream:
        excluded_fixtures.extend(
            item for item in requested_fixtures
            if item not in UPSTREAM_AVAILABLE and item not in excluded_fixtures
        )
    fixtures = [item for item in requested_fixtures if item not in excluded_fixtures]
    if not fixtures:
        raise SystemExit("no correctness-gated fixtures selected")
    output_root = args.output or (
        HERE / "results" / f"matrix-{args.measurement}-{args.size}.json"
    )
    sample_root = output_root.with_suffix("")
    sample_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    affinity = physical_cpu_set(args.core, args.threads)
    backends = ("julia", "rust", "upstream") if args.include_upstream else ("julia", "rust")
    samples: dict[tuple[str, str], list[dict]] = {
        (fixture, backend): []
        for fixture in fixtures
        for backend in backends
    }

    sessions: dict[str, SampleSession] = {}
    try:
        for round_index in range(args.maximum_samples):
            cells = list(samples)
            if round_index >= args.minimum_samples:
                cells = [
                    cell for cell in cells
                    if not cell_converged(
                        [sample["elapsed_seconds"] for sample in samples[cell]],
                        args.minimum_samples,
                        args.seed + round_index - 1,
                    )
                ]
            if not cells:
                break
            rng.shuffle(cells)
            for cell_index, (fixture, backend) in enumerate(cells, start=1):
                destination = sample_root / f"{round_index:02d}-{fixture}-{backend}.json"
                print(
                    f"round {round_index + 1}/{args.maximum_samples} "
                    f"cell {cell_index}/{len(cells)}: starting "
                    f"{fixture}/{backend}",
                    flush=True,
                )
                if args.resume and destination.is_file():
                    result = json.loads(destination.read_text(encoding="utf-8"))
                elif not args.fresh_process_per_sample:
                    if backend not in sessions:
                        sessions[backend] = SampleSession(
                            backend, affinity, args.threads, args.timeout,
                            sample_root, args.upstream_project)
                    result = sessions[backend].run(
                        fixture, args.size, args.measurement, destination)
                    session_rss = sessions[backend].rss_bytes()
                    if session_rss > args.session_rss_limit_gib * 1024**3:
                        sessions[backend].protocol_file.write(
                            f"__AUDIT_RECYCLE_RSS__\t{session_rss}\n")
                        sessions[backend].protocol_file.flush()
                        sessions[backend].close()
                        del sessions[backend]
                else:
                    result = run_sample(
                        fixture,
                        args.size,
                        backend,
                        args.measurement,
                        destination,
                        affinity,
                        args.threads,
                        args.timeout,
                        args.upstream_project,
                    )
                samples[(fixture, backend)].append(result)
                print(
                    f"round {round_index + 1}/{args.maximum_samples} "
                    f"cell {cell_index}/{len(cells)}: completed "
                    f"{fixture}/{backend} in "
                    f"{result['elapsed_seconds']:.6g}s",
                    flush=True,
                )
                # Full fields can be tens of MiB for large free-space cells. The
                # final sample in each cell is sufficient for the post-matrix
                # correctness check; retain every timing JSON but only that field.
                if len(samples[(fixture, backend)]) > 1:
                    prior_field = pathlib.Path(
                        samples[(fixture, backend)][-2]["field_path"]
                    )
                    if prior_field.is_file():
                        prior_field.unlink()

            if round_index + 1 >= args.minimum_samples:
                converged = all(
                    cell_converged(
                        [sample["elapsed_seconds"] for sample in cell_samples],
                        args.minimum_samples,
                        args.seed + round_index,
                    )
                    for cell_samples in samples.values()
                )
                if converged:
                    break
    finally:
        for session in sessions.values():
            session.close()

    summary_cells = []
    correctness = []
    for fixture in fixtures:
        julia_field = read_complex_field(
            latest_existing_field(sample_root, fixture, "julia")
        )
        rust_field = read_complex_field(
            latest_existing_field(sample_root, fixture, "rust")
        )
        batch_error = relative_error(rust_field, julia_field)
        if args.measurement == "fixed_rhs":
            error = gate_results[fixture]["single_step_relative_error"]
            tolerance = gate_results[fixture]["single_step_tolerance"]
            correctness_source = "frozen-strict-single-step-gate"
        else:
            error = batch_error
            tolerance = gate_results[fixture]["fixed_solve_tolerance"]
            correctness_source = "final-timed-sample-field"
        correctness.append(
            {"fixture": fixture, "backend": "rust",
             "relative_error": error, "tolerance": tolerance,
             "passed": error < tolerance,
             "source": correctness_source,
             "timed_batch_final_field_relative_error": batch_error}
        )
        if args.include_upstream:
            upstream_field = read_complex_field(
                latest_existing_field(sample_root, fixture, "upstream")
            )
            upstream_error = relative_error(upstream_field, julia_field)
            correctness.append(
                {"fixture": fixture, "backend": "upstream",
                 "relative_error": upstream_error, "tolerance": tolerance,
                 "passed": upstream_error < tolerance}
            )
        for backend in backends:
            cell_samples = samples[(fixture, backend)]
            values = [sample["elapsed_seconds"] for sample in cell_samples]
            rss_values = [
                sample["peak_rss_bytes"] for sample in cell_samples
                if sample.get("process_mode", "fresh") == "fresh"
            ]
            check_rng = random.Random(args.seed)
            summary_cells.append(
                {
                    "fixture": fixture,
                    "backend": backend,
                    "samples": len(values),
                    "median_seconds": statistics.median(values),
                    "relative_mad": relative_mad(values),
                    "bootstrap_ci_half_width_relative": bootstrap_half_width(values, check_rng),
                    "allocated_bytes_median": statistics.median(
                        sample["allocated_bytes"] for sample in cell_samples
                    ),
                    "peak_rss_bytes_median": (
                        statistics.median(rss_values) if rss_values else None
                    ),
                    "peak_rss_samples": len(rss_values),
                    "accepted_steps": cell_samples[-1]["accepted_steps"],
                    "rejected_steps": cell_samples[-1]["rejected_steps"],
                    "derived_rhs_evaluations": cell_samples[-1]["derived_rhs_evaluations"],
                }
            )
    result = {
        "schema_version": 2,
        "size": args.size,
        "measurement": args.measurement,
        "seed": args.seed,
        "core": args.core,
        "affinity": affinity,
        "threads": args.threads,
        "correctness_gate": str(gate_path),
        "correctness_excluded_fixtures": excluded_fixtures,
        "requested_excluded_fixtures": args.exclude_fixture or [],
        "include_upstream": args.include_upstream,
        "process_protocol": (
            "fresh-per-sample" if args.fresh_process_per_sample else
            "persistent-isolated-backend-sessions"
        ),
        "cells": summary_cells,
        "correctness": correctness,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_root)
    if not all(item["passed"] for item in correctness):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
