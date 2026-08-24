#!/usr/bin/env python3
"""Capture the immutable inputs and host state for the CPU performance audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(*argv: str, cwd: pathlib.Path = ROOT) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        return {"argv": list(argv), "available": False, "error": str(exc)}
    return {
        "argv": list(argv),
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.rstrip(),
        "stderr": proc.stderr.rstrip(),
    }


def output(*argv: str, cwd: pathlib.Path = ROOT) -> str | None:
    result = run(*argv, cwd=cwd)
    if result.get("returncode") != 0:
        return None
    return str(result["stdout"])


def read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def sha256(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_cpu_record() -> dict[str, str]:
    text = read_text(pathlib.Path("/proc/cpuinfo")) or ""
    record: dict[str, str] = {}
    for line in text.split("\n\n", 1)[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            record[key.strip()] = value.strip()
    keep = ("vendor_id", "cpu family", "model", "model name", "stepping", "microcode")
    return {key: record[key] for key in keep if key in record}


def sysfs_values(pattern: str) -> dict[str, str | None]:
    return {
        str(path): read_text(path)
        for path in sorted(pathlib.Path("/").glob(pattern.lstrip("/")))
    }


def relevant_environment() -> dict[str, str | None]:
    names = (
        "JULIA_NUM_THREADS",
        "JULIA_CPU_TARGET",
        "JULIA_DEPOT_PATH",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
        "RUSTFLAGS",
        "AMALTHEA_CUDA_BUILD",
        "AMALTHEA_USE_RUST_NATIVE",
        "AMALTHEA_USE_RUST_CUDA_NATIVE",
        "AMALTHEA_NATIVE_GPU",
        "AMALTHEA_NATIVE_FFTW_WISDOM",
        "AMALTHEA_NATIVE_DETERMINISTIC",
        "AMALTHEA_QDHT_BLAS",
    )
    return {name: os.environ.get(name) for name in names}


def parse_kv(text: str | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (text or "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def artifact(path: pathlib.Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "exists": path.is_file(),
        "sha256": sha256(path),
    }
    if path.is_file():
        stat = path.stat()
        result.update({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        result["file"] = output("file", str(path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=HERE / "results" / "baseline.json",
    )
    parser.add_argument(
        "--expected-amalthea",
        default="73e32dcf45d93f11136d419faeae3b3641c9577d",
    )
    parser.add_argument(
        "--expected-upstream",
        default="0a52ffbba6d5dd6820bb3dc3c300b8b38d724214",
    )
    args = parser.parse_args()

    amalthea_commit = output("git", "rev-parse", "HEAD")
    upstream_commit = output("git", "rev-parse", "upstream/master")
    status_lines = (output("git", "status", "--porcelain=v1") or "").splitlines()
    dirty_paths = [line[3:] for line in status_lines if len(line) >= 4]
    runtime_dirty = [
        path
        for path in dirty_paths
        if path.startswith(("src/", "amalthea/src/", "Project.toml", "Manifest.toml"))
    ]

    julia_probe = output(
        "julia",
        "--startup-file=no",
        "--project",
        "-e",
        "using FFTW, LinearAlgebra; "
        "println(\"julia_version=\", VERSION); "
        "println(\"julia_commit=\", Base.GIT_VERSION_INFO.commit); "
        "println(\"machine=\", Sys.MACHINE); "
        "println(\"kernel=\", Sys.KERNEL); "
        "println(\"cpu_name=\", Sys.CPU_NAME); "
        "println(\"julia_threads=\", Threads.nthreads()); "
        "println(\"fftw_version=\", FFTW.version); "
        "println(\"fftw_provider=\", FFTW.get_provider()); "
        "println(\"fftw_library=\", FFTW.FFTW_jll.libfftw3); "
        "println(\"blas=\", BLAS.get_config())",
    )

    manifest_files = (
        "Project.toml",
        "Manifest.toml",
        "amalthea/Cargo.toml",
        "amalthea/Cargo.lock",
        "test/performance_audit/upstream/Project.toml",
        "test/performance_audit/upstream/Manifest.toml",
    )
    data: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline": {
            "amalthea_commit": amalthea_commit,
            "amalthea_expected_commit": args.expected_amalthea,
            "amalthea_commit_matches": amalthea_commit == args.expected_amalthea,
            "upstream_url": "https://github.com/LupoLab/Luna.jl.git",
            "upstream_ref": "refs/remotes/upstream/master",
            "upstream_commit": upstream_commit,
            "upstream_expected_commit": args.expected_upstream,
            "upstream_commit_matches": upstream_commit == args.expected_upstream,
            "upstream_resolved_manifest_sha256": sha256(
                ROOT / "test/performance_audit/upstream/Manifest.toml"
            ),
            "git_status": status_lines,
            "dirty_paths": dirty_paths,
            "runtime_source_dirty": runtime_dirty,
            "portable_build_contract": {
                "command": "AMALTHEA_RUST_SKIP_DOWNLOAD=1 AMALTHEA_CUDA_BUILD=off RUSTFLAGS='' julia --startup-file=no --project deps/build.jl",
                "rustflags": "",
                "cuda_build": "off",
                "profile": "release",
                "acceptance_role": "installed-default portable resident CPU backend",
            },
        },
        "inputs": {
            name: {"sha256": sha256(ROOT / name)} for name in manifest_files
        },
        "artifacts": {
            "portable_release": artifact(ROOT / "amalthea/target/release/libamalthea.so")
        },
        "toolchain": {
            "julia": parse_kv(julia_probe),
            "rustc": run("rustc", "-vV"),
            "cargo": run("cargo", "-V"),
            "perf": run("perf", "--version"),
        },
        "host": {
            "uname": run("uname", "-a"),
            "lscpu_json": run("lscpu", "--json"),
            "cpu": first_cpu_record(),
            "kernel_cmdline": read_text(pathlib.Path("/proc/cmdline")),
            "memory": {
                key: value
                for key, value in (
                    line.split(":", 1)
                    for line in (read_text(pathlib.Path("/proc/meminfo")) or "").splitlines()
                    if ":" in line
                )
                if key in {"MemTotal", "SwapTotal", "HugePages_Total", "Hugepagesize"}
            },
            "affinity": run("taskset", "-pc", str(os.getpid())),
            "governors": sysfs_values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"),
            "boost": {
                "cpufreq_boost": read_text(pathlib.Path("/sys/devices/system/cpu/cpufreq/boost")),
                "amd_pstate": read_text(pathlib.Path("/sys/devices/system/cpu/amd_pstate/status")),
            },
            "perf_event_paranoid": read_text(pathlib.Path("/proc/sys/kernel/perf_event_paranoid")),
        },
        "environment": relevant_environment(),
    }

    problems: list[str] = []
    if not data["baseline"]["amalthea_commit_matches"]:
        problems.append("Amalthea HEAD differs from the frozen baseline")
    if not data["baseline"]["upstream_commit_matches"]:
        problems.append("upstream/master differs from the frozen upstream baseline")
    if runtime_dirty:
        problems.append("runtime source or dependency metadata is dirty")
    if not data["artifacts"]["portable_release"]["exists"]:
        problems.append("portable release library is missing")
    data["validation"] = {"passed": not problems, "problems": problems}

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    temporary.replace(destination)
    print(destination)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
