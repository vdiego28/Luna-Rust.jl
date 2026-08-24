#!/usr/bin/env python3
"""5--10 minute Apple Silicon correctness and performance diagnostic."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import platform
import shutil
import struct
import subprocess
import tempfile
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AMALTHEA = ROOT / "amalthea"


def run(argv: list[str], *, cwd: pathlib.Path = ROOT,
        env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(argv)}\n{proc.stdout}\n{proc.stderr}")
    return proc


def text(argv: list[str]) -> str | None:
    try:
        proc = run(argv, check=False)
    except FileNotFoundError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def sysctl(name: str) -> str | None:
    return text(["sysctl", "-n", name])


def julia_metadata() -> dict[str, str]:
    code = (
        "using FFTW, LinearAlgebra; "
        "println(\"julia=\",VERSION); println(\"machine=\",Sys.MACHINE); "
        "println(\"cpu=\",Sys.CPU_NAME); println(\"threads=\",Threads.nthreads()); "
        "println(\"fftw_version=\",FFTW.version); "
        "println(\"fftw_library=\",FFTW.FFTW_jll.libfftw3); "
        "println(\"blas=\",replace(string(BLAS.get_config()),'\\n'=>' '))"
    )
    proc = run(["julia", "--startup-file=no", "--project", "-e", code])
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def artifact_path() -> pathlib.Path:
    if platform.system() == "Darwin":
        name = "libamalthea.dylib"
    elif platform.system() == "Windows":
        name = "amalthea.dll"
    else:
        name = "libamalthea.so"
    return AMALTHEA / "target" / "release" / name


def build_portable(env: dict[str, str]) -> None:
    build_env = env | {
        "AMALTHEA_RUST_SKIP_DOWNLOAD": "1",
        "AMALTHEA_CUDA_BUILD": "off",
        "RUSTFLAGS": "",
    }
    run(["julia", "--startup-file=no", "--project", "deps/build.jl"], env=build_env)


def build_native_diagnostic(env: dict[str, str]) -> None:
    build_env = env | {
        "AMALTHEA_CUDA_BUILD": "off",
        "RUSTFLAGS": "-C target-cpu=native",
        "CARGO_PROFILE_RELEASE_LTO": "thin",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
    }
    run(["cargo", "build", "--release"], cwd=AMALTHEA, env=build_env)


def read_complex_field(path: pathlib.Path) -> list[complex]:
    raw = path.read_bytes()
    values = struct.unpack(f"={len(raw) // 8}d", raw)
    return [complex(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def relative_error(a: list[complex], b: list[complex]) -> float:
    if len(a) != len(b):
        return math.inf
    den = math.sqrt(sum(abs(x) ** 2 for x in a))
    return math.sqrt(sum(abs(x - y) ** 2 for x, y in zip(a, b))) / max(den, 1e-300)


def sample(build: str, fixture: str, threads: int, sample_index: int,
           result_dir: pathlib.Path, env: dict[str, str]) -> dict[str, Any]:
    output = result_dir / f"{build}-{fixture}-t{threads}-s{sample_index}.json"
    sample_env = env | {
        "JULIA_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "AMALTHEA_QDHT_BLAS": "auto",
        "AMALTHEA_AUDIT_WARMUPS": "1",
        "AMALTHEA_AUDIT_REPETITIONS": "3",
    }
    run(["julia", "--startup-file=no", "--project",
         str(HERE / "run_sample.jl"), fixture, "medium", "rust", "fixed_step", str(output)],
        env=sample_env)
    data = json.loads(output.read_text())
    data["field"] = read_complex_field(pathlib.Path(data["field_path"]))
    return data


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def markdown(data: dict[str, Any]) -> str:
    host = data["host"]
    lines = [
        "# Amalthea Apple Silicon quick test",
        "",
        f"Generated: {data['captured_at_utc']}",
        "",
        f"Machine: {host.get('chip') or host.get('machine')}; "
        f"performance/efficiency cores: {host.get('performance_cores')}/"
        f"{host.get('efficiency_cores')}",
        f"Julia: {data['toolchain']['julia'].get('julia')}; Rust: {data['toolchain']['rustc']}",
        f"BLAS: {data['toolchain']['julia'].get('blas')}",
        f"FFTW: {data['toolchain']['julia'].get('fftw_library')}",
        "",
        "## Principal levers",
        "",
        "| Lever | Portable result | Host-native thin-LTO result | Correctness |",
        "|---|---:|---:|---:|",
    ]
    for lever in ("neon_raman", "configured_blas_qdht"):
        item = data["levers"].get(lever, {})
        lines.append(
            f"| {lever.replace('_', ' ')} | {item.get('portable_seconds', 'n/a')} | "
            f"{item.get('native_seconds', 'n/a')} | rel {item.get('relative_error', 'n/a')} |")
    topology = data["levers"].get("process_thread_topology", {})
    lines.append(
        f"| process/thread topology | modal {topology.get('modal_seconds', 'n/a')} | "
        f"scan {topology.get('scan_seconds', 'n/a')} | {topology.get('correct', 'n/a')} |")
    lines += ["", f"LTO recommendation: **{data['lto_recommendation']}**", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path,
                        default=HERE / "results" / "apple-quick.json")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--allow-non-apple", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    if not is_apple and not args.allow_non_apple:
        parser.error("this runner targets Apple Silicon; pass --allow-non-apple for diagnostics")
    args.samples >= 1 or parser.error("--samples must be positive")

    env = os.environ.copy()
    host = {
        "system": platform.system(), "release": platform.release(),
        "machine": platform.machine(), "chip": sysctl("machdep.cpu.brand_string"),
        "physical_cores": sysctl("hw.physicalcpu"),
        "logical_cores": sysctl("hw.logicalcpu"),
        "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
    }
    planned = {
        "portable_build": "AMALTHEA_CUDA_BUILD=off RUSTFLAGS='' julia --project deps/build.jl",
        "diagnostic_build": "RUSTFLAGS='-C target-cpu=native' LTO=thin codegen-units=1 cargo build --release",
        "threads": [1, 2, 4],
        "fixtures": ["modeavg_real_raman_rotational", "radial_real_kerr"],
    }
    data: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host,
        "environment": {key: env.get(key) for key in (
            "JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS", "RUSTFLAGS")},
        "planned": planned,
        "dry_run": args.dry_run,
        "levers": {},
    }
    if args.dry_run:
        data["toolchain"] = {"julia": {}, "rustc": text(["rustc", "-V"])}
        data["lto_recommendation"] = "not evaluated (dry run)"
    else:
        data["toolchain"] = {"julia": julia_metadata(), "rustc": text(["rustc", "-V"])}
        artifact = artifact_path()
        with tempfile.TemporaryDirectory(prefix="amalthea-apple-quick-") as tmp:
            tmpdir = pathlib.Path(tmp)
            results: dict[str, dict[str, list[dict[str, Any]]]] = {}
            build_portable(env)
            portable_copy = tmpdir / artifact.name
            shutil.copy2(artifact, portable_copy)
            try:
                for build in ("portable", "native"):
                    if build == "native":
                        build_native_diagnostic(env)
                    results[build] = {}
                    for fixture in planned["fixtures"]:
                        results[build][fixture] = []
                        for threads in planned["threads"]:
                            rows = [sample(build, fixture, threads, i, tmpdir, env)
                                    for i in range(args.samples)]
                            results[build][fixture].append({
                                "threads": threads,
                                "seconds": median([float(row["elapsed_seconds"]) for row in rows]),
                                "field": rows[-1]["field"],
                            })
                shutil.copy2(portable_copy, artifact)
            finally:
                # A failed diagnostic must never leave the host-native artifact active.
                if portable_copy.is_file():
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(portable_copy, artifact)

            aux: dict[str, Any] = {}
            for threads in planned["threads"]:
                path = tmpdir / f"modal-t{threads}.json"
                run(["julia", f"--threads={threads}", "--startup-file=no", "--project",
                     str(HERE / "apple_quick_aux.jl"), "modal", str(path)], env=env)
                aux[f"modal_t{threads}"] = json.loads(path.read_text())
            scan_path = tmpdir / "scan.json"
            run(["julia", "--threads=1", "--startup-file=no", "--project",
                 str(HERE / "apple_quick_aux.jl"), "scan", str(scan_path)], env=env)
            aux["scan"] = json.loads(scan_path.read_text())

            fixture_to_lever = {
                "modeavg_real_raman_rotational": "neon_raman",
                "radial_real_kerr": "configured_blas_qdht",
            }
            speedups = []
            for fixture, lever in fixture_to_lever.items():
                portable = results["portable"][fixture]
                native = results["native"][fixture]
                portable_reference = next(row for row in portable if row["threads"] == 1)["field"]
                native_reference = next(row for row in native if row["threads"] == 1)["field"]
                thread_errors = []
                cross_build_errors = []
                for prow, nrow in zip(portable, native, strict=True):
                    prow["thread_relative_error"] = relative_error(
                        portable_reference, prow["field"])
                    nrow["thread_relative_error"] = relative_error(
                        native_reference, nrow["field"])
                    cross_error = relative_error(prow["field"], nrow["field"])
                    prow["cross_build_relative_error"] = cross_error
                    nrow["cross_build_relative_error"] = cross_error
                    thread_errors.extend([
                        prow["thread_relative_error"], nrow["thread_relative_error"]])
                    cross_build_errors.append(cross_error)
                p4 = next(row for row in portable if row["threads"] == 4)
                n4 = next(row for row in native if row["threads"] == 4)
                rel = p4["cross_build_relative_error"]
                for row in portable + native:
                    row.pop("field", None)
                gain = p4["seconds"] / n4["seconds"] - 1.0
                speedups.append(gain)
                data["levers"][lever] = {
                    "portable_seconds": p4["seconds"], "native_seconds": n4["seconds"],
                    "native_gain": gain, "relative_error": rel,
                    "max_thread_relative_error": max(thread_errors),
                    "max_cross_build_relative_error": max(cross_build_errors),
                    "correct": max(thread_errors + cross_build_errors) <= 1e-6,
                    "portable_threads": portable, "native_threads": native,
                }
            modal4 = aux["modal_t4"]
            scan = aux["scan"]
            data["levers"]["process_thread_topology"] = {
                "modal_seconds": modal4["elapsed_seconds"],
                "modal_by_threads": {
                    key: value for key, value in aux.items() if key.startswith("modal_t")
                },
                "scan_seconds": scan["elapsed_seconds"],
                "correct": bool(modal4["exact"] and scan["exact_once"]
                                and scan["worker_threads_ok"] and scan["cleanup_ok"]),
            }
            # This runner alone never supplies the required second-host/local audit.
            qualifies = (all(gain >= 0.05 for gain in speedups)
                         and all(data["levers"][lever]["correct"]
                                 for lever in fixture_to_lever.values())
                         and data["levers"]["process_thread_topology"]["correct"])
            data["lto_recommendation"] = (
                "candidate only; run the local end-to-end audit before promotion"
                if qualifies else "do not promote; Apple diagnostic gain is below 5%")

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    destination.with_suffix(".md").write_text(markdown(data))
    print(destination)
    print(destination.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
