#!/usr/bin/env python3
"""Combine converged audit matrices into report-ready machine data."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import tomllib


HERE = pathlib.Path(__file__).resolve().parent
with (HERE / "workloads.toml").open("rb") as inventory_file:
    INVENTORY = {
        item["id"]: item for item in tomllib.load(inventory_file)["fixture"]
    }


def geometric_mean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else None


def load_matrix(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") not in (1, 2):
        raise ValueError(f"unsupported schema in {path}")
    failed = [item for item in data["correctness"] if not item["passed"]]
    if failed:
        raise ValueError(f"post-timing correctness failure in {path}: {failed}")
    unstable = [
        (item["fixture"], item["backend"], item["samples"])
        for item in data["cells"]
        if item["relative_mad"] > 0.03
        or item["bootstrap_ci_half_width_relative"] > 0.05
    ]
    if unstable:
        raise ValueError(f"non-converged timing cells in {path}: {unstable}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=HERE / "results" / "matrix-analysis.json",
    )
    args = parser.parse_args()

    rows = []
    for path in args.matrix:
        matrix = load_matrix(path)
        by_cell = {
            (cell["fixture"], cell["backend"]): cell for cell in matrix["cells"]
        }
        fixtures = sorted({fixture for fixture, _ in by_cell})
        for fixture in fixtures:
            julia = by_cell[(fixture, "julia")]
            rust = by_cell[(fixture, "rust")]
            meta = INVENTORY[fixture]
            row = {
                "fixture": fixture,
                "geometry": meta["geometry"],
                "grid": meta["grid"],
                "features": meta["features"],
                "size": matrix["size"],
                "measurement": matrix["measurement"],
                "threads": matrix["threads"],
                "julia_seconds": julia["median_seconds"],
                "rust_seconds": rust["median_seconds"],
                "rust_speedup_over_julia": julia["median_seconds"] / rust["median_seconds"],
                "julia_samples": julia["samples"],
                "rust_samples": rust["samples"],
                "julia_relative_mad": julia["relative_mad"],
                "rust_relative_mad": rust["relative_mad"],
                "julia_ci_half_width_relative": julia["bootstrap_ci_half_width_relative"],
                "rust_ci_half_width_relative": rust["bootstrap_ci_half_width_relative"],
                "rust_allocated_bytes": rust["allocated_bytes_median"],
                "julia_allocated_bytes": julia["allocated_bytes_median"],
                "rust_peak_rss_bytes": rust["peak_rss_bytes_median"],
                "julia_peak_rss_bytes": julia["peak_rss_bytes_median"],
                "rust_steps": rust["accepted_steps"],
                "julia_steps": julia["accepted_steps"],
                "rust_rejected_steps": rust["rejected_steps"],
                "julia_rejected_steps": julia["rejected_steps"],
                "source": str(path),
            }
            if (fixture, "upstream") in by_cell:
                upstream = by_cell[(fixture, "upstream")]
                row.update(
                    {
                        "upstream_seconds": upstream["median_seconds"],
                        "rust_speedup_over_upstream": upstream["median_seconds"] / rust["median_seconds"],
                        "julia_speedup_over_upstream": upstream["median_seconds"] / julia["median_seconds"],
                        "upstream_steps": upstream["accepted_steps"],
                        "upstream_rejected_steps": upstream["rejected_steps"],
                    }
                )
            rows.append(row)

    adaptive = [row for row in rows if row["measurement"] == "adaptive_solve"]
    representative = [row for row in adaptive if row["size"] in ("medium", "large")]
    regressions = [row for row in adaptive if row["rust_speedup_over_julia"] < 0.95]
    by_branch = {}
    for geometry in ("modeavg", "radial", "modal", "free"):
        branch_rows = [row for row in representative if row["geometry"] == geometry]
        by_branch[geometry] = {
            "cells": len(branch_rows),
            "geometric_mean_speedup": geometric_mean(
                [row["rust_speedup_over_julia"] for row in branch_rows]
            ),
            "minimum_speedup": min(
                (row["rust_speedup_over_julia"] for row in branch_rows),
                default=None,
            ),
        }

    result = {
        "schema_version": 2,
        "rows": rows,
        "adaptive_summary": {
            "medium_large_cells": len(representative),
            "geometric_mean_speedup": geometric_mean(
                [row["rust_speedup_over_julia"] for row in representative]
            ),
            "regressions_over_five_percent": regressions,
            "by_geometry": by_branch,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
