#!/usr/bin/env python3
"""Validate the audit workload inventory before any benchmark consumes it."""

from __future__ import annotations

import pathlib
import sys
import tomllib


HERE = pathlib.Path(__file__).resolve().parent
INVENTORY = HERE / "workloads.toml"
REQUIRED_GEOMETRIES = {"modeavg", "radial", "modal", "free"}
REQUIRED_GRIDS = {"real", "env"}
REQUIRED_SIZES = {"small", "medium", "large"}


def main() -> int:
    with INVENTORY.open("rb") as stream:
        data = tomllib.load(stream)
    fixtures = data.get("fixture", [])
    errors: list[str] = []
    ids: set[str] = set()
    covered_geometries: set[str] = set()
    covered_grids: set[str] = set()
    covered_features: set[str] = set()

    for fixture in fixtures:
        fixture_id = fixture.get("id")
        if not fixture_id or fixture_id in ids:
            errors.append(f"missing or duplicate fixture id: {fixture_id!r}")
        ids.add(fixture_id)
        geometry = fixture.get("geometry")
        grid = fixture.get("grid")
        covered_geometries.add(geometry)
        covered_grids.add(grid)
        covered_features.update(fixture.get("features", []))
        if set(fixture.get("sizes", {})) != REQUIRED_SIZES:
            errors.append(f"{fixture_id}: sizes must be small/medium/large")
        if fixture.get("upstream") not in {"required", "unavailable", "probe"}:
            errors.append(f"{fixture_id}: invalid upstream classification")
        if not fixture.get("oracle_test"):
            errors.append(f"{fixture_id}: missing oracle_test provenance")
        else:
            for relative in fixture["oracle_test"].split(";"):
                path = HERE.parents[1] / relative.strip()
                if not path.is_file():
                    errors.append(f"{fixture_id}: oracle-test path does not exist: {relative.strip()}")
        if not fixture.get("guard_basis"):
            errors.append(f"{fixture_id}: missing guard_basis provenance")

    if covered_geometries != REQUIRED_GEOMETRIES:
        errors.append(f"geometry coverage is {covered_geometries}, expected {REQUIRED_GEOMETRIES}")
    if covered_grids != REQUIRED_GRIDS:
        errors.append(f"grid coverage is {covered_grids}, expected {REQUIRED_GRIDS}")

    required_features = {
        "kerr",
        "ppt",
        "adk",
        "raman_sdo",
        "raman_rotational",
        "raman_sio2",
        "thg_true",
        "thg_false",
        "npol_1",
        "npol_2",
        "full_false",
        "full_true",
        "shotnoise",
        "mixture",
        "z_dependent",
        "wrapper_modes",
    }
    missing = required_features - covered_features
    if missing:
        errors.append(f"missing required feature classes: {sorted(missing)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"validated {len(fixtures)} fixtures; "
        f"geometries={sorted(covered_geometries)} grids={sorted(covered_grids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
