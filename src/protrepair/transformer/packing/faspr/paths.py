"""Installed-path helpers for the packaged FASPR tool assets."""

import os
import sys
from importlib.resources import files
from pathlib import Path


def faspr_binary_directory() -> Path:
    """Return the installed directory that contains the packaged FASPR assets."""

    for path in candidate_binary_directories():
        if path.exists():
            return path

    raise FileNotFoundError(
        "FASPR assets are missing from the installed protrepair package"
    )


def faspr_executable_path() -> Path:
    """Return the installed path to the packaged FASPR executable."""

    executable_name = "FASPR.exe" if os.name == "nt" else "FASPR"
    path = faspr_binary_directory() / executable_name
    if not path.exists():
        raise FileNotFoundError(
            "FASPR executable is missing from the installed protrepair package"
        )

    return path


def faspr_rotamer_library_path() -> Path:
    """Return the installed path to the packaged Dunbrack rotamer library."""

    path = faspr_binary_directory() / "dun2010bbdep.bin"
    if not path.exists():
        raise FileNotFoundError(
            "dun2010bbdep.bin is missing from the installed protrepair package"
        )

    return path


def candidate_binary_directories() -> tuple[Path, ...]:
    """Return candidate directories that may contain packaged FASPR assets."""

    candidates: list[Path] = []
    seen: set[Path] = set()

    resource = files("protrepair").joinpath("packing").joinpath("faspr").joinpath("bin")
    initial_candidate = Path(str(resource))
    candidates.append(initial_candidate)
    seen.add(initial_candidate)

    for entry in sys.path:
        if not entry:
            continue

        candidate = Path(entry) / "protrepair" / "packing" / "faspr" / "bin"
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    return tuple(candidates)
