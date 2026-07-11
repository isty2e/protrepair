"""Install an explicit ProtRepair artifact and run functional smoke checks."""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_PATH = REPOSITORY_ROOT / ".tmp" / "release-artifact-smoke"
DEFAULT_CONSTRAINTS_PATH = REPOSITORY_ROOT / "constraints" / "release.txt"


def main() -> None:
    """Install one explicit release artifact and run the shared smoke payload."""

    arguments = parse_args()
    artifact_path = resolve_artifact_path(arguments.artifact_path)
    constraints_path = resolve_constraints_path(arguments.constraints_path)
    venv_path = prepare_virtual_environment(
        arguments.venv_path,
        keep_venv=arguments.keep_venv,
    )

    python = venv_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install_artifact(python, artifact_path, constraints_path)
    run(
        (
            str(python),
            "-I",
            "-c",
            installed_smoke_program(),
        ),
        cwd=REPOSITORY_ROOT,
        env=clean_python_environment(),
    )

    print(f"installed {artifact_kind(artifact_path)} smoke passed: {artifact_path}")


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for one explicit release artifact."""

    parser = argparse.ArgumentParser(
        description=(
            "Install an explicit ProtRepair wheel or sdist and run functional "
            "smoke tests."
        ),
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        required=True,
        help="Built ProtRepair wheel or source distribution to install.",
    )
    parser.add_argument(
        "--venv-path",
        type=Path,
        default=DEFAULT_VENV_PATH,
        help="Virtual environment path used for the smoke install.",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Reuse the target virtual environment instead of recreating it.",
    )
    parser.add_argument(
        "--constraints-path",
        type=Path,
        default=DEFAULT_CONSTRAINTS_PATH,
        help="Constraints file used for the smoke install.",
    )
    return parser.parse_args()


def resolve_artifact_path(artifact_path: Path) -> Path:
    """Validate and resolve one supported ProtRepair release artifact."""

    resolved_artifact_path = artifact_path.resolve()
    if not resolved_artifact_path.is_file():
        raise SystemExit(f"artifact path does not exist: {resolved_artifact_path}")
    artifact_kind(resolved_artifact_path)
    return resolved_artifact_path


def artifact_kind(artifact_path: Path) -> str:
    """Return the supported release artifact kind for a validated filename."""

    artifact_name = artifact_path.name
    if artifact_name.startswith("protrepair-") and artifact_name.endswith(".whl"):
        return "wheel"
    if artifact_name.startswith("protrepair-") and artifact_name.endswith(".tar.gz"):
        return "sdist"
    raise SystemExit(
        "artifact path must name a ProtRepair wheel or source distribution "
        f"(.whl or .tar.gz): {artifact_path}"
    )


def resolve_constraints_path(constraints_path: Path) -> Path:
    """Validate and resolve the constraints file used for every smoke install."""

    resolved_constraints_path = constraints_path.resolve()
    if not resolved_constraints_path.is_file():
        raise SystemExit(
            f"constraints path does not exist: {resolved_constraints_path}"
        )
    return resolved_constraints_path


def prepare_virtual_environment(venv_path: Path, *, keep_venv: bool) -> Path:
    """Create or recreate the smoke-test virtual environment safely."""

    resolved_venv_path = venv_path.resolve()
    if resolved_venv_path.exists():
        if keep_venv:
            _assert_existing_virtual_environment(resolved_venv_path)
            return resolved_venv_path
        _assert_recreatable_virtual_environment(resolved_venv_path)
        shutil.rmtree(resolved_venv_path)
    if not resolved_venv_path.exists():
        _assert_managed_virtual_environment_path(resolved_venv_path)
        run((sys.executable, "-m", "venv", str(resolved_venv_path)))
    return resolved_venv_path


def _assert_managed_virtual_environment_path(venv_path: Path) -> None:
    """Allow managed smoke environments only below the repository scratch root."""

    temporary_root = (REPOSITORY_ROOT / ".tmp").resolve()
    if venv_path == temporary_root or not venv_path.is_relative_to(temporary_root):
        raise SystemExit(
            "refusing to manage venv path outside repository temporary root "
            f"{temporary_root}: {venv_path}"
        )


def _assert_recreatable_virtual_environment(venv_path: Path) -> None:
    """Reject destructive recreation unless the target already looks like a venv."""

    _assert_managed_virtual_environment_path(venv_path)
    _assert_existing_virtual_environment(venv_path)


def _assert_existing_virtual_environment(venv_path: Path) -> None:
    """Reject an existing path unless it is an initialized virtual environment."""

    if not venv_path.is_dir():
        raise SystemExit(f"venv path is not a directory: {venv_path}")
    if not (venv_path / "pyvenv.cfg").is_file():
        raise SystemExit(
            "refusing to use existing --venv-path without pyvenv.cfg: "
            f"{venv_path}"
        )


def install_artifact(
    python: Path,
    artifact_path: Path,
    constraints_path: Path,
) -> None:
    """Force-install the exact artifact into the isolated constrained venv."""

    run(
        (
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "-c",
            str(constraints_path),
            str(artifact_path),
        ),
        env=clean_install_environment(constraints_path),
    )


def run(
    command: tuple[str, ...],
    *,
    cwd: Path = REPOSITORY_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    """Run one subprocess command with visible output."""

    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def clean_python_environment() -> dict[str, str]:
    """Return an environment that cannot import package code from the checkout."""

    environment = dict(os.environ)
    for variable_name in (
        "PIP_PREFIX",
        "PIP_TARGET",
        "PIP_USER",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
    ):
        environment.pop(variable_name, None)
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def clean_install_environment(constraints_path: Path) -> dict[str, str]:
    """Return the constrained environment inherited by isolated sdist builds."""

    environment = clean_python_environment()
    environment["PIP_CONSTRAINT"] = str(constraints_path.resolve())
    environment["PIP_BUILD_CONSTRAINT"] = str(constraints_path.resolve())
    return environment


def installed_smoke_program() -> str:
    """Return the shared program executed inside the installed environment."""

    refinement_block = """
        from rdkit import Chem
        from protrepair.scope import WholeStructureScope
        from protrepair.state import HydrogenCoverageState
        from protrepair.structure.slots import AtomIndex
        from protrepair.transformer.continuous.settings import (
            ContinuousRelaxationConfig,
            ContinuousRelaxationForceField,
        )
        from protrepair.transformer.local import (
            DirectRegionTransformationSpec,
            LocalScopeSpec,
            transform_local_region,
        )
        from protrepair.workflow.contracts import requested_process_goal

        assert Chem is not None
        hydrogenated = process_structure(
            input_path,
            requested_goals=(
                requested_process_goal(
                    scope=WholeStructureScope(),
                    value=HydrogenCoverageState.COMPLETE,
                ),
            ),
        )
        assert not hydrogenated.has_errors()
        selected_atom_ref = None
        for atom_index_value, atom_site in enumerate(
            hydrogenated.structure.iter_atom_sites()
        ):
            if atom_site.is_hydrogen():
                selected_atom_ref = hydrogenated.structure.constitution.atom_ref_at(
                    AtomIndex(atom_index_value)
                )
                break
        assert selected_atom_ref is not None
        refined = transform_local_region(
            hydrogenated.structure,
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_atoms(
                    (selected_atom_ref,)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
                config=ContinuousRelaxationConfig(
                    context_radius_angstrom=2.5,
                    max_iterations=10,
                ),
            ),
        )
        assert refined.backend_name == "rdkit"
        """
    return textwrap.dedent(
        f"""
        import sys
        from importlib.metadata import distribution
        from importlib.resources import files
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import protrepair
        from protrepair import process_structure
        from protrepair.io import read_structure, write_structure
        from protrepair.transformer.packing import PackingSpec
        from protrepair.transformer.packing.runtime import execute_sidechain_packing

        checkout_source_root = (
            Path({str(REPOSITORY_ROOT)!r}) / "src" / "protrepair"
        ).resolve()
        installed_package_root = Path(
            distribution("protrepair").locate_file("protrepair")
        ).resolve()
        imported_module_path = Path(protrepair.__file__).resolve()
        assert imported_module_path.is_relative_to(installed_package_root)
        assert installed_package_root.is_relative_to(Path(sys.prefix).resolve())
        assert not imported_module_path.is_relative_to(checkout_source_root)

        root = Path.cwd()
        input_path = root / "tests/fixtures/pdb/1aho.pdb"
        faspr_input_path = root / "tests/fixtures/pdb/1aho_faspr_input.pdb"

        for resource_name in (
            "nonstandard_components.json.gz",
            "retained_non_polymer_components.json.gz",
        ):
            assert files("protrepair.chemistry.resources").joinpath(
                resource_name
            ).is_file()

        for asset_name in ("FASPR", "dun2010bbdep.bin"):
            assert files("protrepair.packing.faspr.bin").joinpath(asset_name).is_file()

        processed = process_structure(input_path)
        assert not processed.has_errors()
        assert processed.structure.geometry.atom_count() > 0

        reread = read_structure(input_path)
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "processed.pdb"
            write_structure(reread, output_path)
            assert "ATOM" in output_path.read_text()

        faspr_structure = read_structure(faspr_input_path)
        packed = execute_sidechain_packing(
            faspr_structure,
            PackingSpec(backend_name="faspr"),
        )
        assert packed.backend_name == "faspr"
        assert packed.packed_structure.chain_ids() == faspr_structure.chain_ids()

        {textwrap.indent(textwrap.dedent(refinement_block), "        ")}
        """
    )


if __name__ == "__main__":
    main()
