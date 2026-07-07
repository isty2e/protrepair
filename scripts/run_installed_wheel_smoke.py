"""Build and smoke-test an installed ProtRepair wheel."""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_PATH = REPOSITORY_ROOT / ".tmp" / "release-wheel-smoke"
DEFAULT_CONSTRAINTS_PATH = REPOSITORY_ROOT / "constraints" / "release.txt"


def main() -> None:
    """Run the installed-wheel functional smoke."""

    arguments = parse_args()
    wheel_path = arguments.wheel_path
    if wheel_path is None:
        build_wheel()
        wheel_path = newest_wheel()

    venv_path = prepare_virtual_environment(
        arguments.venv_path,
        keep_venv=arguments.keep_venv,
    )

    python = venv_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install_target = (
        f"{wheel_path.resolve()}[refinement]"
        if arguments.with_refinement
        else str(wheel_path.resolve())
    )
    install_command = [str(python), "-m", "pip", "install"]
    if arguments.constraints_path is not None:
        install_command.extend(("-c", str(arguments.constraints_path.resolve())))
    install_command.append(install_target)
    run(tuple(install_command))
    run(
        (
            str(python),
            "-c",
            installed_smoke_program(with_refinement=arguments.with_refinement),
        ),
        cwd=REPOSITORY_ROOT,
        env=clean_python_environment(),
    )

    print(f"installed wheel smoke passed: {wheel_path}")


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build/install a ProtRepair wheel and run functional smoke tests.",
    )
    parser.add_argument(
        "--wheel-path",
        type=Path,
        default=None,
        help="Existing wheel to install instead of building a new wheel.",
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
        "--with-refinement",
        action="store_true",
        help="Install the wheel with the refinement extra and run RDKit smoke.",
    )
    parser.add_argument(
        "--constraints-path",
        type=Path,
        default=DEFAULT_CONSTRAINTS_PATH,
        help="Constraints file used for the smoke install.",
    )
    return parser.parse_args()


def build_wheel() -> None:
    """Build source and wheel artifacts using the project build backend."""

    run((sys.executable, "-m", "hatchling", "build"), cwd=REPOSITORY_ROOT)


def newest_wheel() -> Path:
    """Return the newest built ProtRepair wheel artifact."""

    wheel_paths = sorted(
        (REPOSITORY_ROOT / "dist").glob("protrepair-*.whl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not wheel_paths:
        raise SystemExit("no ProtRepair wheel found in dist/")
    return wheel_paths[-1]


def prepare_virtual_environment(venv_path: Path, *, keep_venv: bool) -> Path:
    """Create or recreate the smoke-test virtual environment safely."""

    resolved_venv_path = venv_path.resolve()
    if resolved_venv_path.exists() and not keep_venv:
        _assert_recreatable_virtual_environment(resolved_venv_path)
        shutil.rmtree(resolved_venv_path)
    if not resolved_venv_path.exists():
        run((sys.executable, "-m", "venv", str(resolved_venv_path)))
    return resolved_venv_path


def _assert_recreatable_virtual_environment(venv_path: Path) -> None:
    """Reject destructive recreation unless the target already looks like a venv."""

    repository_root = REPOSITORY_ROOT.resolve()
    if venv_path == repository_root or venv_path.parent == venv_path:
        raise SystemExit(f"refusing to remove unsafe venv path: {venv_path}")
    if not venv_path.is_dir():
        raise SystemExit(f"refusing to remove non-directory venv path: {venv_path}")
    if not (venv_path / "pyvenv.cfg").is_file():
        raise SystemExit(
            "refusing to remove existing --venv-path without pyvenv.cfg: "
            f"{venv_path}"
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
    """Return an environment that does not import from the source checkout."""

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def installed_smoke_program(*, with_refinement: bool) -> str:
    """Return the Python program executed inside the installed environment."""

    refinement_block = (
        """
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
            if atom_site.element == "H":
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
        if with_refinement
        else """
        import importlib.util

        assert importlib.util.find_spec("rdkit") is None
        """
    )
    return textwrap.dedent(
        f"""
        from importlib.resources import files
        from pathlib import Path

        from protrepair import process_structure
        from protrepair.io import read_structure, write_structure
        from protrepair.transformer.packing import PackingSpec
        from protrepair.transformer.packing.runtime import execute_sidechain_packing

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
        output_path = root / ".tmp/installed-wheel-smoke-output.pdb"
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
