"""Release artifact sanity tests for packaged resources and metadata."""

from importlib.resources import files
from pathlib import Path


def test_bundled_chemistry_resources_are_package_visible() -> None:
    """Bundled chemistry JSON resources should be visible to importlib.resources."""

    resource_root = files("protrepair.chemistry.resources")

    assert resource_root.joinpath("nonstandard_components.json.gz").is_file()
    assert resource_root.joinpath(
        "retained_non_polymer_components.json.gz"
    ).is_file()


def test_release_metadata_declares_dependency_boundary() -> None:
    """Release metadata should keep required and optional backends distinct."""

    pyproject = Path("pyproject.toml").read_text()
    project_dependencies, optional_dependencies = pyproject.split(
        "[project.optional-dependencies]",
        maxsplit=1,
    )

    assert '"gemmi>=0.7.5",' in project_dependencies
    assert '"rdkit",' not in project_dependencies
    assert "[project.optional-dependencies]" in pyproject
    assert "analysis = [" not in optional_dependencies
    assert "matplotlib" not in optional_dependencies
    assert "gemmi" not in optional_dependencies
    assert "refinement = [" in pyproject
    assert '"rdkit",' in optional_dependencies
    assert "dev = [" in pyproject


def test_ci_exercises_required_and_refinement_dependency_worlds() -> None:
    """CI should cover both lean installs and the optional RDKit backend."""

    workflow = Path(".github/workflows/ci.yml").read_text()

    assert "  checks:" in workflow
    assert "  lean:" in workflow
    assert 'run: .venv/bin/python -m pip install ".[dev]"' in workflow
    assert '".[dev,refinement]"' in workflow
    full_checks_job, lean_job = workflow.split("  lean:", maxsplit=1)
    assert "Basedpyright" in full_checks_job
    assert "Lean optional dependency boundary" in workflow
    assert "Verify RDKit is outside the lean environment" in lean_job
    assert "rdkit_blocker" in lean_job
    assert "tests/unit/test_parser_repair_performance_support.py" in lean_job
    assert "test_process_structure_preserves_rdkit_coordinate_digest" in workflow
    assert "tests/workflow/test_process_representatives.py" in workflow
    assert '-m "not benchmark"' in workflow


def test_release_docs_state_faspr_installed_asset_contract() -> None:
    """Release docs should not imply arbitrary source-tree FASPR availability."""

    readme = Path("README.md").read_text()
    normalized_readme = " ".join(readme.split())

    assert "Built packages and wheels include the vendored FASPR executable" in (
        normalized_readme
    )
    assert "Direct source-tree imports are not guaranteed" in normalized_readme
    assert "explicit FASPR `executable_path`" in normalized_readme
