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
