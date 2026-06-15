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


def test_release_metadata_declares_expected_optional_dependencies() -> None:
    """Release metadata should expose documented optional dependency groups."""

    pyproject = Path("pyproject.toml").read_text()

    assert "[project.optional-dependencies]" in pyproject
    assert "refinement = [" in pyproject
    assert '"rdkit",' in pyproject
    assert "dev = [" in pyproject
