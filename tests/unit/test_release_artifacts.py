"""Release artifact sanity tests for packaged resources and metadata."""

from importlib.resources import files
from pathlib import Path


def test_bundled_chemistry_resources_are_package_visible() -> None:
    """Bundled chemistry JSON resources should be visible to importlib.resources."""

    resource_root = files("protrepair.chemistry.resources")

    assert resource_root.joinpath("nonstandard_components.json.gz").is_file()
    assert resource_root.joinpath("retained_non_polymer_components.json.gz").is_file()


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
    assert "tests/unit/test_retained_non_polymer_no_rdkit_release.py" in lean_job
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


def test_faspr_runtime_policy_documents_hydrogen_merge_contract() -> None:
    """FASPR runtime policy should pin hydrogen ownership boundaries."""

    policy = Path("docs/faspr-runtime-policy.md").read_text()
    normalized_policy = " ".join(policy.split())

    assert "heavy-atom side-chain packing backend" in normalized_policy
    assert "Preserve original polymer hydrogens only when" in normalized_policy
    assert "heavy-atom state is equivalent" in normalized_policy
    assert "fixed-vs-packed labels as hints, not proof" in normalized_policy
    assert (
        "Preserve atom sites, coordinates, and topology bonds together"
        in normalized_policy
    )
    assert "Drop hydrogen atom sites and hydrogen topology bonds together" in (
        normalized_policy
    )
    assert "must not place replacement hydrogens itself" in normalized_policy


def test_faspr_provenance_and_notices_are_release_artifacts() -> None:
    """FASPR vendoring metadata should ship with release artifacts."""

    pyproject = Path("pyproject.toml").read_text()
    third_party_notices = Path("THIRD_PARTY_NOTICES.md").read_text()
    provenance = Path("vendor/faspr/PROVENANCE.md").read_text()

    assert '"THIRD_PARTY_NOTICES.md"' in pyproject
    assert '"vendor/faspr/LICENSE"' in pyproject
    assert '"vendor/faspr/PROVENANCE.md"' in pyproject
    assert '"vendor/faspr/README.upstream.md"' in pyproject
    assert "vendor/faspr" in pyproject
    assert "vendor/faspr/PROVENANCE.md" in third_party_notices
    assert "vendor/faspr/README.upstream.md" in third_party_notices
    assert "upstream README includes academic-user wording" in third_party_notices
    assert "0d55732fd6307f373018c6bddd842291c355c5f7" in provenance
    assert "No local source patches are applied" in provenance
    assert "The vendored code is treated under the supplied MIT license" in provenance


def test_faspr_fast_math_requires_explicit_build_option() -> None:
    """FASPR unsafe math optimization should be an explicit build choice."""

    cmake = Path("CMakeLists.txt").read_text()
    provenance = Path("vendor/faspr/PROVENANCE.md").read_text()

    assert "option(\n  PROTREPAIR_FASPR_ENABLE_FAST_MATH" in cmake
    assert '  "Build the vendored FASPR executable with -ffast-math' in cmake
    assert "  OFF\n)" in cmake
    assert "if(PROTREPAIR_FASPR_ENABLE_FAST_MATH AND NOT MSVC)" in cmake
    assert "target_compile_options(FASPR PRIVATE -ffast-math)" in cmake
    assert "PROTREPAIR_FASPR_ENABLE_FAST_MATH=OFF" in provenance
    assert "floating-point reproducibility" in provenance
    assert "explicitly overridden at CMake configuration time" in provenance


def test_readme_documents_retained_ligand_fallback_contract() -> None:
    """README should document the retained-ligand optional-backend contract."""

    readme = Path("README.md").read_text()

    assert "RETAINED_NON_POLYMER_FALLBACK_USED" in readme
    assert "RETAINED_NON_POLYMER_FALLBACK_BLOCKED" in readme
    assert (
        "explicit overrides used without optional RDKit support raise `ValueError`"
        in readme
    )
