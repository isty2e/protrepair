"""Release artifact sanity tests for packaged resources and metadata."""

import hashlib
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

    assert 'requires-python = ">=3.10,<3.13"' in pyproject
    assert '"Programming Language :: Python :: 3.10"' in pyproject
    assert '"Programming Language :: Python :: 3.11"' in pyproject
    assert '"Programming Language :: Python :: 3.12"' in pyproject
    assert "Python :: 3.13" not in pyproject
    assert '"Operating System :: POSIX :: Linux"' in pyproject
    assert '"scikit-build-core>=0.12,<0.13",' in pyproject
    assert 'license = { text = "MIT AND CC-BY-4.0" }' in pyproject
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
    assert "  installed-wheel-smoke:" in workflow
    assert 'python-version: ["3.10", "3.11", "3.12"]' in workflow
    assert 'python-version: "3.12"' in workflow
    assert "ubuntu-latest" in workflow
    assert "macos-latest" not in workflow
    assert "windows-latest" not in workflow
    assert (
        'run: .venv/bin/python -m pip install -c constraints/release.txt ".[dev]"'
        in workflow
    )
    assert '".[dev,refinement]"' in workflow
    assert "constraints/release.txt" in workflow
    full_checks_job, lean_job = workflow.split("  lean:", maxsplit=1)
    assert 'PROTREPAIR_RELEASE_STRICT_RDKIT_DIGESTS: "1"' in full_checks_job
    assert "PROTREPAIR_RELEASE_STRICT_RDKIT_DIGESTS" not in lean_job
    assert "Basedpyright" in full_checks_job
    assert "Lean optional dependency boundary" in workflow
    assert "Verify RDKit is outside the lean environment" in lean_job
    assert "rdkit_blocker" in lean_job
    assert "tests/unit/test_release_public_api_contract.py" in lean_job
    assert "tests/unit/test_readme_usage.py" in lean_job
    assert "tests/unit/test_retained_non_polymer_no_rdkit_release.py" in lean_job
    assert "tests/unit/test_parser_repair_performance_support.py" in lean_job
    assert "test_process_structure_preserves_rdkit_coordinate_digest" in workflow
    assert "tests/workflow/test_process_representatives.py" in workflow
    assert '-m "not benchmark"' in workflow
    installed_wheel_job = workflow.split("  installed-wheel-smoke:", maxsplit=1)[1]
    assert "Installed wheel FASPR smoke" in installed_wheel_job
    assert (
        "pip install -c constraints/release.txt hatchling hatch-vcs "
        "scikit-build-core"
    ) in installed_wheel_job
    assert (
        "run: .venv/bin/python scripts/run_installed_wheel_smoke.py"
        in installed_wheel_job
    )
    assert "continue-on-error" not in installed_wheel_job


def test_release_gate_sources_are_sdist_visible() -> None:
    """Release gates should not live only in ignored local checkout files."""

    checklist = Path("docs/release-checklist.md").read_text()
    gitignore_lines = {
        line.strip()
        for line in Path(".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    pyproject = Path("pyproject.toml").read_text()

    assert Path("docs/release-checklist.md").is_file()
    assert Path("scripts/run_installed_wheel_smoke.py").is_file()
    assert Path("constraints/release.txt").is_file()
    assert "docs/release-checklist.md" not in gitignore_lines
    assert "scripts/" not in gitignore_lines
    assert "scripts/*" in gitignore_lines
    assert "!scripts/run_installed_wheel_smoke.py" in gitignore_lines
    assert '"docs",' in pyproject
    assert '"constraints/release.txt",' in pyproject
    assert '"scripts/run_installed_wheel_smoke.py",' in pyproject
    assert '"scripts",' not in pyproject
    assert "python scripts/run_installed_wheel_smoke.py" in checklist
    assert "tests/unit/test_release_artifacts.py" in checklist
    assert "constraints/release.txt" in checklist
    assert "CPython 3.10, 3.11, and 3.12" in checklist
    assert "Linux through GitHub Actions `ubuntu-latest`" in checklist
    assert "Python 3.13+, macOS, and Windows are not advertised" in checklist
    assert "PROTREPAIR_RELEASE_STRICT_RDKIT_DIGESTS=1" in checklist
    assert "rdkit==2026.3.2" in checklist
    assert "2026.03.2" in checklist


def test_release_constraints_pin_release_environment() -> None:
    """Release constraints should pin CI tools without narrowing metadata ranges."""

    constraint_lines = [
        line.strip()
        for line in Path("constraints/release.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    constraints = dict(line.split("==", maxsplit=1) for line in constraint_lines)
    workflow = Path(".github/workflows/ci.yml").read_text()
    smoke_script = Path("scripts/run_installed_wheel_smoke.py").read_text()

    assert all("==" in line for line in constraint_lines)
    assert constraints == {
        "basedpyright": "1.39.9",
        "gemmi": "0.7.5",
        "hatch-vcs": "0.5.0",
        "hatchling": "1.30.1",
        "numpy": "2.2.6",
        "pytest": "8.4.2",
        "pytest-cov": "7.1.0",
        "rdkit": "2026.3.2",
        "ruff": "0.15.20",
        "scikit-build-core": "0.12.2",
        "typing_extensions": "4.16.0",
    }
    assert "numpy==2.4" not in Path("constraints/release.txt").read_text()
    assert 'pip install -c constraints/release.txt ".[dev,refinement]"' in workflow
    assert "DEFAULT_CONSTRAINTS_PATH" in smoke_script


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
    third_party_notices = " ".join(Path("THIRD_PARTY_NOTICES.md").read_text().split())
    provenance = Path("vendor/faspr/PROVENANCE.md").read_text()
    dunbrack_notice = Path(
        "vendor/faspr/DUNBRACK_ROTAMER_LIBRARY_NOTICE.md"
    ).read_text()

    assert '"THIRD_PARTY_NOTICES.md"' in pyproject
    assert '"vendor/faspr/LICENSE"' in pyproject
    assert '"vendor/faspr/DUNBRACK_ROTAMER_LIBRARY_NOTICE.md"' in pyproject
    assert '"vendor/faspr/PROVENANCE.md"' in pyproject
    assert '"vendor/faspr/README.upstream.md"' in pyproject
    assert "vendor/faspr" in pyproject
    assert "DUNBRACK_ROTAMER_LIBRARY_NOTICE.md" in third_party_notices
    assert "vendor/faspr/PROVENANCE.md" in third_party_notices
    assert "vendor/faspr/README.upstream.md" in third_party_notices
    assert "upstream README includes academic-user wording" in third_party_notices
    assert "Creative Commons Attribution 4.0 International" in third_party_notices
    assert "0d55732fd6307f373018c6bddd842291c355c5f7" in provenance
    assert "DUNBRACK_ROTAMER_LIBRARY_NOTICE.md" in provenance
    assert "No local source patches are applied" in provenance
    assert "The vendored code is treated under the supplied MIT license" in provenance
    assert "Dunbrack Lab Smooth Backbone-Dependent Rotamer Library 2010" in (
        dunbrack_notice
    )
    assert "CC BY 4.0" in dunbrack_notice
    assert "Shapovalov" in dunbrack_notice


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


def test_faspr_build_config_declares_packaged_runtime_assets() -> None:
    """Build config should install packaged FASPR runtime assets."""

    cmake = Path("CMakeLists.txt").read_text()
    pyproject = Path("pyproject.toml").read_text()
    faspr_asset_destination = (
        'DESTINATION "${SKBUILD_PLATLIB_DIR}/protrepair/packing/faspr/bin"'
    )

    assert "install(\n  TARGETS FASPR" in cmake
    assert f"RUNTIME {faspr_asset_destination}" in cmake
    assert "FILES vendor/faspr/dun2010bbdep.bin" in cmake
    assert faspr_asset_destination in cmake
    assert '"CMakeLists.txt"' in pyproject
    assert '"vendor/faspr",' in pyproject


def test_faspr_provenance_checksums_match_vendored_files() -> None:
    """FASPR provenance checksums should match the vendored snapshot."""

    provenance = Path("vendor/faspr/PROVENANCE.md").read_text()

    expected_file_digests = {
        "vendor/faspr/LICENSE": (
            "de1d50743d73db2567b0688e7d5e156f17055234c3d6b15fea13453ccb2a9300"
        ),
        "vendor/faspr/README.upstream.md": (
            "20ebfbe9e6227abb29a0d4bd09ff0e2fbb70c275294758b8399a02caad27e1be"
        ),
        "vendor/faspr/dun2010bbdep.bin": (
            "ed3f7be5f33b5fa947ac5e83cb024c6a6af6440bb50a1c8073aacabe6d792d0e"
        ),
    }
    for path_text, expected_digest in expected_file_digests.items():
        assert expected_digest in provenance
        assert sha256_file(Path(path_text)) == expected_digest

    expected_src_manifest_digest = (
        "cc2ca78f30069e0e9308a0b68674da00864fa1ca09386128b96b36148a3c09d7"
    )
    assert expected_src_manifest_digest in provenance
    assert sha256_sorted_file_manifest(Path("vendor/faspr/src")) == (
        expected_src_manifest_digest
    )


def test_readme_documents_retained_ligand_fallback_contract() -> None:
    """README should document the retained-ligand optional-backend contract."""

    readme = Path("README.md").read_text()

    assert "RETAINED_NON_POLYMER_FALLBACK_USED" in readme
    assert "RETAINED_NON_POLYMER_FALLBACK_BLOCKED" in readme
    assert (
        "explicit overrides used without optional RDKit support raise `ValueError`"
        in readme
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_sorted_file_manifest(root: Path) -> str:
    """Return a stable SHA-256 digest over per-file SHA-256 manifest lines."""

    payload = bytearray()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        relative_path = f"./{path.relative_to(root).as_posix()}"
        payload.extend(f"{sha256_file(path)}  {relative_path}\n".encode())

    return hashlib.sha256(payload).hexdigest()
