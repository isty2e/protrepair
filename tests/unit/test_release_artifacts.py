"""Release artifact sanity tests for packaged resources and metadata."""

import hashlib
import importlib.util
import re
import subprocess
import sys
from importlib.resources import files
from pathlib import Path


def test_bundled_chemistry_resources_are_package_visible() -> None:
    """Bundled chemistry JSON resources should be visible to importlib.resources."""

    resource_root = files("protrepair.chemistry.resources")

    assert resource_root.joinpath("nonstandard_components.json.gz").is_file()
    assert resource_root.joinpath("retained_non_polymer_components.json.gz").is_file()


def test_required_rdkit_dependency_is_importable() -> None:
    """RDKit is part of the required runtime dependency set."""

    assert importlib.util.find_spec("rdkit") is not None


def test_release_metadata_declares_dependency_boundary() -> None:
    """Release metadata should make RDKit part of the default runtime."""

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
    assert 'license = { text = "MIT AND BSD-3-Clause AND CC-BY-4.0" }' in pyproject
    assert '"gemmi>=0.7.5",' in project_dependencies
    assert '"rdkit",' in project_dependencies
    assert "[project.optional-dependencies]" in pyproject
    assert "analysis = [" not in optional_dependencies
    assert "matplotlib" not in optional_dependencies
    assert "gemmi" not in optional_dependencies
    assert "refinement = [" not in pyproject
    assert '"rdkit",' not in optional_dependencies
    assert "dev = [" in pyproject


def test_ci_exercises_required_rdkit_dependency_world() -> None:
    """CI should smoke-test both built artifact kinds with required RDKit."""

    workflow = Path(".github/workflows/ci.yml").read_text()

    assert "permissions:\n  contents: read" in workflow
    assert "  checks:" in workflow
    assert "  installed-artifact-smoke:" in workflow
    assert "  artifact-content:" in workflow
    assert 'python-version: ["3.10", "3.11", "3.12"]' in workflow
    assert 'python-version: "3.12"' in workflow
    assert "ubuntu-latest" in workflow
    assert "macos-latest" not in workflow
    assert "windows-latest" not in workflow
    assert (
        'run: .venv/bin/python -m pip install -c constraints/release.txt ".[dev]"'
        in workflow
    )
    assert '".[dev,refinement]"' not in workflow
    assert "constraints/release.txt" in workflow
    full_checks_job = workflow.split("  installed-artifact-smoke:", maxsplit=1)[0]
    assert 'PROTREPAIR_RELEASE_STRICT_RDKIT_DIGESTS: "1"' in full_checks_job
    assert "Basedpyright" in full_checks_job
    assert "  lean:" not in workflow
    assert "Verify RDKit is outside the lean environment" not in workflow
    assert "rdkit_blocker" not in workflow
    assert "tests/unit/test_retained_non_polymer_no_rdkit_release.py" not in workflow
    assert "tests/workflow" in workflow
    assert '-m "not benchmark"' in workflow
    installed_artifact_job = workflow.split(
        "  installed-artifact-smoke:",
        maxsplit=1,
    )[1]
    assert "Installed wheel and sdist FASPR/RDKit smoke" in installed_artifact_job
    assert (
        "pip install -c constraints/release.txt hatchling hatch-vcs "
        "scikit-build-core"
    ) in installed_artifact_job
    assert "Build release artifacts" in installed_artifact_job
    assert (
        "scripts/run_installed_artifact_smoke.py --artifact-path "
        "dist/protrepair-*.whl --venv-path .tmp/release-wheel-smoke"
    ) in installed_artifact_job
    assert (
        "scripts/run_installed_artifact_smoke.py --artifact-path "
        "dist/protrepair-*.tar.gz --venv-path .tmp/release-sdist-smoke"
    ) in installed_artifact_job
    assert "hatchling build" in installed_artifact_job
    assert "--artifact-path" in installed_artifact_job
    assert "--wheel-path" not in installed_artifact_job
    assert "newest_wheel" not in installed_artifact_job
    assert "--with-refinement" not in installed_artifact_job
    assert "continue-on-error" not in installed_artifact_job
    artifact_content_job = workflow.split("  artifact-content:", maxsplit=1)[1]
    assert "Release artifact content" in artifact_content_job
    assert (
        "pip install -c constraints/release.txt pytest hatchling hatch-vcs "
        "scikit-build-core"
    ) in artifact_content_job
    assert (
        "run: .venv/bin/python -m pytest "
        "tests/release/test_artifact_contents.py -q"
    ) in artifact_content_job
    assert "continue-on-error" not in artifact_content_job


def test_ci_action_refs_follow_release_pinning_policy() -> None:
    """Action refs should follow the documented release supply-chain policy."""

    workflow = Path(".github/workflows/ci.yml").read_text()
    checklist = Path("docs/release-checklist.md").read_text()
    allowed_major_tag_refs = {
        "actions/checkout@v4",
        "actions/setup-python@v5",
    }
    sha_pinned_ref = re.compile(r".+@[0-9a-f]{40}$")
    action_refs = re.findall(r"^\s*uses:\s*(\S+)\s*$", workflow, re.MULTILINE)

    assert action_refs
    for action_ref in action_refs:
        assert action_ref in allowed_major_tag_refs or sha_pinned_ref.fullmatch(
            action_ref
        ), action_ref

    normalized_checklist = " ".join(checklist.split())
    assert "first-party GitHub actions pinned to reviewed major-version tags" in (
        normalized_checklist
    )
    assert "actions/checkout@v4" in normalized_checklist
    assert "actions/setup-python@v5" in normalized_checklist
    assert "full 40-character commit SHA" in normalized_checklist
    assert "contents: read" in normalized_checklist


def test_release_gate_sources_are_sdist_visible() -> None:
    """Release gates should not live only in ignored local checkout files."""

    checklist = Path("docs/release-checklist.md").read_text()
    normalized_checklist = " ".join(checklist.split())
    gitignore_lines = {
        line.strip()
        for line in Path(".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    pyproject = Path("pyproject.toml").read_text()

    assert Path("docs/release-checklist.md").is_file()
    assert Path("docs/radius-policy.md").is_file()
    assert Path("scripts/run_installed_artifact_smoke.py").is_file()
    assert Path("constraints/release.txt").is_file()
    assert "docs/release-checklist.md" not in gitignore_lines
    assert "scripts/" not in gitignore_lines
    assert "scripts/*" in gitignore_lines
    assert "!scripts/run_installed_artifact_smoke.py" in gitignore_lines
    assert '"docs",' in pyproject
    assert '"constraints/release.txt",' in pyproject
    assert '"scripts/run_installed_artifact_smoke.py",' in pyproject
    assert '"scripts",' not in pyproject
    assert "python scripts/run_installed_artifact_smoke.py --artifact-path" in checklist
    assert "tests/release/test_artifact_contents.py" in checklist
    assert "tests/unit/test_release_artifacts.py" in checklist
    assert "tests/unit/test_radii.py" in checklist
    assert "tests/corpus" in checklist
    assert "tests/workflow" in checklist
    assert "-m \"not benchmark\"" in checklist
    assert "constraints/release.txt" in checklist
    assert "docs/radius-policy.md" in checklist
    assert "Release CI builds wheel and sdist artifacts" in normalized_checklist
    assert "--with-refinement" not in normalized_checklist
    assert "CPython 3.10, 3.11, and 3.12" in checklist
    assert "Linux through GitHub Actions `ubuntu-latest`" in checklist
    assert "Python 3.13+, macOS, and Windows are not advertised" in checklist
    assert "PROTREPAIR_RELEASE_STRICT_RDKIT_DIGESTS=1" in checklist
    assert "rdkit==2026.3.2" in checklist
    assert "2026.03.2" in checklist
    assert "all atomic numbers 1 through 118" in normalized_checklist
    assert "live RDKit PeriodicTable verifier" in normalized_checklist
    assert "more than one known coordinate digest" in normalized_checklist
    assert "2026.03.3" in checklist
    assert "CMake 3.18 or newer" in normalized_checklist
    assert "working C++ compiler toolchain" in normalized_checklist


def test_installed_artifact_smoke_exercises_required_rdkit_by_default() -> None:
    """Installed artifact smoke should not keep a hidden no-RDKit mode."""

    smoke_script = Path("scripts/run_installed_artifact_smoke.py").read_text()

    assert "[refinement]" not in smoke_script
    assert "with_refinement" not in smoke_script
    assert "--with-refinement" not in smoke_script
    assert "find_spec(\"rdkit\") is None" not in smoke_script
    assert "from rdkit import Chem" in smoke_script
    assert "transform_local_region" in smoke_script
    assert '"--artifact-path"' in smoke_script
    assert "required=True" in smoke_script
    assert "build_wheel" not in smoke_script
    assert "newest_wheel" not in smoke_script
    assert "--force-reinstall" in smoke_script
    assert "PIP_CONSTRAINT" in smoke_script
    assert "PIP_BUILD_CONSTRAINT" in smoke_script
    assert "clean_install_environment" in smoke_script
    assert 'distribution("protrepair")' in smoke_script
    assert "is_relative_to" in smoke_script
    assert '"src" / "protrepair"' in smoke_script


def test_installed_artifact_smoke_rejects_missing_or_unsupported_artifacts(
    tmp_path: Path,
) -> None:
    """The smoke boundary should reject ambiguous or non-release inputs early."""

    smoke_script = Path("scripts/run_installed_artifact_smoke.py")
    unsupported_path = tmp_path / "protrepair-0.1.0.zip"
    unsupported_path.write_text("not a release artifact")
    cases = (
        (tmp_path / "protrepair-0.1.0.whl", "does not exist"),
        (unsupported_path, "wheel or source distribution"),
    )

    for artifact_path, expected_message in cases:
        result = subprocess.run(
            (
                sys.executable,
                str(smoke_script),
                "--artifact-path",
                str(artifact_path),
            ),
            capture_output=True,
            check=False,
            text=True,
        )

        assert result.returncode != 0
        assert expected_message in result.stderr


def test_rdkit_radius_snapshot_provenance_is_documented() -> None:
    """RDKit runtime and static radius snapshot provenance should be explicit."""

    pyproject = Path("pyproject.toml").read_text()
    notices = " ".join(Path("THIRD_PARTY_NOTICES.md").read_text().split())
    radius_policy = " ".join(Path("docs/radius-policy.md").read_text().split())
    radii_source = Path("src/protrepair/chemistry/radii.py").read_text()

    assert '"rdkit",' in pyproject
    assert "Runtime dependency: `rdkit`" in notices
    assert "RDKit source license: BSD 3-Clause" in notices
    assert "PyPI package license: BSD 3-Clause" in notices
    rdkit_license = Path("vendor/rdkit/LICENSE").read_text()
    assert "BSD 3-Clause License" in rdkit_license
    assert "Redistribution and use in source and binary forms" in rdkit_license
    assert "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS" in rdkit_license
    assert '"vendor/rdkit/LICENSE"' in pyproject
    assert "src/protrepair/chemistry/radii.py" in notices
    assert "rdkit==2026.3.2" in notices
    assert "rdBase.rdkitVersion=2026.03.2" in notices
    assert "GetRvdw" in notices
    assert "GetRcovalent" in notices
    assert "static snapshot of RDKit PeriodicTable radii" in radius_policy
    assert "must not depend on vdW clash overlap" in radius_policy
    assert (
        'RDKIT_PERIODIC_TABLE_RADIUS_SNAPSHOT_VERSION = "2026.03.2"'
        in radii_source
    )
    assert "rdkit==2026.3.2" in radii_source


def test_release_constraints_pin_release_environment() -> None:
    """Release constraints should pin CI tools without narrowing metadata ranges."""

    constraint_lines = [
        line.strip()
        for line in Path("constraints/release.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    constraints = dict(line.split("==", maxsplit=1) for line in constraint_lines)
    workflow = Path(".github/workflows/ci.yml").read_text()
    smoke_script = Path("scripts/run_installed_artifact_smoke.py").read_text()

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
    assert 'pip install -c constraints/release.txt ".[dev]"' in workflow
    assert "DEFAULT_CONSTRAINTS_PATH" in smoke_script
    assert "PIP_CONSTRAINT" in smoke_script
    assert "PIP_BUILD_CONSTRAINT" in smoke_script


def test_release_docs_state_faspr_installed_asset_contract() -> None:
    """Release docs should not imply arbitrary source-tree FASPR availability."""

    readme = Path("README.md").read_text()
    normalized_readme = " ".join(readme.split())

    assert "Built packages and wheels include the vendored FASPR executable" in (
        normalized_readme
    )
    assert "Prefer a built wheel" in normalized_readme
    assert "direct GitHub installs" in normalized_readme
    assert "CMake 3.18 or newer" in normalized_readme
    assert "working C++ compiler toolchain" in normalized_readme
    assert "Direct source-tree imports are not guaranteed" in normalized_readme
    assert "explicit FASPR `executable_path`" in normalized_readme
    assert "dun2010bbdep.bin" in normalized_readme


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
    """README should document the retained-ligand fallback contract."""

    readme = Path("README.md").read_text()

    assert "RETAINED_NON_POLYMER_FALLBACK_USED" in readme
    assert "RETAINED_NON_POLYMER_FALLBACK_BLOCKED" in readme
    assert "optional RDKit" not in readme
    assert "refinement extra" not in readme


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
