"""Fresh release artifact content tests."""

import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class BuiltReleaseArtifacts:
    """Fresh wheel and sdist paths built for release artifact inspection."""

    wheel_path: Path
    sdist_path: Path


@pytest.fixture(scope="module")
def built_release_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> BuiltReleaseArtifacts:
    """Build fresh release artifacts in an isolated dist directory."""

    dist_dir = tmp_path_factory.mktemp("release-artifacts")
    subprocess.run(
        (
            sys.executable,
            "-m",
            "hatchling",
            "build",
            "-d",
            str(dist_dir),
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    wheels = tuple(dist_dir.glob("protrepair-*.whl"))
    sdists = tuple(dist_dir.glob("protrepair-*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
    return BuiltReleaseArtifacts(wheel_path=wheels[0], sdist_path=sdists[0])


def test_built_wheel_contains_runtime_assets_and_notices(
    built_release_artifacts: BuiltReleaseArtifacts,
) -> None:
    """Built wheels should carry runtime data, FASPR assets, and notices."""

    with ZipFile(built_release_artifacts.wheel_path) as archive:
        names = frozenset(archive.namelist())

    assert {
        "protrepair/py.typed",
        "protrepair/chemistry/radii.py",
        "protrepair/chemistry/resources/nonstandard_components.json.gz",
        "protrepair/chemistry/resources/retained_non_polymer_components.json.gz",
        "protrepair/packing/faspr/bin/FASPR",
        "protrepair/packing/faspr/bin/dun2010bbdep.bin",
    } <= names
    assert _has_suffix(names, ".dist-info/licenses/LICENSE")
    assert _has_suffix(names, ".dist-info/licenses/THIRD_PARTY_NOTICES.md")
    assert _has_suffix(names, ".dist-info/licenses/vendor/faspr/LICENSE")
    assert _has_suffix(
        names,
        ".dist-info/licenses/vendor/faspr/DUNBRACK_ROTAMER_LIBRARY_NOTICE.md",
    )
    assert _has_suffix(names, ".dist-info/licenses/vendor/faspr/PROVENANCE.md")
    assert _has_suffix(names, ".dist-info/licenses/vendor/faspr/README.upstream.md")

    with ZipFile(built_release_artifacts.wheel_path) as archive:
        notices_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md")
        )
        notices = archive.read(notices_name).decode("utf-8")

    assert "src/protrepair/chemistry/radii.py" in notices
    assert "rdkit==2026.3.2" in notices
    assert "GetRvdw" in notices
    assert "GetRcovalent" in notices


def test_built_sdist_contains_release_sources_and_vendor_snapshot(
    built_release_artifacts: BuiltReleaseArtifacts,
) -> None:
    """Built sdists should carry release gates and complete vendored sources."""

    with tarfile.open(built_release_artifacts.sdist_path) as archive:
        names = frozenset(archive.getnames())

    assert _has_suffix(names, "/src/protrepair/py.typed")
    assert _has_suffix(names, "/src/protrepair/chemistry/radii.py")
    assert _has_suffix(
        names,
        "/src/protrepair/chemistry/resources/nonstandard_components.json.gz",
    )
    assert _has_suffix(
        names,
        "/src/protrepair/chemistry/resources/retained_non_polymer_components.json.gz",
    )
    assert _has_suffix(names, "/constraints/release.txt")
    assert _has_suffix(names, "/docs/release-checklist.md")
    assert _has_suffix(names, "/docs/radius-policy.md")
    assert _has_suffix(names, "/scripts/run_installed_wheel_smoke.py")
    assert _has_suffix(names, "/THIRD_PARTY_NOTICES.md")

    vendor_root = REPOSITORY_ROOT / "vendor/faspr"
    for vendor_path in sorted(vendor_root.rglob("*")):
        if vendor_path.is_file():
            relative_vendor_path = vendor_path.relative_to(vendor_root).as_posix()
            assert _has_suffix(
                names,
                f"/vendor/faspr/{relative_vendor_path}",
            )

    assert not any("/.tickets/" in name for name in names)
    assert not any("/.tmp/" in name for name in names)
    assert not any("/dist/" in name for name in names)
    smoke_script = "/scripts/run_installed_wheel_smoke.py"
    assert not any(
        "/scripts/" in name and not name.endswith(smoke_script)
        for name in names
    )


def _has_suffix(names: frozenset[str], suffix: str) -> bool:
    """Return whether any archive member has the expected suffix."""

    return any(name.endswith(suffix) for name in names)
