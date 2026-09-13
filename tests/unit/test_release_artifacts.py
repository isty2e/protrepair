"""Release artifact sanity tests for packaged resources and metadata."""

import hashlib
import subprocess
import sys
from pathlib import Path


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


def test_faspr_vendored_snapshot_integrity() -> None:
    """The source and assets must match the reviewed vendored snapshot."""

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
        assert sha256_file(Path(path_text)) == expected_digest

    expected_src_manifest_digest = (
        "cc2ca78f30069e0e9308a0b68674da00864fa1ca09386128b96b36148a3c09d7"
    )
    assert sha256_sorted_file_manifest(Path("vendor/faspr/src")) == (
        expected_src_manifest_digest
    )


def test_rdkit_license_payload_integrity() -> None:
    """Preserve the complete reviewed license, not selected wording from it."""

    assert sha256_file(Path("vendor/rdkit/LICENSE")) == (
        "daeb8d194502cbcf34c05c39541a0d02be65bc9bada5b891c1974cd24e9fca30"
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
