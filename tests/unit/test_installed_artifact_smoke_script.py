"""Unit tests for installed-artifact smoke release tooling."""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class SmokeScriptModule(Protocol):
    """Typed surface used from the release smoke script."""

    REPOSITORY_ROOT: Path

    def prepare_virtual_environment(
        self,
        venv_path: Path,
        *,
        keep_venv: bool,
    ) -> Path: ...

    def install_artifact(
        self,
        python: Path,
        artifact_path: Path,
        constraints_path: Path,
    ) -> None: ...


def test_prepare_virtual_environment_refuses_non_venv_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing arbitrary directories must not be removed as smoke venvs."""

    smoke_script = _load_smoke_script()
    target_path = _managed_venv_path(smoke_script, tmp_path, monkeypatch)
    target_path.mkdir(parents=True)

    with pytest.raises(SystemExit, match="without pyvenv.cfg"):
        smoke_script.prepare_virtual_environment(target_path, keep_venv=False)

    assert target_path.is_dir()


def test_prepare_virtual_environment_refuses_paths_outside_temporary_root(
    tmp_path: Path,
) -> None:
    """New smoke environments must remain within the repository scratch directory."""

    smoke_script = _load_smoke_script()

    with pytest.raises(SystemExit, match="outside repository temporary root"):
        smoke_script.prepare_virtual_environment(
            tmp_path / "outside-repository",
            keep_venv=False,
        )


def test_prepare_virtual_environment_recreates_existing_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing venv recreation should delete only venv-shaped targets."""

    smoke_script = _load_smoke_script()
    target_path = _managed_venv_path(smoke_script, tmp_path, monkeypatch)
    target_path.mkdir(parents=True)
    (target_path / "pyvenv.cfg").write_text("home = /tmp/python\n")
    stale_file = target_path / "stale.txt"
    stale_file.write_text("old")
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...]) -> None:
        commands.append(command)

    monkeypatch.setattr(smoke_script, "run", fake_run)

    prepared_path = smoke_script.prepare_virtual_environment(
        target_path,
        keep_venv=False,
    )

    assert prepared_path == target_path.resolve()
    assert not stale_file.exists()
    assert commands == [
        (sys.executable, "-m", "venv", str(target_path.resolve())),
    ]


def test_prepare_virtual_environment_refuses_external_existing_venv(
    tmp_path: Path,
) -> None:
    """An explicit external venv must not become a deletable smoke target."""

    smoke_script = _load_smoke_script()
    target_path = tmp_path / "external-venv"
    target_path.mkdir()
    (target_path / "pyvenv.cfg").write_text("home = /tmp/python\n")
    retained_file = target_path / "retain.txt"
    retained_file.write_text("keep")

    with pytest.raises(SystemExit, match="outside repository temporary root"):
        smoke_script.prepare_virtual_environment(target_path, keep_venv=False)

    assert retained_file.read_text() == "keep"


def test_prepare_virtual_environment_keep_venv_does_not_delete_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--keep-venv should preserve an existing directory instead of recreating it."""

    smoke_script = _load_smoke_script()
    target_path = tmp_path / "kept"
    target_path.mkdir()
    (target_path / "pyvenv.cfg").write_text("home = /tmp/python\n")
    kept_file = target_path / "kept.txt"
    kept_file.write_text("keep")
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...]) -> None:
        commands.append(command)

    monkeypatch.setattr(smoke_script, "run", fake_run)

    prepared_path = smoke_script.prepare_virtual_environment(
        target_path,
        keep_venv=True,
    )

    assert prepared_path == target_path.resolve()
    assert kept_file.read_text() == "keep"
    assert commands == []


def test_prepare_virtual_environment_keep_venv_rejects_arbitrary_directory(
    tmp_path: Path,
) -> None:
    """--keep-venv must not trust an arbitrary existing directory."""

    smoke_script = _load_smoke_script()
    target_path = tmp_path / "not-a-venv"
    target_path.mkdir()

    with pytest.raises(SystemExit, match="without pyvenv.cfg"):
        smoke_script.prepare_virtual_environment(target_path, keep_venv=True)


def test_install_artifact_constrains_pep517_build_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct and isolated pip resolvers should share release constraints."""

    smoke_script = _load_smoke_script()
    artifact_path = tmp_path / "protrepair-0.1.0.whl"
    artifact_path.write_text("artifact")
    constraints_path = tmp_path / "release.txt"
    constraints_path.write_text("rdkit==2026.3.2\n")
    commands: list[tuple[str, ...]] = []
    install_environments: list[dict[str, str] | None] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        cwd: Path = REPOSITORY_ROOT,
        env: dict[str, str] | None = None,
    ) -> None:
        commands.append(command)
        install_environments.append(env)

    monkeypatch.setenv("PIP_CONSTRAINT", "/untrusted/constraints.txt")
    monkeypatch.setenv("PIP_BUILD_CONSTRAINT", "/untrusted/build-constraints.txt")
    monkeypatch.setenv("PIP_TARGET", "/untrusted/target")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/source")
    monkeypatch.setattr(smoke_script, "run", fake_run)

    smoke_script.install_artifact(
        Path("/tmp/smoke-venv/bin/python"),
        artifact_path,
        constraints_path,
    )

    assert commands == [
        (
            "/tmp/smoke-venv/bin/python",
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "-c",
            str(constraints_path.resolve()),
            str(artifact_path),
        )
    ]
    assert install_environments[0] is not None
    install_environment = install_environments[0]
    assert install_environment["PIP_CONSTRAINT"] == str(constraints_path.resolve())
    assert install_environment["PIP_BUILD_CONSTRAINT"] == str(
        constraints_path.resolve()
    )
    assert install_environment["PYTHONNOUSERSITE"] == "1"
    assert install_environment["PIP_CONFIG_FILE"] == os.devnull
    assert "PIP_TARGET" not in install_environment
    assert "PYTHONPATH" not in install_environment


def _managed_venv_path(
    smoke_script: SmokeScriptModule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Return a test-only venv path under the script's managed scratch root."""

    repository_root = tmp_path / "repository"
    monkeypatch.setattr(smoke_script, "REPOSITORY_ROOT", repository_root)
    return repository_root / ".tmp" / "release-artifact-smoke"


def _load_smoke_script() -> SmokeScriptModule:
    script_path = REPOSITORY_ROOT / "scripts" / "run_installed_artifact_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "run_installed_artifact_smoke_for_tests",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(SmokeScriptModule, cast(ModuleType, module))
