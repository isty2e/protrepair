"""Unit tests for installed-wheel smoke release tooling."""

import importlib.util
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


def test_prepare_virtual_environment_refuses_non_venv_directory(
    tmp_path: Path,
) -> None:
    """Existing arbitrary directories must not be removed as smoke venvs."""

    smoke_script = _load_smoke_script()
    target_path = tmp_path / "not-a-venv"
    target_path.mkdir()

    with pytest.raises(SystemExit, match="without pyvenv.cfg"):
        smoke_script.prepare_virtual_environment(target_path, keep_venv=False)

    assert target_path.is_dir()


def test_prepare_virtual_environment_refuses_repository_root() -> None:
    """A bad --venv-path must not be able to remove the repository checkout."""

    smoke_script = _load_smoke_script()

    with pytest.raises(SystemExit, match="unsafe venv path"):
        smoke_script.prepare_virtual_environment(
            smoke_script.REPOSITORY_ROOT,
            keep_venv=False,
        )


def test_prepare_virtual_environment_recreates_existing_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing venv recreation should delete only venv-shaped targets."""

    smoke_script = _load_smoke_script()
    target_path = tmp_path / "release-wheel-smoke"
    target_path.mkdir()
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


def test_prepare_virtual_environment_keep_venv_does_not_delete_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--keep-venv should preserve an existing directory instead of recreating it."""

    smoke_script = _load_smoke_script()
    target_path = tmp_path / "kept"
    target_path.mkdir()
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


def _load_smoke_script() -> SmokeScriptModule:
    script_path = REPOSITORY_ROOT / "scripts" / "run_installed_wheel_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "run_installed_wheel_smoke_for_tests",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(SmokeScriptModule, cast(ModuleType, module))
