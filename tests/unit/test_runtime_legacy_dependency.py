from pathlib import Path


def test_runtime_code_does_not_import_protrepair_server() -> None:
    runtime_root = Path("src/protrepair")
    runtime_python_files = tuple(sorted(runtime_root.rglob("*.py")))

    offending_files = [
        path
        for path in runtime_python_files
        if "Pras_Server" in path.read_text(encoding="utf-8")
    ]

    assert offending_files == []
