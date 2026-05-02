from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Provide a Windows-safe replacement for pytest's temporary path fixture."""
    base_dir = Path.cwd() / ".pytest-safe-tmp"
    path = base_dir / uuid4().hex
    remove_tree = shutil.rmtree
    path.mkdir(parents=True)

    try:
        yield path
    finally:
        try:
            remove_tree(path)
        except FileNotFoundError:
            pass

        try:
            base_dir.rmdir()
        except OSError:
            pass
