"""pytest plugin: collect the repository root as a plain directory.

Loaded through `-p _collect_root_as_dir` in pyproject.toml; see the comment
there for why. Without it pytest imports the node's `__init__.py`, which needs
a running ComfyUI, before the first test runs.
"""

import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def pytest_collect_directory(path, parent):
    if pathlib.Path(path).resolve() == _ROOT:
        return pytest.Dir.from_parent(parent, path=path)
    return None
