import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_expected(name):
    """Your golden values for a chapter, or a skip if you have not written them yet."""
    path = ROOT / "expected" / f"{name}.json"
    if not path.exists():
        pytest.skip(
            f"no expected/{name}.json yet; see expected/README.md for the shape",
            allow_module_level=True,
        )
    return json.loads(path.read_text())
