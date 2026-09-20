"""Chapter 3 verifier.

Checks your implementation against your own golden values in
expected/ch03.json. Skips until you write that file.

Tolerance is 1e-4. If you generate the goldens from torch, that is float32
while your numpy is float64; measured drift on these runs is under 5e-6, so
1e-4 is loose enough for that and tight enough to catch a wrong derivative.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ch03_backprop import TINYGPS, linreg_run, tinygps_run

from conftest import load_expected

EXPECTED = load_expected("ch03")
ATOL = 1e-4


def _check(got, key):
    exp = EXPECTED[key]
    for field in ("weights", "grads", "losses", "final"):
        np.testing.assert_allclose(
            np.asarray(got[field], dtype=float), exp[field], atol=ATOL,
            err_msg=f"{key}: {field} does not match",
        )


@pytest.mark.parametrize("city", sorted(TINYGPS))
def test_tinygps(city):
    lons, y = TINYGPS[city]
    _check(tinygps_run(lons, y), f"tinygps_{city}")


@pytest.mark.parametrize("kind", ["mse", "mae"])
def test_linreg(kind):
    _check(linreg_run(kind), f"linreg_{kind}")
