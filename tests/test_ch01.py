"""Chapter 1 verifier.

Checks your implementation against your own golden values in
expected/ch01.json. Skips until you write that file.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ch01_perceptron import ORDERS, Y, classify, delta_rule, make_X, perceptron_rule, three_neuron_net

from conftest import load_expected

EXPECTED = load_expected("ch01")


@pytest.mark.parametrize("order", sorted(ORDERS))
def test_perceptron_rule(order):
    exp = EXPECTED["orders"][order]
    w, updates = perceptron_rule(make_X(order), Y)
    assert updates == exp["perceptron_updates"], f"order {order}: wrong number of weight updates"
    np.testing.assert_allclose(np.asarray(w, dtype=float), exp["perceptron_w"], atol=1e-9)


@pytest.mark.parametrize("order", sorted(ORDERS))
def test_perceptron_accuracy(order):
    exp = EXPECTED["orders"][order]
    X = make_X(order)
    w, _ = perceptron_rule(X, Y)
    assert int((classify(X, w) == Y).sum()) == exp["perceptron_correct"]


@pytest.mark.parametrize("order", sorted(ORDERS))
def test_delta_rule(order):
    exp = EXPECTED["orders"][order]
    w = delta_rule(make_X(order), Y)
    np.testing.assert_allclose(np.asarray(w, dtype=float), exp["delta_w"], atol=1e-9)


@pytest.mark.parametrize("point", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
def test_three_neuron_net(point):
    exp = EXPECTED["three_neuron"][f"{point[0]},{point[1]}"]
    assert list(three_neuron_net(*point)) == exp
