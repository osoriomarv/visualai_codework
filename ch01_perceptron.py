"""Chapter 1 exercises: the perceptron.

Hand-code every function marked TODO. Do not open the author's notebook and do
not open expected/ch01.json until pytest has told you you are wrong twice.

Run:  ~/.venvs/welch-ai/bin/pytest tests/test_ch01.py
"""

import numpy as np

# Given. The four two-bit examples in the three orders the book uses, and the
# targets. Building X is not the rep.
ORDERS = {
    "A": [[-1, -1], [-1, 1], [1, -1], [1, 1]],
    "B": [[-1, -1], [1, -1], [-1, 1], [1, 1]],
    "C": [[1, -1], [-1, 1], [-1, -1], [1, 1]],
}
Y = np.array([-1, -1, 1, 1])


def make_X(order):
    """Examples as rows with a bias column of ones appended. Given."""
    return np.hstack((np.array(ORDERS[order], dtype=float), np.ones((len(Y), 1))))


def perceptron_rule(X, y, lr=1.0, steps=12):
    """Exercises 1.4, 1.5, 1.6, 1.7.

    Start at w = zeros(3). Walk the examples in order, cycling with i % len(y).
    At each step compute yhat = w . x. Update ONLY when the sign is wrong:
    add lr*x when the target is positive and yhat is not, subtract lr*x when
    the target is negative and yhat is not. Treat yhat == 0 as wrong either way.

    Returns:
        (w, n_updates) - final weight vector, and how many of the `steps`
        steps actually changed w.
    """
    raise NotImplementedError("1.4")


def delta_rule(X, y, lr=0.2, steps=12):
    """Exercises 1.10, 1.13, 1.14.

    Same setup, different update: every step moves the weights, by an amount
    proportional to the error rather than by a fixed size.

        w <- w + lr * x * (y - yhat)

    Returns:
        w - final weight vector.
    """
    raise NotImplementedError("1.10")


def classify(X, w):
    """Sign of w . x for every row, with 0 counted as -1. Given."""
    out = np.sign(X @ w)
    out[out == 0] = -1
    return out


def three_neuron_net(x1, x2):
    """Exercise 1.15. Nothing here trains; the thresholds are fixed.

        h1 = sign(x1 + x2 + 0.5)
        h2 = sign(x1 + x2 - 1.5)
        out = sign(h1 - h2 - 0.5)

    sign() here returns +1 when the expression is strictly greater than 0 and
    -1 otherwise.

    Returns:
        (h1, h2, out) as ints.
    """
    raise NotImplementedError("1.15")


if __name__ == "__main__":
    for name in ORDERS:
        X = make_X(name)
        print(name, perceptron_rule(X, Y), delta_rule(X, Y))
