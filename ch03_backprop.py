"""Chapter 3 exercises: backpropagation, by hand, in numpy.

No torch in this file. Deriving and writing the backward pass is the rep;
torch is the thing you check against afterwards.

Run:  ~/.venvs/welch-ai/bin/pytest tests/test_ch03.py
"""

import numpy as np

# Given. Longitudes only, in the order the author uses.
TINYGPS = {
    "berlin": ([2.3514, 2.2945, 13.4050, 13.3777], [0, 0, 1, 1]),
    "madrid": ([2.3514, 2.2945, -3.7033, -3.6835], [0, 0, 1, 1]),
}
LINREG_X = np.array([1.0, 2.0, 3.0, 4.0])
LINREG_Y = np.array([3.0, 5.0, 7.0, 9.0])


def tinygps_run(lons, y, lr=0.1, steps=10):
    """Runs 1 and 2. One linear layer, 1 input, 2 outputs, softmax cross-entropy.

    Initial parameters, set by hand, not random:
        W = [-1.0, 1.0]   (one weight per output class)
        b = [ 0.0, 0.0]

    One example per step, cycling with i % len(y), plain SGD.

    Record the state BEFORE each step's update, so `weights[0]` is the
    initialisation and `grads[0]` is the gradient computed at it.

    Returns a dict:
        weights - (steps, 4) array, each row [W0, W1, b0, b1]
        grads   - (steps, 4) array, each row [dW0, dW1, db0, db1]
        losses  - (steps,) array of the scalar loss at each step
        final   - (4,) array of parameters after the last update

    Softmax overflows if you exponentiate raw logits. Subtract the max first.
    """
    raise NotImplementedError("run 1")


def linreg_run(kind, lr=0.1, steps=8):
    """Runs 3 and 4. y = wx + b against four points on y = 2x + 1.

    Initial parameters w = 1.0, b = 0.0. One example per step, cycling,
    plain SGD.

    Args:
        kind: "mse" or "mae".

    Returns the same dict shape as tinygps_run, with 2-wide rows [w, b].

    Two traps. torch's MSELoss on a single element differentiates to
    2*(yhat - y), not (yhat - y); if your weights drift by exactly a factor
    of two that is why. And MAE's derivative at zero error is a choice, not
    a fact. Decide what you think it should be, then find out what torch does.
    """
    raise NotImplementedError("run 3")


if __name__ == "__main__":
    for name, (lons, y) in TINYGPS.items():
        print(name, tinygps_run(lons, y)["final"])
    for kind in ("mse", "mae"):
        print(kind, linreg_run(kind)["final"])
