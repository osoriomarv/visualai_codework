# Expected values

Yours to write. Nothing in here is generated for you.

The tests in `../tests/` read `ch01.json` and `ch03.json` from this directory
and skip cleanly when a file is missing. Produce them however you like: run the
author's notebook, derive the numbers on paper, or trace the book's printed
tables.

## ch01.json

```json
{
  "orders": {
    "A": {
      "perceptron_w": [w1, w2, b],
      "perceptron_updates": int,
      "perceptron_correct": int,
      "delta_w": [w1, w2, b]
    },
    "B": { ... },
    "C": { ... }
  },
  "three_neuron": {
    "1,1":   [h1, h2, out],
    "-1,1":  [h1, h2, out],
    "1,-1":  [h1, h2, out],
    "-1,-1": [h1, h2, out]
  }
}
```

## ch03.json

One entry per run. `weights` and `grads` record the state *before* each step's
update, so row 0 is the initialisation.

```json
{
  "tinygps_berlin": {
    "weights": [[W0, W1, b0, b1], ...],
    "grads":   [[dW0, dW1, db0, db1], ...],
    "losses":  [float, ...],
    "final":   [W0, W1, b0, b1]
  },
  "tinygps_madrid": { ... },
  "linreg_mse": { "weights": [[w, b], ...], "grads": [[dw, db], ...], "losses": [...], "final": [w, b] },
  "linreg_mae": { ... }
}
```
