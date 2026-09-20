# visualai_codework

Exercise code for The Welch Labs Illustrated Guide to AI. Blank stubs: every
function that matters raises `NotImplementedError`. Filling them in cold, from
the docstring, is the rep.

Chapter notes, predictions and rep logs live in the vault at
`Books/The-Welch-Labs-Illustrated-Guide-to-AI/`. Code lives here.

## Environment

`~/.venvs/welch-ai`, Python 3.12, with numpy, matplotlib, pytest and torch
installed. The venv lives outside `~/Documents` on purpose: venvs under iCloud
break on newer Pythons.

```
~/.venvs/welch-ai/bin/python ch01_perceptron.py
~/.venvs/welch-ai/bin/python -m pytest tests/ -q
```

Chapter 2 also wants `transformers`, which is not installed yet. Add it when
you get there.

## Layout

```
ch01_perceptron.py        perceptron and delta rules, the three-neuron stack
ch02_gradient_descent.py  softmax, per-position loss table, sequence loss
ch03_backprop.py          TinyGPS and linear regression, backward pass in numpy
expected/                 your golden values, one JSON per chapter
tests/                    verifiers that read expected/ and skip without it
```

## Golden values

`expected/` is empty except for its README, which documents the JSON shape each
test expects. Producing those numbers is your job, not the scaffold's. Until a
chapter's file exists its tests skip rather than fail.

## Loop

1. Predict in the chapter note. Guess before you run anything.
2. Implement the stub cold, chapter note closed.
3. `pytest`. Wrong twice on the same exercise means it becomes a fumble-log row.
4. Log date, minutes, pass or fail, and what broke in the chapter note.
5. Three clean reps on three separate days retires the exercise.

The author's own notebooks are at https://github.com/stephencwelch/ai_book.
They are solutions. Opening one before three clean reps costs you the rep.
