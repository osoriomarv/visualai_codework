"""Chapter 2 exercises: gradient descent, via language-model loss.

The rep here is the reduction, not the model call. Given a sequence of logits
from a causal LM, build the per-position table and reduce it to the loss.

The author runs meta-llama/Llama-3.2-1B on CUDA. On a CPU box substitute gpt2
and accept different numbers, or save a logit tensor from the 4080 and rep
against it offline. See the chapter note.
"""

import numpy as np

SENTENCES = {
    "2.7": "The capital of France is Paris",
    "2.12": "An apple a day keeps the doctor away",
    "2.17": "I've had a perfectly wonderful evening, but this wasn't it",
}


def softmax(logits, axis=-1):
    """Numerically stable softmax. Subtract the max before exponentiating."""
    raise NotImplementedError("2.7")


def loss_table(logits, token_ids):
    """Per-position next-token table for one sequence.

    Args:
        logits: (T, V) array. Row t holds the scores for predicting token t+1.
        token_ids: (T,) array of the actual token ids in the sequence.

    Position t predicts token t+1, so the last row has nothing to score and
    the first token has no row predicting it. Get the off-by-one right and
    the rest is arithmetic.

    Returns:
        A list of T-1 dicts, one per scored position:
            {"position": int, "target_id": int, "prob": float, "neg_log_prob": float}
    """
    raise NotImplementedError("2.7")


def sequence_loss(logits, token_ids):
    """Mean of the negative log probabilities from loss_table. The cross-entropy."""
    raise NotImplementedError("2.7")


def load_logits(sentence, model_id="gpt2"):
    """Optional helper. Needs `pip install transformers` in ~/.venvs/welch-ai.

    Returns (logits, token_ids) as numpy arrays for the given sentence.
    Not part of the rep; fill it in once and leave it alone.
    """
    raise NotImplementedError("optional")
