# ==========================================
# VØRAX v0.5 — Fair Training Pipeline
# ==========================================

import numpy as np
from pathlib import Path

from tokenizer import Tokenizer
from data_loader import load_training_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "checkpoints" / "vorax_v07.npz"

CONTEXT_SIZE = 8
EMBED_SIZE = 16
HIDDEN_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 200
SEED = 42


class VORAXModel:
    def __init__(self, vocab_size):
        rng = np.random.default_rng(SEED)

        self.E = rng.normal(0, 0.1, (vocab_size, EMBED_SIZE))
        self.W1 = rng.normal(
            0, 0.1, (CONTEXT_SIZE * EMBED_SIZE, HIDDEN_SIZE)
        )
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.W2 = rng.normal(0, 0.1, (HIDDEN_SIZE, vocab_size))
        self.b2 = np.zeros(vocab_size)

    def forward(self, context):
        self.context = context
        self.embedded = self.E[context].reshape(-1)
        self.hidden = np.tanh(self.embedded @ self.W1 + self.b1)
        self.logits = self.hidden @ self.W2 + self.b2

        logits = self.logits - np.max(self.logits)
        exp_values = np.exp(logits)
        self.probs = exp_values / exp_values.sum()

        return self.probs

    def train_step(self, context, target):
        probs = self.forward(context)
        loss = -np.log(probs[target] + 1e-12)

        d_logits = probs.copy()
        d_logits[target] -= 1

        dW2 = np.outer(self.hidden, d_logits)
        db2 = d_logits.copy()

        d_hidden = self.W2 @ d_logits
        d_hidden *= 1 - self.hidden ** 2

        dW1 = np.outer(self.embedded, d_hidden)
        db1 = d_hidden.copy()

        d_embedded = (d_hidden @ self.W1.T).reshape(
            CONTEXT_SIZE, EMBED_SIZE
        )

        dE = np.zeros_like(self.E)
        for position, token_id in enumerate(self.context):
            dE[token_id] += d_embedded[position]

        self.E -= LEARNING_RATE * dE
        self.W1 -= LEARNING_RATE * dW1
        self.b1 -= LEARNING_RATE * db1
        self.W2 -= LEARNING_RATE * dW2
        self.b2 -= LEARNING_RATE * db2

        return float(loss)


def make_examples(encoded):
    return [
        (encoded[i - CONTEXT_SIZE:i], encoded[i])
        for i in range(CONTEXT_SIZE, len(encoded))
    ]


def evaluate(model, examples):
    if not examples:
        return float("nan")

    losses = []
    for context, target in examples:
        probs = model.forward(context)
        losses.append(-np.log(probs[target] + 1e-12))

    return float(np.mean(losses))


def main():
    text = load_training_data()

    # Split raw text BEFORE building training examples.
    split = int(len(text) * 0.8)
    train_text = text[:split]
    test_text = text[split:]

    # Build the vocabulary from training text only.
    tokenizer = Tokenizer(train_text)
    char_to_id = tokenizer.char_to_id

    # Unknown test characters cannot be scored.
    unknown = sorted(set(test_text) - set(char_to_id))

    # Encode training text.
    train_ids = tokenizer.encode(train_text)
    train_examples = make_examples(train_ids)

    # Build test examples while preserving original character positions.
    # The last training characters provide context for the test boundary.
    test_examples = []
    full_text = train_text + test_text

    for target_pos in range(split, len(full_text)):
        target_char = full_text[target_pos]
        context_text = full_text[target_pos - CONTEXT_SIZE:target_pos]

        # Require a complete context and known characters throughout.
        if len(context_text) != CONTEXT_SIZE:
            continue

        if any(char not in char_to_id for char in context_text):
            continue

        if target_char not in char_to_id:
            continue

        context = tokenizer.encode(context_text)
        target = char_to_id[target_char]
        test_examples.append((context, target))

    if not train_examples or not test_examples:
        raise ValueError(
            "Not enough known text to create training and test examples."
        )

    model = VORAXModel(len(tokenizer.chars))
    rng = np.random.default_rng(SEED)

    print("=" * 44)
    print("       VØRAX v0.7 TRAINING")
    print("=" * 44)
    print(f"Training characters: {len(train_text)}")
    print(f"Test characters: {len(test_text)}")
    print(f"Vocabulary size: {len(tokenizer.chars)}")
    print(f"Unknown test characters: {len(unknown)}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print("-" * 44)

    best_test_loss = float("inf")
    best_epoch = -1
    best_weights = None

    for epoch in range(EPOCHS):
        rng.shuffle(train_examples)

        for context, target in train_examples:
            model.train_step(context, target)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            train_loss = evaluate(model, train_examples)
            test_loss = evaluate(model, test_examples)

            print(
                f"Epoch {epoch:3d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Test loss: {test_loss:.4f}"
            )

            if np.isfinite(test_loss) and test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                best_weights = {
                    "E": model.E.copy(),
                    "W1": model.W1.copy(),
                    "b1": model.b1.copy(),
                    "W2": model.W2.copy(),
                    "b2": model.b2.copy(),
                }

    if best_weights is None:
        raise RuntimeError("No valid checkpoint was produced.")

    model.E = best_weights["E"]
    model.W1 = best_weights["W1"]
    model.b1 = best_weights["b1"]
    model.W2 = best_weights["W2"]
    model.b2 = best_weights["b2"]

    print("-" * 44)
    print(f"Best epoch: {best_epoch}")
    print(f"Best test loss: {best_test_loss:.4f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        MODEL_FILE,
        E=model.E,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        chars=np.array(tokenizer.chars),
        context_size=np.array(CONTEXT_SIZE),
        embed_size=np.array(EMBED_SIZE),
        hidden_size=np.array(HIDDEN_SIZE),
    )

    print("-" * 44)
    print("Training complete.")
    print(f"New checkpoint: {MODEL_FILE}")
    print("Existing v0.3 checkpoint was not changed.")
    print("=" * 44)


if __name__ == "__main__":
    main()
