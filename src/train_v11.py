# ==========================================
# VORAX v0.8 — Train / Validation / Test
# ==========================================

import numpy as np
from pathlib import Path

from tokenizer import Tokenizer
from data_loader import load_training_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "checkpoints" / "vorax_v11.npz"

CONTEXT_SIZE = 16
EMBED_SIZE = 12
HIDDEN_SIZE = 48
LEARNING_RATE = 0.003
EPOCHS = 150
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

        WEIGHT_DECAY = 0.001

        self.E -= LEARNING_RATE * (dE + WEIGHT_DECAY * self.E)
        self.W1 -= LEARNING_RATE * (dW1 + WEIGHT_DECAY * self.W1)
        self.b1 -= LEARNING_RATE * db1
        self.W2 -= LEARNING_RATE * (dW2 + WEIGHT_DECAY * self.W2)
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


def build_boundary_examples(text, start, end, tokenizer):
    examples = []
    char_to_id = tokenizer.char_to_id

    for target_pos in range(start, end):
        target_char = text[target_pos]
        context_text = text[target_pos - CONTEXT_SIZE:target_pos]

        if len(context_text) != CONTEXT_SIZE:
            continue

        if any(char not in char_to_id for char in context_text):
            continue

        if target_char not in char_to_id:
            continue

        context = tokenizer.encode(context_text)
        target = char_to_id[target_char]
        examples.append((context, target))

    return examples


def main():
    text = load_training_data()

    # Chronological split: 70% train, 15% validation, 15% test.
    train_end = int(len(text) * 0.70)
    val_end = int(len(text) * 0.85)

    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]

    # Learn vocabulary from training data only.
    tokenizer = Tokenizer(train_text)
    char_to_id = tokenizer.char_to_id

    train_ids = tokenizer.encode(train_text)
    train_examples = make_examples(train_ids)

    # Preserve context across split boundaries.
    full_text = train_text + val_text + test_text
    val_examples = build_boundary_examples(
        full_text, train_end, val_end, tokenizer
    )
    test_examples = build_boundary_examples(
        full_text, val_end, len(full_text), tokenizer
    )

    if not train_examples or not val_examples or not test_examples:
        raise ValueError(
            "Not enough known text to create train, validation, and test examples."
        )

    unknown_val = sum(
        char not in char_to_id for char in val_text
    )
    unknown_test = sum(
        char not in char_to_id for char in test_text
    )

    model = VORAXModel(len(tokenizer.chars))
    rng = np.random.default_rng(SEED)

    print("=" * 44)
    print("       VØRAX v0.11 TRAINING")
    print("=" * 44)
    print(f"Training characters: {len(train_text)}")
    print(f"Validation characters: {len(val_text)}")
    print(f"Test characters: {len(test_text)}")
    print(f"Vocabulary size: {len(tokenizer.chars)}")
    print(f"Unknown validation characters: {unknown_val}")
    print(f"Unknown test characters: {unknown_test}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print("-" * 44)

    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = None

    for epoch in range(EPOCHS):
        rng.shuffle(train_examples)

        for context, target in train_examples:
            model.train_step(context, target)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            train_loss = evaluate(model, train_examples)
            val_loss = evaluate(model, val_examples)

            print(
                f"Epoch {epoch:3d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f}"
            )

            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
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

    # Restore weights selected using validation loss.
    model.E = best_weights["E"]
    model.W1 = best_weights["W1"]
    model.b1 = best_weights["b1"]
    model.W2 = best_weights["W2"]
    model.b2 = best_weights["b2"]

    # Evaluate test data only after checkpoint selection is complete.
    final_test_loss = evaluate(model, test_examples)

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
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final test loss: {final_test_loss:.4f}")
    print("Training complete.")
    print(f"New checkpoint: {MODEL_FILE}")
    print("Existing checkpoints were not overwritten.")
    print("=" * 44)


if __name__ == "__main__":
    main()

