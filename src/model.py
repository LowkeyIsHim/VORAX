# ==========================================
# VØRAX — Neural Network v0.2
# Character-Level Learning Engine
# ==========================================

import numpy as np
from pathlib import Path

from tokenizer import Tokenizer
from data_loader import load_training_data


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "checkpoints" / "vorax_weights.npz"

SEED = 42
HIDDEN_SIZE = 64
LEARNING_RATE = 0.05
EPOCHS = 1500


class VORAXModel:
    def __init__(self, vocab_size):
        rng = np.random.default_rng(SEED)

        self.W1 = rng.normal(
            0, 0.1, (vocab_size, HIDDEN_SIZE)
        )
        self.b1 = np.zeros(HIDDEN_SIZE)

        self.W2 = rng.normal(
            0, 0.1, (HIDDEN_SIZE, vocab_size)
        )
        self.b2 = np.zeros(vocab_size)

    def forward(self, x):
        self.hidden = np.tanh(
            self.W1[x] + self.b1
        )

        self.logits = (
            self.hidden @ self.W2 + self.b2
        )

        # Stable softmax
        shifted = self.logits - np.max(self.logits)
        exp_values = np.exp(shifted)
        self.probs = exp_values / np.sum(exp_values)

        return self.probs

    def train_step(self, x, target):
        probs = self.forward(x)

        loss = -np.log(probs[target] + 1e-12)

        # Gradient of cross-entropy with softmax
        d_logits = probs.copy()
        d_logits[target] -= 1

        dW2 = np.outer(self.hidden, d_logits)
        db2 = d_logits.copy()

        d_hidden = self.W2 @ d_logits
        d_hidden *= 1 - self.hidden ** 2

        dW1 = np.zeros_like(self.W1)
        dW1[x] = d_hidden
        db1 = d_hidden.copy()

        # Gradient descent
        self.W1 -= LEARNING_RATE * dW1
        self.b1 -= LEARNING_RATE * db1
        self.W2 -= LEARNING_RATE * dW2
        self.b2 -= LEARNING_RATE * db2

        return float(loss)


def build_examples(text, tokenizer):
    encoded = tokenizer.encode(text)

    # Each character predicts the following character.
    return [
        (encoded[i], encoded[i + 1])
        for i in range(len(encoded) - 1)
    ]


def main():
    text = load_training_data()
    tokenizer = Tokenizer(text)

    examples = build_examples(text, tokenizer)

    model = VORAXModel(len(tokenizer.chars))

    print("=" * 44)
    print("       VØRAX LEARNING ENGINE")
    print("=" * 44)
    print(f"Training characters: {len(text)}")
    print(f"Vocabulary size: {len(tokenizer.chars)}")
    print(f"Training examples: {len(examples)}")
    print(f"Hidden neurons: {HIDDEN_SIZE}")
    print(f"Training epochs: {EPOCHS}")
    print("-" * 44)

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for x, target in examples:
            total_loss += model.train_step(x, target)

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            average_loss = total_loss / len(examples)
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {average_loss:.4f}"
            )

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        MODEL_FILE,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        chars=np.array(tokenizer.chars),
    )

    print("-" * 44)
    print("Training complete.")
    print(f"Weights saved to: {MODEL_FILE}")
    print("=" * 44)


if __name__ == "__main__":
    main()
