# ==========================================
# VØRAX — Context-Based Neural Network
# Version 0.3
# ==========================================

import numpy as np
from pathlib import Path

from tokenizer import Tokenizer
from data_loader import load_training_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "checkpoints" / "vorax_weights.npz"

CONTEXT_SIZE = 8
EMBED_SIZE = 16
HIDDEN_SIZE = 64
LEARNING_RATE = 0.03
EPOCHS = 1200
SEED = 42


class VORAXModel:
    def __init__(self, vocab_size):
        rng = np.random.default_rng(SEED)

        self.vocab_size = vocab_size
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

        shifted = self.logits - np.max(self.logits)
        exp_values = np.exp(shifted)
        self.probs = exp_values / np.sum(exp_values)

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

        d_embedded = d_hidden @ self.W1.T
        d_embedded = d_embedded.reshape(
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


def build_examples(text, tokenizer):
    encoded = tokenizer.encode(text)
    examples = []

    for i in range(CONTEXT_SIZE, len(encoded)):
        context = encoded[i - CONTEXT_SIZE:i]
        target = encoded[i]
        examples.append((context, target))

    return examples


def main():
    text = load_training_data()
    tokenizer = Tokenizer(text)
    examples = build_examples(text, tokenizer)

    model = VORAXModel(len(tokenizer.chars))

    print("=" * 44)
    print("       VØRAX CONTEXT LEARNING ENGINE")
    print("=" * 44)
    print(f"Training characters: {len(text)}")
    print(f"Vocabulary size: {len(tokenizer.chars)}")
    print(f"Context size: {CONTEXT_SIZE} characters")
    print(f"Training examples: {len(examples)}")
    print(f"Training epochs: {EPOCHS}")
    print("-" * 44)

    rng = np.random.default_rng(SEED)

    for epoch in range(EPOCHS):
        rng.shuffle(examples)
        total_loss = 0.0

        for context, target in examples:
            total_loss += model.train_step(context, target)

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            average_loss = total_loss / len(examples)
            print(f"Epoch {epoch:4d} | Loss: {average_loss:.4f}")

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
    print(f"Weights saved to: {MODEL_FILE}")
    print("=" * 44)


if __name__ == "__main__":
    main()
