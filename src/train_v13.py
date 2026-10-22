
import numpy as np
from pathlib import Path

from tokenizer import Tokenizer
from data_loader import load_training_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "checkpoints" / "vorax_v13.npz"

CONTEXT_SIZE = 16
EMBED_SIZE = 12
HIDDEN_SIZE = 48
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
EPOCHS = 100
SEED = 42


class VORAXModel:
    def __init__(self, vocab_size):
        rng = np.random.default_rng(SEED)
        fan_in = CONTEXT_SIZE * EMBED_SIZE

        self.E = rng.normal(0, 0.1, (vocab_size, EMBED_SIZE))
        self.W1 = rng.normal(0, np.sqrt(2 / fan_in),
                             (fan_in, HIDDEN_SIZE))
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.W2 = rng.normal(0, np.sqrt(2 / HIDDEN_SIZE),
                             (HIDDEN_SIZE, HIDDEN_SIZE))
        self.b2 = np.zeros(HIDDEN_SIZE)
        self.W3 = rng.normal(0, 0.05, (HIDDEN_SIZE, vocab_size))
        self.b3 = np.zeros(vocab_size)

    def forward(self, context):
        self.context = context
        self.embedded = self.E[context].reshape(-1)
        self.hidden1 = np.tanh(self.embedded @ self.W1 + self.b1)
        self.hidden2 = np.tanh(self.hidden1 @ self.W2 + self.b2)
        self.logits = self.hidden2 @ self.W3 + self.b3

        logits = self.logits - np.max(self.logits)
        exp_values = np.exp(logits)
        self.probs = exp_values / exp_values.sum()
        return self.probs

    def train_step(self, context, target):
        probs = self.forward(context)
        loss = -np.log(probs[target] + 1e-12)

        d3 = probs.copy()
        d3[target] -= 1

        dW3 = np.outer(self.hidden2, d3)
        db3 = d3.copy()

        d2 = (self.W3 @ d3) * (1 - self.hidden2 ** 2)
        dW2 = np.outer(self.hidden1, d2)
        db2 = d2.copy()

        d1 = (self.W2 @ d2) * (1 - self.hidden1 ** 2)
        dW1 = np.outer(self.embedded, d1)
        db1 = d1.copy()

        dembedded = (d1 @ self.W1.T).reshape(
            CONTEXT_SIZE, EMBED_SIZE
        )
        dE = np.zeros_like(self.E)

        for position, token_id in enumerate(self.context):
            dE[token_id] += dembedded[position]

        lr = LEARNING_RATE
        wd = WEIGHT_DECAY

        self.E -= lr * (dE + wd * self.E)
        self.W1 -= lr * (dW1 + wd * self.W1)
        self.b1 -= lr * db1
        self.W2 -= lr * (dW2 + wd * self.W2)
        self.b2 -= lr * db2
        self.W3 -= lr * (dW3 + wd * self.W3)
        self.b3 -= lr * db3

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

    for pos in range(start, end):
        target_char = text[pos]
        context_text = text[pos - CONTEXT_SIZE:pos]

        if len(context_text) != CONTEXT_SIZE:
            continue
        if any(ch not in char_to_id for ch in context_text):
            continue
        if target_char not in char_to_id:
            continue

        examples.append((
            tokenizer.encode(context_text),
            char_to_id[target_char]
        ))

    return examples


def main():
    text = load_training_data()

    train_end = int(len(text) * 0.70)
    val_end = int(len(text) * 0.85)

    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]

    tokenizer = Tokenizer(train_text)
    char_to_id = tokenizer.char_to_id

    train_examples = make_examples(tokenizer.encode(train_text))

    val_examples = build_boundary_examples(
        text, train_end, val_end, tokenizer
    )
    test_examples = build_boundary_examples(
        text, val_end, len(text), tokenizer
    )

    if not train_examples or not val_examples or not test_examples:
        raise ValueError("Insufficient known text for training and evaluation.")

    model = VORAXModel(len(tokenizer.chars))
    rng = np.random.default_rng(SEED)

    print("=" * 44)
    print("       VØRAX v0.13 TRAINING")
    print("=" * 44)
    print(f"Training characters: {len(train_text)}")
    print(f"Validation characters: {len(val_text)}")
    print(f"Test characters: {len(test_text)}")
    print(f"Vocabulary size: {len(tokenizer.chars)}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print("-" * 44)

    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = None
    patience = 10
    checks_without_improvement = 0

    for epoch in range(EPOCHS):
        rng.shuffle(train_examples)

        for context, target in train_examples:
            model.train_step(context, target)

        train_loss = evaluate(model, train_examples)
        val_loss = evaluate(model, val_examples)

        print(
            f"Epoch {epoch:3d} | Train loss: {train_loss:.4f} "
            f"| Val loss: {val_loss:.4f}"
        )

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {
                name: getattr(model, name).copy()
                for name in ("E", "W1", "b1", "W2", "b2", "W3", "b3")
            }
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        if checks_without_improvement >= patience:
            print("Early stopping: validation loss stopped improving.")
            break

    if best_weights is None:
        raise RuntimeError("No valid checkpoint was produced.")

    for name, weights in best_weights.items():
        setattr(model, name, weights)

    final_test_loss = evaluate(model, test_examples)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_FILE,
        E=model.E, W1=model.W1, b1=model.b1,
        W2=model.W2, b2=model.b2,
        W3=model.W3, b3=model.b3,
        chars=np.array(tokenizer.chars),
        context_size=np.array(CONTEXT_SIZE),
        embed_size=np.array(EMBED_SIZE),
        hidden_size=np.array(HIDDEN_SIZE),
        num_hidden_layers=np.array(2),
    )

    print("-" * 44)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"New checkpoint: {MODEL_FILE}")
    print("Training complete.")


if __name__ == "__main__":
    main()

