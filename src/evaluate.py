# ==========================================
# VØRAX — Model Evaluation
# ==========================================

import numpy as np
from pathlib import Path

from data_loader import load_training_data
from tokenizer import Tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "checkpoints" / "vorax_weights.npz"
CONTEXT_SIZE = 8


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "No checkpoint found. Train VØRAX first."
        )

    with np.load(MODEL_FILE) as saved:
        model = {
            key: saved[key].copy()
            for key in ("E", "W1", "b1", "W2", "b2", "chars")
        }

    model["chars"] = model["chars"].tolist()
    model["context_size"] = int(
        np.load(MODEL_FILE)["context_size"]
    )
    return model


def predict(context, model):
    embedded = model["E"][context].reshape(-1)
    hidden = np.tanh(embedded @ model["W1"] + model["b1"])
    logits = hidden @ model["W2"] + model["b2"]

    logits -= np.max(logits)
    exp_values = np.exp(logits)
    return exp_values / exp_values.sum()


def main():
    text = load_training_data()
    model = load_model()

    chars = model["chars"]
    char_to_id = {char: i for i, char in enumerate(chars)}
    context_size = model["context_size"]

    # Hold out the final 20% of the text.
    split = int(len(text) * 0.8)
    train_text = text[:split]
    test_text = text[split:]

    # Report characters the model has never encountered.
    unknown = sorted(set(test_text) - set(chars))

    # Keep the context from the training section,
    # but score only characters in the test section.
    combined = train_text + test_text
    encoded = [char_to_id[c] for c in combined if c in char_to_id]

    losses = []
    correct = 0
    total = 0

    for i in range(split, len(combined)):
        target_char = combined[i]

        if target_char not in char_to_id:
            continue

        context_chars = combined[i - context_size:i]

        if len(context_chars) != context_size:
            continue

        if any(c not in char_to_id for c in context_chars):
            continue

        context = [char_to_id[c] for c in context_chars]
        target = char_to_id[target_char]

        probabilities = predict(context, model)
        losses.append(-np.log(probabilities[target] + 1e-12))

        if int(np.argmax(probabilities)) == target:
            correct += 1

        total += 1

    print("=" * 44)
    print("       VØRAX MODEL EVALUATION")
    print("=" * 44)
    print(f"Total text characters: {len(text)}")
    print(f"Training section: {len(train_text)} characters")
    print(f"Test section: {len(test_text)} characters")
    print(f"Scored test characters: {total}")
    print(f"Unknown test characters: {len(unknown)}")

    if total:
        mean_loss = float(np.mean(losses))
        accuracy = 100 * correct / total

        print(f"Test loss: {mean_loss:.4f}")
        print(f"Next-character accuracy: {accuracy:.2f}%")
        print(f"Perplexity: {np.exp(min(mean_loss, 50)):.2f}")
    else:
        print("No test characters could be scored.")

    if unknown:
        print(f"Unknown characters: {unknown}")

    print("=" * 44)
    print("Lower test loss is generally better.")
    print("Compare models using the same test split.")
    print("=" * 44)


if __name__ == "__main__":
    main()	
