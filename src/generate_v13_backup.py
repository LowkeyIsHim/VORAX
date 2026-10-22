# ==========================================
# VØRAX — Context-Based Text Generator
# ==========================================

import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "checkpoints" / "vorax_v10.npz"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model weights not found. Run python src/model.py first."
        )

    with np.load(MODEL_FILE) as saved:
        return {
            "E": saved["E"].copy(),
            "W1": saved["W1"].copy(),
            "b1": saved["b1"].copy(),
            "W2": saved["W2"].copy(),
            "b2": saved["b2"].copy(),
            "chars": saved["chars"].tolist(),
            "context_size": int(saved["context_size"]),
        }



def predict_next(context, model, temperature=0.7):
    embedded = model["E"][context].reshape(-1)

    hidden = np.tanh(
        embedded @ model["W1"] + model["b1"]
    )

    logits = hidden @ model["W2"] + model["b2"]
    logits /= max(temperature, 0.01)

    # Keep only the 5 most likely next characters.
    k = min(5, len(logits))
    top_ids = np.argpartition(logits, -k)[-k:]

    filtered_logits = logits[top_ids]
    filtered_logits -= np.max(filtered_logits)

    probabilities = np.exp(filtered_logits)
    probabilities /= probabilities.sum()

    result = np.zeros_like(logits, dtype=float)
    result[top_ids] = probabilities

    return result


def generate_text(seed, model, max_new_chars=200, temperature=0.7):
    chars = model["chars"]
    char_to_id = {char: i for i, char in enumerate(chars)}
    context_size = model["context_size"]

    unknown = [char for char in seed if char not in char_to_id]

    if unknown:
        raise ValueError(
            f"Unknown characters: {unknown}. "
            "Use characters present in the training data."
        )

    if not seed:
        seed = "hello"

    result = seed
    encoded = [char_to_id[char] for char in seed]

    # Pad short prompts with spaces if available.
    pad_id = char_to_id.get(" ", encoded[0])
    context = ([pad_id] * context_size + encoded)[-context_size:]

    rng = np.random.default_rng()

    for _ in range(max_new_chars):
        probabilities = predict_next(
            context, model, temperature
        )

        next_id = int(
            rng.choice(len(chars), p=probabilities)
        )

        result += chars[next_id]
        context = context[1:] + [next_id]

    return result


def main():
    model = load_model()

    print("=" * 44)
    print("       VØRAX AI v0.10")
    print("=" * 44)
    print("Context-based language model loaded.")
    print("Type 'exit' to shut down VØRAX.")

    while True:
        seed = input("\nYou: ").strip()

        if seed.lower() in ("exit", "quit", "shutdown"):
            print("VØRAX: Shutting down.")
            break

        if not seed:
            continue

        try:
            output = generate_text(
                seed,
                model,
                max_new_chars=150,
                temperature=0.5,
            )

            # Display only the generated continuation.
            print("\nVØRAX:", output[len(seed):].strip())

        except ValueError as error:
            print(f"VØRAX: {error}")


if __name__ == "__main__":
    main()
