
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "checkpoints" / "vorax_v14.npz"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_FILE}")

    with np.load(MODEL_FILE) as saved:
        return {key: saved[key].copy() for key in saved.files}


def predict_next(context, model, temperature=0.8):
    embedded = model["E"][context].reshape(-1)

    hidden1 = np.tanh(embedded @ model["W1"] + model["b1"])
    hidden2 = np.tanh(hidden1 @ model["W2"] + model["b2"])
    logits = (hidden2 @ model["W3"] + model["b3"]) / max(temperature, 0.01)

    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return probabilities


def generate_text(seed, model, max_new_chars=200, temperature=0.8):
    chars = model["chars"].tolist()
    char_to_id = {char: i for i, char in enumerate(chars)}
    context_size = int(model["context_size"])

    unknown = sorted(set(ch for ch in seed if ch not in char_to_id))
    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")

    if not seed:
        seed = "hello"

    result = seed
    encoded = [char_to_id[ch] for ch in seed]
    pad_id = char_to_id.get(" ", encoded[0])
    context = ([pad_id] * context_size + encoded)[-context_size:]
    rng = np.random.default_rng()

    for _ in range(max_new_chars):
        probabilities = predict_next(context, model, temperature)
        next_id = int(rng.choice(len(chars), p=probabilities))
        result += chars[next_id]
        context = context[1:] + [next_id]

    return result


def main():
    model = load_model()

    print("=" * 44)
    print("       VØRAX AI v0.14")
    print("=" * 44)
    print("Two-hidden-layer language model loaded.")
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
                seed, model, max_new_chars=150, temperature=0.8
            )
            print("\nVØRAX:", output[len(seed):].strip())
        except ValueError as error:
            print(f"VØRAX: {error}")


if __name__ == "__main__":
    main()

