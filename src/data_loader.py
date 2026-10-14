# ==========================================
# VØRAX — Data Loader
# Version: 0.1.0
# ==========================================

from pathlib import Path

from tokenizer import Tokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "training.txt"


def load_training_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_FILE}"
        )

    return DATA_FILE.read_text(encoding="utf-8")


def main():
    text = load_training_data()

    tokenizer = Tokenizer(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("=" * 40)
    print("       VØRAX DATA LOADER")
    print("=" * 40)
    print(f"Characters in dataset: {len(text)}")
    print(f"Vocabulary size: {len(tokenizer.chars)}")
    print(f"Encoded IDs: {len(encoded)}")
    print(f"Round-trip successful: {decoded == text}")
    print("=" * 40)


if __name__ == "__main__":
    main()
