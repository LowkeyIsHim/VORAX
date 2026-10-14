# ==========================================
# VØRAX — Character Tokenizer
# Version: 0.1.0
# ==========================================

class Tokenizer:
    def __init__(self, text):
        # Build a vocabulary of unique characters.
        self.chars = sorted(set(text))

        # Assign each character a unique integer.
        self.char_to_id = {
            char: index
            for index, char in enumerate(self.chars)
        }

        # Create the reverse mapping.
        self.id_to_char = {
            index: char
            for char, index in self.char_to_id.items()
        }

    def encode(self, text):
        """Convert characters into integer IDs."""
        return [self.char_to_id[char] for char in text]

    def decode(self, ids):
        """Convert integer IDs back into text."""
        return "".join(self.id_to_char[index] for index in ids)


if __name__ == "__main__":
    sample_text = "hello vorax"

    tokenizer = Tokenizer(sample_text)

    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)

    print("VØRAX TOKENIZER TEST")
    print("-" * 30)
    print("Vocabulary size:", len(tokenizer.chars))
    print("Original text:  hello")
    print("Encoded IDs:", encoded)
    print("Decoded text:", decoded)
