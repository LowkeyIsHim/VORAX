import numpy as np


class VORAXTransformer:
    def __init__(
        self,
        vocab_size,
        context_size=64,
        embed_size=32,
        num_heads=4,
        seed=42,
    ):
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must divide evenly by num_heads")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads

        rng = np.random.default_rng(seed)

        def weight(shape, fan_in):
            return rng.normal(0, np.sqrt(2 / fan_in), shape)

        self.E = rng.normal(0, 0.1, (vocab_size, embed_size))
        self.P = rng.normal(0, 0.02, (context_size, embed_size))

        self.Wq = weight((embed_size, embed_size), embed_size)
        self.Wk = weight((embed_size, embed_size), embed_size)
        self.Wv = weight((embed_size, embed_size), embed_size)
        self.Wo = weight((embed_size, embed_size), embed_size)

        self.W1 = weight((embed_size, 4 * embed_size), embed_size)
        self.b1 = np.zeros(4 * embed_size)
        self.W2 = weight((4 * embed_size, embed_size), 4 * embed_size)
        self.b2 = np.zeros(embed_size)

        self.Wout = weight((embed_size, vocab_size), embed_size)
        self.bout = np.zeros(vocab_size)

    def forward(self, token_ids):
        token_ids = np.asarray(token_ids, dtype=np.int64)

        if token_ids.ndim != 1 or len(token_ids) == 0:
            raise ValueError("Input must be a non-empty 1D token sequence")

        if len(token_ids) > self.context_size:
            token_ids = token_ids[-self.context_size:]

        length = len(token_ids)

        x = self.E[token_ids] + self.P[:length]

        q = (x @ self.Wq).reshape(
            length, self.num_heads, self.head_size
        ).transpose(1, 0, 2)

        k = (x @ self.Wk).reshape(
            length, self.num_heads, self.head_size
        ).transpose(1, 0, 2)

        v = (x @ self.Wv).reshape(
            length, self.num_heads, self.head_size
        ).transpose(1, 0, 2)

        scores = q @ k.transpose(0, 2, 1)
        scores /= np.sqrt(self.head_size)

        # Causal mask: positions cannot attend to future characters.
        mask = np.triu(
            np.ones((length, length), dtype=bool), k=1
        )
        scores = np.where(mask[None, :, :], -1e9, scores)

        scores -= np.max(scores, axis=-1, keepdims=True)
        attention = np.exp(scores)
        attention /= attention.sum(axis=-1, keepdims=True)

        attended = attention @ v
        attended = attended.transpose(1, 0, 2).reshape(
            length, self.embed_size
        )

        x = x + attended @ self.Wo

        hidden = np.tanh(x @ self.W1 + self.b1)
        x = x + hidden @ self.W2 + self.b2

        logits = x @ self.Wout + self.bout

        return logits

    def predict_next(self, token_ids, temperature=1.0):
        logits = self.forward(token_ids)[-1]
        logits = logits / max(temperature, 0.01)
        logits -= np.max(logits)

        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        return probabilities
