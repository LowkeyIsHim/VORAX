
import torch
import torch.nn as nn


class VORAXTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=64,
        embed_size=32,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            vocab_size, embed_size
        )
        self.position_embedding = nn.Embedding(
            context_size, embed_size
        )

        self.attention = nn.MultiheadAttention(
            embed_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

        self.output = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        batch_size, length = token_ids.shape

        if length > self.context_size:
            raise ValueError("Input exceeds context size")

        positions = torch.arange(
            length, device=token_ids.device
        )

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        # Prevent each position from seeing future characters.
        causal_mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )

        attended, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask,
            need_weights=False,
        )

        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.feed_forward(x))

        return self.output(x)

