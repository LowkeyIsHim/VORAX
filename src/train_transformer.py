
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenizer import Tokenizer
from transformer_torch import VORAXTransformer


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Training settings
CONTEXT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 5
PATIENCE = 8

DATA_FILE = BASE_DIR / "data" / "training_expanded.txt"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MODEL_FILE = CHECKPOINT_DIR / "vorax_transformer_v1.pt"

device = torch.device("cpu")


def make_examples(text, tokenizer, context_size):
    ids = tokenizer.encode(text)
    examples = []

    for i in range(context_size, len(ids)):
        x = ids[i - context_size:i]
        y = ids[i]
        examples.append((x, y))

    return examples


def batch_examples(examples, batch_size):
    chosen = random.sample(
        examples,
        min(batch_size, len(examples)),
    )

    x = torch.tensor(
        [item[0] for item in chosen],
        dtype=torch.long,
        device=device,
    )
    y = torch.tensor(
        [item[1] for item in chosen],
        dtype=torch.long,
        device=device,
    )

    return x, y


def evaluate(model, examples, loss_function):
    if not examples:
        return float("inf")

    model.eval()
    losses = []

    with torch.no_grad():
        for start in range(0, len(examples), 64):
            batch = examples[start:start + 64]

            x = torch.tensor(
                [item[0] for item in batch],
                dtype=torch.long,
                device=device,
            )
            y = torch.tensor(
                [item[1] for item in batch],
                dtype=torch.long,
                device=device,
            )

            logits = model(x)[:, -1, :]
            loss = loss_function(logits, y)
            losses.append(loss.item() * len(batch))

    model.train()
    return sum(losses) / len(examples)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    text = DATA_FILE.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n")

    # Chronological split: later text is held out.
    n = len(text)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]

    # Fit vocabulary on training text only.
    tokenizer = Tokenizer(train_text)

    train_examples = make_examples(
        train_text, tokenizer, CONTEXT_SIZE
    )
    val_examples = make_examples(
        val_text, tokenizer, CONTEXT_SIZE
    )
    test_examples = make_examples(
        test_text, tokenizer, CONTEXT_SIZE
    )

    if not train_examples or not val_examples or not test_examples:
        raise ValueError("Not enough data to create all splits.")

    print("VØRAX Transformer v1")
    print("-" * 40)
    print(f"Device: {device}")
    print(f"Dataset: {DATA_FILE.name}")
    print(f"Characters: {n}")
    print(f"Train/validation/test: "
          f"{len(train_text)}/{len(val_text)}/{len(test_text)}")
    print(f"Vocabulary: {len(tokenizer.chars)}")
    print(f"Examples: {len(train_examples)}/"
          f"{len(val_examples)}/{len(test_examples)}")
    print(f"Parameters: "
          f"{sum(p.numel() for p in VORAXTransformer(len(tokenizer.chars)).parameters()):,}")
    print("-" * 40)

    model = VORAXTransformer(
        vocab_size=len(tokenizer.chars),
        context_size=CONTEXT_SIZE,
        embed_size=32,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_time = time.time()

    steps_per_epoch = max(
        1, min(200, len(train_examples) // BATCH_SIZE)
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            x, y = batch_examples(
                train_examples, BATCH_SIZE
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)[:, -1, :]
            loss = loss_function(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / steps_per_epoch

        if epoch % EVAL_INTERVAL == 0 or epoch == 1:
            val_loss = evaluate(
                model, val_examples, loss_function
            )

            print(
                f"Epoch {epoch:3d} | "
                f"train {train_loss:.4f} | "
                f"val {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                stale_epochs = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "vocab_chars": tokenizer.chars,
                        "context_size": CONTEXT_SIZE,
                        "embed_size": 32,
                        "num_heads": 4,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                    },
                    MODEL_FILE,
                )
                print("  Best checkpoint saved.")
            else:
                stale_epochs += 1

            if stale_epochs >= PATIENCE:
                print("Early stopping.")
                break

    # Evaluate the best saved model, not the final epoch.
    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(
        model, test_examples, loss_function
    )
    test_perplexity = float(np.exp(min(test_loss, 20)))

    print("-" * 40)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {test_perplexity:.2f}")
    print(f"Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Checkpoint: {MODEL_FILE}")


if __name__ == "__main__":
    main()

