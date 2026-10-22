
# VORAX

A custom AI project built from scratch using Python, NumPy, and PyTorch.

## Project Status

VORAX is an experimental character-level language model project.

- Earlier versions use custom NumPy neural networks.
- Version 0.14 uses a two-hidden-layer feedforward architecture.
- The Transformer prototype uses PyTorch and causal self-attention.
- Transformer training is not yet complete.

## Project Structure

```text
VORAX/
├── checkpoints/       # Saved model weights
├── data/              # Training datasets
├── src/               # Model, training, and generation code
├── requirements.txt   # Python dependencies
└── README.md
```

## Requirements

- Python 3.13 or a compatible Python version
- NumPy
- PyTorch

## Setup

Clone the repository:

```bash
git clone https://github.com/LowkeyIsHim/VORAX.git
cd VORAX
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Existing NumPy Model

The existing generation program is:

```bash
python src/generate.py
```

It expects the checkpoint configured in that script to exist.

## Transformer Development

Test the Transformer model:

```bash
python -c "import sys, torch; sys.path.insert(0, 'src'); from transformer_torch import VORAXTransformer; model = VORAXTransformer(28); print(model)"
```

Train the Transformer when sufficient computing resources are available:

```bash
python src/train_transformer.py
```

The Transformer training script uses the expanded training dataset and saves its checkpoint to:

```text
checkpoints/vorax_transformer_v1.pt
```

## Notes

- Existing checkpoints should be preserved.
- Transformer training has not yet been completed.
- Training and generation scripts may require different checkpoints.
- Model quality depends on dataset size, training, and architecture.

