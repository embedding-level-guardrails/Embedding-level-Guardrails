# Commands for Running Training


## 1. Create the environment with uv

Install [uv](https://docs.astral.sh/uv/) and make sure `make` is available.
The project requires Python 3.13 or later. From the repository root, run:

```bash
cd rq3_training/ettin_guardrails
uv venv --python 3.13
uv sync --locked
```

## 2. Log in to Weights & Biases

```bash
uv run wandb login
```


## 3. Log in to Hugging Face

```bash
uv run hf auth login
```

The default configuration downloads the
`jhu-clsp/ettin-encoder-68m` backbone and the `allenai/wildguardmix` training data
from Hugging Face.

## 4. Start training through a Makefile target

Linear probe:

```bash
make linear-probe batch_size=64 num_per_category=32
```

Classifier:

```bash
make classifier batch_size=32 num_per_category=16
```

Embedder:

```bash
make embedder batch_size=32 num_per_category=4
```
