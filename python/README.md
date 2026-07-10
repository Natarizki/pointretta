![github repo](https://img.shields.io/badge/github-repo-orange?style=plastic&logo=github&link=https%3A%2F%2FGitHub.com%2FNatarizki%2Fpointretta)



# PointRetta

A retention-based language model, built from scratch in C — trainable and runnable directly from Python.

PointRetta replaces self-attention with a **retention mechanism**: **O(N) training** (fully parallelizable) and **O(1) inference per token** (fixed-size state, doesn't grow with context length — unlike attention's ever-expanding KV-cache).

## Install

```bash
pip install pointretta
```
**No external Python dependencies — the core engine is self-contained C, compiled into the extension at install time.**

## Quick Start
```Python
import pointretta

# Train a small model on your own text
loss = pointretta.train_and_save(
    text=open("corpus.txt").read(),
    output_path="model.pr",
    dim=64, heads=4, head_dim=16, layers=4,
    iters=500,
)
print(f"Final loss: {loss:.4f}")

# Load it back and generate
text = pointretta.load_and_generate("model.pr", prompt="once upon a time", max_tokens=50)
print(text)
```
API
train_and_save(text, output_path, **kwargs) -> float
Trains a byte-level BPE tokenizer on text, builds a retention-based model, trains it, and saves everything (architecture + tokenizer + weights) into a single .pr file. Returns the final training loss.

| Parameter              |         Default      |           Description                                   |
|:----------------------:|:--------------------:|:-------------------------------------------------------:|
| dim                    |         32           |           Model dimension (must equal heads * head_dim) |
| heads                  |         04           |           Number of retention heads                     |
| head_dim               |         08           |           Dimension per head                            |
| ffn_hidden             |         64           |           SwiGLU hidden dimension                       |
| layers                 |         02           |           Number of layers                              |
| vocab_size             |         280          |           Target BPE vocabulary size                    |
| iters                  |         300          |           Training iterations                           |
| lr                     |         0.05         |           Learning rate (Adafactor)                     |
| window                 |         32           |           Training window size (tokens)                 |
| decay_min / decay_max  |         0.7 / 0.95   |           Multi-scale retention decay range             |

** load_and_generate(pr_path, prompt, max_tokens=20) -> str **
**Loads a .pr file and greedily generates a continuation of prompt.**

## What Makes This Different

- No attention, no softmax in the core loop —
**retention uses a linear decay mechanism instead, which is what makes the O(1) recurrent inference form possible.**
- Every core operation is validated —
**forward/backward equivalence, numerical gradient checking, and forward-parallel vs. forward-recurrent equivalence were all checked before this was trusted to work.**
- Multi-scale decay — 
**each attention head "forgets" at a different rate, so the model retains both short- and long-range information without attention's O(N) memory cost.**

## Native CLI & ASM Kernels
This PyPI package builds with portable C for maximum compatibility. For SIMD-accelerated (NEON/AVX2) native builds and the full CLI (build / train / delete), see the [Github Repo](https://github.com/Natarizki/pointretta).

## License
![Apache](https://img.shields.io/badge/License-apache2.0-orange?style=plastic&logo=apache&logoSize=10&label=LICENSE&link=https%3A%2F%2Fwww.apache.org%2Flicenses%2FLICENSE-2.0.txt)

