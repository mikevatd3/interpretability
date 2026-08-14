# gemma-2-9b-it smoke test

A minimal check that `gemma-2-9b-it` loads and generates through
TransformerLens before building a real experiment around it: one
hand-rolled chat-format prompt, greedy-decoded until `<end_of_turn>`.

gemma-2-9b-it is gated on HuggingFace and needs an `HF_TOKEN` with access
granted (see repo-root `.env`) plus much more disk/VRAM than a local
machine may have -- meant to run on a rented GPU pod (see `../RUNPOD.md`),
not locally.

```
uv sync
uv run main.py
```
