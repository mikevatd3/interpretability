"""Memory diagnostic: load the model step by step and print memory after each.

Run capped so a runaway load fails fast instead of freezing the machine:
    systemd-run --user --scope -p MemoryMax=80G -p MemorySwapMax=0 uv run python load_test.py
"""
import subprocess

import torch
from transformer_lens.model_bridge import TransformerBridge


def mem(tag):
    row = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1]
    print(f"{tag:12s} {row}", flush=True)


mem("start")
model = TransformerBridge.boot_transformers("google/gemma-2-9b", device="cuda", dtype=torch.bfloat16)
mem("loaded")
print("dtype:", next(model.parameters()).dtype, flush=True)

model.enable_compatibility_mode()
mem("compat")
print("dtype:", next(model.parameters()).dtype, flush=True)
