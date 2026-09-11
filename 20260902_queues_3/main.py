import torch
from dotenv import load_dotenv
from sae_lens import SAE, HookedSAETransformer

from config import DEVICE

load_dotenv()

MODEL_NAME = "gemma-2-9b"

# Gemma Scope's canonical, pre-trained (not instruction-tuned) residual-stream
# SAEs. Layer/width are placeholders -- pick per what the analysis needs;
# valid layers are 0-41 and widths are 16k/65k/131k/1m, per the release's
# entry in sae_lens's pretrained_saes.yaml.
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
LAYER = 20
SAE_ID = f"layer_{LAYER}/width_16k/canonical"


def load_model_and_sae() -> tuple[HookedSAETransformer, SAE]:
    """Load gemma-2-9b and the matching Gemma Scope SAE for one of its
    residual-stream layers.

    Unlike gpt2-small-res-jb (see ../20260725_sae_basics/core.py), Gemma
    Scope's canonical SAEs were trained against an unmodified gemma-2-9b
    forward pass -- no from_pretrained_no_processing or extra
    model_from_pretrained_kwargs needed, a plain from_pretrained lines up.
    """
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=DEVICE)
    if isinstance(sae, tuple):
        sae = sae[0]

    model = HookedSAETransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    # get_device() trusts torch's own cuda/mps checks, but on some
    # platforms (e.g. aarch64 + CUDA boxes like GB10/DGX-Spark-class
    # machines) a mismatched torch wheel makes torch.cuda.is_available()
    # silently report False -- so confirm against where the params
    # actually landed rather than trusting DEVICE alone.
    actual_device = next(model.parameters()).device
    print(f"requested device={DEVICE!r}, model parameters are on {actual_device}")
    if DEVICE == "cuda":
        print(f"torch cuda device: {torch.cuda.get_device_name(0)}")

    return model, sae


def main():
    model, sae = load_model_and_sae()
    print(f"Loaded {MODEL_NAME} with SAE {SAE_RELEASE}/{SAE_ID}")


if __name__ == "__main__":
    main()
