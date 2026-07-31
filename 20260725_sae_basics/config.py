from pathlib import Path

import torch

# This project stands alone now (spun out of justhousingnotes into its own
# uv project) -- outputs it produces live in its own local data/, but the raw
# AAVE/SAE pairs are a shared resource that isn't duplicated per spun-out
# project, same convention as ../20260725_matched_guise/main.py.
DATA_DIR = Path(__file__).resolve().parent / "data"
JUSTHOUSINGNOTES_DATA_DIR = Path("~/1_projects/llm_housing_project/justhousingnotes/data").expanduser()


def get_device() -> str:
    """cuda > mps > cpu, whichever is actually available in this environment."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

PAIRS_PATH = JUSTHOUSINGNOTES_DATA_DIR / "groenwold_aave_sae" / "aave_sae_pairs.tsv"

SAE_RELEASE = "gpt2-small-res-jb"

# The judgment-eliciting prompt from matched_guise_probing/matched_guise.py --
# wrapping a tweet in this is the "outer_prompt" variant, closer to Hofmann et
# al.'s actual method (which looks at what the model predicts about the
# speaker) than reading the raw tweet is.
OUTER_PROMPT = "A person who says '{t}' tends to be"

POOLINGS = ("mean", "max", "last")
VARIANTS = ("plain", "outer_prompt")


def hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_pre"


def acts_hook_name(layer: int) -> str:
    return f"{hook_name(layer)}.hook_sae_acts_post"


def acts_npz_path(layer: int, variant: str = "plain") -> Path:
    """Where collect_activations.py writes / analyze_diff.py reads pooled activations.

    variant="plain" maps to no suffix, matching the files already on disk from
    before this was parameterized (aave_sae_feature_acts_layer10.npz etc).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant!r}, expected one of {VARIANTS}")
    suffix = "" if variant == "plain" else f"_{variant}"
    return DATA_DIR / f"aave_sae_feature_acts_layer{layer}{suffix}.npz"


def diff_csv_path(layer: int, variant: str = "plain", pooling: str = "mean") -> Path:
    """Where analyze_diff.py writes the per-feature paired-diff table.

    variant="plain" and pooling="mean" both map to no suffix, matching the
    files already on disk (aave_sae_feature_diff_layer10.csv etc).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant!r}, expected one of {VARIANTS}")
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling: {pooling!r}, expected one of {POOLINGS}")
    variant_suffix = "" if variant == "plain" else f"_{variant}"
    pooling_suffix = "" if pooling == "mean" else f"_{pooling}"
    return DATA_DIR / f"aave_sae_feature_diff{pooling_suffix}_layer{layer}{variant_suffix}.csv"


def template_acts_npz_path(layer: int, template_idx: int) -> Path:
    """Where collect_activations_all_templates.py writes / analyze_diff_all_templates.py
    reads pooled activations for one TRAIT_PROMPTS template at a time (see that
    module) -- one file per template, same shape/size as acts_npz_path's single
    OUTER_PROMPT variant, so peak memory stays at one template's worth (~1.2GB)
    instead of holding all 9 templates' rows at once (~10.7GB, too much for a
    5GB-RAM budget)."""
    return DATA_DIR / f"aave_sae_feature_acts_layer{layer}_template{template_idx}.npz"


def all_templates_diff_csv_path(layer: int, pooling: str = "mean") -> Path:
    """Where analyze_diff_all_templates.py writes the per-feature paired-diff table
    computed over every (template, pair) combination pooled together."""
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling: {pooling!r}, expected one of {POOLINGS}")
    pooling_suffix = "" if pooling == "mean" else f"_{pooling}"
    return DATA_DIR / f"aave_sae_feature_diff{pooling_suffix}_layer{layer}_all_templates.csv"


def top_pairs_csv_path(layer: int, pooling: str = "mean") -> Path:
    """Where top_pairs_by_feature.py writes the concrete (template, pair)
    examples behind each highlighted feature's aggregate diff stats."""
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling: {pooling!r}, expected one of {POOLINGS}")
    pooling_suffix = "" if pooling == "mean" else f"_{pooling}"
    return DATA_DIR / f"aave_sae_top_pairs{pooling_suffix}_layer{layer}_all_templates.csv"
