from analyze_diff import build_diff_table
from collect_activations import collect
from config import acts_npz_path, diff_csv_path
from plots import show_plots

LAYER = 10

STEPS = [
    ("plain", "mean"),
    ("outer_prompt", "last"),
]
PLOT_STEP = ("plain", "mean")  # which step's charts to actually pop open at the end

N_LABELED = 200  # how many top rows per step get a neuronpedia label

# Collection is the ~10-minute step (a full model+SAE pass over 2019 pairs);
# analysis is a lot cheaper but still makes up to N_LABELED live Neuronpedia
# requests. Both stages are skipped if their output already exists, so
# re-running this script is safe and cheap once everything's been collected
# at least once. Flip these to True to force a redo.
FORCE_RECOLLECT = False
FORCE_REANALYZE = False


def ensure_collected(layer: int, variant: str) -> None:
    path = acts_npz_path(layer, variant)
    if path.exists() and not FORCE_RECOLLECT:
        print(f"[collect] {path.name} already exists, skipping (FORCE_RECOLLECT=True to redo)")
        return
    print(f"[collect] running model + SAE over '{variant}' texts, layer {layer} (~10 min)...")
    collect(layer, variant)


def ensure_analyzed(layer: int, variant: str, pooling: str) -> None:
    path = diff_csv_path(layer, variant, pooling)
    if path.exists() and not FORCE_REANALYZE:
        print(f"[analyze] {path.name} already exists, skipping (FORCE_REANALYZE=True to redo)")
        return
    print(f"[analyze] building paired-diff table for {variant}/{pooling}, layer {layer}...")
    diff_df = build_diff_table(layer, variant, pooling, N_LABELED)
    diff_df.to_csv(path, index=False)
    print(f"[analyze] wrote {len(diff_df)} rows to {path.name}")


def main():
    variants_needed = {variant for variant, _pooling in STEPS}
    for variant in variants_needed:
        ensure_collected(LAYER, variant)

    for variant, pooling in STEPS:
        ensure_analyzed(LAYER, variant, pooling)

    plot_variant, plot_pooling = PLOT_STEP
    print(f"[plot] showing charts for {plot_variant}/{plot_pooling}...")
    show_plots(LAYER, plot_variant, plot_pooling)


if __name__ == "__main__":
    main()
