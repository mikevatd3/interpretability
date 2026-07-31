# sae_basics

SAE-based interpretability pass over the AAVE/SAE (African-American Vernacular
English / Standard American English) intent-equivalent tweet pairs, using the
`gpt2-small-res-jb` SAE release and Neuronpedia feature labels.

## File inventory

- **`config.py`** — Shared paths (`DATA_DIR`, `JUSTHOUSINGNOTES_DATA_DIR`), device
  selection (`get_device`), and path-builder helpers for the various npz/csv
  outputs this project produces.

- **`data_prep.py`** — Loads the 2019 AAVE/SAE tweet pairs (`load_pairs`) and
  applies the judgment-prompt wrapper for the `outer_prompt` variant
  (`prompt_text`, `prompted_pairs`).

- **`collect_activations.py`** — Loads the model + SAE (`load_model_and_sae`)
  and runs a forward pass over the AAVE/SAE pairs to collect pooled SAE
  feature activations, saved to an npz (`collect`).

- **`collect_activations_all_templates.py`** — Same collection idea as
  `collect_activations.py`, but repeated across the 9 dialect-prejudice
  `TRAIT_PROMPTS` templates instead of a single plain/outer_prompt variant.

- **`analyze_diff.py`** — Builds the paired-diff table (t-stats, effect sizes)
  between AAVE and SAE pooled activations per feature, with optional
  Neuronpedia labels for the top features (`build_diff_table`,
  `neuronpedia_label`).

- **`analyze_diff_all_templates.py`** — Same paired-diff analysis, combined
  across all templates collected by `collect_activations_all_templates.py`

  (`combine_grouped_stats`).
- **`top_pairs_by_feature.py`** — For a set of top features, surfaces the
  specific (template, pair) examples that activate them most
  (`top_pairs_for_features`).

- **`top_activating_examples.py`** — Standalone scan of Wikitext for the
  lines that most activate a single chosen SAE feature (`FEATURE`), for
  sanity-checking what a feature actually represents.

- **`plots.py`** — Plotly charts over a diff table: t-stat distribution,
  and other AAVE-vs-SAE comparisons (`show_plots`).

- **`main.py`** — Orchestrates the plain end-to-end pipeline: collect (if
  needed) → analyze (if needed) → plot, across the `STEPS` list of
  (variant, pooling) combinations.

