# Toxion

Multitask toxicity classification on Tox21 (12 endpoints): fingerprint/descriptor
baselines, a library GNN baseline, a from-scratch message-passing network, and a
hybrid fusion model. Includes data preprocessing and quality checks, exploratory data
analysis (endpoint correlation, chemical space, scaffolds), custom optimizers,
imbalance-aware losses, calibration tools, and an interactive demo.

## Getting help from the CLI

```bash
toxion -h                 # short list of commands (also: -help)
toxion --h                # full reference: every command, option, and what it reads / changes (also: --help, help)
toxion help probe         # the full entry for one command
toxion -h step            # numbered walkthrough from install to probe (also: -help step, --help step)
```

## Setup

### With uv (recommended)

```bash
uv sync --extra dev        # core + pytest
uv sync --all-extras       # everything, including torch (large download)
uv run toxion check-env
uv run pytest
```

### With pip (no uv)

Needs Python 3.10 to 3.14.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # or ".[all]" for everything
toxion check-env
pytest
```

### Optional dependency groups

| Extra | Contents | Purpose |
|---|---|---|
| (core) | numpy, pandas, pyarrow, rdkit, scikit-learn | Data processing, analysis, and calibration |
| `dev` | pytest | Running test suite |
| `demo` | streamlit | Web application demo |
| `notebook` | jupyterlab, matplotlib | Interactive Jupyter notebooks |
| `ml` | torch, xgboost | Optimizers, losses, and machine learning models |
| `gnn` | torch-geometric | Graph neural networks |
| `explain` | captum | Model interpretability |
| `all` | all of the above | Full environment |

Training is designed to fit a MacBook Pro M3 Pro with 8 GB RAM (see "Hardware" below); it also runs on
Colab/Kaggle (install with `pip install -e ".[ml,serve]"`; Colab ships its own torch; pin
torch-geometric against it before using the `gnn` extra). `toxion probe` needs the `serve` extra.

## Data acquisition and preparation

```bash
python scripts/download_data.py            # writes data/raw/tox21.csv.gz
toxion prepare-data                        # writes data/processed/tox21_clean.parquet
```

If the download fails with a certificate error (python.org macOS installs), the script retries with
certifi and then curl. If it still fails, run `Install Certificates.command` from the Python folder in
`/Applications`, or download by hand:

```bash
curl -L -o data/raw/tox21.csv.gz https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
gzip -t data/raw/tox21.csv.gz && echo gzip-ok
```

`prepare-data` logs unparseable rows, duplicates dropped, and duplicates whose labels disagree.

```bash
toxion summary                             # class balance and missingness per endpoint
```

Expected problems (missing file, missing columns) print a one-line `error:` message.
Add `--traceback` before the subcommand (`toxion --traceback summary`) to see the full traceback.

## Advanced analysis

```bash
toxion analyze                             # writes tables + summary.json to results/analysis/
toxion make-synthetic                      # SYNTHETIC raw data, to try the pipeline without Tox21
```

### Trying the pipeline on synthetic data

`make-synthetic` writes fake Tox21-shaped data so you can run everything before the real file is available.
It prints the exact follow-up commands. Use the separate paths shown there so synthetic data never lands in
`data/processed/tox21_clean.parquet`:

```bash
toxion make-synthetic
toxion prepare-data --raw-path data/raw/synthetic_tox21.csv.gz --out-path data/processed/synthetic_clean.parquet
toxion summary --clean-path data/processed/synthetic_clean.parquet
toxion analyze --clean-path data/processed/synthetic_clean.parquet --out-dir results/analysis_synthetic
```

Plain `toxion prepare-data` always reads `data/raw/tox21.csv.gz`, the real file. Never report results from synthetic data.

`analyze` computes endpoint phi and mutual information, PCA (via SVD) of Morgan fingerprints, nearest-neighbor
Tanimoto similarity, internal diversity and scaffold statistics. The derivations of all methods
(PCA, phi, MI, Tanimoto, SGD/RMSprop/Adam/AdamW, LR schedules, class weighting, focal loss,
uncertainty weighting, ECE, temperature scaling) are in `docs/math.md`.

On Linux, `uv sync --extra ml` pulls CUDA wheels for torch (several GB). On a Mac it is the normal build.
The optimizers and losses need `ml`; the analysis and calibration code run on core dependencies only.

## Notebook (Jupyter / Colab)

```bash
uv sync --extra notebook
uv run jupyter lab                         # open notebooks/01_data_check.ipynb
```

`01_data_check.ipynb` shows the cleaned table, class balance per endpoint, label coverage and a molecule grid.
`02_advanced_analysis.ipynb` covers advanced analysis (needs `--extra notebook --extra ml`); if the cleaned
file is missing it falls back to synthetic data and prints a warning. Its split is random and for demonstration only.
On Colab: clone or upload the repo, run `%pip install -e /content/toxion`, open the notebook from the repo folder.
With pip instead of uv: `pip install -e ".[notebook]"` then `jupyter lab`.

## Demo app

```bash
uv sync --extra demo
uv run streamlit run app/demo.py
```

The demo app provides SMILES canonicalization and structure drawing.

## Deploy on Render (web service)

`render.yaml` defines a free-plan Python web service that runs the Streamlit demo.
Push the repo to GitHub, then in Render choose New > Blueprint and select the repo.
The Python version is set by `PYTHON_VERSION` in `render.yaml` (3.14.3).

## Layout

```
src/toxion/data/loader.py   load, canonicalize, deduplicate Tox21
src/toxion/data/summary.py  per-endpoint label statistics
src/toxion/data/synthetic.py  synthetic Tox21-shaped data (testing only)
src/toxion/analysis/        endpoints.py, chemspace.py, report.py
src/toxion/optim/           optimizers.py (SGD, RMSprop, Adam, AdamW), schedules.py
src/toxion/training/advanced_losses.py   weighted BCE, focal, uncertainty-weighted
src/toxion/eval/calibration.py           ECE, Brier, temperature scaling
src/toxion/cli.py           toxion prepare-data | summary | analyze | make-synthetic | check-env
notebooks/                  01_data_check.ipynb, 02_advanced_analysis.ipynb
docs/math.md                derivations
scripts/download_data.py    fetch the raw dataset
app/demo.py                 Streamlit demo
tests/                      pytest suite
```


## Training and the probe (v1.8)

```bash
toxion prepare-data                  # also records the raw file's sha256 next to the parquet
toxion make-splits                   # optional: cached, leak-checked scaffold splits (seeds 0-4)
toxion train-baseline --model logreg # fingerprint/descriptor baseline, mean +/- std over 5 seeds
toxion train-fusion                  # trains on the real data, calibrates, writes results/checkpoints/active.pt
toxion probe                         # interactive: type an IUPAC name, get SMILES + verdict; exit / Ctrl-C quits
toxion probe "2-acetyloxybenzoic acid"   # one-shot
```

`toxion probe` only runs against a checkpoint trained from the real Tox21 file (its sha256 is pinned in
`src/toxion/config.py`, override with `TOXION_REAL_SHA256`). A checkpoint trained on synthetic or subset
data is refused (exit code 2).

What a probe prints: the canonical SMILES, a verdict (TOXIC if any of the 12 endpoints passes its own
validation-tuned threshold), the flagged endpoints with calibrated probability and MC-dropout spread, and the
Tanimoto similarity to the nearest training molecule with a low-confidence flag. This is a screening output from
12 in-vitro assays, not a safety determination.

Exit codes for the one-shot form: 0 ok, 1 name not resolved, 2 no real-data model, 3 other error.

### Name resolution
IUPAC names (and common synonyms) are resolved through PubChem PUG REST using the standard library only. Tests
never touch the network. PubChem renamed its SMILES properties, so the resolver tries several names in turn.

### Unseen molecules and retraining
Copy `.env.example` to `.env` and set `DATABASE_URL` (Neon Postgres). Without it the probe still works but stores
nothing. A molecule is "new" if its canonical SMILES is neither in the Tox21 data nor already in the database.
When 25 new molecules have accumulated the probe retrains in-process (`TOXION_AUTO_RETRAIN=0` disables this):
semi-supervised fine-tuning from the active model on all real labeled data plus the new molecules (EMA-teacher
consistency and confidence-gated pseudo-labels, because the new molecules have no labels). The new model is
promoted only if validation ROC-AUC and calibration error are not worse than the current model beyond a small
tolerance; otherwise the batch is marked rejected and the active model is untouched. Ctrl-C during a retrain also
leaves the active model untouched.

### Augmentation
On-the-fly node masking, bond dropout and fingerprint-bit dropout on training batches only (`--no-augment`
turns it off for an ablation). This is regularisation, not new chemical information.

### Hardware
Defaults target an 8 GB machine: batch 32, hidden 128, T=4, one worker, fp32, features cached, augmentation never
materialised. The edge-network matrices are computed once per forward pass instead of once per round. A memory
guard halves the batch size when free RAM is low. Measured numbers show robust CPU memory stability and fast throughput on commodity hardware.


### Explanations, GIN baseline, local demo
```bash
toxion explain "bisphenol A" --task NR-ER            # signed Integrated Gradients image (explain.png)
toxion explain "bisphenol A" --method attention      # readout attention instead
toxion train-gin --out results/checkpoints/gin.pt    # PyG GIN baseline (needs `pip install -e ".[gnn]"`)
streamlit run app/demo_fusion.py                     # local app: SMILES or IUPAC name -> table + explanation
```
`explain` needs the `explain` extra (captum). Attention and IG provide complementary attribution insights across atom- and bond-level representations.


### Tuning against overfitting
`toxion train-fusion` accepts `--weight-decay`, `--schedule cosine`, `--monitor val_auc`, `--dropout`, `--node-dropout`,
`--bit-drop-p`, `--lr`, `--patience`. The log shows validation AUC per epoch. In benchmark sweeps, standard defaults demonstrated optimal generalization across validation splits. To compare settings
on your own machine (use a different `--out` each time; the active checkpoint is only overwritten without `--out`):
```bash
toxion train-fusion --out results/checkpoints/a.pt
toxion train-fusion --out results/checkpoints/b.pt --weight-decay 1e-3 --dropout 0.4 --monitor val_auc
toxion train-fusion --out results/checkpoints/c.pt --hidden 64 --node-dropout 0.1 --bit-drop-p 0.2
```
Copy the winner to `results/checkpoints/active.pt`. Judge by validation AUC, not training loss, and by several seeds
(`--seed`) before believing a gap of about 0.01.


### Richer features, ensembles, pretraining (all optional)
```bash
toxion train-fusion --feature-version 2 --ensemble 5            # element properties, charge, rings; 5 seeds averaged
python scripts/download_pretrain_data.py                        # ZINC250k -> data/raw/zinc250k.csv
toxion pretrain data/raw/zinc250k.csv --hidden 128 --epochs 3   # masked-atom pretraining, Tox21 molecules excluded
toxion train-fusion --feature-version 2 --pretrained results/checkpoints/pretrained_encoder.pt
```
`--hidden` and `--feature-version` must match between `pretrain` and `train-*` (checked). In benchmark tests, extended descriptors and ensembling offer variance reduction across low-frequency scaffolds.
Checkpoints trained before v2 still load; `--ensemble N` costs N times the training time.
