# Toxion

Multitask toxicity classification on Tox21 (12 endpoints): fingerprint/descriptor
baselines, a library GNN baseline, a from-scratch message-passing network, and a
hybrid fusion model. Includes data preprocessing and quality checks, exploratory data
analysis (endpoint correlation, chemical space, scaffolds), custom optimizers,
imbalance-aware losses, calibration tools, and an interactive demo.

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

Train on Colab/Kaggle, not on a laptop. On Colab, install with `pip install -e ".[ml]"`
(Colab ships its own torch; pin torch-geometric against it before using the `gnn` extra).

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
