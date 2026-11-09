
from __future__ import annotations

import argparse
import textwrap

WIDTH = 100
SHORT_FLAGS = {"-h", "-help"}
FULL_FLAGS = {"--h", "--help", "help"}
STEP_WORDS = {"step", "steps"}

COMMANDS: dict[str, dict] = {
    "check-env": {
        "summary": "report which Python packages (per optional extra) are installed",
        "does": "Prints the version of each dependency group (core, ml, serve, gnn, explain, demo, ...) or says it is "
                "missing, without importing heavy packages. Use it first when something fails to import.",
        "usage": ["toxion check-env"],
        "reads": ["installed package metadata"],
        "changes": ["nothing"],
        "next": "install any missing extra, e.g. `uv sync --extra ml --extra serve --extra explain --extra gnn`",
    },
    "prepare-data": {
        "summary": "clean the raw Tox21 file into a parquet table and record its hash",
        "does": "Reads the raw Tox21 CSV, converts every SMILES to RDKit's canonical form, drops rows RDKit cannot "
                "parse, removes duplicate molecules (first one kept; label conflicts are logged), keeps missing "
                "labels as NaN (never 0, because 0 would mean 'tested inactive'), and writes the 12 endpoint columns. "
                "It also records the raw file's sha256 next to the output: that hash is how the probe knows a model "
                "was trained on the real data.",
        "usage": ["toxion prepare-data", "toxion prepare-data --raw-path data/raw/tox21.csv.gz "
                  "--out-path data/processed/tox21_clean.parquet"],
        "reads": ["the raw CSV (default data/raw/tox21.csv.gz)"],
        "changes": ["writes/overwrites the clean parquet (default data/processed/tox21_clean.parquet)",
                    "writes/overwrites <parquet>.manifest.json (raw file hash and row count)",
                    "does not touch checkpoints or the database"],
        "next": "toxion summary, then toxion train-baseline / toxion train-fusion",
        "warn": "Re-running it on a different raw file changes the recorded hash; models trained earlier on the old "
                "data will no longer match and `retrain` will refuse them.",
    },
    "summary": {
        "summary": "print class balance and missing-label counts per endpoint",
        "does": "Shows, for each of the 12 Tox21 endpoints, how many molecules were tested, how many are positive, and "
                "the positive rate. Read-only; a quick check that the data loaded as expected.",
        "usage": ["toxion summary"],
        "reads": ["the clean parquet"],
        "changes": ["nothing"],
        "next": "toxion analyze (optional) or toxion train-baseline",
    },
    "analyze": {
        "summary": "exploratory analysis: endpoint dependence, PCA, chemical space, scaffolds",
        "does": "Computes endpoint co-occurrence, a PCA of Morgan fingerprints, nearest-neighbour Tanimoto "
                "similarity, internal diversity and scaffold statistics, and writes them as tables. Slow on the full "
                "set (about 15 minutes on a laptop); use --max-molecules to sample.",
        "usage": ["toxion analyze", "toxion analyze --max-molecules 2000"],
        "reads": ["the clean parquet"],
        "changes": ["writes tables into the output directory (default results/analysis)"],
        "next": "informational only",
    },
    "make-synthetic": {
        "summary": "write a fake Tox21-shaped raw file for testing the pipeline",
        "does": "Generates random molecules with random labels in the same format as the real file, so the code can be "
                "exercised without the real data. Models trained from it can never pass the probe's real-data gate.",
        "usage": ["toxion make-synthetic --n 400 --out data/raw/synthetic_tox21.csv.gz"],
        "reads": ["nothing"],
        "changes": ["writes the synthetic raw file (default data/raw/synthetic_tox21.csv.gz)"],
        "next": "toxion prepare-data --raw-path <that file> --out-path data/processed/synthetic.parquet "
                "(use a different --out-path so the real parquet is not overwritten)",
        "warn": "Never point prepare-data at it with the default --out-path unless you want to overwrite the real "
                "clean data.",
    },
    "make-splits": {
        "summary": "write leak-checked scaffold splits for several seeds",
        "does": "Groups molecules by Bemis-Murcko scaffold, puts very large groups in train first, fills train / "
                "validation / test (80/10/10) with whole groups, and asserts that no scaffold appears in two sets. "
                "Writes one .npz of row indices per seed.",
        "usage": ["toxion make-splits", "toxion make-splits --seeds 0,1,2"],
        "reads": ["the clean parquet"],
        "changes": ["writes split_seed<N>.npz files into the output directory (default data/processed/splits)"],
        "next": "informational: the training commands recompute the identical split from --seed, so they do not "
                "read these files; use them to inspect or reuse a split",
    },
    "train-baseline": {
        "summary": "fit fingerprint/descriptor baselines (logreg, random forest, xgboost) over 5 splits",
        "does": "Fits one classifier per endpoint on Morgan counts + descriptors + structural alerts, using only "
                "molecules that have a label for that endpoint, on each scaffold-split seed, and reports mean ROC-AUC "
                "with standard deviation across seeds. This is the number the neural models must be compared against.",
        "usage": ["toxion train-baseline --model logreg", "toxion train-baseline --model rf --seeds 0,1,2"],
        "reads": ["the clean parquet"],
        "changes": ["writes results/tables/baseline_<model>.csv",
                    "creates a feature cache next to the parquet (<parquet>.features.pkl, safe to delete)",
                    "does not touch checkpoints"],
        "next": "toxion train-fusion",
    },
    "train-fusion": {
        "summary": "train the fusion model (graph network + fingerprint) on the real data  [the main model]",
        "does": "Builds molecular graphs and fingerprints, splits by scaffold, trains the model with a NaN-masked "
                "loss and on-the-fly augmentation, keeps the best-validation weights, fits per-endpoint temperature "
                "scaling and decision thresholds on the validation set, evaluates on the held-out test set, and saves "
                "everything (weights, scaler, thresholds, training-set fingerprints, data hash) in one checkpoint. "
                "With --ensemble N it trains N seeds and averages them. Uses the Apple GPU (MPS) if it passes a small "
                "self-test, else the CPU; halves the batch size when free RAM is low.",
        "usage": ["toxion train-fusion", "toxion train-fusion --feature-version 2 --ensemble 5",
                  "toxion train-fusion --out results/checkpoints/trial.pt --weight-decay 1e-3 --monitor val_auc",
                  "toxion train-fusion --resume"],
        "reads": ["the clean parquet and its .manifest.json (both required)",
                  "--pretrained encoder file, if given"],
        "changes": ["WRITES results/checkpoints/active.pt, the model `toxion probe` uses, replacing the previous one "
                    "unless you pass --out",
                    "creates a feature cache next to the parquet (safe to delete)",
                    "writes temporary .state / .member files next to the checkpoint while running, removed on success "
                    "(kept if interrupted so --resume can continue)"],
        "next": "toxion probe",
        "warn": "A run with --max-molecules is a debug run: its checkpoint is rejected by `toxion probe`.",
    },
    "train-mpnn": {
        "summary": "train the graph-only message-passing network (no fingerprint input)",
        "does": "Same procedure and options as train-fusion, but the model sees only the molecular graph. Useful as "
                "an ablation: does the fingerprint half of the fusion model help?",
        "usage": ["toxion train-mpnn --out results/checkpoints/mpnn.pt"],
        "reads": ["same as train-fusion"],
        "changes": ["same as train-fusion: overwrites results/checkpoints/active.pt unless --out is given"],
        "next": "compare its test ROC-AUC with train-fusion's",
    },
    "train-gin": {
        "summary": "train the PyTorch Geometric GIN baseline (library GNN)",
        "does": "Same procedure and options as train-fusion, with a standard GIN network instead of the custom edge "
                "network. It ignores bond types. Needs the `gnn` extra.",
        "usage": ["toxion train-gin --out results/checkpoints/gin.pt"],
        "reads": ["same as train-fusion"],
        "changes": ["same as train-fusion: overwrites results/checkpoints/active.pt unless --out is given"],
        "next": "compare with train-mpnn: if the custom network does clearly worse, suspect a bug",
    },
    "pretrain": {
        "summary": "self-supervised masked-atom pretraining of the graph encoder on any SMILES files",
        "does": "Hides 15% of atoms in each molecule and trains the encoder to predict the hidden atom's element from "
                "its neighbourhood. Uses no toxicity labels, so any CSV with a smiles column works (e.g. ZINC250k from "
                "scripts/download_pretrain_data.py). Tox21 molecules are excluded unless --no-exclude. The result is "
                "a starting point for train-fusion --pretrained; it is never used by the probe directly.",
        "usage": ["python scripts/download_pretrain_data.py",
                  "toxion pretrain data/raw/zinc250k.csv --hidden 128 --epochs 3"],
        "reads": ["the SMILES CSV file(s) given", "the clean parquet (to exclude Tox21 molecules)"],
        "changes": ["writes the encoder file (default results/checkpoints/pretrained_encoder.pt)",
                    "does not touch the active checkpoint"],
        "next": "toxion train-fusion --feature-version 2 --pretrained results/checkpoints/pretrained_encoder.pt "
                "(hidden size and feature version must match)",
    },
    "probe": {
        "summary": "IUPAC name -> structure -> toxicity verdict, flagged endpoints, Morgan fingerprint",
        "does": "Looks the name up on PubChem, converts it to a canonical SMILES, and prints: the SMILES and Morgan "
                "fingerprint, TOXIC / NOT TOXIC (toxic = at least one endpoint over its own threshold) with the "
                "flagged endpoints and their probability and uncertainty, and how similar the nearest training "
                "molecule is (with a low-confidence warning when it is far). With no name it asks repeatedly "
                "(type exit or press Ctrl-C to quit). A molecule that is neither in Tox21 nor already stored is saved "
                "to the database; when 25 new ones have accumulated the model retrains itself and is replaced only "
                "if validation scores do not get worse. Refuses any model not trained on the real Tox21 file. "
                "This is a screening output, not a safety determination.",
        "usage": ["toxion probe", "toxion probe \"2-acetyloxybenzoic acid\""],
        "reads": ["the active checkpoint (TOXION_CHECKPOINT, default results/checkpoints/active.pt)",
                  "PubChem over the internet", "DATABASE_URL / .env if set"],
        "changes": ["with DATABASE_URL: creates the tables if missing; adds rows to `molecules` and `name_aliases` "
                    "for new molecules",
                    "when the 25th new molecule is added (and TOXION_AUTO_RETRAIN is not 0): retrains and, if "
                    "promoted, REPLACES results/checkpoints/active.pt; marks the batch trained or rejected",
                    "without DATABASE_URL it stores nothing"],
        "next": "toxion explain \"<name>\" to see which atoms drove the verdict",
        "warn": "Exit codes for the one-name form: 0 ok, 1 name not found, 2 no real-data model, 3 other error.",
    },
    "explain": {
        "summary": "draw which atoms drove the prediction for an IUPAC name",
        "does": "Resolves the name, then colours each atom of the molecule. --method ig (default) uses Integrated "
                "Gradients for one endpoint (--task): red pushes toward that endpoint being active, blue away. "
                "--method attention shows what the readout weighted overall, which is not endpoint-specific. Prints "
                "the five most influential atoms and the IG convergence check.",
        "usage": ["toxion explain \"bisphenol A\" --task NR-ER", "toxion explain aspirin --method attention "
                  "--out aspirin.png"],
        "reads": ["the active checkpoint", "PubChem over the internet"],
        "changes": ["writes the PNG (default ./explain.png), overwriting it if it exists"],
        "next": "informational",
    },
    "retrain": {
        "summary": "(maintenance) run the semi-supervised retraining job by hand",
        "does": "Normally `probe` triggers this automatically at 25 new molecules; this runs it on demand. It claims "
                "exactly one full batch of stored new molecules, fine-tunes the active model on all real labeled data "
                "plus those unlabeled molecules, and promotes the result only if validation ROC-AUC and calibration "
                "error are not worse than the current model by more than a small tolerance. Ctrl-C leaves everything "
                "unchanged.",
        "usage": ["toxion retrain"],
        "reads": ["DATABASE_URL (required)", "the active checkpoint", "the clean parquet and its manifest"],
        "changes": ["database: batch status new -> claimed -> trained or rejected; one row in `retrain_runs`",
                    "if promoted: REPLACES results/checkpoints/active.pt; if rejected: the model is untouched"],
        "next": "toxion probe uses the (possibly new) model",
    },
}

_TRAIN = ("train-fusion", "train-mpnn", "train-gin")
OPTION_NOTES: dict[tuple, str] = {
    (None, "--traceback"): "show the full Python traceback instead of a one-line error (put it before the command)",
    (None, "--clean-path"): "the cleaned parquet to read (default data/processed/tox21_clean.parquet)",
    ("prepare-data", "--raw-path"): "the raw Tox21 CSV to clean (default data/raw/tox21.csv.gz)",
    ("prepare-data", "--out-path"): "where to write the clean parquet; the manifest goes next to it",
    ("analyze", "--out-dir"): "directory for the output tables (default results/analysis)",
    ("analyze", "--n-bits"): "Morgan fingerprint length used for the PCA and similarity (default 2048)",
    ("analyze", "--max-molecules"): "analyse only a random sample of this many molecules (default: all)",
    ("make-synthetic", "--n"): "number of synthetic molecules to generate (default 400)",
    ("make-synthetic", "--seed"): "random seed for the synthetic data (default 0)",
    ("make-synthetic", "--out"): "output path (default data/raw/synthetic_tox21.csv.gz)",
    ("make-splits", "--out-dir"): "directory for split_seed<N>.npz (default data/processed/splits)",
    ("make-splits", "--seeds"): "comma-separated split seeds (default 0,1,2,3,4)",
    ("train-baseline", "--model"): "logreg, rf (random forest) or xgboost (needs the ml extra); default logreg",
    ("train-baseline", "--seeds"): "comma-separated split seeds to average over (default 0,1,2,3,4)",
    (_TRAIN, "--out"): "checkpoint path to write. Default: the ACTIVE checkpoint that probe uses, which is replaced",
    (_TRAIN, "--epochs"): "maximum epochs (default 100); early stopping usually ends sooner",
    (_TRAIN, "--batch-size"): "molecules per step (default 32); halved automatically when free RAM is low",
    (_TRAIN, "--hidden"): "hidden width of the network (default 128); 64 is about 4x cheaper in memory and time",
    (_TRAIN, "--seed"): "split and initialisation seed (default 0); with --ensemble N seeds seed..seed+N-1 are used",
    (_TRAIN, "--no-augment"): "turn off training augmentation (for an ablation)",
    (_TRAIN, "--feature-version"): "1 = original atom/bond features (default); 2 = adds element properties, charge, "
                                   "ring and stereo features. The checkpoint remembers which one",
    (_TRAIN, "--ensemble"): "train and average N models from different seeds (default 1); costs N times the time",
    (_TRAIN, "--pretrained"): "encoder file from `toxion pretrain`; hidden size and feature version must match",
    (_TRAIN, "--lr"): "learning rate (default 0.001)",
    (_TRAIN, "--weight-decay"): "L2 penalty; any value above 0 switches Adam to AdamW (default 0)",
    (_TRAIN, "--dropout"): "dropout in the prediction head (default: model default, 0.3 fusion / 0.2 mpnn)",
    (_TRAIN, "--node-dropout"): "dropout on atom states between message-passing rounds (default 0)",
    (_TRAIN, "--node-mask-p"): "augmentation: probability of blanking an atom's features (default 0.10)",
    (_TRAIN, "--edge-drop-p"): "augmentation: probability of dropping a bond (default 0.10)",
    (_TRAIN, "--bit-drop-p"): "augmentation: probability of zeroing a fingerprint entry (default 0.10)",
    (_TRAIN, "--patience"): "stop after this many epochs without improvement (default 15)",
    (_TRAIN, "--monitor"): "what picks the best epoch: val_loss (default) or val_auc",
    (_TRAIN, "--schedule"): "learning-rate schedule: none (default) or cosine",
    (_TRAIN, "--resume"): "continue an interrupted run from its saved state files",
    (_TRAIN, "--max-molecules"): "DEBUG: use only the first N molecules; the checkpoint is rejected by probe",
    ("probe", "name"): "IUPAC name (or common name) to look up; omit it for the interactive prompt",
    ("explain", "name"): "IUPAC name (or common name) to explain",
    ("explain", "--task"): "endpoint to explain with --method ig (default SR-MMP); one of the 12 Tox21 names",
    ("explain", "--method"): "ig = Integrated Gradients for one endpoint (default); attention = readout attention",
    ("explain", "--steps"): "Integrated Gradients interpolation steps (default 50); raise it if the convergence "
                            "check is large",
    ("explain", "--out"): "image path to write (default explain.png)",
    ("pretrain", "data"): "one or more CSV / CSV.GZ files with a smiles column",
    ("pretrain", "--out"): "encoder file to write (default results/checkpoints/pretrained_encoder.pt)",
    ("pretrain", "--hidden"): "hidden width (default 128); must equal the training --hidden",
    ("pretrain", "--feature-version"): "atom/bond feature set (default 2); must equal the training one",
    ("pretrain", "--epochs"): "passes over the pretraining molecules (default 3)",
    ("pretrain", "--batch-size"): "molecules per step (default 32)",
    ("pretrain", "--max-molecules"): "use at most this many molecules, sampled at random (default 30000)",
    ("pretrain", "--seed"): "sampling / masking seed (default 0)",
    ("pretrain", "--clean-path"): "Tox21 parquet whose molecules are excluded from pretraining",
    ("pretrain", "--no-exclude"): "do not exclude Tox21 molecules from the pretraining set",
}

ENV_VARS = [
    ("DATABASE_URL", "Postgres (Neon) connection string for storing new molecules; probe works without it but stores nothing"),
    ("TOXION_CHECKPOINT", "which checkpoint probe / explain / retrain use (default results/checkpoints/active.pt)"),
    ("TOXION_CLEAN_PATH", "clean parquet used by retrain (default data/processed/tox21_clean.parquet)"),
    ("TOXION_AUTO_RETRAIN", "0 stops probe from retraining automatically (default 1)"),
    ("TOXION_RETRAIN_BATCH", "number of new molecules that triggers retraining (default 25)"),
    ("TOXION_FPR_CAP", "at TRAIN time, e.g. 0.10: cap each endpoint's false-positive rate instead of using Youden's J"),
    ("TOXION_DEVICE", "force cpu or mps instead of auto-detecting"),
    ("TOXION_REAL_SHA256", "override the pinned hash of the real Tox21 file"),
    ("TOXION_ENV_FILE", "path of the .env file to read (default ./.env)"),
]

STEPS = [
    {"title": "Install and check the environment",
     "run": ["uv sync --extra ml --extra serve --extra explain --extra gnn --extra demo --extra dev", "toxion check-env"],
     "does": "Creates the virtual environment with torch, SQLAlchemy/psycopg, captum, PyG and Streamlit, then reports "
             "what is installed. Without uv: python -m venv .venv && pip install -e \".[all]\".",
     "changes": ["creates .venv and installs packages; no project data is touched"],
     "ok": "check-env lists every group as installed"},
    {"title": "Get the real Tox21 file",
     "run": ["python scripts/download_data.py"],
     "does": "Downloads tox21.csv.gz (7,831 rows) and checks it really is gzip data.",
     "changes": ["writes data/raw/tox21.csv.gz"],
     "ok": "prints 'Saved data/raw/tox21.csv.gz (122925 bytes)'"},
    {"title": "Clean the data",
     "run": ["toxion prepare-data"],
     "does": "Canonicalises SMILES, drops 8 unparseable rows and duplicates, keeps missing labels as NaN, records the "
             "raw file hash.",
     "changes": ["writes data/processed/tox21_clean.parquet and tox21_clean.parquet.manifest.json"],
     "ok": "'Clean dataset: 7823 molecules, 12 tasks' and the recorded sha256 starts with 45d09792"},
    {"title": "Look at the data (optional)",
     "run": ["toxion summary", "toxion analyze --max-molecules 2000"],
     "does": "summary prints class balance per endpoint; analyze writes PCA / chemical-space / scaffold tables.",
     "changes": ["analyze writes tables into results/analysis; summary changes nothing"],
     "ok": "summary shows 12 endpoints with positive rates of about 3 to 16 percent"},
    {"title": "Cache the splits (optional)",
     "run": ["toxion make-splits"],
     "does": "Writes 5 leak-checked scaffold splits (80/10/10). Training recomputes the same split from its --seed, so "
             "this is for inspection.",
     "changes": ["writes data/processed/splits/split_seed0..4.npz"],
     "ok": "'Cached leak-checked scaffold splits for seeds [0, 1, 2, 3, 4]'"},
    {"title": "Get the baseline number to beat",
     "run": ["toxion train-baseline --model logreg"],
     "does": "Trains a fingerprint logistic regression per endpoint on 5 splits.",
     "changes": ["writes results/tables/baseline_logreg.csv and a feature cache next to the parquet"],
     "ok": "prints 'logreg: ROC-AUC 0.76 +/- 0.03' (about; the spread across splits is large)"},
    {"title": "Train the model the probe will use",
     "run": ["toxion train-fusion", "toxion train-fusion --feature-version 2 --ensemble 5   # optional, slower"],
     "does": "Trains, picks the best epoch, calibrates, tests, and saves one checkpoint. Close other apps first: low "
             "free RAM halves the batch size. Watch the validation AUC column, not the training loss.",
     "changes": ["REPLACES results/checkpoints/active.pt (use --out to keep the old one)",
                 "creates a feature cache next to the parquet"],
     "ok": "prints 'checkpoint: results/checkpoints/active.pt' and val/test ROC-AUC around 0.81 / 0.77. "
           "Commit your work now."},
    {"title": "Compare variants (optional)",
     "run": ["toxion train-fusion --out results/checkpoints/b.pt --weight-decay 1e-3 --monitor val_auc",
             "toxion train-mpnn --out results/checkpoints/mpnn.pt", "toxion train-gin --out results/checkpoints/gin.pt"],
     "does": "Tries other settings or model types without touching the active model. Judge by validation AUC and by "
             "several --seed values; differences below about 0.01 are noise on this data.",
     "changes": ["writes only the --out files you name"],
     "ok": "each run prints its own val/test ROC-AUC; copy the winner over active.pt yourself"},
    {"title": "Add more molecules for pretraining (optional)",
     "run": ["python scripts/download_pretrain_data.py", "toxion pretrain data/raw/zinc250k.csv --hidden 128",
             "toxion train-fusion --feature-version 2 --pretrained results/checkpoints/pretrained_encoder.pt"],
     "does": "Pretrains the graph encoder on 30,000 unlabeled molecules, then fine-tunes on Tox21 starting from it.",
     "changes": ["writes data/raw/zinc250k.csv and results/checkpoints/pretrained_encoder.pt",
                 "the last command REPLACES active.pt unless you add --out"],
     "ok": "pretrain prints held-out masked-atom accuracy above the always-carbon baseline"},
    {"title": "Connect the database (optional, needed to store new molecules)",
     "run": ["cp .env.example .env", "# edit .env and set DATABASE_URL=postgresql://...?sslmode=require"],
     "does": "Neon Postgres holds the molecules the model has never seen. Without it, probe still predicts but stores "
             "nothing.",
     "changes": ["creates .env (git-ignored). Tables are created automatically on the first probe"],
     "ok": "the first probe of a new molecule ends with 'New molecule stored (1/25 until retrain)'"},
    {"title": "Probe a compound",
     "run": ["toxion probe", "toxion probe \"2-acetyloxybenzoic acid\""],
     "does": "Name -> SMILES and Morgan fingerprint -> verdict, flagged endpoints, nearest-training similarity. "
             "Interactive: type exit or press Ctrl-C to quit. A molecule already in Tox21 is reported as such; use a "
             "molecule NOT in Tox21 to test generalisation.",
     "changes": ["new molecules are added to the database (molecules, name_aliases)",
                 "at the 25th new molecule the model retrains and may replace results/checkpoints/active.pt "
                 "(set TOXION_AUTO_RETRAIN=0 to prevent this)"],
     "ok": "prints SMILES, Morgan FP, Verdict, similarity. Exit code 2 means no real-data model yet (do the train step)"},
    {"title": "See why (optional)",
     "run": ["toxion explain \"bisphenol A\" --task NR-ER", "toxion explain \"bisphenol A\" --method attention"],
     "does": "Draws the molecule with atoms coloured by influence.",
     "changes": ["writes explain.png in the current folder"],
     "ok": "prints the top atoms and 'wrote explain.png'; a large convergence delta means raise --steps"},
    {"title": "Retrain on the stored molecules by hand (maintenance)",
     "run": ["toxion retrain"],
     "does": "Same job probe triggers at 25 new molecules. The new model replaces the old only if validation scores "
             "do not get worse.",
     "changes": ["database batch status and a retrain_runs row", "if promoted: REPLACES active.pt"],
     "ok": "prints 'promoted: ...' or 'rejected: kept the current model ...' (or 'skipped' if under 25)"},
    {"title": "Local web demo (optional)",
     "run": ["streamlit run app/demo_fusion.py"],
     "does": "Browser interface for the trained model: SMILES or name in, table and explanation out. Local only.",
     "changes": ["nothing on disk except a temporary image"],
     "ok": "the page opens at localhost:8501"},
    {"title": "Test the pipeline without real data (optional, development)",
     "run": ["toxion make-synthetic --n 400 --out data/raw/synthetic_tox21.csv.gz",
             "toxion prepare-data --raw-path data/raw/synthetic_tox21.csv.gz --out-path data/processed/synthetic.parquet"],
     "does": "Builds a fake dataset. Models trained from it are always refused by probe.",
     "changes": ["writes the synthetic raw file and the synthetic parquet; use a different --out-path so the real "
                 "clean data is not overwritten"],
     "ok": "no errors; then `toxion summary --clean-path data/processed/synthetic.parquet`"},
]


def _wrap(text: str, indent: int = 4, first: str = "") -> str:
    return textwrap.fill(text, width=WIDTH, initial_indent=" " * indent + first,
                         subsequent_indent=" " * (indent + len(first)))


def _bullets(items: list[str], indent: int = 6) -> str:
    return "\n".join(_wrap(i, indent, "- ") for i in items)


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _note(command: str, flag: str) -> str | None:
    for key in ((command, flag), *[(grp, flag) for grp in (_TRAIN,) if command in grp], (None, flag)):
        if key in OPTION_NOTES:
            return OPTION_NOTES[key]
    return None


def option_rows(parser: argparse.ArgumentParser, command: str) -> list[tuple[str, str]]:
    sub = _subparsers(parser)[command]
    rows = []
    for a in sub._actions:
        if isinstance(a, argparse._HelpAction):
            continue
        name = ", ".join(a.option_strings) if a.option_strings else a.dest
        if a.option_strings and a.nargs != 0 and not isinstance(a, argparse._StoreTrueAction):
            name += f" <{(a.metavar or a.dest).upper()}>"
        if a.choices:
            name += "  {" + "|".join(map(str, a.choices)) + "}"
        flag = a.option_strings[0] if a.option_strings else a.dest
        rows.append((name, _note(command, flag) or (a.help if a.help not in (None, argparse.SUPPRESS) else "")))
    return rows


def command_reference(parser: argparse.ArgumentParser, command: str) -> str:
    info = COMMANDS[command]
    out = [f"toxion {command}  -  {info['summary']}", "", _wrap(info["does"], 2), "", "  Usage:"]
    out += [f"      {u}" for u in info["usage"]]
    rows = option_rows(parser, command)
    if rows:
        out += ["", "  Options:"]
        for name, note in rows:
            out += [f"      {name}", _wrap(note or "(no description)", 10)]
    out += ["", "  Reads:", _bullets(info["reads"]), "", "  Changes / writes:", _bullets(info["changes"])]
    if info.get("warn"):
        out += ["", _wrap(info["warn"], 2, "Note: ")]
    out += ["", _wrap(info["next"], 2, "Next: ")]
    return "\n".join(out)


def full_reference(parser: argparse.ArgumentParser) -> str:
    bar = "=" * WIDTH
    out = [bar, "TOXION: complete command reference", bar, "",
           _wrap("Toxion predicts Tox21 toxicity endpoints for a compound named by its IUPAC name. Typical order: "
                 "prepare-data -> train-fusion -> probe. For that order, with what each step changes, run:", 0),
           "    toxion -h step", "",
           "How to get help:",
           "  toxion -h | -help         short list of commands",
           "  toxion --h | --help | help this full reference (every command, option, file and database change)",
           "  toxion help <command>     the full entry for one command",
           "  toxion -h step            numbered walkthrough (also: -help step, --help step, --h step, help step)",
           "  toxion <command> -h       argparse's own option summary for one command",
           "", "Global option:", _wrap(_note(None, "--traceback") or "", 4, "--traceback  "), ""]
    names = list(_subparsers(parser))
    order = [n for n in COMMANDS if n in names] + [n for n in names if n not in COMMANDS]
    for i, name in enumerate(order, 1):
        out += ["-" * WIDTH, f"[{i}/{len(order)}] " + (command_reference(parser, name) if name in COMMANDS
                                                      else f"toxion {name}  (undocumented)"), ""]
    out += ["-" * WIDTH, "Environment variables (can also be set in a .env file; real environment variables win):"]
    out += [textwrap.fill(desc, width=WIDTH, initial_indent=f"  {k:<22} ", subsequent_indent=" " * 25)
            for k, desc in ENV_VARS]
    out += ["", "Files you will see:",
            "  data/raw/tox21.csv.gz                      the raw dataset (input)",
            "  data/processed/tox21_clean.parquet         cleaned dataset (+ .manifest.json hash, + feature cache .pkl)",
            "  results/checkpoints/active.pt              the model probe / explain / retrain use",
            "  results/tables, results/analysis           baseline tables, analysis tables",
            "  .env                                       your DATABASE_URL and switches (never commit it)", ""]
    return "\n".join(out)


def steps_guide() -> str:
    bar = "=" * WIDTH
    out = [bar, "TOXION: step-by-step guide (what to run, what it does, what it changes)", bar, "",
           _wrap("Steps marked optional can be skipped. 'Changes' lists every file, database row or model that the "
                 "step creates or replaces, so you know what a re-run will overwrite.", 0)]
    for i, s in enumerate(STEPS, 1):
        out += ["", f"STEP {i}: {s['title']}", "  Run:"] + [f"      {r}" for r in s["run"]]
        out += [_wrap(s["does"], 2, "What it does: "), "  Changes:", _bullets(s["changes"]),
                _wrap(s["ok"], 2, "You know it worked when: ")]
    out += ["", "Milestones to commit: after step 3 (data clean), after step 7 (model trained), after step 11 "
            "(probe works end to end).", "For every option of every command: toxion --h", ""]
    return "\n".join(out)


def handle_help(argv: list[str], parser: argparse.ArgumentParser, out=print) -> bool:
    if not argv:
        return False
    first, rest = argv[0], [a.lower() for a in argv[1:]]
    if first in SHORT_FLAGS:
        if rest[:1] and rest[0] in STEP_WORDS:
            out(steps_guide())
            return True
        return False
    if first in FULL_FLAGS:
        if not rest:
            out(full_reference(parser))
        elif rest[0] in STEP_WORDS:
            out(steps_guide())
        elif rest[0] in COMMANDS:
            out(command_reference(parser, rest[0]))
        else:
            out(f"no command named {argv[1]!r}. Commands: {', '.join(COMMANDS)}. Try: toxion --h")
        return True
    return False
