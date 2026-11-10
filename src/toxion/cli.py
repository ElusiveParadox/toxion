
from __future__ import annotations

import argparse
import logging
import os
import sys

DEFAULT_RAW_PATH = "data/raw/tox21.csv.gz"
DEFAULT_CLEAN_PATH = "data/processed/tox21_clean.parquet"
DEFAULT_ANALYSIS_DIR = "results/analysis"
DEPENDENCY_GROUPS = {
    "core": [
        ("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
        ("rdkit", "rdkit"), ("scikit-learn", "sklearn"), ("scipy", "scipy"),
    ],
    "ml": [("torch", "torch"), ("xgboost", "xgboost")],
    "serve": [("sqlalchemy", "sqlalchemy"), ("psycopg", "psycopg"), ("psutil", "psutil")],
    "gnn": [("torch-geometric", "torch_geometric")],
    "explain": [("captum", "captum")],
    "demo": [("streamlit", "streamlit")],
    "notebook": [("jupyterlab", "jupyterlab"), ("matplotlib", "matplotlib")],
    "dev": [("pytest", "pytest")],
}


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from toxion.data.loader import load_and_clean, save_clean

    if not os.path.exists(args.raw_path):
        raise FileNotFoundError(
            f"Raw file not found: {args.raw_path}. "
            "Download it first with: python scripts/download_data.py"
        )
    df = load_and_clean(args.raw_path)
    save_clean(df, args.out_path)
    from toxion.training.pipeline import write_manifest

    manifest = write_manifest(args.out_path, args.raw_path, len(df))
    print(f"Wrote {len(df)} clean rows to {args.out_path}")
    print(f"Recorded raw file sha256 {manifest['raw_sha256'][:16]}... in {args.out_path}.manifest.json")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from toxion.data.loader import load_clean
    from toxion.data.summary import overall_stats, summarize_labels

    if not os.path.exists(args.clean_path):
        raise FileNotFoundError(
            f"Clean file not found: {args.clean_path}. Create it first with: toxion prepare-data"
        )
    df = load_clean(args.clean_path)
    stats = overall_stats(df)
    print(f"molecules: {stats['molecules']}")
    print(
        f"labeled entries: {stats['labeled_entries']} of {stats['total_entries']} "
        f"({stats['missing_pct']}% missing)"
    )
    print(f"mean labels per molecule: {stats['mean_labels_per_molecule']}")
    print(f"molecules with no labels: {stats['molecules_without_labels']}")
    print()
    print(summarize_labels(df).to_string())
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from toxion.analysis.report import run_analysis
    from toxion.data.loader import load_clean

    if not os.path.exists(args.clean_path):
        raise FileNotFoundError(
            f"Clean file not found: {args.clean_path}. Create it first with: toxion prepare-data"
        )
    df = load_clean(args.clean_path)
    summary = run_analysis(df, args.out_dir, n_bits=args.n_bits, max_molecules=args.max_molecules)
    nn = summary["nearest_neighbor_tanimoto"]
    sc = summary["scaffolds"]
    print(f"molecules analyzed: {summary['n_molecules']}")
    print(f"PCA: top-3 variance ratio {[round(v, 3) for v in summary['pca']['variance_ratio_top'][:3]]}, "
          f"{summary['pca']['components_for_90pct_variance']} components for 90% variance")
    print(f"nearest-neighbor Tanimoto: mean {nn['mean']:.3f}, median {nn['median']:.3f}, "
          f"{100 * nn['fraction_above_0.7']:.1f}% above 0.7")
    print(f"internal diversity (1 - Tanimoto): {summary['internal_diversity']:.3f}")
    print(f"scaffolds: {sc['n_scaffolds']} distinct, largest group {100 * sc['largest_group_fraction']:.1f}%, "
          f"effective number {sc['effective_n_scaffolds']:.1f}")
    print(f"tables written to {args.out_dir}")
    return 0


def cmd_make_synthetic(args: argparse.Namespace) -> int:
    import shlex

    from toxion.data.synthetic import make_synthetic_raw

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    raw = make_synthetic_raw(args.n, args.seed)
    raw.to_csv(args.out, index=False)
    print(f"Wrote {len(raw)} SYNTHETIC molecules to {args.out} (not real data; never report results from it)")
    clean = "data/processed/synthetic_clean.parquet"
    print("Next (synthetic data only, separate output paths so it is never mixed with the real data):")
    print(f"  toxion prepare-data --raw-path {shlex.quote(args.out)} --out-path {clean}")
    print(f"  toxion summary --clean-path {clean}")
    print(f"  toxion analyze --clean-path {clean} --out-dir results/analysis_synthetic")
    return 0


def _parse_seeds(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip() != ""]


def cmd_make_splits(args: argparse.Namespace) -> int:
    from toxion.data.loader import load_clean
    from toxion.data.splits import make_and_cache_splits

    df = load_clean(args.clean_path)
    seeds = _parse_seeds(args.seeds)
    make_and_cache_splits(df, seeds, args.out_dir)
    print(f"Cached leak-checked scaffold splits for seeds {seeds} in {args.out_dir}")
    return 0


def cmd_train_baseline(args: argparse.Namespace) -> int:
    import numpy as np

    from toxion import config
    from toxion.data.featurize import FlatScaler
    from toxion.data.loader import TOX21_TASKS
    from toxion.data.splits import scaffold_split
    from toxion.eval.metrics import aggregate_across_seeds, masked_pr_auc, masked_roc_auc
    from toxion.models import baselines
    from toxion.training import pipeline as pl

    factories = {"logreg": baselines.make_logreg_factory, "rf": baselines.make_random_forest_factory,
                 "xgboost": baselines.make_xgboost_factory}
    factory = factories[args.model]()
    df = pl.load_clean(args.clean_path)
    graphs_unused, flat = pl.featurize_dataframe(df, pl.feature_cache_path(args.clean_path))
    y = df[TOX21_TASKS].to_numpy(dtype="float32")
    roc_runs, pr_runs = [], []
    for seed in _parse_seeds(args.seeds):
        tr, va, te = scaffold_split(df, seed)
        scaler = FlatScaler.fit(flat[tr])
        model = baselines.MaskedMultitaskBaseline(factory, TOX21_TASKS)
        model.fit(scaler.transform(flat[tr]), y[tr])
        pred = model.predict_proba(scaler.transform(flat[te]))
        roc_runs.append(masked_roc_auc(y[te], pred, TOX21_TASKS))
        pr_runs.append(masked_pr_auc(y[te], pred, TOX21_TASKS))
        print(f"seed {seed}: mean test ROC-AUC {np.mean(list(roc_runs[-1].values())):.4f}")
    roc, pr = aggregate_across_seeds(roc_runs), aggregate_across_seeds(pr_runs)
    os.makedirs(config.DEFAULT_TABLE_DIR, exist_ok=True)
    out = os.path.join(config.DEFAULT_TABLE_DIR, f"baseline_{args.model}.csv")
    import pandas as pd

    pd.DataFrame({"roc_auc_mean": {k: v[0] for k, v in roc.items()}, "roc_auc_std": {k: v[1] for k, v in roc.items()},
                  "pr_auc_mean": {k: v[0] for k, v in pr.items()},
                  "pr_auc_std": {k: v[1] for k, v in pr.items()}}).to_csv(out)
    print(f"{args.model}: ROC-AUC {roc['__overall__'][0]:.4f} +/- {roc['__overall__'][1]:.4f}; table: {out}")
    return 0


def _train(args: argparse.Namespace, model_type: str) -> int:
    from toxion import config
    from toxion.training.pipeline import TrainConfig
    from toxion.training.runner import run_training

    overrides = {k: getattr(args, k) for k in ("lr", "weight_decay", "dropout", "node_dropout", "node_mask_p",
                                              "edge_drop_p", "bit_drop_p", "patience", "monitor", "schedule")
                 if getattr(args, k) is not None}
    cfg = TrainConfig(model_type=model_type, hidden_dim=args.hidden, batch_size=args.batch_size,
                      epochs=args.epochs, seed=args.seed, augment=not args.no_augment,
                      max_molecules=args.max_molecules, feature_version=args.feature_version,
                      ensemble_size=args.ensemble, pretrained=args.pretrained, **overrides)
    out = args.out or ("results/checkpoints/debug.pt" if args.max_molecules else config.checkpoint_path())
    metrics = run_training(args.clean_path, cfg, out, resume=args.resume)
    print(f"checkpoint: {out}")
    if "members" in metrics:
        import numpy as np

        for split in ("val", "test"):
            a = metrics["members"][f"{split}_auc"]
            print(f"{split} members ({len(a)}): ROC-AUC {np.mean(a):.4f} +/- {np.std(a):.4f}  (each: "
                  + ", ".join(f"{x:.4f}" for x in a) + ")")
    for split in ("val", "test"):
        m = metrics[split]
        print(f"{split}{' ensemble' if 'members' in metrics else ''}: ROC-AUC {m['roc_auc_mean']:.4f}  PR-AUC {m['pr_auc_mean']:.4f}  ECE {m['ece_mean']:.4f}")
    if args.max_molecules:
        print("debug run on a subset: this checkpoint cannot be used by `toxion probe`")
    else:
        print("Milestone: commit your work (git add -A && git commit) before moving on.")
    return 0


def cmd_train_fusion(args: argparse.Namespace) -> int:
    return _train(args, "fusion")


def cmd_train_gin(args: argparse.Namespace) -> int:
    return _train(args, "gin")


def cmd_train_mpnn(args: argparse.Namespace) -> int:
    return _train(args, "mpnn")


def cmd_probe(args: argparse.Namespace) -> int:
    from toxion import config

    config.load_dotenv()
    from toxion.chem.resolver import PubChemResolver, ResolveError
    from toxion.inference.probe import DISCLAIMER, ProbeSession, format_result, interactive_loop
    from toxion.inference.registry import RealDataGateError, load_checkpoint

    try:
        loaded = load_checkpoint(config.checkpoint_path(), gate=True, device="cpu")
    except (RealDataGateError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    store = None
    url = config.database_url()
    if url:
        try:
            from toxion.db.store import MoleculeStore

            store = MoleculeStore(url)
        except Exception as exc:
            print(f"warning: database unavailable ({type(exc).__name__}); new molecules will not be stored",
                  file=sys.stderr)
    else:
        print("warning: DATABASE_URL is not set; new molecules will not be stored", file=sys.stderr)

    def retrain_fn(s):
        from toxion.training.retrain import retrain

        return retrain(s)

    session = ProbeSession(loaded, PubChemResolver(), store, retrain_fn)
    if args.name:
        try:
            result = session.handle(args.name)
        except ResolveError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3
        print(format_result(result))
        print(DISCLAIMER)
        return 0
    return interactive_loop(session)


def cmd_explain(args: argparse.Namespace) -> int:
    from toxion import config

    config.load_dotenv()
    from toxion.chem.resolver import PubChemResolver, ResolveError
    from toxion.explain.attention_viz import attention_for_smiles, render_atom_scores
    from toxion.explain.integrated_gradients import ig_for_smiles
    from toxion.inference.registry import RealDataGateError, load_checkpoint

    try:
        loaded = load_checkpoint(config.checkpoint_path(), gate=True, device="cpu")
        smiles = PubChemResolver().resolve(args.name).canonical_smiles
        if args.method == "ig":
            scores, delta = ig_for_smiles(loaded, smiles, args.task, args.steps)
            signed = True
            note = f"integrated gradients for {args.task} (convergence delta {delta:.3g}; large means raise --steps)"
        else:
            scores, signed = attention_for_smiles(loaded, smiles), False
            note = "Set2Set readout attention (not endpoint-specific)"
    except (RealDataGateError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ResolveError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    render_atom_scores(smiles, scores, args.out, signed=signed)
    order = sorted(range(len(scores)), key=lambda i: -abs(scores[i]))[:5]
    print(f"SMILES: {smiles}\n{note}\nwrote {args.out}\ntop atoms (index: score): "
          + ", ".join(f"{i}: {scores[i]:+.3f}" for i in order))
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    from toxion import device as device_mod
    from toxion.data.loader import load_clean
    from toxion.training.pretrain import run_pretraining

    exclude = None
    if not args.no_exclude and os.path.exists(args.clean_path):
        exclude = set(load_clean(args.clean_path)["smiles_canonical"])
    meta = run_pretraining(args.data, args.out, hidden_dim=args.hidden, feature_version=args.feature_version,
                           epochs=args.epochs, batch_size=args.batch_size, max_molecules=args.max_molecules,
                           exclude=exclude, seed=args.seed, device=device_mod.pick_device())
    print(f"pretrained encoder: {args.out}\n{meta['n_molecules']} molecules, held-out masked-atom accuracy "
          f"{meta['val_masked_atom_acc']:.3f} (always guessing the commonest element: {meta['majority_class_acc']:.3f})")
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    from toxion import config

    config.load_dotenv()
    url = config.database_url()
    if not url:
        raise ValueError("DATABASE_URL is not set")
    from toxion.db.store import MoleculeStore
    from toxion.training.retrain import retrain

    outcome = retrain(MoleculeStore(url))
    print(f"{outcome.decision}: {outcome.message}")
    return 0


def cmd_check_env(args: argparse.Namespace) -> int:
    from importlib import metadata
    from importlib.util import find_spec

    core_missing = False
    print(f"python {sys.version.split()[0]}")
    for group, packages in DEPENDENCY_GROUPS.items():
        for dist, module in packages:
            if find_spec(module) is None:
                if group == "core":
                    core_missing = True
                    status = "MISSING"
                else:
                    status = f"MISSING (install extra: {group})"
            else:
                status = metadata.version(dist)
            print(f"  [{group:<7}] {dist:<16} {status}")
    return 1 if core_missing else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toxion",
        add_help=False,
        epilog="more help:  toxion --h  (every command in detail)   |   toxion -h step  (step-by-step guide)",
    )
    parser.add_argument(
        "-h", "-help", action="help", default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    parser.add_argument(
        "--traceback", action="store_true",
        help="show the full Python traceback instead of a one-line error",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    orig_add_parser = sub.add_parser

    def custom_add_parser(*args, **kwargs):
        kwargs["add_help"] = False
        sp = orig_add_parser(*args, **kwargs)
        sp.add_argument("-h", "-help", action="help", dest="help", default=argparse.SUPPRESS, help="show this help message and exit")
        return sp

    sub.add_parser = custom_add_parser

    p1 = sub.add_parser("prepare-data", help="clean raw Tox21 CSV into parquet")
    p1.add_argument("--raw-path", default=DEFAULT_RAW_PATH)
    p1.add_argument("--out-path", default=DEFAULT_CLEAN_PATH)
    p1.set_defaults(func=cmd_prepare_data)

    p_sum = sub.add_parser("summary", help="class balance and missingness per endpoint")
    p_sum.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH)
    p_sum.set_defaults(func=cmd_summary)

    p_an = sub.add_parser("analyze", help="advanced EDA: endpoint dependence, PCA, chemical space")
    p_an.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH)
    p_an.add_argument("--out-dir", default=DEFAULT_ANALYSIS_DIR)
    p_an.add_argument("--n-bits", type=int, default=2048)
    p_an.add_argument("--max-molecules", type=int, default=None)
    p_an.set_defaults(func=cmd_analyze)

    p_syn = sub.add_parser("make-synthetic", help="write a synthetic raw Tox21-shaped CSV for pipeline testing")
    p_syn.add_argument("--n", type=int, default=400)
    p_syn.add_argument("--seed", type=int, default=0)
    p_syn.add_argument("--out", default="data/raw/synthetic_tox21.csv.gz")
    p_syn.set_defaults(func=cmd_make_synthetic)

    p_sp = sub.add_parser("make-splits", help="cache leak-checked scaffold splits for several seeds")
    p_sp.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH)
    p_sp.add_argument("--out-dir", default="data/processed/splits")
    p_sp.add_argument("--seeds", default="0,1,2,3,4")
    p_sp.set_defaults(func=cmd_make_splits)

    p_bl = sub.add_parser("train-baseline", help="fingerprint/descriptor baselines across scaffold-split seeds")
    p_bl.add_argument("--model", choices=["logreg", "rf", "xgboost"], default="logreg")
    p_bl.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH)
    p_bl.add_argument("--seeds", default="0,1,2,3,4")
    p_bl.set_defaults(func=cmd_train_baseline)

    for name, func, help_ in (("train-fusion", cmd_train_fusion, "train the fusion model (graph + fingerprint) on the real data"),
                              ("train-gin", cmd_train_gin, "train the PyG GIN baseline (needs the gnn extra)"),
                              ("train-mpnn", cmd_train_mpnn, "train the standalone MPNN on the real data")):
        p_t = sub.add_parser(name, help=help_)
        p_t.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH)
        p_t.add_argument("--out", default=None, help="checkpoint path (default: the active checkpoint)")
        p_t.add_argument("--epochs", type=int, default=100)
        p_t.add_argument("--batch-size", type=int, default=32)
        p_t.add_argument("--hidden", type=int, default=128)
        p_t.add_argument("--seed", type=int, default=0)
        p_t.add_argument("--no-augment", action="store_true")
        p_t.add_argument("--feature-version", type=int, choices=[1, 2], default=1,
                         help="2 adds element properties, charge, ring and stereo features")
        p_t.add_argument("--ensemble", type=int, default=1, help="average N models trained from different seeds")
        p_t.add_argument("--pretrained", default=None, help="encoder file written by `toxion pretrain`")
        p_t.add_argument("--lr", type=float, default=None)
        p_t.add_argument("--weight-decay", type=float, default=None, help="> 0 switches Adam to AdamW")
        p_t.add_argument("--dropout", type=float, default=None, help="head dropout")
        p_t.add_argument("--node-dropout", type=float, default=None, help="dropout on atom states between rounds")
        p_t.add_argument("--node-mask-p", type=float, default=None)
        p_t.add_argument("--edge-drop-p", type=float, default=None)
        p_t.add_argument("--bit-drop-p", type=float, default=None)
        p_t.add_argument("--patience", type=int, default=None)
        p_t.add_argument("--monitor", choices=["val_loss", "val_auc"], default=None)
        p_t.add_argument("--schedule", choices=["none", "cosine"], default=None)
        p_t.add_argument("--resume", action="store_true")
        p_t.add_argument("--max-molecules", type=int, default=None, help="debug only; checkpoint is not probe-eligible")
        p_t.set_defaults(func=func)

    p_pr = sub.add_parser("probe", help="IUPAC name -> toxicity verdict (interactive when no name is given)")
    p_pr.add_argument("name", nargs="?", default=None)
    p_pr.set_defaults(func=cmd_probe)

    p_ex = sub.add_parser("explain", help="atom-level explanation image for an IUPAC name")
    p_ex.add_argument("name")
    p_ex.add_argument("--task", default="SR-MMP", help="endpoint for --method ig (default SR-MMP)")
    p_ex.add_argument("--method", choices=["ig", "attention"], default="ig")
    p_ex.add_argument("--steps", type=int, default=50)
    p_ex.add_argument("--out", default="explain.png")
    p_ex.set_defaults(func=cmd_explain)

    p_pt = sub.add_parser("pretrain", help="self-supervised masked-atom pretraining on any SMILES files")
    p_pt.add_argument("data", nargs="+", help="CSV / CSV.GZ files with a smiles column")
    p_pt.add_argument("--out", default="results/checkpoints/pretrained_encoder.pt")
    p_pt.add_argument("--hidden", type=int, default=128)
    p_pt.add_argument("--feature-version", type=int, choices=[1, 2], default=2)
    p_pt.add_argument("--epochs", type=int, default=3)
    p_pt.add_argument("--batch-size", type=int, default=32)
    p_pt.add_argument("--max-molecules", type=int, default=30000)
    p_pt.add_argument("--seed", type=int, default=0)
    p_pt.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH, help="Tox21 molecules are excluded from pretraining")
    p_pt.add_argument("--no-exclude", action="store_true")
    p_pt.set_defaults(func=cmd_pretrain)

    p_rt = sub.add_parser("retrain", help="(maintenance) retrain on the stored new molecules now")
    p_rt.set_defaults(func=cmd_retrain)

    p2 = sub.add_parser("check-env", help="report which dependencies are installed")
    p2.set_defaults(func=cmd_check_env)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from toxion.help_text import handle_help

    parser = build_parser()
    try:
        if handle_help(argv, parser):
            return 0
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
