# CLI dispatch engine with full multi-command registration
"""Single entrypoint for the pipeline: `toxion <subcommand>` or
`python -m toxion.cli <subcommand>`. Each subcommand is thin glue; the logic
lives in the modules under toxion/. Heavy imports happen inside the command
functions so `--help` and `check-env` work even when extras are missing.
"""

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
    print(f"Wrote {len(df)} clean rows to {args.out_path}")
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
    parser = argparse.ArgumentParser(prog="toxion")
    parser.add_argument(
        "--traceback", action="store_true",
        help="show the full Python traceback instead of a one-line error",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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

    p2 = sub.add_parser("check-env", help="report which dependencies are installed")
    p2.set_defaults(func=cmd_check_env)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
