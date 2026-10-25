from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

DEFAULT_CLEAN_PATH = "data/processed/tox21_clean.parquet"
MAX_AUTO_EPOCHS = 300
OPTIMIZERS = ("sgd", "rmsprop", "adam", "adamw")
DEFAULT_LR = {"sgd": 0.05, "rmsprop": 1e-3, "adam": 1e-3, "adamw": 1e-3}
DEFAULT_WEIGHT_DECAY = {"sgd": 0.0, "rmsprop": 0.0, "adam": 0.0, "adamw": 1e-2}
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "r2"]

OPTIMIZER_INFO = {
    "sgd": {
        "name": "SGD with momentum 0.9",
        "rule": "buf <- 0.9*buf + g ;  w <- w - lr*buf",
        "use": "Follows a smoothed gradient direction with one global step size. Needs lr tuned; "
               "can be slow when features are scaled very differently.",
    },
    "rmsprop": {
        "name": "RMSprop (alpha 0.99)",
        "rule": "v <- 0.99*v + 0.01*g^2 ;  w <- w - lr * g / (sqrt(v) + 1e-8)",
        "use": "Divides each weight's step by a running RMS of its own gradient, so rarely-active "
               "fingerprint bits still get usable steps.",
    },
    "adam": {
        "name": "Adam (beta 0.9, 0.999)",
        "rule": "m <- 0.9*m + 0.1*g ; v <- 0.999*v + 0.001*g^2 ; "
                "w <- w - lr * (m/(1-0.9^t)) / (sqrt(v/(1-0.999^t)) + 1e-8)",
        "use": "Momentum plus per-weight step sizes with bias correction. Usually the fastest "
               "starter on sparse binary inputs like fingerprints.",
    },
    "adamw": {
        "name": "AdamW (beta 0.9, 0.999, decoupled weight decay)",
        "rule": "w <- w*(1 - lr*wd) ; then the Adam step",
        "use": "Adam plus weight decay applied directly to the weights (not through the gradient). "
               "Shrinks weights each step, which reduces overfitting on 2048 mostly-zero features.",
    },
}


class _Fmt(argparse.RawDescriptionHelpFormatter):
    """Raw epilog text, and '(default: X)' appended to option help."""

    def _get_help_string(self, action):
        text = action.help or ""
        if "%(default)" in text or action.default is None or action.default is False \
                or action.default is argparse.SUPPRESS or not action.option_strings:
            return text
        return f"{text} (default: {action.default})"


# --------------------------------------------------------------------------- argument parsing helpers

def _epochs_arg(value: str):
    if value == "auto":
        return None
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return n


def _threshold_arg(value: str):
    if value == "auto":
        return None
    try:
        t = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be 'auto' or a number between 0 and 1") from None
    if not 0.0 < t < 1.0:
        raise argparse.ArgumentTypeError("must be 'auto' or a number between 0 and 1")
    return t


def _add_clean_path(p):
    p.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH,
                   help="cleaned parquet written by `toxion prepare-data`")


def _add_split_args(p):
    p.add_argument("--split-method", choices=["scaffold", "random"], default="scaffold",
                   help="scaffold keeps every Bemis-Murcko scaffold group on one side (harder, realistic); "
                        "random mixes molecules freely (optimistic, near-duplicates leak across sets)")
    p.add_argument("--train-frac", type=float, default=0.8, help="fraction of molecules for training")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction for validation (early stopping, optimizer choice, threshold choice)")
    p.add_argument("--test-frac", type=float, default=0.1,
                   help="fraction for test (never used to make any choice)")
    p.add_argument("--split-seed", type=int, default=0, help="random seed for the split")


# --------------------------------------------------------------------------- data helpers

def _load_clean(path: str):
    from toxion.data.loader import load_clean

    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean file not found: {path}. Create it first with: toxion prepare-data")
    return load_clean(path)


def _check_fracs(train_frac, val_frac, test_frac):
    if min(train_frac, val_frac, test_frac) < 0 or abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("--train-frac, --val-frac and --test-frac must be non-negative and sum to 1")


def random_split(n: int, fracs, seed: int):
    import numpy as np

    order = np.random.default_rng(seed).permutation(n)
    n_train, n_val = int(fracs[0] * n), int(fracs[1] * n)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def scaffold_groups(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups = defaultdict(list)
    for pos, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        key = "" if mol is None else MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        groups[key].append(pos)
    return groups


def scaffold_split(smiles, fracs, seed: int):
    import numpy as np

    groups = list(scaffold_groups(smiles).values())
    random.Random(seed).shuffle(groups)
    n = len(smiles)
    n_train_target, n_val_target = int(fracs[0] * n), int(fracs[1] * n)
    train, val, test = [], [], []
    for group in groups:
        if len(train) + len(group) <= n_train_target:
            train.extend(group)
        elif len(val) + len(group) <= n_val_target:
            val.extend(group)
        else:
            test.extend(group)
    return tuple(np.array(part, dtype=np.int64) for part in (train, val, test))


def size_warning(n_total, parts, fracs):
    """One-line warning if actual split sizes differ from the requested fractions by more than 5 points."""
    actual = [len(p) / n_total for p in parts]
    if any(abs(a - f) > 0.05 for a, f in zip(actual, fracs)):
        return ("WARNING: actual split sizes "
                f"{actual[0]:.2f}/{actual[1]:.2f}/{actual[2]:.2f} differ from the requested "
                f"{fracs[0]}/{fracs[1]}/{fracs[2]}; a few very large scaffold groups cannot be divided. "
                "Try another --split-seed or --split-method random.")
    return None


def make_split(smiles, method, fracs, seed):
    _check_fracs(*fracs)
    if method == "random":
        return random_split(len(smiles), fracs, seed)
    return scaffold_split(smiles, fracs, seed)


# --------------------------------------------------------------------------- data commands

def _show(df):
    import pandas as pd

    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 40):
        print(df)


def cmd_data_head(args):
    _show(_load_clean(args.clean_path).head(args.n))
    return 0


def cmd_data_tail(args):
    _show(_load_clean(args.clean_path).tail(args.n))
    return 0


def cmd_data_shape(args):
    df = _load_clean(args.clean_path)
    print(f"{df.shape}  ->  {df.shape[0]} molecules x {df.shape[1]} columns "
          f"(smiles_canonical + {df.shape[1] - 1} endpoints)")
    return 0


def cmd_data_missing(args):
    import pandas as pd

    df = _load_clean(args.clean_path)
    n = len(df)
    table = pd.DataFrame({"missing": df.isna().sum(), "missing_pct": (100 * df.isna().mean()).round(1)})
    table["labeled"] = n - table["missing"]
    print(table.to_string())
    total = int(df.drop(columns="smiles_canonical").isna().to_numpy().sum())
    cells = n * (df.shape[1] - 1)
    print(f"\nmissing label entries: {total} of {cells} ({100 * total / cells:.1f} percent). "
          "Missing means 'not tested', never 'inactive'.")
    return 0


def cmd_data_split(args):
    import numpy as np

    from toxion.data.loader import TOX21_TASKS

    df = _load_clean(args.clean_path)
    smiles = df["smiles_canonical"].tolist()
    fracs = (args.train_frac, args.val_frac, args.test_frac)
    parts = dict(zip(("train", "val", "test"), make_split(smiles, args.split_method, fracs, args.split_seed)))
    Y = df[TOX21_TASKS].to_numpy(dtype=float)

    print(f"split: {args.split_method}, seed {args.split_seed}, requested fractions "
          f"{fracs[0]}/{fracs[1]}/{fracs[2]}")
    for name, idx in parts.items():
        y = Y[idx]
        labeled = int((~np.isnan(y)).sum())
        positives = int((y == 1).sum())
        rate = f"{positives / labeled:.3f}" if labeled else "n/a"
        print(f"  {name:<5} {len(idx):>6} molecules ({100 * len(idx) / len(df):5.1f} percent)  "
              f"labeled entries {labeled:>7}  positive rate {rate}")
    warning = size_warning(len(df), list(parts.values()), fracs)
    if warning:
        print(warning)
    if args.split_method == "scaffold":
        groups = scaffold_groups(smiles)
        of = {pos: key for key, members in groups.items() for pos in members}
        sets = {name: {of[i] for i in idx} for name, idx in parts.items()}
        leaked = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
        print(f"  scaffolds: train {len(sets['train'])}, val {len(sets['val'])}, test {len(sets['test'])}; "
              f"{'no scaffold appears in two splits' if not leaked else f'LEAK: {len(leaked)} shared'}")
    if args.save:
        out_dir = os.path.dirname(args.save)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez(args.save, **{f"{k}_idx": v for k, v in parts.items()})
        print(f"saved indices to {args.save}")
    return 0


# --------------------------------------------------------------------------- metrics

def compute_metrics(y, p, threshold: float):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                                 r2_score, recall_score, roc_auc_score)

    from toxion.data.loader import TOX21_TASKS

    nan = float("nan")
    rows = {}
    for t, task in enumerate(TOX21_TASKS):
        keep = ~np.isnan(y[:, t])
        if not keep.any():
            continue
        yt, pt = y[keep, t].astype(int), p[keep, t]
        pred = (pt >= threshold).astype(int)
        has_pos = bool(yt.any())
        has_both = has_pos and not bool(yt.all())
        rows[task] = {
            "n_labeled": int(keep.sum()),
            "accuracy": accuracy_score(yt, pred),
            "precision": precision_score(yt, pred, zero_division=0) if has_pos else nan,
            "recall": recall_score(yt, pred, zero_division=0) if has_pos else nan,
            "f1": f1_score(yt, pred, zero_division=0) if has_pos else nan,
            "roc_auc": roc_auc_score(yt, pt) if has_both else nan,
            "pr_auc": average_precision_score(yt, pt) if has_both else nan,
            "r2": r2_score(yt, pt) if has_both else nan,
        }
    per_task = pd.DataFrame.from_dict(rows, orient="index", columns=["n_labeled", *METRIC_NAMES])
    macro = {m: float(per_task[m].mean()) if len(per_task) else nan for m in METRIC_NAMES}
    labeled = int(per_task["n_labeled"].sum()) if len(per_task) else 0
    positives = int((y == 1).sum())
    macro["n_labeled"] = labeled
    macro["positive_rate"] = positives / labeled if labeled else nan
    return macro, per_task


def _quick_metrics(y, p):
    """Mean ROC-AUC and mean F1 at 0.5, cheap enough to compute every epoch."""
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    rocs, f1s = [], []
    for t in range(y.shape[1]):
        keep = ~np.isnan(y[:, t])
        yt = y[keep, t].astype(int)
        if yt.size == 0 or not yt.any():
            continue
        f1s.append(f1_score(yt, (p[keep, t] >= 0.5).astype(int), zero_division=0))
        if not yt.all():
            rocs.append(roc_auc_score(yt, p[keep, t]))
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    return mean(rocs), mean(f1s)


def _best_threshold(y, p):
    """Threshold in 0.05..0.95 that maximizes mean per-endpoint F1 on the validation split."""
    import numpy as np
    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        scores = []
        for t in range(y.shape[1]):
            keep = ~np.isnan(y[:, t])
            yt = y[keep, t].astype(int)
            if yt.size and yt.any():
                scores.append(f1_score(yt, (p[keep, t] >= thr).astype(int), zero_division=0))
        if scores and np.mean(scores) > best_f1:
            best_t, best_f1 = float(thr), float(np.mean(scores))
    return best_t


# --------------------------------------------------------------------------- training

def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ValueError("`train` needs torch. Install it with: uv sync --extra ml") from None


def _make_optimizer(name, params, lr, weight_decay):
    from toxion.optim.optimizers import SGD, Adam, AdamW, RMSprop

    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def _make_loss(args, Ytr):
    import torch

    from toxion.training.advanced_losses import compute_pos_weights, masked_focal_loss, masked_weighted_bce

    if args.loss == "weighted":
        pos_w = torch.tensor(compute_pos_weights(Ytr.numpy()))
        return lambda z, y: masked_weighted_bce(z, y, pos_w)
    if args.loss == "focal":
        return lambda z, y: masked_focal_loss(z, y, gamma=args.focal_gamma)
    return lambda z, y: masked_weighted_bce(z, y)


def describe_optimizer(name, lr, wd, n_weights, steps_per_epoch, batch_size, schedule):
    info = OPTIMIZER_INFO[name]
    lines = [
        f"Optimizer: {info['name']}  (lr={lr:g}, weight_decay={wd:g}, schedule={schedule})",
        f"  update rule : {info['rule']}   (g = gradient, w = parameters)",
        f"  how it is used here: every step it takes the gradient of the masked BCE loss on a batch of "
        f"{batch_size} molecules (labels that are NaN contribute nothing) and updates all {n_weights} "
        f"parameters (a 2048x12 weight matrix plus 12 biases); {steps_per_epoch} steps make one epoch.",
        f"  why: {info['use']}",
    ]
    return "\n".join(lines)


def _train_once(opt_name, lr, wd, data, args, max_epochs, patience, track, log_every, quiet, loss_fn):
    """Mini-batch training with per-epoch evaluation. Early stopping and the returned weights use the
    plain masked BCE on the validation split; the test split is only recorded, never used to decide."""
    import numpy as np
    import torch
    import torch.nn as nn

    from toxion.optim.schedules import make_scheduler, warmup_cosine_factor
    from toxion.training.advanced_losses import masked_weighted_bce

    X, Y, split = data["X"], data["Y"], data["split"]
    Xtr, Ytr = X[split["train"]], Y[split["train"]]
    torch.manual_seed(args.seed)
    model = nn.Linear(X.shape[1], Y.shape[1])
    opt = _make_optimizer(opt_name, list(model.parameters()), lr, wd)
    sched = None
    if args.schedule == "cosine":
        warm = min(3, max(0, max_epochs // 10))
        sched = make_scheduler(opt, lambda e: warmup_cosine_factor(e, warm, max_epochs, min_factor=0.05))

    gen = torch.Generator().manual_seed(args.seed)
    n_train = Xtr.shape[0]
    hist = {k: [] for k in ["train_loss", "val_loss", "test_loss", "train_roc", "val_roc", "test_roc",
                            "train_f1", "val_f1", "test_f1"]}
    best_val, best_epoch, bad, best_state = float("inf"), 0, 0, None
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        if sched:
            sched.step()
        epochs_run = epoch

        with torch.no_grad():
            logits = {name: model(X[idx]) for name, idx in split.items()}
        for name, idx in split.items():
            if not track and name == "test":
                continue
            hist[f"{name}_loss"].append(masked_weighted_bce(logits[name], Y[idx]).item())
            if track:
                roc, f1 = _quick_metrics(Y[idx].numpy(), torch.sigmoid(logits[name]).numpy())
                hist[f"{name}_roc"].append(roc)
                hist[f"{name}_f1"].append(f1)

        val_loss = hist["val_loss"][-1]
        if val_loss < best_val - 1e-4:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if not quiet and log_every and (epoch % log_every == 0 or epoch == 1):
            extra = f" val_roc_auc {hist['val_roc'][-1]:.3f}" if track else ""
            print(f"  epoch {epoch:>4}/{max_epochs}  train_loss {hist['train_loss'][-1]:.4f}  "
                  f"val_loss {val_loss:.4f}{extra}")
        if patience and bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "hist": hist, "epochs_run": epochs_run, "best_epoch": best_epoch,
            "best_val_loss": best_val, "stopped_early": epochs_run < max_epochs}


def _plot(hist, best_epoch, final, out, show):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ValueError("--plot needs matplotlib. Install it with: uv sync --extra notebook") from None

    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for name in ("train", "val", "test"):
        style = "--" if name == "test" else "-"
        axes[0].plot(epochs, hist[f"{name}_loss"], style, color=colors[name], label=name)
        axes[1].plot(epochs, hist[f"{name}_roc"], style, color=colors[name], label=name)
    for ax, title in ((axes[0], "loss (masked BCE)"), (axes[1], "mean ROC-AUC")):
        ax.axvline(best_epoch, color="gray", linestyle=":", label=f"best epoch {best_epoch}")
        ax.set(title=title, xlabel="epoch")
        ax.legend()
    axes[0].set_title("loss (masked BCE); test shown but never used for choices")

    width = 0.27
    x = np.arange(len(METRIC_NAMES))
    for k, name in enumerate(("train", "val", "test")):
        values = [final[name][m] for m in METRIC_NAMES]
        axes[2].bar(x + (k - 1) * width, np.nan_to_num(values), width, label=name, color=colors[name])
    axes[2].set_xticks(x, METRIC_NAMES, rotation=30)
    axes[2].set(title="final metrics by split")
    axes[2].legend()
    plt.tight_layout()
    if out:
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out, dpi=130)
        print(f"saved plot to {out}")
    if show:
        plt.show()
    plt.close(fig)


def cmd_train(args):
    import numpy as np
    import pandas as pd

    _require_torch()
    import torch

    from toxion.analysis.chemspace import morgan_bits
    from toxion.data.loader import TOX21_TASKS

    if args.batch_size < 1 or args.patience < 0 or args.auto_epochs < 1:
        raise ValueError("--batch-size and --auto-epochs must be >= 1 and --patience >= 0")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.val_frac <= 0 or args.train_frac <= 0:
        raise ValueError("--train-frac and --val-frac must be greater than 0 (validation drives early stopping)")

    df = _load_clean(args.clean_path)
    smiles = df["smiles_canonical"].tolist()
    fracs = (args.train_frac, args.val_frac, args.test_frac)
    parts = make_split(smiles, args.split_method, fracs, args.split_seed)
    if len(parts[0]) == 0 or len(parts[1]) == 0:
        raise ValueError(
            f"the {args.split_method} split left {'train' if len(parts[0]) == 0 else 'validation'} empty "
            f"(sizes {len(parts[0])}/{len(parts[1])}/{len(parts[2])}). Run `toxion data split` to inspect it; "
            "try another --split-seed or --split-method random.")
    X = torch.tensor(morgan_bits(smiles, n_bits=args.n_bits), dtype=torch.float32)
    Y = torch.tensor(df[TOX21_TASKS].to_numpy(dtype=np.float32))
    split = {name: torch.as_tensor(idx, dtype=torch.long) for name, idx in zip(("train", "val", "test"), parts)}
    data = {"X": X, "Y": Y, "split": split}
    loss_fn = _make_loss(args, Y[split["train"]])

    print(f"Toxion train: multitask logistic regression on Morgan bits ({len(df)} molecules, "
          f"{len(TOX21_TASKS)} endpoints, {args.n_bits} bits)")
    print(f"Split ({args.split_method}, seed {args.split_seed}): train {len(parts[0])} | val {len(parts[1])} "
          f"| test {len(parts[2])}   loss: {args.loss}   batch size: {args.batch_size}")
    warning = size_warning(len(df), parts, fracs)
    if warning:
        print(warning)

    max_epochs = args.epochs or MAX_AUTO_EPOCHS
    wd_arg = args.weight_decay

    chosen = args.optimizer
    if chosen == "auto":
        print(f"\nChoosing the optimizer: each candidate trains {args.auto_epochs} epochs with its default "
              f"learning rate; the lowest validation loss wins.")
        scores = {}
        for name in OPTIMIZERS:
            lr = args.lr or DEFAULT_LR[name]
            wd = DEFAULT_WEIGHT_DECAY[name] if wd_arg is None else wd_arg
            run = _train_once(name, lr, wd, data, args, args.auto_epochs, 0, False, 0, True, loss_fn)
            scores[name] = run["best_val_loss"]
            print(f"  {name:<8} lr={lr:<7g} best val loss {run['best_val_loss']:.4f} (epoch {run['best_epoch']})")
        chosen = min(scores, key=scores.get)
        print(f"  -> chose {chosen}")

    lr = args.lr or DEFAULT_LR[chosen]
    wd = DEFAULT_WEIGHT_DECAY[chosen] if wd_arg is None else wd_arg
    n_train = len(parts[0])
    print()
    print(describe_optimizer(chosen, lr, wd, args.n_bits * len(TOX21_TASKS) + len(TOX21_TASKS),
                             -(-n_train // args.batch_size), args.batch_size, args.schedule))
    patience_text = f"patience {args.patience}" if args.patience else "early stopping off"
    print(f"\nTraining for up to {max_epochs} epochs ({'auto' if args.epochs is None else 'fixed'} maximum, "
          f"{patience_text}):")
    run = _train_once(chosen, lr, wd, data, args, max_epochs, args.patience, True, args.log_every,
                      args.quiet, loss_fn)
    model, hist = run["model"], run["hist"]

    with torch.no_grad():
        probs = {name: torch.sigmoid(model(X[idx])).numpy() for name, idx in split.items()}
        probs["all"] = torch.sigmoid(model(X)).numpy()
    labels = {name: Y[idx].numpy() for name, idx in split.items()}
    labels["all"] = Y.numpy()

    threshold = args.threshold
    threshold_note = "fixed by --threshold"
    if threshold is None:
        threshold = _best_threshold(labels["val"], probs["val"])
        threshold_note = "auto: maximizes mean F1 on the validation split"
    final, per_task = {}, {}
    for name in ("train", "val", "test", "all"):
        final[name], per_task[name] = compute_metrics(labels[name], probs[name], threshold)

    print(f"\nEpochs run: {run['epochs_run']} (maximum {max_epochs}). Best epoch: {run['best_epoch']} "
          f"(validation loss {run['best_val_loss']:.4f}). "
          f"Stopped early: {'yes' if run['stopped_early'] else 'no'}. Weights from the best epoch are used.")
    print(f"Optimizer used: {chosen}.  Decision threshold: {threshold:.2f} ({threshold_note}).")
    table = pd.DataFrame({name: final[name] for name in ("train", "val", "test", "all")})
    print("\nMetrics (mean over endpoints; 'all' = train+val+test pooled, includes training data):")
    print(table.loc[METRIC_NAMES].round(4).to_string())
    counts = table.loc[["n_labeled", "positive_rate"]]
    counts.loc["n_labeled"] = counts.loc["n_labeled"].astype(int).astype(str)
    counts.loc["positive_rate"] = counts.loc["positive_rate"].map(lambda v: f"{v:.4f}")
    print("\nLabeled entries and positive rate:")
    print(counts.to_string())
    print("\nNotes: accuracy is misleading when positives are rare; compare it with positive_rate. "
          "r2 is computed on predicted probabilities. The test column was not used to choose anything.")

    if args.per_task:
        print("\nPer-endpoint metrics, test split:")
        print(per_task["test"].round(4).to_string())

    if args.plot or args.show:
        _plot(hist, run["best_epoch"], final, args.plot_out, args.show)

    if args.save_metrics:
        out_dir = os.path.dirname(args.save_metrics)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        clean = lambda d: {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}
        payload = {
            "config": {k: v for k, v in vars(args).items() if k != "func"} | {"optimizer_used": chosen,
                                                                                "lr_used": lr, "wd_used": wd,
                                                                                "threshold_used": threshold},
            "epochs_run": run["epochs_run"], "best_epoch": run["best_epoch"], "max_epochs": max_epochs,
            "metrics": {name: clean(final[name]) for name in final},
        }
        with open(args.save_metrics, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"saved metrics to {args.save_metrics}")
    return 0


def cmd_optimizers(args):
    print("Optimizers available to `toxion train --optimizer` (written from scratch in toxion/optim/optimizers.py,\n"
          "checked against torch.optim in the tests). 'auto' tries all four briefly and keeps the best.\n")
    for name in OPTIMIZERS:
        info = OPTIMIZER_INFO[name]
        print(f"{name}: {info['name']}")
        print(f"  default lr {DEFAULT_LR[name]:g}, default weight decay {DEFAULT_WEIGHT_DECAY[name]:g}")
        print(f"  update rule: {info['rule']}")
        print(f"  use: {info['use']}\n")
    print("Math and derivations: docs/math.md, section 4. Learning-rate schedule `--schedule cosine`: "
          "3-epoch linear warmup then cosine decay to 5 percent of the base rate.")
    return 0


# --------------------------------------------------------------------------- registration

TRAIN_EPILOG = """\
examples:
  toxion train                                   all defaults: scaffold split, auto optimizer, auto epochs
  toxion train --optimizer adamw --epochs 100    fixed optimizer and at most 100 epochs
  toxion train --split-method random --plot      random split, save curves to results/figures/train_curves.png
  toxion train --per-task --save-metrics results/tables/run1.json

what the defaults do:
  epochs      'auto' = up to 300 epochs with early stopping: training stops after --patience epochs without
              the validation loss improving, and the weights from the best epoch are kept.
              A number sets the maximum; early stopping still applies unless you pass --patience 0.
  optimizer   'auto' = train each of sgd, rmsprop, adam, adamw for --auto-epochs epochs and keep the one with
              the lowest validation loss. See `toxion optimizers` for the update rules. Caution: a short budget
              favors optimizers that start fast, not necessarily the best final model; raise --auto-epochs
              or pass --optimizer yourself if the choice looks suspicious.
  threshold   'auto' = the cutoff (0.05 to 0.95) that maximizes mean F1 on the validation split. It affects
              accuracy, precision, recall and F1 only; ROC-AUC, PR-AUC and R2 use raw probabilities.
  test split  never used for any choice (optimizer, epochs, threshold); it is only reported.

'optimal' here means best validation loss for this linear baseline, not a guarantee of the best possible model.
"""


def register(sub):
    """Add the `data`, `train` and `optimizers` commands to the main parser's subparsers."""
    p_data = sub.add_parser(
        "data", formatter_class=_Fmt, help="inspect the cleaned data: head, tail, shape, missing, split",
        description="Look at the cleaned Tox21 table (like pandas head/tail/shape/isna) and the train/val/test "
                    "split. Run `toxion prepare-data` first.")
    data_sub = p_data.add_subparsers(dest="data_command", required=True, metavar="{head,tail,shape,missing,split}")

    for name, func, text in (("head", cmd_data_head, "first N rows"), ("tail", cmd_data_tail, "last N rows")):
        p = data_sub.add_parser(name, formatter_class=_Fmt, help=text, description=f"Show the {text} of the cleaned table.")
        _add_clean_path(p)
        p.add_argument("-n", type=int, default=5, help="number of rows to show")
        p.set_defaults(func=func)

    p = data_sub.add_parser("shape", formatter_class=_Fmt, help="rows x columns",
                            description="Print (rows, columns) of the cleaned table.")
    _add_clean_path(p)
    p.set_defaults(func=cmd_data_shape)

    p = data_sub.add_parser("missing", formatter_class=_Fmt, help="missing values per column",
                            description="Count missing (not tested) labels per column and overall.")
    _add_clean_path(p)
    p.set_defaults(func=cmd_data_missing)

    p = data_sub.add_parser("split", formatter_class=_Fmt, help="show train/val/test sizes for a split config",
                            description="Show what a split configuration produces: sizes, positive rate per split, "
                                        "and (for scaffold) a check that no scaffold is shared between splits.")
    _add_clean_path(p)
    _add_split_args(p)
    p.add_argument("--save", default=None, help="write the split indices to this .npz file")
    p.set_defaults(func=cmd_data_split)

    p = sub.add_parser(
        "train", formatter_class=_Fmt, epilog=TRAIN_EPILOG,
        help="train a baseline and report accuracy, F1, ROC-AUC, R2 on train/val/test",
        description="Train a multitask logistic regression on Morgan fingerprints with the project's own "
                    "optimizers, then report metrics for the train, validation, test and whole dataset. "
                    "Requires: uv sync --extra ml (and --extra notebook for --plot).")
    _add_clean_path(p)
    _add_split_args(p)
    p.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint length")
    p.add_argument("--optimizer", choices=["auto", *OPTIMIZERS], default="auto",
                   help="auto tries sgd, rmsprop, adam, adamw briefly and keeps the lowest validation loss")
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate; if omitted, a per-optimizer default is used (see `toxion optimizers`)")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="weight decay; if omitted: 0.01 for adamw, 0 for the others")
    p.add_argument("--auto-epochs", type=int, default=15, help="epochs per candidate when --optimizer auto")
    p.add_argument("--epochs", type=_epochs_arg, default="auto",
                   help="'auto' (up to 300 with early stopping) or a number = maximum epochs")
    p.add_argument("--patience", type=int, default=15,
                   help="stop after this many epochs without validation-loss improvement; 0 disables early stopping")
    p.add_argument("--batch-size", type=int, default=256, help="molecules per optimizer step")
    p.add_argument("--schedule", choices=["none", "cosine"], default="none",
                   help="learning-rate schedule: cosine = short warmup then cosine decay")
    p.add_argument("--loss", choices=["bce", "weighted", "focal"], default="bce",
                   help="training loss: bce, weighted (positives weighted by n_neg/n_pos per endpoint) or focal")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="focal loss gamma (use 0 or at least 1)")
    p.add_argument("--threshold", type=_threshold_arg, default="auto",
                   help="'auto' (best mean F1 on validation) or a number in (0, 1) for accuracy/precision/recall/F1")
    p.add_argument("--seed", type=int, default=0, help="seed for weight initialization and batch order")
    p.add_argument("--log-every", type=int, default=10, help="print a progress line every N epochs (0 = none)")
    p.add_argument("--quiet", action="store_true", help="hide per-epoch progress lines")
    p.add_argument("--per-task", action="store_true", help="also print the per-endpoint metrics for the test split")
    p.add_argument("--plot", action="store_true", help="save loss, ROC-AUC curves and a train/val/test metric bar chart")
    p.add_argument("--plot-out", default="results/figures/train_curves.png", help="where --plot saves the figure")
    p.add_argument("--show", action="store_true", help="also open the plot in a window (implies --plot)")
    p.add_argument("--save-metrics", default=None, help="write config, epochs and metrics to this JSON file")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("optimizers", formatter_class=_Fmt, help="explain the optimizers used by train",
                       description="List the optimizers with their update rules and how `train` uses them.")
    p.set_defaults(func=cmd_optimizers)