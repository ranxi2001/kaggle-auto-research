"""CLI entry point for kaggle-auto-research (kar command)."""

import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env from project root
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

app = typer.Typer(
    name="kar",
    help="kaggle-auto-research: AI Agent powered Kaggle competition framework",
)
console = Console()


def _run_kaggle(args: list[str], retries: int = 3) -> subprocess.CompletedProcess:
    """Run Kaggle CLI with small retry budget for flaky network connections."""
    cmd = [sys.executable, "-m", "kaggle", *args]
    result = subprocess.CompletedProcess(cmd, 1)
    for attempt in range(1, retries + 1):
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return result
        if attempt < retries:
            console.print(f"[yellow]Kaggle command failed, retrying ({attempt}/{retries})...[/yellow]")
            time.sleep(2 * attempt)
    return result


@app.command()
def auth():
    """Log in to Kaggle and show credential status."""
    console.print("[yellow]Opening Kaggle login...[/yellow]")
    result = _run_kaggle(["auth", "login"], retries=1)
    if result.returncode != 0:
        console.print("[red]Kaggle login failed.[/red]")
        raise typer.Exit(result.returncode)

    console.print("\n[yellow]Checking Kaggle credentials...[/yellow]")
    status = _run_kaggle(["config", "view"], retries=1)
    if status.returncode == 0:
        console.print("[green]Kaggle auth OK.[/green]")
    else:
        console.print("[red]Kaggle auth check failed.[/red]")
        raise typer.Exit(status.returncode)


@app.command()
def data(name: str = typer.Argument(..., help="Competition workspace name")):
    """Download and extract Kaggle competition data for a workspace."""
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config

    workspace = get_workspace(name)
    config = load_config(workspace)
    raw_dir = workspace / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    slug = config.competition.name
    zip_files = sorted(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if zip_files:
        console.print(f"[dim]Using existing archive:[/dim] {zip_files[0].name}")
    else:
        console.print(f"[yellow]Downloading data:[/yellow] {slug}")
        result = _run_kaggle(["competitions", "download", slug, "-p", str(raw_dir)])
        if result.returncode != 0:
            console.print("[red]Download failed.[/red]")
            raise typer.Exit(result.returncode)

        zip_files = sorted(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        console.print("[dim]No zip file found to extract.[/dim]")
        return

    archive = zip_files[0]
    console.print(f"[yellow]Extracting:[/yellow] {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)

    files = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() != ".zip")
    console.print("[green]Data ready.[/green]")
    for p in files:
        console.print(f"  {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")


@app.command()
def leaderboard(
    name: str = typer.Argument(..., help="Competition workspace name"),
    top: bool = typer.Option(False, "--top", help="Show public leaderboard top rows"),
    page_size: int = typer.Option(20, "--page-size", "-n", help="Rows to request from Kaggle"),
):
    """Show Kaggle submissions or public leaderboard for a workspace."""
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config

    workspace = get_workspace(name)
    config = load_config(workspace)
    slug = config.competition.name

    if top:
        console.print(f"[cyan]Leaderboard top:[/cyan] {slug}")
        result = _run_kaggle([
            "competitions", "leaderboard", slug,
            "--show", "--page-size", str(page_size),
        ])
    else:
        console.print(f"[cyan]Your submissions:[/cyan] {slug}")
        result = _run_kaggle([
            "competitions", "submissions", slug,
            "--page-size", str(page_size),
        ])

    if result.returncode != 0:
        console.print("[red]Failed to fetch leaderboard/submissions.[/red]")
        raise typer.Exit(result.returncode)


@app.command("drw-clean")
def drw_clean(
    name: str = typer.Argument("drw-crypto", help="DRW workspace name"),
    top_k: int = typer.Option(350, "--top-k", help="Keep top-K features by absolute target correlation"),
    n_estimators: int = typer.Option(700, "--n-estimators", help="LightGBM estimators per fold"),
    learning_rate: float = typer.Option(0.025, "--learning-rate", help="LightGBM learning rate"),
    num_leaves: int = typer.Option(31, "--num-leaves", help="LightGBM num leaves"),
    min_child_samples: int = typer.Option(400, "--min-child-samples", help="LightGBM min child samples"),
    reg_alpha: float = typer.Option(2.0, "--reg-alpha", help="LightGBM L1 regularization"),
    reg_lambda: float = typer.Option(8.0, "--reg-lambda", help="LightGBM L2 regularization"),
    tail_frac: float = typer.Option(1.0, "--tail-frac", help="Use only the latest fraction of train rows"),
    corr_threshold: float = typer.Option(0.999, "--corr-threshold", help="Drop feature pairs above this correlation"),
):
    """Train a DRW-specific cleaned LightGBM baseline and generate a submission."""
    import json
    import gc

    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    from sklearn.model_selection import TimeSeriesSplit

    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter

    workspace = get_workspace(name)
    config = load_config(workspace)
    train_path = workspace / config.data.train
    test_path = workspace / config.data.test

    console.print(f"[yellow]Loading raw data:[/yellow] {name}")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    if tail_frac <= 0 or tail_frac > 1:
        console.print("[red]--tail-frac must be in (0, 1].[/red]")
        raise typer.Exit(1)
    if tail_frac < 1:
        start = int(len(train) * (1 - tail_frac))
        train = train.iloc[start:].reset_index(drop=True)
        console.print(f"  using latest {tail_frac:.0%} of train rows: {len(train):,}")

    target_col = config.data.target_column
    y = train[target_col].astype("float32").to_numpy()
    feature_cols = [c for c in train.columns if c != target_col]

    console.print(f"  train={train.shape}, test={test.shape}, raw_features={len(feature_cols)}")

    nunique = train[feature_cols].nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    feature_cols = [c for c in feature_cols if c not in constant_cols]
    console.print(f"  dropped constant: {len(constant_cols)}")

    console.print("  ranking features by target correlation...")
    corr_to_target = train[feature_cols].corrwith(train[target_col]).abs()
    ranked = corr_to_target.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    selected = ranked.head(min(top_k, len(ranked))).index.tolist()
    console.print(f"  selected top target-corr features: {len(selected)}")

    console.print("  dropping near-duplicate correlated features...")
    sample = train[selected].sample(min(80_000, len(train)), random_state=config.model.seed)
    corr_matrix = sample.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_drop = [col for col in upper.columns if (upper[col] > corr_threshold).any()]
    selected = [c for c in selected if c not in corr_drop]
    console.print(f"  dropped correlated: {len(corr_drop)}, final_features={len(selected)}")
    del sample, corr_matrix, upper
    gc.collect()

    X = train[selected].astype("float32")
    X_test = test[selected].astype("float32")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "max_depth": 5,
        "min_child_samples": min_child_samples,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "tail_frac": tail_frac,
        "random_state": config.model.seed,
        "verbosity": -1,
        "n_jobs": -1,
    }

    tscv = TimeSeriesSplit(n_splits=config.model.cv_folds)
    oof = np.zeros(len(X), dtype="float32")
    test_preds = np.zeros(len(X_test), dtype="float32")
    fold_scores = []
    models = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        console.print(f"[yellow]Fold {fold}/{config.model.cv_folds}[/yellow]")
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(X.iloc[va_idx])
        oof[va_idx] = pred
        score = pearsonr(y[va_idx], pred)[0]
        fold_scores.append(float(score))
        test_preds += model.predict(X_test).astype("float32") / config.model.cv_folds
        models.append(model)
        console.print(f"  R2={score:.6f}, best_iter={getattr(model, 'best_iteration_', None)}")

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    console.print(f"[green]CV Pearson:[/green] {mean_score:.6f} +/- {std_score:.6f}")

    models_dir = workspace / "models"
    version = 1
    while (models_dir / f"v{version:03d}").exists():
        version += 1
    model_path = models_dir / f"v{version:03d}"
    model_path.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(model_path / "model.pkl", "wb") as f:
        pickle.dump(models, f)
    np.save(model_path / "oof_preds.npy", oof)
    np.save(model_path / "test_preds.npy", test_preds)
    pd.DataFrame({
        "feature": selected,
        "importance": ranked.reindex(selected).fillna(0).values,
        "target_corr": ranked.reindex(selected).fillna(0).values,
    }).to_csv(
        model_path / "importance.csv", index=False
    )
    with open(model_path / "cv_scores.json", "w") as f:
        json.dump({
            "fold_scores": fold_scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "model_type": "lightgbm_clean",
            "features": selected,
            "params": params,
            "dropped_constant": constant_cols,
            "dropped_correlated": corr_drop,
        }, f, indent=2)

    submitter = Submitter(workspace, config)
    sub_path = submitter.generate_submission(test_preds, model_version=f"v{version:03d}")
    validation = submitter.validate(sub_path)
    console.print(f"  Model: {model_path}")
    console.print(f"  Submission: {sub_path}")
    console.print(f"  Valid: {'Yes' if validation.is_valid else 'No'}")
    if not validation.is_valid:
        for error in validation.errors:
            console.print(f"    [red]{error}[/red]")


@app.command("drw-ensemble")
def drw_ensemble(
    name: str = typer.Argument("drw-crypto", help="DRW workspace name"),
    models: str = typer.Option("v010,v011,v007,v003", "--models", help="Comma-separated model versions"),
    step: int = typer.Option(20, "--step", help="Weight grid denominator, e.g. 20 means 0.05 steps"),
):
    """Build a simple OOF-optimized ensemble for DRW models."""
    import itertools
    import json

    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr

    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter

    workspace = get_workspace(name)
    config = load_config(workspace)
    versions = [m.strip() for m in models.split(",") if m.strip()]
    if len(versions) < 2:
        console.print("[red]Need at least two model versions.[/red]")
        raise typer.Exit(1)

    y = pd.read_parquet(workspace / config.data.train, columns=[config.data.target_column])[
        config.data.target_column
    ].to_numpy()

    loaded = []
    for version in versions:
        model_dir = workspace / "models" / version
        oof_path = model_dir / "oof_preds.npy"
        pred_path = model_dir / "test_preds.npy"
        score_path = model_dir / "cv_scores.json"
        if not (oof_path.exists() and pred_path.exists() and score_path.exists()):
            console.print(f"[red]Missing artifacts for {version}[/red]")
            raise typer.Exit(1)
        oof = np.load(oof_path)
        preds = np.load(pred_path)
        if len(oof) != len(y):
            console.print(f"[red]OOF length mismatch for {version}[/red]")
            raise typer.Exit(1)
        score = json.load(open(score_path)).get("mean_score")
        loaded.append({"version": version, "score": score, "oof": oof, "preds": preds})

    console.print("[yellow]Optimizing ensemble weights...[/yellow]")
    best = None
    n = len(loaded)

    def compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in compositions(total - i, parts - 1):
                yield (i, *rest)

    for raw_weights in compositions(step, n):
        if sum(raw_weights) == 0:
            continue
        weights = np.array(raw_weights, dtype=float) / step
        blended = sum(weights[i] * loaded[i]["oof"] for i in range(n))
        score = pearsonr(y, blended)[0]
        if best is None or score > best["score"]:
            best = {"score": float(score), "weights": weights}

    assert best is not None
    test_preds = sum(best["weights"][i] * loaded[i]["preds"] for i in range(n))
    model_tag = "_".join(item["version"] for item in loaded)
    sub_path = workspace / "submissions" / f"sub_ensemble_{model_tag}.csv"

    sample = pd.read_csv(workspace / config.data.sample_submission)
    sample[sample.columns[1]] = test_preds
    sample.to_csv(sub_path, index=False)

    meta_path = workspace / "submissions" / f"sub_ensemble_{model_tag}.json"
    meta = {
        "models": [{"version": item["version"], "cv_score": item["score"]} for item in loaded],
        "weights": {loaded[i]["version"]: float(best["weights"][i]) for i in range(n)},
        "metric": "pearson",
        "oof_pearson": best["score"],
        "submission": sub_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    validation = Submitter(workspace, config).validate(sub_path)
    console.print(f"[green]Ensemble OOF Pearson:[/green] {best['score']:.6f}")
    console.print(f"  Weights: {meta['weights']}")
    console.print(f"  Submission: {sub_path}")
    console.print(f"  Valid: {'Yes' if validation.is_valid else 'No'}")
    if not validation.is_valid:
        for error in validation.errors:
            console.print(f"    [red]{error}[/red]")


@app.command("drw-public")
def drw_public(
    name: str = typer.Argument("drw-crypto", help="DRW workspace name"),
    model: str = typer.Option("lgbm", "--model", help="Model: lgbm|xgb"),
    n_folds: int = typer.Option(3, "--folds", help="KFold splits"),
    decay: float = typer.Option(0.95, "--decay", help="Time-decay sample weight"),
):
    """Reproduce public DRW 25-feature time-slice baseline locally."""
    import gc
    import json
    import pickle

    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    from sklearn.model_selection import KFold

    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter

    workspace = get_workspace(name)
    config = load_config(workspace)

    public_features = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603",
        "X860", "X674", "X415", "X345", "X137", "X855", "X174", "X302",
        "X178", "X532", "X168", "X612", "bid_qty", "ask_qty", "buy_qty",
        "sell_qty", "volume",
    ]

    train_path = workspace / config.data.train
    test_path = workspace / config.data.test
    target_col = config.data.target_column

    console.print("[yellow]Loading public feature subset...[/yellow]")
    import pyarrow.parquet as pq
    available_train_cols = set(pq.ParquetFile(train_path).schema.names)
    available_test_cols = set(pq.ParquetFile(test_path).schema.names)
    available_features = [
        col for col in public_features
        if col in available_train_cols and col in available_test_cols
    ]
    missing_features = [col for col in public_features if col not in available_features]
    if missing_features:
        console.print(f"[yellow]Missing public features skipped:[/yellow] {missing_features}")
    if not available_features:
        console.print("[red]No public features are available in this data version.[/red]")
        raise typer.Exit(1)

    train = pd.read_parquet(train_path, columns=available_features + [target_col]).reset_index(drop=True)
    test = pd.read_parquet(test_path, columns=available_features).reset_index(drop=True)

    def create_time_decay_weights(n: int, d: float) -> np.ndarray:
        positions = np.arange(n)
        normalized = positions / float(max(n - 1, 1))
        weights = d ** (1.0 - normalized)
        return weights * n / weights.sum()

    if model == "xgb":
        from xgboost import XGBRegressor
        estimator_cls = XGBRegressor
        params = {
            "tree_method": "hist",
            "colsample_bylevel": 0.4778015829774066,
            "colsample_bynode": 0.362764358742407,
            "colsample_bytree": 0.7107423488010493,
            "gamma": 1.7094857725240398,
            "learning_rate": 0.02213323588455387,
            "max_depth": 20,
            "max_leaves": 12,
            "min_child_weight": 16,
            "n_estimators": 1667,
            "subsample": 0.06566669853471274,
            "reg_alpha": 39.352415706891264,
            "reg_lambda": 75.44843704068275,
            "verbosity": 0,
            "random_state": config.model.seed,
            "n_jobs": -1,
        }
    else:
        from lightgbm import LGBMRegressor
        estimator_cls = LGBMRegressor
        params = {
            "boosting_type": "gbdt",
            "colsample_bytree": 0.5625888953382505,
            "learning_rate": 0.029312951475451557,
            "min_child_samples": 63,
            "min_child_weight": 0.11456572852335424,
            "n_estimators": 126,
            "n_jobs": -1,
            "num_leaves": 37,
            "random_state": config.model.seed,
            "reg_alpha": 85.2476527854083,
            "reg_lambda": 99.38305361388907,
            "subsample": 0.450669817684892,
            "verbose": -1,
        }

    n_samples = len(train)
    slices = [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples)},
    ]
    y = train[target_col].to_numpy()
    full_weights = create_time_decay_weights(n_samples, decay)
    kf = KFold(n_splits=n_folds, shuffle=False)

    oof_by_slice = {s["name"]: np.zeros(n_samples, dtype="float32") for s in slices}
    test_by_slice = {s["name"]: np.zeros(len(test), dtype="float32") for s in slices}
    fold_scores = {s["name"]: [] for s in slices}
    models = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train), 1):
        console.print(f"[yellow]Fold {fold}/{n_folds}[/yellow]")
        X_valid = train.iloc[valid_idx][available_features]
        y_valid = y[valid_idx]

        for slice_spec in slices:
            slice_name = slice_spec["name"]
            cutoff = slice_spec["cutoff"]
            rel_idx = train_idx[train_idx >= cutoff] - cutoff
            if len(rel_idx) == 0:
                continue

            subset = train.iloc[cutoff:].reset_index(drop=True)
            X_train = subset.iloc[rel_idx][available_features]
            y_train = subset.iloc[rel_idx][target_col].to_numpy()
            if cutoff == 0:
                sample_weight = full_weights[train_idx]
            else:
                sample_weight = create_time_decay_weights(len(subset), decay)[rel_idx]

            fitted = estimator_cls(**params)
            fit_kwargs = {"sample_weight": sample_weight}
            if model == "lgbm":
                fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            else:
                fit_kwargs["eval_set"] = [(X_valid.to_numpy(), y_valid)]
                X_train = X_train.to_numpy()

            fitted.fit(X_train, y_train, **fit_kwargs)
            models.append({"fold": fold, "slice": slice_name, "model": fitted})

            mask = valid_idx >= cutoff
            if mask.any():
                idxs = valid_idx[mask]
                pred_input = train.iloc[idxs][available_features]
                if model == "xgb":
                    pred_input = pred_input.to_numpy()
                oof_by_slice[slice_name][idxs] = fitted.predict(pred_input)
            if cutoff > 0 and (~mask).any():
                oof_by_slice[slice_name][valid_idx[~mask]] = oof_by_slice["full_data"][valid_idx[~mask]]

            test_input = test[available_features]
            if model == "xgb":
                test_input = test_input.to_numpy()
            test_by_slice[slice_name] += fitted.predict(test_input).astype("float32") / n_folds
            score = pearsonr(y, oof_by_slice[slice_name])[0]
            fold_scores[slice_name].append(float(score))
            console.print(f"  {slice_name}: running Pearson={score:.6f}")
            gc.collect()

    slice_scores = {name: float(pearsonr(y, preds)[0]) for name, preds in oof_by_slice.items()}
    simple_oof = np.mean(list(oof_by_slice.values()), axis=0)
    simple_test = np.mean(list(test_by_slice.values()), axis=0)
    simple_score = float(pearsonr(y, simple_oof)[0])

    positive_total = sum(max(score, 0.0) for score in slice_scores.values())
    if positive_total > 0:
        weights = {name: max(score, 0.0) / positive_total for name, score in slice_scores.items()}
    else:
        weights = {name: 1 / len(slice_scores) for name in slice_scores}
    weighted_oof = sum(weights[name] * oof_by_slice[name] for name in weights)
    weighted_test = sum(weights[name] * test_by_slice[name] for name in weights)
    weighted_score = float(pearsonr(y, weighted_oof)[0])

    if weighted_score >= simple_score:
        final_oof = weighted_oof
        final_test = weighted_test
        final_score = weighted_score
        ensemble_mode = "weighted_slices"
    else:
        final_oof = simple_oof
        final_test = simple_test
        final_score = simple_score
        ensemble_mode = "simple_slices"

    models_dir = workspace / "models"
    version = 1
    while (models_dir / f"v{version:03d}").exists():
        version += 1
    model_path = models_dir / f"v{version:03d}"
    model_path.mkdir(parents=True, exist_ok=True)

    with open(model_path / "model.pkl", "wb") as f:
        pickle.dump(models, f)
    np.save(model_path / "oof_preds.npy", final_oof)
    np.save(model_path / "test_preds.npy", final_test)
    pd.DataFrame({"feature": available_features, "importance": 1.0}).to_csv(
        model_path / "importance.csv", index=False
    )
    with open(model_path / "cv_scores.json", "w") as f:
        json.dump({
            "fold_scores": list(slice_scores.values()),
            "mean_score": final_score,
            "std_score": float(np.std(list(slice_scores.values()))),
            "metric": "pearson",
            "model_type": f"public_{model}",
            "features": available_features,
            "missing_public_features": missing_features,
            "params": params,
            "slice_scores": slice_scores,
            "slice_weights": weights,
            "ensemble_mode": ensemble_mode,
            "decay": decay,
            "n_folds": n_folds,
        }, f, indent=2)

    submitter = Submitter(workspace, config)
    sub_path = submitter.generate_submission(final_test, model_version=f"v{version:03d}")
    validation = submitter.validate(sub_path)

    console.print(f"[green]Public baseline Pearson:[/green] {final_score:.6f} ({ensemble_mode})")
    console.print(f"  Slice scores: {slice_scores}")
    console.print(f"  Weights: {weights}")
    console.print(f"  Model: {model_path}")
    console.print(f"  Submission: {sub_path}")
    console.print(f"  Valid: {'Yes' if validation.is_valid else 'No'}")
    if not validation.is_valid:
        for error in validation.errors:
            console.print(f"    [red]{error}[/red]")


@app.command()
def init(
    name: str = typer.Argument(..., help="Competition name (used as workspace directory)"),
    type: str = typer.Option("tabular", "--type", "-t", help="Competition type: tabular|crypto|llm"),
    url: str = typer.Option("", "--url", "-u", help="Kaggle competition URL"),
    metric: str = typer.Option("", "--metric", "-m", help="Evaluation metric override"),
):
    """Initialize a new competition workspace."""
    from kaggle_auto.workspace import init_workspace

    try:
        path = init_workspace(name, competition_type=type, url=url, metric=metric)
        console.print(f"[green]Workspace created:[/green] {path}")
        console.print(f"[dim]Type: {type} | Metric: {metric or 'default'}[/dim]")
        console.print("\nNext steps:")
        console.print(f"  1. Download data: kaggle competitions download {name} -p {path}/data/raw/")
        console.print(f"  2. Run: kar research {name}")
    except FileExistsError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def ls():
    """List all competition workspaces."""
    from kaggle_auto.workspace import list_workspaces
    from kaggle_auto.config import load_config

    workspaces = list_workspaces()
    if not workspaces:
        console.print("[dim]No workspaces found. Run 'kar init' to create one.[/dim]")
        return

    table = Table(title="Competition Workspaces")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Metric", style="yellow")
    table.add_column("Status")

    for ws in workspaces:
        config = load_config(ws)
        state_file = ws / ".state" / "pipeline_state.json"
        status = "new"
        if state_file.exists():
            import json
            state = json.loads(state_file.read_text())
            stages = state.get("stages_completed", [])
            status = f"{len(stages)} stages done"

        table.add_row(
            config.competition.name,
            config.competition.type,
            config.competition.metric,
            status,
        )

    console.print(table)


@app.command()
def research(name: str = typer.Argument(..., help="Competition workspace name")):
    """Run competition research and analysis."""
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.pipeline.stages import run_research

    workspace = get_workspace(name)
    config = load_config(workspace)

    console.print(f"[yellow]Researching:[/yellow] {name}")
    result = run_research(workspace, config)

    if result["status"] == "completed":
        console.print(f"[green]Done![/green] Report: {result['report_path']}")
        console.print(f"  Notebooks found: {result['notebooks_found']}")
    else:
        console.print(f"[red]Failed:[/red] {result.get('reason', 'unknown')}")


@app.command()
def eda(
    name: str = typer.Argument(..., help="Competition workspace name"),
    features_only: bool = typer.Option(False, "--features-only", help="Skip EDA, only generate features"),
):
    """Run EDA and/or feature engineering."""
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.pipeline.stages import run_eda, run_features

    workspace = get_workspace(name)
    config = load_config(workspace)

    if not features_only:
        console.print(f"[yellow]Running EDA for:[/yellow] {name}")
        result = run_eda(workspace, config)
        if result["status"] == "completed":
            console.print(f"[green]EDA done![/green] {result['n_rows']:,} rows x {result['n_cols']} cols")
            console.print(f"  Report: {result['report_path']}")
        else:
            console.print(f"[red]EDA failed:[/red] {result.get('reason')}")
            return

    console.print(f"[yellow]Generating features...[/yellow]")
    feat_result = run_features(workspace, config)
    if feat_result["status"] == "completed":
        console.print(f"[green]Features done![/green] +{feat_result['new_columns']} columns")
        console.print(f"  Built: {', '.join(feat_result['features_built'])}")
        console.print(f"  Version: {feat_result['version']}")
    else:
        console.print(f"[dim]Features skipped:[/dim] {feat_result.get('reason')}")


@app.command()
def train(
    name: str = typer.Argument(..., help="Competition workspace name"),
    model: str = typer.Option("", "--model", "-m", help="Model type override"),
    trials: int = typer.Option(0, "--trials", "-n", help="Optuna trial count (0=no tuning)"),
):
    """Train models with optional hyperparameter tuning."""
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.pipeline.stages import run_train

    workspace = get_workspace(name)
    config = load_config(workspace)

    console.print(f"[yellow]Training for:[/yellow] {name}")
    result = run_train(workspace, config)

    if result["status"] == "completed":
        console.print(f"[green]Training done![/green]")
        console.print(f"  CV Score: {result['cv_mean']:.6f} (+/- {result['cv_std']:.6f})")
        console.print(f"  Folds: {result['fold_scores']}")
        console.print(f"  Model: {result['model_version']} at {result['model_path']}")

        if trials > 0:
            console.print(f"\n[yellow]Tuning with {trials} trials...[/yellow]")
            _run_tuning(workspace, config, trials)
    else:
        console.print(f"[red]Training failed:[/red] {result.get('reason')}")


def _run_tuning(workspace: Path, config, trials: int):
    """Run Optuna hyperparameter tuning."""
    import numpy as np
    import pandas as pd
    from kaggle_auto.models import LightGBMModel
    from kaggle_auto.tuning import OptunaTuner
    from kaggle_auto.utils.paths import get_latest_features

    feat_path = get_latest_features(workspace)
    if feat_path:
        df = pd.read_parquet(feat_path)
    else:
        train_path = workspace / config.data.train
        df = pd.read_csv(train_path)

    target_col = config.data.target_column
    y = df[target_col].values
    X = df.drop(columns=[target_col, config.data.id_column], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    task = "classification" if df[target_col].nunique() <= 20 else "regression"
    direction = config.competition.metric_direction

    tuner = OptunaTuner(
        model_cls=LightGBMModel,
        task=task,
        cv_strategy=config.model.cv_strategy,
        n_splits=config.model.cv_folds,
        direction=direction,
    )

    result = tuner.tune(X, y, n_trials=trials)
    console.print(f"[green]Tuning done![/green] Best score: {result['best_score']:.6f}")
    console.print(f"  Best params: {result['best_params']}")

    tuner.save_results(workspace / "models" / "tuning")


@app.command()
def submit(
    name: str = typer.Argument(..., help="Competition workspace name"),
    force: bool = typer.Option(False, "--force", help="Skip threshold check (still respects budget)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only, don't submit"),
    history: bool = typer.Option(False, "--history", help="Show submission history"),
    status: bool = typer.Option(False, "--status", help="Show budget and queue status"),
    flush: bool = typer.Option(False, "--flush", help="Submit best from reserve queue"),
    file: str = typer.Option("", "--file", "-f", help="Submit a specific file"),
):
    """Submit predictions with budget protection."""
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter, ScoreTracker

    workspace = get_workspace(name)
    config = load_config(workspace)
    submitter = Submitter(workspace, config)

    # Status view
    if status:
        budget = submitter.status()
        console.print(f"[cyan]Submission Budget:[/cyan] {name}")
        console.print(f"  Today: {budget['submitted_today']}/{budget['max_daily']} used")
        console.print(f"  Remaining: [{'green' if budget['remaining_today'] > 0 else 'red'}]{budget['remaining_today']}[/]")
        if budget["reserved_queue"] > 0:
            console.print(f"\n  [yellow]Reserve queue ({budget['reserved_queue']}):[/yellow]")
            for r in budget["reserved"]:
                console.print(f"    CV={r.get('cv_score', '?'):.4f} | {r['reason'][:60]}")
        return

    # Flush reserved
    if flush:
        results = submitter.submit_reserved(n=1)
        if not results:
            console.print("[dim]No reserved submissions to flush.[/dim]")
        for r in results:
            if r.get("success"):
                console.print(f"[green]Submitted![/green] Remaining: {r.get('remaining_today', '?')}")
            else:
                console.print(f"[red]Failed:[/red] {r.get('errors', ['unknown'])}")
        return

    # History view
    if history:
        tracker = ScoreTracker(workspace)
        entries = tracker.get_history()
        if not entries:
            console.print("[dim]No submissions yet.[/dim]")
            return

        table = Table(title="Submission History")
        table.add_column("ID")
        table.add_column("Time")
        table.add_column("CV Score")
        table.add_column("LB Score")
        table.add_column("Model")

        for e in entries:
            table.add_row(
                e["id"],
                e["timestamp"][:16],
                f"{e['cv_score']:.6f}" if e.get("cv_score") else "-",
                f"{e['lb_score']:.6f}" if e.get("lb_score") else "-",
                e.get("model_version", "-"),
            )
        console.print(table)
        return

    # Submit specific file
    if file:
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = workspace / file_path
        if not file_path.exists():
            console.print(f"[red]File not found:[/red] {file_path}")
            raise typer.Exit(1)

        # Show budget before submitting
        budget = submitter.status()
        console.print(f"  Budget: {budget['remaining_today']}/{budget['max_daily']} remaining")

        if dry_run:
            validation = submitter.validate(file_path)
            console.print(f"  Valid: {'Yes' if validation.is_valid else 'No'}")
            if not validation.is_valid:
                for e in validation.errors:
                    console.print(f"    [red]{e}[/red]")
            return

        result = submitter.submit(file_path, message="Manual submit", force=force)
        if result.get("success"):
            console.print(f"[green]Submitted![/green] Remaining: {result.get('remaining_today', '?')}")
        elif result.get("queued"):
            console.print(f"[yellow]Queued:[/yellow] {result['errors'][0]}")
        else:
            console.print(f"[red]Failed:[/red] {result.get('errors', ['unknown'])}")
        return

    # Default: show status and instructions
    budget = submitter.status()
    console.print(f"[cyan]Submission Status:[/cyan] {name}")
    console.print(f"  Budget: {budget['remaining_today']}/{budget['max_daily']} remaining today")
    if budget["reserved_queue"] > 0:
        console.print(f"  Reserve queue: {budget['reserved_queue']} candidates waiting")
    console.print(f"\n  Submit a file:  kar submit {name} -f submissions/sub_007.csv")
    console.print(f"  From queue:     kar submit {name} --flush")
    console.print(f"  Check status:   kar submit {name} --status")


@app.command()
def pipeline(
    name: str = typer.Argument(..., help="Competition workspace name"),
    full: bool = typer.Option(False, "--full", help="Run all stages from scratch"),
    from_stage: str = typer.Option("", "--from", help="Resume from specific stage"),
    iterate: int = typer.Option(0, "--iterate", "-i", help="Run N improvement iterations"),
):
    """Run the competition pipeline."""
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.pipeline import PipelineRunner
    from kaggle_auto.pipeline.stages import register_all_stages

    register_all_stages()

    workspace = get_workspace(name)
    runner = PipelineRunner(workspace)

    if iterate > 0:
        console.print(f"[yellow]Running {iterate} iterations for:[/yellow] {name}")
        result = runner.iterate(iterate)
        console.print(f"[green]Done![/green] {result['iterations_run']} iterations completed")
        if result.get("best_score"):
            console.print(f"  Best score: {result['best_score']:.6f}")
        console.print(f"\n{result['tree_summary']}")
        if result.get("idea_pool_summary"):
            console.print(f"\n{result['idea_pool_summary']}")
        if result.get("recommendations"):
            console.print("\n[yellow]Next recommendations:[/yellow]")
            for r in result["recommendations"]:
                console.print(f"  → {r}")
    else:
        console.print(f"[yellow]Running pipeline for:[/yellow] {name}")
        if full:
            console.print("[dim]Full run from scratch[/dim]")

        result = runner.run(from_stage=from_stage or None, full=full)

        if result.get("status") == "all_stages_completed":
            console.print("  [green]All stages already completed[/green]")
        elif result.get("error"):
            console.print(f"  [red]Error:[/red] {result['error']}")
        else:
            for stage_name, stage_result in result.items():
                if not isinstance(stage_result, dict):
                    continue
                status = stage_result.get("status", "unknown")
                icon = "[green]OK[/green]" if status == "completed" else "[dim]--[/dim]"
                console.print(f"  {icon} {stage_name}: {status}")


@app.command()
def status(name: str = typer.Argument(..., help="Competition workspace name")):
    """Show pipeline status and experiment tree."""
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.pipeline import PipelineRunner
    from kaggle_auto.pipeline.stages import register_all_stages

    register_all_stages()
    workspace = get_workspace(name)
    runner = PipelineRunner(workspace)

    status = runner.get_status()
    console.print(f"[cyan]Pipeline Status:[/cyan] {name}")
    console.print(f"  Completed stages: {status['completed_stages']}")
    console.print(f"  Total experiments: {status['total_experiments']}")

    if status.get("best_node"):
        console.print(f"  Best score: {status['best_node'].metric_value}")

    if status["tree_summary"]:
        console.print(f"\n{status['tree_summary']}")


@app.command()
def ensemble(
    name: str = typer.Argument(..., help="Competition workspace name"),
    top_n: int = typer.Option(3, "--top", "-n", help="Number of top models to ensemble"),
):
    """Build an optimized ensemble from top models."""
    import numpy as np
    import pandas as pd
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.pipeline.ensemble_builder import EnsembleBuilder

    workspace = get_workspace(name)
    config = load_config(workspace)

    # Load target
    train_path = workspace / config.data.train
    if str(train_path).endswith(".parquet"):
        train_df = pd.read_parquet(train_path, columns=[config.data.target_column])
    else:
        train_df = pd.read_csv(train_path, usecols=[config.data.target_column])
    target = train_df[config.data.target_column].values

    builder = EnsembleBuilder(workspace)
    # Internal CV uses model's native metric (logloss/RMSE) which is always minimized
    result = builder.build_ensemble(target, top_n=top_n, minimize=True)

    if result["status"] != "completed":
        console.print(f"[red]Ensemble failed:[/red] {result.get('reason', 'unknown')}")
        return

    console.print(f"[green]Ensemble built![/green]")
    console.print(f"  Models used: {result['models_used']}")
    console.print(f"  Weights: {[f'{w:.3f}' for w in result['weights']]}")
    console.print(f"  Ensemble score: {result['ensemble_score']:.6f}")
    console.print(f"  Best single:    {result['best_single_score']:.6f}")
    improvement = result['improvement']
    if improvement > 0:
        console.print(f"  [green]Improvement: {improvement:.6f}[/green]")
    else:
        console.print(f"  [dim]No improvement over best single model[/dim]")


@app.command()
def analyze(name: str = typer.Argument(..., help="Competition workspace name")):
    """Analyze model performance and get recommendations."""
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.pipeline.analyzer import IterationAnalyzer

    workspace = get_workspace(name)
    analyzer = IterationAnalyzer(workspace)

    console.print(f"[cyan]Analysis:[/cyan] {name}\n")

    # Model comparison
    comparisons = analyzer.compare_models()
    if comparisons:
        table = Table(title="Model Comparison")
        table.add_column("Version")
        table.add_column("CV Score")
        table.add_column("Patch")
        table.add_column("Model")
        for c in comparisons:
            table.add_row(
                c["version"],
                f"{c['cv_mean']:.6f} ± {c['cv_std']:.6f}",
                c["patch"],
                c["model_type"],
            )
        console.print(table)

    # Recommendations
    recs = analyzer.get_recommendations()
    console.print("\n[yellow]Recommendations:[/yellow]")
    for r in recs:
        console.print(f"  → {r}")


@app.command()
def improve(
    name: str = typer.Argument(..., help="Competition workspace name"),
    rounds: int = typer.Option(10, "--rounds", "-n", help="Max improvement rounds"),
):
    """Full intelligent improvement loop: iterate → analyze → ensemble → submit.

    Combines tree-search iteration, idea pool, ensemble building, and analysis
    into a single automated workflow.
    """
    import numpy as np
    import pandas as pd
    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.pipeline import PipelineRunner, IdeaPool, IterationAnalyzer
    from kaggle_auto.pipeline.stages import register_all_stages
    from kaggle_auto.pipeline.ensemble_builder import EnsembleBuilder

    register_all_stages()
    workspace = get_workspace(name)
    config = load_config(workspace)
    runner = PipelineRunner(workspace)

    console.print(f"[bold cyan]Improvement Loop:[/bold cyan] {name}")
    console.print(f"  Target metric: {config.competition.metric} ({config.competition.metric_direction})")
    console.print(f"  Max rounds: {rounds}\n")

    # Phase 1: Run pipeline if not done
    status = runner.get_status()
    if "train" not in status["completed_stages"]:
        console.print("[yellow]Phase 1:[/yellow] Running baseline pipeline...")
        result = runner.run(full=True)
        for stage_name, stage_result in result.items():
            if isinstance(stage_result, dict):
                s = stage_result.get("status", "unknown")
                icon = "OK" if s == "completed" else "--"
                console.print(f"  {icon} {stage_name}: {s}")
        console.print()

    # Phase 2: Tree-search iterations
    console.print(f"[yellow]Phase 2:[/yellow] Running {rounds} iterations...")
    iter_result = runner.iterate(rounds)
    console.print(f"  Completed: {iter_result['iterations_run']} iterations")
    if iter_result.get("best_score"):
        console.print(f"  Best CV: {iter_result['best_score']:.6f}")
    console.print()

    # Phase 3: Ensemble
    console.print("[yellow]Phase 3:[/yellow] Building ensemble...")
    train_path = workspace / config.data.train
    if str(train_path).endswith(".parquet"):
        train_df = pd.read_parquet(train_path, columns=[config.data.target_column])
    else:
        train_df = pd.read_csv(train_path, usecols=[config.data.target_column])
    target = train_df[config.data.target_column].values

    builder = EnsembleBuilder(workspace)
    ens_result = builder.build_ensemble(target, top_n=5, minimize=True)
    if ens_result["status"] == "completed":
        console.print(f"  Ensemble score: {ens_result['ensemble_score']:.6f}")
        console.print(f"  Models: {ens_result['models_used']}")
        if ens_result["improvement"] > 0:
            console.print(f"  [green]Improvement: +{ens_result['improvement']:.6f}[/green]")
    console.print()

    # Phase 4: Analysis and recommendations
    console.print("[yellow]Phase 4:[/yellow] Analysis")
    analyzer = IterationAnalyzer(workspace)
    recs = analyzer.get_recommendations()
    for r in recs:
        console.print(f"  → {r}")

    # Phase 5: Idea pool status
    pool = IdeaPool(workspace)
    untried = pool.get_next(3)
    if untried:
        console.print(f"\n[yellow]Next ideas to try:[/yellow]")
        for idea in untried:
            console.print(f"  [{idea.priority:.1f}] {idea.title}")

    console.print(f"\n[bold green]Done![/bold green] Tree summary:")
    console.print(iter_result["tree_summary"])


if __name__ == "__main__":
    app()
