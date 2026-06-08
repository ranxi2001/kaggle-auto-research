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


@app.command("sync-lb")
def sync_lb(name: str = typer.Argument(..., help="Competition workspace name")):
    """Sync Kaggle submission scores into local reports and history."""
    import csv
    import json
    from datetime import datetime

    from kaggle_auto.config import load_config
    from kaggle_auto.submission import ScoreTracker
    from kaggle_auto.workspace import get_workspace

    workspace = get_workspace(name)
    config = load_config(workspace)
    slug = config.competition.name

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "submissions",
        slug,
        "--csv",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print("[red]Failed to fetch submissions from Kaggle.[/red]")
        if result.stderr:
            console.print(result.stderr)
        raise typer.Exit(result.returncode)

    rows = list(csv.DictReader(result.stdout.splitlines()))
    report_rows = []
    for row in rows:
        report_rows.append({
            "ref": row.get("ref", ""),
            "fileName": row.get("fileName", ""),
            "date": row.get("date", ""),
            "description": row.get("description", ""),
            "status": row.get("status", ""),
            "publicScore": row.get("publicScore", ""),
            "privateScore": row.get("privateScore", ""),
        })

    report_path = workspace / "reports" / "lb_sync.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ref", "fileName", "date", "description", "status", "publicScore", "privateScore"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    tracker = ScoreTracker(workspace)
    history = tracker.get_history()
    updated = 0
    by_file = {row["fileName"]: row for row in report_rows if row.get("fileName")}
    for entry in history:
        match = by_file.get(entry.get("file", ""))
        if not match:
            continue
        public_score = match.get("publicScore")
        if public_score not in (None, ""):
            try:
                entry["lb_score"] = float(public_score)
                entry["kaggle_ref"] = match.get("ref", "")
                entry["kaggle_status"] = match.get("status", "")
                entry["kaggle_private_score"] = (
                    float(match["privateScore"]) if match.get("privateScore") not in (None, "") else None
                )
                entry["lb_synced_at"] = datetime.now().isoformat()
                updated += 1
            except ValueError:
                pass

    if updated:
        tracker.history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")

    table = Table(title=f"LB Sync: {name}")
    table.add_column("ref")
    table.add_column("file")
    table.add_column("status")
    table.add_column("public")
    table.add_column("private")
    for row in report_rows:
        table.add_row(
            row.get("ref", ""),
            row.get("fileName", ""),
            row.get("status", ""),
            row.get("publicScore", ""),
            row.get("privateScore", ""),
        )
    console.print(table)
    console.print(f"  Report: {report_path}")
    console.print(f"  History entries updated: {updated}")


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


@app.command("drw-ridge")
def drw_ridge(
    name: str = typer.Argument("drw-crypto", help="DRW workspace name"),
    top_k: int = typer.Option(140, "--top-k", help="Top target-correlation features to use"),
    folds: int = typer.Option(5, "--folds", help="CV splits"),
    cv: str = typer.Option("kfold", "--cv", help="CV strategy: kfold|timeseries"),
    alphas: str = typer.Option("1,10,100,1000,10000", "--alphas", help="Comma-separated Ridge alphas"),
):
    """Train a fast closed-form Ridge candidate for DRW and generate a submission."""
    import json
    import pickle
    import time

    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    from sklearn.model_selection import KFold, TimeSeriesSplit

    from kaggle_auto.workspace import get_workspace
    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter

    workspace = get_workspace(name)
    config = load_config(workspace)
    train_path = workspace / config.data.train
    test_path = workspace / config.data.test
    target_col = config.data.target_column
    alpha_values = [float(a.strip()) for a in alphas.split(",") if a.strip()]
    if not alpha_values:
        console.print("[red]Need at least one alpha.[/red]")
        raise typer.Exit(1)

    console.print("[yellow]Ranking DRW features by target correlation...[/yellow]")
    train_all = pd.read_parquet(train_path)
    test_columns = set(pd.read_parquet(test_path).columns)
    raw_features = [
        col for col in train_all.columns
        if col != target_col and col in test_columns and train_all[col].nunique(dropna=False) > 1
    ]
    corr_to_target = train_all[raw_features].corrwith(train_all[target_col]).abs()
    ranked = corr_to_target.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    selected = ranked.head(min(top_k, len(ranked))).index.tolist()
    public_diverse = [
        "X344", "X598", "X385", "X603", "X674", "X415", "X345", "X137", "X174",
        "X302", "X178", "X532", "X168", "X612", "bid_qty", "ask_qty", "buy_qty",
        "sell_qty", "volume",
    ]
    for col in public_diverse:
        if col in train_all.columns and col in test_columns and col not in selected:
            selected.append(col)
    console.print(f"  selected_features={len(selected)}")

    train = train_all[selected + [target_col]]
    del train_all
    test = pd.read_parquet(test_path, columns=selected)
    y = train[target_col].to_numpy(dtype="float64")
    x = train[selected].to_numpy(dtype="float64")
    x_test = test[selected].to_numpy(dtype="float64")
    del train, test

    def fit_predict_ridge(x_train, y_train, x_valid, x_target, alpha):
        med = np.nanmedian(x_train, axis=0)
        x_train = np.where(np.isnan(x_train), med, x_train)
        x_valid = np.where(np.isnan(x_valid), med, x_valid)
        x_target = np.where(np.isnan(x_target), med, x_target)
        mu = x_train.mean(axis=0)
        sigma = x_train.std(axis=0)
        sigma[sigma == 0] = 1.0
        x_train = (x_train - mu) / sigma
        x_valid = (x_valid - mu) / sigma
        x_target = (x_target - mu) / sigma
        intercept = y_train.mean()
        centered_y = y_train - intercept
        system = x_train.T @ x_train
        system.flat[:: system.shape[0] + 1] += alpha
        coef = np.linalg.solve(system, x_train.T @ centered_y)
        model_state = {
            "median": med,
            "mean": mu,
            "std": sigma,
            "coef": coef,
            "intercept": intercept,
            "alpha": alpha,
        }
        return x_valid @ coef + intercept, x_target @ coef + intercept, model_state

    if cv == "kfold":
        splitter = KFold(n_splits=folds, shuffle=False)
    elif cv == "timeseries":
        splitter = TimeSeriesSplit(n_splits=folds)
    else:
        console.print("[red]--cv must be kfold or timeseries.[/red]")
        raise typer.Exit(1)

    best = None
    started_at = time.time()
    for alpha in alpha_values:
        oof = np.zeros(len(y), dtype="float32")
        test_preds = np.zeros(len(x_test), dtype="float64")
        fold_scores = []
        fold_models = []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x), 1):
            pred, test_pred, model_state = fit_predict_ridge(
                x[train_idx], y[train_idx], x[valid_idx], x_test, alpha
            )
            oof[valid_idx] = pred.astype("float32")
            test_preds += test_pred / folds
            score = float(pearsonr(y[valid_idx], pred)[0])
            fold_scores.append(score)
            fold_models.append(model_state)
            console.print(f"  alpha={alpha:g} fold={fold}/{folds} Pearson={score:.6f}")
        scored_mask = oof != 0
        oof_score = float(pearsonr(y[scored_mask], oof[scored_mask])[0])
        console.print(f"[cyan]alpha={alpha:g} OOF Pearson={oof_score:.6f}[/cyan]")
        if best is None or oof_score > best["score"]:
            best = {
                "alpha": alpha,
                "score": oof_score,
                "fold_scores": fold_scores,
                "models": fold_models,
                "oof": oof.copy(),
                "test_preds": test_preds.astype("float32"),
            }

    assert best is not None
    models_dir = workspace / "models"
    version = 1
    while (models_dir / f"v{version:03d}").exists():
        version += 1
    model_path = models_dir / f"v{version:03d}"
    model_path.mkdir(parents=True, exist_ok=True)

    with open(model_path / "model.pkl", "wb") as f:
        pickle.dump(best["models"], f)
    np.save(model_path / "oof_preds.npy", best["oof"])
    np.save(model_path / "test_preds.npy", best["test_preds"])
    pd.DataFrame({
        "feature": selected,
        "target_corr": ranked.reindex(selected).fillna(0).values,
    }).to_csv(model_path / "importance.csv", index=False)
    (model_path / "cv_scores.json").write_text(json.dumps({
        "fold_scores": best["fold_scores"],
        "mean_score": best["score"],
        "std_score": float(np.std(best["fold_scores"])),
        "metric": "pearson",
        "model_type": "ridge_closed_form_corr_features",
        "features": selected,
        "params": {
            "alpha": best["alpha"],
            "alphas": alpha_values,
            "top_k": top_k,
            "n_features": len(selected),
            "cv_strategy": "time_series_split" if cv == "timeseries" else "kfold_no_shuffle",
            "cv": cv,
            "folds": folds,
        },
        "runtime_sec": time.time() - started_at,
    }, indent=2), encoding="utf-8")

    submitter = Submitter(workspace, config)
    sub_path = submitter.generate_submission(best["test_preds"], model_version=f"v{version:03d}")
    validation = submitter.validate(sub_path)
    console.print(f"[green]Best Ridge OOF Pearson:[/green] {best['score']:.6f}")
    console.print(f"  alpha: {best['alpha']:g}")
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
    method: str = typer.Option("grid", "--method", help="Weight search: grid|optimize"),
    mask_zero_oof: bool = typer.Option(True, "--mask-zero-oof/--no-mask-zero-oof", help="Ignore rows where any model has zero OOF prediction"),
    transform: str = typer.Option("raw", "--transform", help="Prediction transform: raw|zscore|ranknorm|clip001"),
):
    """Build a simple OOF-optimized ensemble for DRW models."""
    import itertools
    import json

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    from scipy.stats import pearsonr
    from scipy.stats import norm, rankdata

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

    eval_mask = np.ones(len(y), dtype=bool)
    if mask_zero_oof:
        for item in loaded:
            eval_mask &= item["oof"] != 0
        if not eval_mask.any():
            console.print("[red]No rows remain after --mask-zero-oof.[/red]")
            raise typer.Exit(1)
        if eval_mask.mean() < 1:
            console.print(
                f"  using common non-zero OOF rows: {int(eval_mask.sum()):,}/{len(eval_mask):,} "
                f"({eval_mask.mean():.1%})"
            )

    y_eval = y[eval_mask]

    def rank_normalize_pair(oof_values: np.ndarray, pred_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = oof_values[eval_mask]
        ranks = rankdata(valid, method="average")
        probs = (ranks - 0.5) / len(valid)
        transformed_oof = np.zeros_like(oof_values, dtype="float64")
        transformed_oof[eval_mask] = norm.ppf(probs)

        sorted_valid = np.sort(valid)
        if (~eval_mask).any():
            idx = np.searchsorted(sorted_valid, oof_values[~eval_mask], side="left")
            probs_missing = np.clip((idx + 0.5) / len(sorted_valid), 1e-6, 1 - 1e-6)
            transformed_oof[~eval_mask] = norm.ppf(probs_missing)

        pred_idx = np.searchsorted(sorted_valid, pred_values, side="left")
        pred_probs = np.clip((pred_idx + 0.5) / len(sorted_valid), 1e-6, 1 - 1e-6)
        return transformed_oof, norm.ppf(pred_probs)

    if transform != "raw":
        for item in loaded:
            oof_values = item["oof"].astype("float64")
            pred_values = item["preds"].astype("float64")
            if transform == "zscore":
                mu = oof_values[eval_mask].mean()
                sigma = oof_values[eval_mask].std()
                sigma = sigma if sigma else 1.0
                item["oof"] = (oof_values - mu) / sigma
                item["preds"] = (pred_values - mu) / sigma
            elif transform == "ranknorm":
                item["oof"], item["preds"] = rank_normalize_pair(oof_values, pred_values)
            elif transform == "clip001":
                lo, hi = np.quantile(oof_values[eval_mask], [0.001, 0.999])
                item["oof"] = np.clip(oof_values, lo, hi)
                item["preds"] = np.clip(pred_values, lo, hi)
            else:
                console.print("[red]--transform must be raw, zscore, ranknorm, or clip001.[/red]")
                raise typer.Exit(1)

    console.print("[yellow]Optimizing ensemble weights...[/yellow]")
    best = None
    n = len(loaded)

    if method == "grid":
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
            blended = sum(weights[i] * loaded[i]["oof"][eval_mask] for i in range(n))
            score = pearsonr(y_eval, blended)[0]
            if best is None or score > best["score"]:
                best = {"score": float(score), "weights": weights}
    elif method == "optimize":
        y_centered = y_eval - y_eval.mean()
        y_norm = np.linalg.norm(y_centered)
        oof_matrix = np.column_stack([item["oof"][eval_mask].astype("float64") for item in loaded])

        def corr(weights: np.ndarray) -> float:
            blended = oof_matrix @ weights
            centered = blended - blended.mean()
            denom = np.linalg.norm(centered) * y_norm
            if denom == 0:
                return -1.0
            return float(centered @ y_centered / denom)

        def objective(weights: np.ndarray) -> float:
            return -corr(weights)

        starts = [np.ones(n, dtype=float) / n]
        for i in range(n):
            point = np.zeros(n, dtype=float)
            point[i] = 1.0
            starts.append(point)

        grid_seed = np.array([item["score"] if item["score"] is not None else 0.0 for item in loaded], dtype=float)
        grid_seed = np.maximum(grid_seed, 0.0)
        if grid_seed.sum() > 0:
            starts.append(grid_seed / grid_seed.sum())

        bounds = [(0.0, 1.0)] * n
        constraints = ({"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},)
        for start in starts:
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            if result.success:
                score = corr(result.x)
                if best is None or score > best["score"]:
                    best = {"score": score, "weights": result.x}
    else:
        console.print("[red]--method must be grid or optimize.[/red]")
        raise typer.Exit(1)

    assert best is not None
    test_preds = sum(best["weights"][i] * loaded[i]["preds"] for i in range(n))
    model_tag = "_".join(item["version"] for item in loaded)
    transform_tag = "" if transform == "raw" else f"_{transform}"
    sub_path = workspace / "submissions" / f"sub_ensemble{transform_tag}_{model_tag}.csv"

    sample = pd.read_csv(workspace / config.data.sample_submission)
    sample[sample.columns[1]] = test_preds
    sample.to_csv(sub_path, index=False)

    meta_path = workspace / "submissions" / f"sub_ensemble{transform_tag}_{model_tag}.json"
    meta = {
        "models": [{"version": item["version"], "cv_score": item["score"]} for item in loaded],
        "weights": {loaded[i]["version"]: float(best["weights"][i]) for i in range(n)},
        "metric": "pearson",
        "transform": transform,
        "oof_pearson": best["score"],
        "oof_mask_rows": int(eval_mask.sum()),
        "oof_mask_fraction": float(eval_mask.mean()),
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


@app.command("drw-tail-ensemble")
def drw_tail_ensemble(
    name: str = typer.Argument("drw-crypto", help="DRW workspace name"),
    models: str = typer.Option("v032,v028,v010,v017,v029,v023", "--models", help="Comma-separated model versions"),
    base_weights: str = typer.Option(
        "0.53696968,0.22322101,0.15869722,0.04722773,0.020876,0.01300836",
        "--base-weights",
        help="Comma-separated base weights matching --models",
    ),
    samples: int = typer.Option(12000, "--samples", help="Random candidate weights per search family"),
    seed: int = typer.Option(47, "--seed", help="Random seed"),
    output_tag: str = typer.Option("tail_cli", "--output-tag", help="Submission filename tag"),
):
    """Build a DRW recency-weighted rank ensemble around a calibrated base vector."""
    import json

    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata, spearmanr

    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter
    from kaggle_auto.workspace import get_workspace

    workspace = get_workspace(name)
    config = load_config(workspace)
    versions = [m.strip() for m in models.split(",") if m.strip()]
    base = np.array([float(x.strip()) for x in base_weights.split(",") if x.strip()], dtype="float64")
    if len(versions) < 2 or len(base) != len(versions):
        console.print("[red]--models and --base-weights must have the same length >= 2.[/red]")
        raise typer.Exit(1)
    base = base / base.sum()

    rng = np.random.default_rng(seed)
    y = pd.read_parquet(workspace / config.data.train, columns=[config.data.target_column])[
        config.data.target_column
    ].to_numpy(dtype="float32")

    first_path = (
        workspace
        / "submissions"
        / "sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv"
    )
    first_preds = pd.read_csv(first_path)["prediction"].to_numpy(dtype="float32") if first_path.exists() else None
    first_rank = rankdata(first_preds) if first_preds is not None else None

    def rank_normalize(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values, method="average")
        return ((ranks - 0.5) / len(ranks) * 2 - 1).astype("float32")

    oof_columns = []
    test_columns = []
    common_mask = np.ones(len(y), dtype=bool)
    for version in versions:
        model_dir = workspace / "models" / version
        oof_path = model_dir / "oof_preds.npy"
        pred_path = model_dir / "test_preds.npy"
        if not (oof_path.exists() and pred_path.exists()):
            console.print(f"[red]Missing artifacts for {version}[/red]")
            raise typer.Exit(1)
        oof = np.load(oof_path)
        preds = np.load(pred_path)
        if len(oof) != len(y):
            console.print(f"[red]OOF length mismatch for {version}[/red]")
            raise typer.Exit(1)
        common_mask &= np.abs(oof) > 1e-12
        oof_columns.append(rank_normalize(oof))
        test_columns.append(rank_normalize(preds))

    oof_matrix = np.vstack(oof_columns).T.astype("float32")
    test_matrix = np.vstack(test_columns).T.astype("float32")
    n_rows = len(y)
    fold_size = n_rows // 6
    segments = {
        "full": (0, n_rows),
        "tail20": (int(n_rows * 0.8), n_rows),
        "tail10": (int(n_rows * 0.9), n_rows),
        "ts_fold5": (fold_size * 5, n_rows),
    }
    segment_data = {}
    for key, (start, end) in segments.items():
        mask = common_mask[start:end]
        x = oof_matrix[start:end][mask]
        target = y[start:end][mask]
        centered_target = target - target.mean()
        segment_data[key] = (
            x - x.mean(axis=0, keepdims=True),
            centered_target,
            float(np.linalg.norm(centered_target)),
        )

    candidate_weights = [base.astype("float32")]
    for scale in (360, 220, 140):
        candidate_weights.append(rng.dirichlet((base + 0.002) * scale, size=samples).astype("float32"))
    gaussian = base + rng.normal(0, 0.018, size=(samples, len(base)))
    gaussian = np.clip(gaussian, 0, 0.65)
    gaussian = gaussian / gaussian.sum(axis=1, keepdims=True)
    candidate_weights.append(gaussian.astype("float32"))
    weights = np.vstack(candidate_weights)

    lower = np.maximum(base - 0.08, 0)
    upper = np.minimum(base + 0.08, 0.70)
    keep = np.ones(len(weights), dtype=bool)
    for i in range(len(base)):
        keep &= (weights[:, i] >= lower[i]) & (weights[:, i] <= upper[i])
    weights = weights[keep]
    if len(weights) == 0:
        console.print("[red]No candidate weights after constraints.[/red]")
        raise typer.Exit(1)

    best_rows = []
    for start in range(0, len(weights), 256):
        batch = weights[start : start + 256]
        values = {}
        for key, (x, target, target_norm) in segment_data.items():
            preds = batch @ x.T
            values[key] = (preds @ target) / (np.linalg.norm(preds, axis=1) * target_norm)
        composite = (
            0.35 * values["ts_fold5"]
            + 0.25 * values["tail10"]
            + 0.20 * values["tail20"]
            + 0.20 * values["full"]
        )
        for local_idx in np.argsort(composite)[-5:]:
            best_rows.append({
                "weights": weights[start + local_idx].astype("float64"),
                "full": float(values["full"][local_idx]),
                "tail20": float(values["tail20"][local_idx]),
                "tail10": float(values["tail10"][local_idx]),
                "ts_fold5": float(values["ts_fold5"][local_idx]),
                "composite": float(composite[local_idx]),
            })

    best_rows = sorted(best_rows, key=lambda row: row["composite"], reverse=True)[:50]
    for row in best_rows:
        test_preds = test_matrix @ row["weights"].astype("float32")
        if first_preds is not None and first_rank is not None:
            test_rank = rankdata(test_preds)
            row["spearman_to_first"] = float(spearmanr(test_preds, first_preds)[0])
            row["mean_rank_delta_to_first"] = float(
                np.mean(np.abs(test_rank - first_rank) / (len(test_rank) - 1))
            )

    best = best_rows[0]
    test_preds = test_matrix @ best["weights"].astype("float32")
    sample = pd.read_csv(workspace / config.data.sample_submission)
    sample[sample.columns[1]] = test_preds

    sub_path = workspace / "submissions" / f"sub_calibrated_{output_tag}.csv"
    sample.to_csv(sub_path, index=False)
    meta = {
        "candidate": output_tag,
        "models": versions,
        "weights": {version: float(weight) for version, weight in zip(versions, best["weights"])},
        "scores": {key: best[key] for key in ["full", "tail20", "tail10", "ts_fold5", "composite"]},
        "spearman_to_first": best.get("spearman_to_first"),
        "mean_rank_delta_to_first": best.get("mean_rank_delta_to_first"),
        "submission": sub_path.name,
        "search": {
            "samples": samples,
            "seed": seed,
            "base_weights": {version: float(weight) for version, weight in zip(versions, base)},
        },
    }
    sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report_rows = []
    for row in best_rows:
        report_rows.append({
            **{key: row[key] for key in ["composite", "full", "tail20", "tail10", "ts_fold5"]},
            **{f"w_{version}": float(weight) for version, weight in zip(versions, row["weights"])},
            "spearman_to_first": row.get("spearman_to_first"),
            "mean_rank_delta_to_first": row.get("mean_rank_delta_to_first"),
        })
    report_path = workspace / "reports" / f"{output_tag}_tail_ensemble_search.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    validation = Submitter(workspace, config).validate(sub_path)
    console.print(f"[green]Tail ensemble composite:[/green] {best['composite']:.6f}")
    console.print(f"  Weights: {meta['weights']}")
    console.print(f"  Submission: {sub_path}")
    console.print(f"  Report: {report_path}")
    console.print(f"  Valid: {'Yes' if validation.is_valid else 'No'}")
    if not validation.is_valid:
        for error in validation.errors:
            console.print(f"    [red]{error}[/red]")


@app.command("drw-anchor-blend")
def drw_anchor_blend(
    name: str = typer.Argument("drw-crypto", help="DRW workspace name"),
    anchor_file: str = typer.Option(
        "sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv",
        "--anchor-file",
        help="Submitted CSV to use as the external LB anchor",
    ),
    failed_file: str = typer.Option(
        "sub_calibrated_tail_cli_full.csv",
        "--failed-file",
        help="Known underperforming submission CSV used as a negative reference",
    ),
    anchor_models: str = typer.Option(
        "v005,v010,v012,v015,v017,v018,v019,v020,v021,v022,v023,v024,v025,v026",
        "--anchor-models",
        help="Comma-separated models used to approximate anchor OOF",
    ),
    groups: str = typer.Option(
        "safe:v016+v017+v031+v032,v032:v032,v017:v017,v016:v016",
        "--groups",
        help="Candidate groups as name:v001+v002,name2:v003",
    ),
    alpha_grid: str = typer.Option(
        "0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.21",
        "--alpha-grid",
        help="Comma-separated blend weights applied to candidate groups",
    ),
    min_spearman: float = typer.Option(0.99, "--min-spearman", help="Minimum Spearman correlation to anchor"),
    max_rank_delta: float = typer.Option(0.035, "--max-rank-delta", help="Maximum mean absolute rank delta to anchor"),
    selection_metric: str = typer.Option(
        "composite",
        "--selection-metric",
        help="Best-candidate metric: composite|utility",
    ),
    failed_threshold: float = typer.Option(
        0.935,
        "--failed-threshold",
        help="Spearman-to-failed level where utility starts applying risk penalty",
    ),
    risk_penalty: float = typer.Option(
        0.60,
        "--risk-penalty",
        help="Penalty multiplier for failed-direction similarity above --failed-threshold",
    ),
    output_tag: str = typer.Option("anchor_blend", "--output-tag", help="Submission filename tag"),
):
    """Build a conservative DRW candidate by small rank blends around a submitted LB anchor."""
    import json

    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, rankdata

    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter
    from kaggle_auto.workspace import get_workspace

    workspace = get_workspace(name)
    config = load_config(workspace)
    if selection_metric not in {"composite", "utility"}:
        console.print("[red]--selection-metric must be composite or utility.[/red]")
        raise typer.Exit(1)
    submissions_dir = workspace / "submissions"
    anchor_path = submissions_dir / anchor_file
    anchor_meta_path = anchor_path.with_suffix(".json")
    failed_path = submissions_dir / failed_file
    if not anchor_path.exists():
        console.print(f"[red]Missing anchor submission:[/red] {anchor_path}")
        raise typer.Exit(1)

    def rank_normalize(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values, method="average").astype("float32")
        return ((ranks - 0.5) / len(ranks) * 2 - 1).astype("float32")

    def parse_versions(text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]

    def load_rank_pair(version: str) -> tuple[np.ndarray, np.ndarray]:
        model_dir = workspace / "models" / version
        oof_path = model_dir / "oof_preds.npy"
        test_path = model_dir / "test_preds.npy"
        if not (oof_path.exists() and test_path.exists()):
            console.print(f"[red]Missing artifacts for {version}[/red]")
            raise typer.Exit(1)
        oof = np.load(oof_path)
        test = np.load(test_path)
        if len(oof) != len(y):
            console.print(f"[red]OOF length mismatch for {version}: {len(oof)} != {len(y)}[/red]")
            raise typer.Exit(1)
        return rank_normalize(oof), rank_normalize(test)

    def parse_groups(text: str) -> dict[str, list[str]]:
        parsed = {}
        for chunk in [part.strip() for part in text.split(",") if part.strip()]:
            if ":" not in chunk:
                console.print(f"[red]Invalid group spec:[/red] {chunk}")
                raise typer.Exit(1)
            group_name, version_text = chunk.split(":", 1)
            versions = [item.strip() for item in version_text.split("+") if item.strip()]
            if not versions:
                console.print(f"[red]Group has no models:[/red] {chunk}")
                raise typer.Exit(1)
            parsed[group_name.strip()] = versions
        return parsed

    def load_anchor_spec() -> tuple[list[str], dict[str, float], str]:
        if anchor_meta_path.exists():
            try:
                meta = json.loads(anchor_meta_path.read_text(encoding="utf-8"))
                meta_models = []
                for item in meta.get("models", []):
                    if isinstance(item, dict) and item.get("version"):
                        meta_models.append(str(item["version"]))
                    elif isinstance(item, str):
                        meta_models.append(item)
                meta_weights = meta.get("weights") if isinstance(meta.get("weights"), dict) else {}
                weights = {
                    version: float(meta_weights[version])
                    for version in meta_models
                    if version in meta_weights and float(meta_weights[version]) > 1e-12
                }
                if weights:
                    total = sum(weights.values())
                    weights = {version: weight / total for version, weight in weights.items()}
                    return list(weights.keys()), weights, f"metadata:{anchor_meta_path.name}"
            except Exception as exc:
                console.print(f"[yellow]Could not read anchor metadata, falling back to --anchor-models: {exc}[/yellow]")

        versions = parse_versions(anchor_models)
        if not versions:
            console.print("[red]Need at least one anchor model.[/red]")
            raise typer.Exit(1)
        weight = 1.0 / len(versions)
        return versions, {version: weight for version in versions}, "equal_weight_cli"

    y = pd.read_parquet(workspace / config.data.train, columns=[config.data.target_column])[
        config.data.target_column
    ].to_numpy(dtype="float32")
    anchor_preds = pd.read_csv(anchor_path)["prediction"].to_numpy(dtype="float32")
    anchor_rank = rank_normalize(anchor_preds)
    failed_rank = None
    if failed_path.exists():
        failed_rank = rank_normalize(pd.read_csv(failed_path)["prediction"].to_numpy(dtype="float32"))

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def cached(version: str) -> tuple[np.ndarray, np.ndarray]:
        if version not in cache:
            cache[version] = load_rank_pair(version)
        return cache[version]

    anchor_versions, anchor_weights, anchor_source = load_anchor_spec()
    anchor_oofs = []
    for version in anchor_versions:
        oof, _ = cached(version)
        anchor_oofs.append(oof * anchor_weights[version])
    anchor_oof = np.sum(anchor_oofs, axis=0).astype("float32")

    group_specs = parse_groups(groups)
    alphas = [float(item.strip()) for item in alpha_grid.split(",") if item.strip()]
    n_rows = len(y)
    segments = {
        "full": (0, n_rows),
        "tail20": (int(n_rows * 0.8), n_rows),
        "tail10": (int(n_rows * 0.9), n_rows),
        "tail5": (int(n_rows * 0.95), n_rows),
    }

    def corr_slice(preds: np.ndarray, start: int, end: int) -> float:
        pred_slice = preds[start:end]
        target_slice = y[start:end]
        mask = np.isfinite(pred_slice)
        return float(pearsonr(target_slice[mask], pred_slice[mask])[0])

    rows = []
    for group_name, versions in group_specs.items():
        group_oofs = []
        group_tests = []
        for version in versions:
            oof, test = cached(version)
            group_oofs.append(oof)
            group_tests.append(test)
        group_oof = np.mean(group_oofs, axis=0).astype("float32")
        group_test = np.mean(group_tests, axis=0).astype("float32")

        for alpha in alphas:
            candidate_oof = ((1 - alpha) * anchor_oof + alpha * group_oof).astype("float32")
            candidate_test = ((1 - alpha) * anchor_rank + alpha * group_test).astype("float32")
            candidate_rank = rank_normalize(candidate_test)
            scores = {key: corr_slice(candidate_oof, start, end) for key, (start, end) in segments.items()}
            scores["ts_fold5"] = scores["tail20"]
            composite = (
                0.45 * scores["full"]
                + 0.20 * scores["tail20"]
                + 0.15 * scores["tail10"]
                + 0.10 * scores["tail5"]
                + 0.10 * scores["ts_fold5"]
            )
            spearman_to_anchor = float(np.corrcoef(candidate_rank, anchor_rank)[0, 1])
            mean_rank_delta = float(np.mean(np.abs(candidate_rank - anchor_rank)) / 2)
            spearman_to_failed = None
            if failed_rank is not None:
                spearman_to_failed = float(np.corrcoef(candidate_rank, failed_rank)[0, 1])
            failed_excess = 0.0
            if spearman_to_failed is not None:
                failed_excess = max(0.0, spearman_to_failed - failed_threshold)
            anchor_shortfall = max(0.0, min_spearman - spearman_to_anchor)
            rank_excess = max(0.0, mean_rank_delta - max_rank_delta)
            utility = composite - risk_penalty * failed_excess - 0.35 * anchor_shortfall - 0.90 * rank_excess
            rows.append({
                "group": group_name,
                "models": "+".join(versions),
                "alpha": alpha,
                **scores,
                "composite": float(composite),
                "utility": float(utility),
                "failed_excess": float(failed_excess),
                "anchor_shortfall": float(anchor_shortfall),
                "rank_delta_excess": float(rank_excess),
                "spearman_to_anchor": spearman_to_anchor,
                "spearman_to_failed": spearman_to_failed,
                "mean_rank_delta_to_anchor": mean_rank_delta,
            })

    report = pd.DataFrame(rows).sort_values(selection_metric, ascending=False)
    safe = report[
        (report["spearman_to_anchor"] >= min_spearman)
        & (report["mean_rank_delta_to_anchor"] <= max_rank_delta)
    ]
    if safe.empty:
        console.print("[red]No candidate passed anchor safety constraints.[/red]")
        report_path = workspace / "reports" / f"{output_tag}_anchor_blend_scan.csv"
        report.to_csv(report_path, index=False)
        console.print(f"  Report: {report_path}")
        raise typer.Exit(1)

    best = safe.sort_values([selection_metric, "spearman_to_anchor"], ascending=[False, False]).iloc[0].to_dict()
    best_versions = group_specs[str(best["group"])]
    best_group_test = np.mean([cached(version)[1] for version in best_versions], axis=0).astype("float32")
    final_test = ((1 - best["alpha"]) * anchor_rank + best["alpha"] * best_group_test).astype("float32")

    sample = pd.read_csv(workspace / config.data.sample_submission)
    sample[sample.columns[1]] = final_test
    sub_path = submissions_dir / f"sub_{output_tag}.csv"
    sample.to_csv(sub_path, index=False)

    meta = {
        "candidate": output_tag,
        "anchor_file": anchor_file,
        "anchor_source": anchor_source,
        "anchor_weights": anchor_weights,
        "failed_reference_file": failed_file if failed_path.exists() else None,
        "group": best["group"],
        "models": best_versions,
        "alpha": float(best["alpha"]),
        "scores": {
            key: float(best[key])
            for key in ["full", "tail20", "tail10", "tail5", "ts_fold5", "composite", "utility"]
        },
        "spearman_to_anchor": float(best["spearman_to_anchor"]),
        "spearman_to_failed": None if pd.isna(best["spearman_to_failed"]) else float(best["spearman_to_failed"]),
        "mean_rank_delta_to_anchor": float(best["mean_rank_delta_to_anchor"]),
        "constraints": {
            "min_spearman": min_spearman,
            "max_rank_delta": max_rank_delta,
            "selection_metric": selection_metric,
            "failed_threshold": failed_threshold,
            "risk_penalty": risk_penalty,
        },
        "submission": sub_path.name,
    }
    sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report_path = workspace / "reports" / f"{output_tag}_anchor_blend_scan.csv"
    report.to_csv(report_path, index=False)

    validation = Submitter(workspace, config).validate(sub_path)
    console.print(f"[green]Anchor blend composite:[/green] {best['composite']:.6f}")
    console.print(f"  Utility: {float(best['utility']):.6f} ({selection_metric})")
    console.print(f"  Group: {best['group']} alpha={float(best['alpha']):.3f}")
    console.print(f"  Spearman to anchor: {float(best['spearman_to_anchor']):.6f}")
    console.print(f"  Submission: {sub_path}")
    console.print(f"  Report: {report_path}")
    console.print(f"  Valid: {'Yes' if validation.is_valid else 'No'}")
    if not validation.is_valid:
        for error in validation.errors:
            console.print(f"    [red]{error}[/red]")


@app.command("drw-compare-submissions")
def drw_compare_submissions(
    name: str = typer.Argument("drw-crypto", help="DRW workspace name"),
    files: str = typer.Option(..., "--files", help="Comma-separated submission CSV names or paths"),
    output_tag: str = typer.Option("submission_compare", "--output-tag", help="Report filename tag"),
):
    """Compare DRW submission candidates by metadata, Spearman correlation, and rank movement."""
    import json

    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter
    from kaggle_auto.workspace import get_workspace

    workspace = get_workspace(name)
    config = load_config(workspace)
    submissions_dir = workspace / "submissions"
    requested = [item.strip() for item in files.split(",") if item.strip()]
    if len(requested) < 2:
        console.print("[red]Need at least two submissions in --files.[/red]")
        raise typer.Exit(1)

    def resolve_submission(value: str) -> Path:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return path
        candidates = [
            workspace / value,
            submissions_dir / value,
            submissions_dir / Path(value).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        console.print(f"[red]Submission not found:[/red] {value}")
        raise typer.Exit(1)

    def rank_normalize(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values, method="average").astype("float64")
        return (ranks - 0.5) / len(ranks) * 2 - 1

    sample = pd.read_csv(workspace / config.data.sample_submission)
    loaded = []
    for value in requested:
        path = resolve_submission(value)
        validation = Submitter(workspace, config).validate(path)
        df = pd.read_csv(path)
        pred = df[df.columns[1]].to_numpy(dtype="float64")
        meta = {}
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        scores = meta.get("scores") if isinstance(meta.get("scores"), dict) else {}
        local_score = None
        score_source = None
        for source, value in [
            ("scores.utility", scores.get("utility")),
            ("scores.composite", scores.get("composite")),
            ("oof_pearson", meta.get("oof_pearson")),
            ("mean_score", meta.get("mean_score")),
        ]:
            if value is not None:
                local_score = float(value)
                score_source = source
                break
        loaded.append({
            "name": path.name,
            "path": path,
            "pred": pred,
            "rank": rank_normalize(pred),
            "valid": validation.is_valid,
            "rows": len(df),
            "id_match": bool((df[df.columns[0]].to_numpy() == sample[sample.columns[0]].to_numpy()).all()),
            "missing": int(df[df.columns[1]].isna().sum()),
            "std": float(np.std(pred)),
            "mean": float(np.mean(pred)),
            "min": float(np.min(pred)),
            "max": float(np.max(pred)),
            "local_score": local_score,
            "score_source": score_source,
            "composite": scores.get("composite"),
            "utility": scores.get("utility"),
            "spearman_to_anchor": meta.get("spearman_to_anchor"),
            "spearman_to_failed": meta.get("spearman_to_failed"),
            "rank_delta_to_anchor": meta.get("mean_rank_delta_to_anchor"),
            "warning": meta.get("warning"),
        })

    summary_rows = []
    for item in loaded:
        summary_rows.append({
            "file": item["name"],
            "valid": item["valid"],
            "rows": item["rows"],
            "id_match": item["id_match"],
            "missing": item["missing"],
            "mean": item["mean"],
            "std": item["std"],
            "min": item["min"],
            "max": item["max"],
            "local_score": item["local_score"],
            "score_source": item["score_source"],
            "composite": item["composite"],
            "utility": item["utility"],
            "spearman_to_anchor": item["spearman_to_anchor"],
            "spearman_to_failed": item["spearman_to_failed"],
            "rank_delta_to_anchor": item["rank_delta_to_anchor"],
            "warning": item["warning"],
        })

    pair_rows = []
    for left_idx, left in enumerate(loaded):
        for right in loaded[left_idx + 1:]:
            pair_rows.append({
                "left": left["name"],
                "right": right["name"],
                "spearman": float(np.corrcoef(left["rank"], right["rank"])[0, 1]),
                "mean_rank_delta": float(np.mean(np.abs(left["rank"] - right["rank"])) / 2),
            })

    report_dir = workspace / "reports"
    summary_path = report_dir / f"{output_tag}_summary.csv"
    pair_path = report_dir / f"{output_tag}_pairs.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)

    table = Table(title=f"Submission Compare: {name}")
    for column in ["file", "valid", "local", "source", "composite", "utility", "spear_failed"]:
        table.add_column(column)
    for row in summary_rows:
        table.add_row(
            str(row["file"]),
            "Y" if row["valid"] else "N",
            "" if row["local_score"] is None else f"{float(row['local_score']):.6f}",
            "" if row["score_source"] is None else str(row["score_source"]),
            "" if row["composite"] is None else f"{float(row['composite']):.6f}",
            "" if row["utility"] is None else f"{float(row['utility']):.6f}",
            "" if row["spearman_to_failed"] is None else f"{float(row['spearman_to_failed']):.6f}",
        )
    console.print(table)
    console.print(f"  Summary: {summary_path}")
    console.print(f"  Pairs: {pair_path}")


@app.command("drw-score-candidates")
def drw_score_candidates(
    name: str = typer.Argument("drw-crypto", help="DRW workspace name"),
    files: str = typer.Option(
        "",
        "--files",
        help="Comma-separated candidate CSV names. Defaults to recent submissions plus known generated candidates.",
    ),
    anchor_file: str = typer.Option(
        "sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv",
        "--anchor-file",
        help="Best real-LB submission used as the positive anchor",
    ),
    failed_files: str = typer.Option(
        "sub_calibrated_tail_cli_full.csv,sub_anchor_blend_utility_scan.csv",
        "--failed-files",
        help="Comma-separated real submissions that underperformed and should be treated as negative directions",
    ),
    output_tag: str = typer.Option("candidate_geometry_score", "--output-tag", help="Report filename tag"),
    json_output: bool = typer.Option(False, "--json-output", help="Also write a strict JSON report"),
):
    """Score DRW candidates against real submission geometry and local metadata."""
    import json

    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter
    from kaggle_auto.workspace import get_workspace

    workspace = get_workspace(name)
    config = load_config(workspace)
    submissions_dir = workspace / "submissions"

    def resolve_submission(value: str) -> Path:
        path = Path(value)
        candidates = [
            path if path.is_absolute() else workspace / path,
            submissions_dir / value,
            submissions_dir / Path(value).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        console.print(f"[red]Submission not found:[/red] {value}")
        raise typer.Exit(1)

    def rank_normalize(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values, method="average").astype("float64")
        return (ranks - 0.5) / len(ranks) * 2 - 1

    def read_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy(dtype="float64")

    def read_meta_score(path: Path) -> tuple[float | None, str | None]:
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            return None, None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None, None
        scores = meta.get("scores") if isinstance(meta.get("scores"), dict) else {}
        for source, value in [
            ("scores.utility", scores.get("utility")),
            ("scores.composite", scores.get("composite")),
            ("oof_pearson", meta.get("oof_pearson")),
            ("mean_score", meta.get("mean_score")),
        ]:
            if value is not None:
                return float(value), source
        return None, None

    anchor_path = resolve_submission(anchor_file)
    failed_paths = [resolve_submission(item.strip()) for item in failed_files.split(",") if item.strip()]
    anchor_ids, anchor_pred = read_pred(anchor_path)
    anchor_rank = rank_normalize(anchor_pred)
    failed_refs = []
    for failed_path in failed_paths:
        ids, pred = read_pred(failed_path)
        if not np.array_equal(anchor_ids, ids):
            console.print(f"[red]ID order mismatch for failed reference:[/red] {failed_path.name}")
            raise typer.Exit(1)
        failed_refs.append((failed_path.name, rank_normalize(pred)))

    if files:
        candidate_paths = [resolve_submission(item.strip()) for item in files.split(",") if item.strip()]
    else:
        patterns = [
            "sub_anchor_blend_*.csv",
            "sub_anti_failed_rank_beta*.csv",
            "sub_calibrated_*.csv",
            "sub_ensemble_ranknorm_*.csv",
        ]
        seen = {}
        for pattern in patterns:
            for path in submissions_dir.glob(pattern):
                seen[path.name] = path
        candidate_paths = sorted(seen.values(), key=lambda p: p.stat().st_mtime, reverse=True)

    validator = Submitter(workspace, config)
    rows = []
    for path in candidate_paths:
        ids, pred = read_pred(path)
        valid = bool(np.array_equal(anchor_ids, ids))
        validation = validator.validate(path) if valid else None
        rank = rank_normalize(pred)
        local_score, score_source = read_meta_score(path)
        spearman_anchor = float(np.corrcoef(anchor_rank, rank)[0, 1])
        rank_delta_anchor = float(np.mean(np.abs(anchor_rank - rank)) / 2)
        failed_spearmans = {
            f"spearman_to_{Path(name).stem[:24]}": float(np.corrcoef(failed_rank, rank)[0, 1])
            for name, failed_rank in failed_refs
        }
        max_failed = max(failed_spearmans.values()) if failed_spearmans else 0.0
        mean_failed = float(np.mean(list(failed_spearmans.values()))) if failed_spearmans else 0.0
        local_bonus = 0.0 if local_score is None else max(0.0, min(1.0, (local_score - 0.123) / 0.030))
        geometry_score = (
            0.50 * spearman_anchor
            + 0.25 * (1.0 - max_failed)
            + 0.15 * (1.0 - rank_delta_anchor)
            + 0.10 * local_bonus
        )
        rows.append({
            "file": path.name,
            "valid": valid and bool(validation and validation.is_valid),
            "rows": len(pred),
            "local_score": local_score,
            "score_source": score_source,
            "spearman_to_anchor": spearman_anchor,
            "max_spearman_to_failed": max_failed,
            "mean_spearman_to_failed": mean_failed,
            "rank_delta_to_anchor": rank_delta_anchor,
            "local_bonus": local_bonus,
            "geometry_score": float(geometry_score),
            **failed_spearmans,
        })

    report = pd.DataFrame(rows).sort_values("geometry_score", ascending=False)
    report_path = workspace / "reports" / f"{output_tag}.csv"
    report.to_csv(report_path, index=False)
    json_path = workspace / "reports" / f"{output_tag}.json"
    if json_output:
        json_rows = report.astype(object).where(pd.notna(report), None).to_dict(orient="records")
        json_path.write_text(json.dumps(json_rows, indent=2, allow_nan=False), encoding="utf-8")

    table = Table(title=f"Candidate Geometry Score: {name}")
    for column in ["file", "local", "anchor", "max_failed", "rank_delta", "score"]:
        table.add_column(column)
    for _, row in report.head(12).iterrows():
        table.add_row(
            str(row["file"]),
            "" if pd.isna(row["local_score"]) else f"{float(row['local_score']):.6f}",
            f"{float(row['spearman_to_anchor']):.6f}",
            f"{float(row['max_spearman_to_failed']):.6f}",
            f"{float(row['rank_delta_to_anchor']):.6f}",
            f"{float(row['geometry_score']):.6f}",
        )
    console.print(table)
    console.print(f"  Report: {report_path}")
    if json_output:
        console.print(f"  JSON: {json_path}")


@app.command("drw-anti-failed")
def drw_anti_failed(
    name: str = typer.Argument("drw-crypto", help="DRW workspace name"),
    anchor_file: str = typer.Option(
        "sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv",
        "--anchor-file",
        help="Public-best submission used as the anchor",
    ),
    failed_file: str = typer.Option(
        "sub_calibrated_tail_cli_full.csv",
        "--failed-file",
        help="Known underperforming submission used as the default negative direction",
    ),
    failed_files: str = typer.Option(
        "",
        "--failed-files",
        help="Optional comma-separated underperforming submissions. Overrides --failed-file.",
    ),
    utility_file: str = typer.Option(
        "sub_anchor_blend_utility_scan.csv",
        "--utility-file",
        help="Optional model-based candidate used as an additional reference",
    ),
    beta_grid: str = typer.Option(
        "0.04,0.06,0.08,0.10,0.12,0.15",
        "--beta-grid",
        help="Comma-separated beta values for rank(anchor + beta * (anchor - failed))",
    ),
    weight_grid: str = typer.Option(
        "",
        "--weight-grid",
        help=(
            "Optional semicolon-separated weights for multiple failed files, "
            "for example '0.08+0.04;0.10+0.06'."
        ),
    ),
    output_tag: str = typer.Option("anti_failed_rank_family", "--output-tag", help="Report filename tag"),
):
    """Generate DRW public-feedback anti-failed rank extrapolation candidates."""
    import json

    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    from kaggle_auto.config import load_config
    from kaggle_auto.submission import Submitter
    from kaggle_auto.workspace import get_workspace

    workspace = get_workspace(name)
    config = load_config(workspace)
    submissions_dir = workspace / "submissions"
    anchor_path = submissions_dir / anchor_file
    failed_names = [item.strip() for item in (failed_files or failed_file).split(",") if item.strip()]
    failed_paths = [submissions_dir / item for item in failed_names]
    utility_path = submissions_dir / utility_file
    if not anchor_path.exists():
        console.print(f"[red]Missing anchor submission:[/red] {anchor_path}")
        raise typer.Exit(1)
    for failed_path in failed_paths:
        if not failed_path.exists():
            console.print(f"[red]Missing failed submission:[/red] {failed_path}")
            raise typer.Exit(1)

    def rank_normalize(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values, method="average").astype("float64")
        return (ranks - 0.5) / len(ranks) * 2 - 1

    def read_submission(path: Path) -> tuple[pd.Series, np.ndarray]:
        df = pd.read_csv(path)
        return df.iloc[:, 0], df.iloc[:, 1].to_numpy(dtype="float64")

    ids, anchor_pred = read_submission(anchor_path)
    failed_refs = []
    for failed_path in failed_paths:
        failed_ids, failed_pred = read_submission(failed_path)
        if not np.array_equal(ids.to_numpy(), failed_ids.to_numpy()):
            console.print(f"[red]Anchor and failed submissions have different ID order:[/red] {failed_path.name}")
            raise typer.Exit(1)
        failed_refs.append((failed_path.name, failed_pred))

    utility_rank = None
    if utility_path.exists():
        utility_ids, utility_pred = read_submission(utility_path)
        if np.array_equal(ids.to_numpy(), utility_ids.to_numpy()):
            utility_rank = rank_normalize(utility_pred)
        else:
            console.print("[yellow]Utility reference skipped because ID order differs.[/yellow]")

    anchor_rank = rank_normalize(anchor_pred)
    failed_ranks = [(name, rank_normalize(pred)) for name, pred in failed_refs]
    if weight_grid:
        weight_specs = []
        for chunk in [item.strip() for item in weight_grid.split(";") if item.strip()]:
            weights = [float(value.strip()) for value in chunk.split("+") if value.strip()]
            if len(weights) != len(failed_ranks):
                console.print(
                    f"[red]Weight spec {chunk!r} has {len(weights)} weights but "
                    f"{len(failed_ranks)} failed files were provided.[/red]"
                )
                raise typer.Exit(1)
            weight_specs.append(weights)
    else:
        betas = [float(item.strip()) for item in beta_grid.split(",") if item.strip()]
        if not betas:
            console.print("[red]Need at least one beta in --beta-grid.[/red]")
            raise typer.Exit(1)
        weight_specs = [[beta] for beta in betas]

    rows = []
    for weights in weight_specs:
        extrapolated = anchor_rank.copy()
        for weight, (_, failed_rank) in zip(weights, failed_ranks):
            extrapolated = extrapolated + weight * (anchor_rank - failed_rank)
        pred = rank_normalize(extrapolated)
        if len(weights) == 1:
            suffix = f"beta{int(round(weights[0] * 1000)):03d}"
        else:
            suffix = "w" + "_".join(f"{int(round(weight * 1000)):03d}" for weight in weights)
        sub_path = submissions_dir / f"sub_anti_failed_rank_{suffix}.csv"
        pd.DataFrame({"ID": ids, "prediction": pred}).to_csv(sub_path, index=False)

        failed_spearmans = {
            f"spearman_to_failed_{idx + 1}": float(np.corrcoef(failed_rank, pred)[0, 1])
            for idx, (_, failed_rank) in enumerate(failed_ranks)
        }
        meta = {
            "method": "anti_failed_extrapolation",
            "mode": "rank",
            "beta": float(weights[0]) if len(weights) == 1 else None,
            "weights": {
                name: float(weight)
                for weight, (name, _) in zip(weights, failed_ranks)
            },
            "anchor_submission": anchor_path.name,
            "failed_submission": failed_ranks[0][0] if len(failed_ranks) == 1 else None,
            "failed_submissions": [name for name, _ in failed_ranks],
            "scores": {},
            "spearman_to_anchor": float(np.corrcoef(anchor_rank, pred)[0, 1]),
            "spearman_to_failed": float(np.corrcoef(failed_ranks[0][1], pred)[0, 1]),
            "max_spearman_to_failed": max(failed_spearmans.values()),
            "spearman_to_utility": None
            if utility_rank is None
            else float(np.corrcoef(utility_rank, pred)[0, 1]),
            "mean_rank_delta_to_anchor": float(np.mean(np.abs(anchor_rank - pred)) / 2),
            "warning": (
                "No OOF score; uses only public-LB feedback geometry. "
                "Diagnostic fallback, not first-choice submit."
            ),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        validation = Submitter(workspace, config).validate(sub_path)
        rows.append({
            "file": sub_path.name,
            **meta,
            **failed_spearmans,
            "valid": validation.is_valid,
            "errors": "; ".join(validation.errors),
        })

    report_path = workspace / "reports" / f"{output_tag}.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)

    table = Table(title=f"Anti-Failed Candidates: {name}")
    for column in ["file", "beta", "valid", "spear_anchor", "spear_failed", "rank_delta"]:
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["file"],
            "" if row["beta"] is None else f"{float(row['beta']):.3f}",
            "Y" if row["valid"] else "N",
            f"{float(row['spearman_to_anchor']):.6f}",
            f"{float(row['max_spearman_to_failed']):.6f}",
            f"{float(row['mean_rank_delta_to_anchor']):.6f}",
        )
    console.print(table)
    console.print(f"  Report: {report_path}")


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
                fit_kwargs["verbose"] = False
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
        console.print(f"  Local date: {budget['today']}")
        console.print(f"  Today: {budget['submitted_today']}/{budget['max_daily']} used")
        console.print(f"  Remaining: [{'green' if budget['remaining_today'] > 0 else 'red'}]{budget['remaining_today']}[/]")
        if budget["remaining_today"] <= 0:
            console.print(f"  Next local reset date: {budget['next_reset_date']}")
        if budget["reserved_queue"] > 0:
            console.print(f"\n  [yellow]Reserve queue ({budget['reserved_queue']}):[/yellow]")
            for r in budget["reserved"]:
                path = Path(r.get("path", ""))
                cv_score = r.get("cv_score")
                cv_text = "?" if cv_score is None else f"{float(cv_score):.6f}"
                console.print(f"    File: {path.name}")
                console.print(f"      CV: {cv_text}")
                console.print(f"      Reserved: {r.get('date_reserved', '-')}")
                console.print(f"      Reason: {r.get('reason', '-')}")
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
        import json

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

        meta_path = file_path.with_suffix(".json")
        cv_score = None
        model_version = ""
        message = f"Manual submit {file_path.name}"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                scores = meta.get("scores") if isinstance(meta.get("scores"), dict) else {}
                cv_score = (
                    scores.get("utility")
                    or scores.get("composite")
                    or meta.get("oof_pearson")
                    or meta.get("mean_score")
                    or meta.get("composite_score")
                )
                versions = []
                for item in meta.get("models", []):
                    if isinstance(item, dict):
                        versions.append(item.get("version", ""))
                    else:
                        versions.append(str(item))
                versions = [v for v in versions if v]
                if versions:
                    model_version = ",".join(versions)
                if cv_score is not None:
                    message = f"Manual submit {file_path.name} CV={float(cv_score):.6f}"
            except Exception as exc:
                console.print(f"[yellow]Could not read metadata {meta_path}: {exc}[/yellow]")

        result = submitter.submit(
            file_path,
            message=message,
            force=force,
            cv_score=float(cv_score) if cv_score is not None else None,
            model_version=model_version,
        )
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
