"""CLI entry point for kaggle-auto-research (kar command)."""

import os
import sys
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
                icon = "[green]✓[/green]" if status == "completed" else "[dim]○[/dim]"
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
                icon = "✓" if s == "completed" else "○"
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
