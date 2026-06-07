"""Code generation and execution for AIDE-style iteration.

Generates Python scripts that create custom features or models,
executes them safely, and captures the output metric.
"""

import json
import subprocess
import sys
from pathlib import Path


class CodeGenerator:
    """Generate and execute experiment scripts."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.scripts_dir = self.workspace / ".state" / "scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to workspace into absolute path for script execution."""
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        return str(p.resolve())

    def generate_feature_script(
        self,
        feature_code: str,
        feature_name: str,
        train_path: str,
        target_col: str,
        id_col: str,
    ) -> Path:
        """Generate a self-contained script that builds a custom feature and evaluates it.

        train_path can be relative (resolved against workspace) or absolute.
        """
        resolved_train = self._resolve_path(train_path)
        header = f'''"""Auto-generated feature experiment: {feature_name}"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Load data
train_path = Path("{resolved_train}")
if str(train_path).endswith(".parquet"):
    df = pd.read_parquet(train_path)
else:
    df = pd.read_csv(train_path)

target_col = "{target_col}"
id_col = "{id_col}"

# --- Custom feature code ---
'''

        footer = f'''
# --- End custom feature code ---

# Prepare for training
y = df[target_col].values
X = df.drop(columns=[target_col, id_col], errors="ignore")
X = X.select_dtypes(include=[np.number])

if X.empty:
    print(json.dumps({{"status": "failed", "reason": "No numeric features"}}))
    sys.exit(1)

# Quick CV evaluation
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

task = "classification" if df[target_col].nunique() <= 20 else "regression"
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    if task == "classification":
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, verbosity=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        pred = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import log_loss
        scores.append(log_loss(y_val, pred))
    else:
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbosity=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        pred = model.predict(X_val)
        from sklearn.metrics import mean_squared_error
        scores.append(mean_squared_error(y_val, pred, squared=False))

result = {{
    "status": "completed",
    "cv_mean": float(np.mean(scores)),
    "cv_std": float(np.std(scores)),
    "n_features": X.shape[1],
    "feature_name": "{feature_name}",
}}
print(json.dumps(result))
'''

        script = header + feature_code + footer
        script_path = self.scripts_dir / f"{feature_name}.py"
        script_path.write_text(script)
        return script_path

    def generate_model_script(
        self,
        model_code: str,
        experiment_name: str,
        train_path: str,
        target_col: str,
        id_col: str,
    ) -> Path:
        """Generate a script with custom model/training logic.

        train_path can be relative (resolved against workspace) or absolute.
        """
        resolved_train = self._resolve_path(train_path)
        header = f'''"""Auto-generated model experiment: {experiment_name}"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Load data
train_path = Path("{resolved_train}")
if str(train_path).endswith(".parquet"):
    df = pd.read_parquet(train_path)
else:
    df = pd.read_csv(train_path)

target_col = "{target_col}"
id_col = "{id_col}"

y = df[target_col].values
X = df.drop(columns=[target_col, id_col], errors="ignore")
X = X.select_dtypes(include=[np.number])

# --- Custom model code ---
'''

        footer = f'''
# --- End custom model code ---

# result should be set by custom code as:
# result = {{"status": "completed", "cv_mean": ..., "cv_std": ...}}
if "result" not in dir():
    result = {{"status": "failed", "reason": "Custom code did not set result"}}
print(json.dumps(result))
'''

        script = header + model_code + footer
        script_path = self.scripts_dir / f"{experiment_name}.py"
        script_path.write_text(script)
        return script_path

    def execute(self, script_path: Path, timeout: int = 300) -> dict:
        """Execute a script and return its JSON result."""
        script_path = Path(script_path).resolve()
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace.resolve()),
            )

            if result.returncode != 0:
                return {
                    "status": "failed",
                    "reason": f"Script error: {result.stderr[-500:]}",
                    "script_path": str(script_path),
                }

            # Parse JSON output from last line
            output_lines = result.stdout.strip().split("\n")
            for line in reversed(output_lines):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)

            return {
                "status": "failed",
                "reason": "No JSON output from script",
                "stdout": result.stdout[-500:],
            }

        except subprocess.TimeoutExpired:
            return {"status": "failed", "reason": f"Timeout ({timeout}s)"}
        except json.JSONDecodeError as e:
            return {"status": "failed", "reason": f"Invalid JSON output: {e}"}

    def list_scripts(self) -> list[dict]:
        """List all generated scripts with their results."""
        scripts = []
        for f in sorted(self.scripts_dir.glob("*.py")):
            result_file = f.with_suffix(".json")
            result = None
            if result_file.exists():
                with open(result_file) as rf:
                    result = json.load(rf)
            scripts.append({
                "name": f.stem,
                "path": str(f),
                "result": result,
            })
        return scripts
