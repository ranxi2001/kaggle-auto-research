import importlib.util
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).parents[1] / "workspaces" / "rogii-wellbore-geology-prediction"
SCRIPT_PATH = WORKSPACE / "scripts" / "build_kaggle_notebook.py"
SPEC = importlib.util.spec_from_file_location("build_rogii_notebook", SCRIPT_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_generated_notebook_has_hidden_test_contract(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "scripts" / "rogii_baseline.py").write_text(
        (WORKSPACE / "scripts" / "rogii_baseline.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    notebook_path, metadata_path = builder.build(workspace, "example-user")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = "\n".join(cell["source"] for cell in notebook["cells"])

    assert metadata["enable_internet"] is False
    assert metadata["id"] == "example-user/kar-rogii-v004-grouped-residual-beam-ensemble"
    assert metadata["competition_sources"] == ["rogii-wellbore-geology-prediction"]
    assert metadata["code_file"] == "rogii-v004.ipynb"
    assert 'rglob("sample_submission.csv")' in source
    assert 'glob(f"*{HORIZONTAL_SUFFIX}")' in source
    assert "sample_fallback_predictions" in source
    assert "beam fallback for" in source
    assert "/kaggle/working/submission.csv" in source
    assert "public_score_7_159_submission.csv" not in source
    assert "000d7d20" not in source
    assert "@njit(cache=False)" in source

    library_source = notebook["cells"][1]["source"]
    namespace = {}
    exec(compile(library_source, "<ipython-input-1-rogii>", "exec"), namespace)
    assert "_beam_path" in namespace


def test_generated_state_space_notebook_is_target_free_and_fail_closed(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "scripts" / "rogii_state_space.py").write_text(
        (WORKSPACE / "scripts" / "rogii_state_space.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    notebook_path, metadata_path = builder.build_state_space(workspace, "example-user")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    driver = notebook["cells"][2]["source"]

    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is False
    assert metadata["id"] == "example-user/kar-rogii-v006-latent-surface-particle-filter"
    assert metadata["competition_sources"] == ["rogii-wellbore-geology-prediction"]
    assert metadata["code_file"] == "rogii-v006.ipynb"
    assert 'rglob("sample_submission.csv")' in source
    assert '"pf_scale_12_hold_0p2"' in driver
    assert "predict_visible_test(" in driver
    assert "if failures:" in driver
    assert driver.index("if failures:") < driver.index("submission.to_csv")
    assert "/kaggle/working" in driver
    assert "KAGGLE_INPUT_ROOT" in driver
    assert "KAGGLE_WORKING_ROOT" in driver
    assert "KAR_RUNTIME_WORKSPACE" in driver
    assert "public_score_7_159_submission.csv" not in source
    assert "000d7d20" not in source
    assert "kaggle competitions submit" not in source
    assert "@njit(cache=False, nogil=True)" in source

    library_source = notebook["cells"][1]["source"]
    namespace = {}
    exec(compile(library_source, "<ipython-input-1-rogii-v006>", "exec"), namespace)
    assert "particle_candidates" in namespace
