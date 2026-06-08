import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DRW = ROOT / "workspaces" / "drw-crypto"


def test_drw_next_submit_plan_matches_queue_and_ranking():
    plan = (DRW / "reports" / "next_submit_plan.md").read_text(encoding="utf-8")
    budget = json.loads((DRW / ".state" / "submission_budget.json").read_text(encoding="utf-8"))
    ranking = pd.read_csv(DRW / "reports" / "next_with_random_pool_score.csv")

    queued = budget["reserved"]
    assert len(queued) == 1
    queued_file = Path(queued[0]["path"]).name
    assert queued_file == "sub_anchor_blend_micro_scan.csv"
    assert queued[0]["cv_score"] == 0.12804221734404564

    top = ranking.iloc[0]
    assert top["file"] == queued_file
    assert bool(top["valid"])
    assert top["geometry_score"] == 0.664044220618396

    assert "sub_anchor_blend_micro_scan.csv" in plan
    assert ".\\kar.cmd submit drw-crypto --status" in plan
    assert ".\\kar.cmd submit drw-crypto --flush" in plan
