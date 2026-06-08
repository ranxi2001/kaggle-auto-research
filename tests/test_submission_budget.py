from pathlib import Path

from kaggle_auto.submission.submitter import SubmissionBudget


def test_reserve_updates_existing_submission_path(tmp_path):
    budget = SubmissionBudget(tmp_path, max_daily=2)
    submission = tmp_path / "submissions" / "sub_a.csv"
    submission.parent.mkdir()
    submission.write_text("ID,prediction\n1,0.1\n", encoding="utf-8")

    budget.reserve(str(submission), 0.12, "first reason")
    budget.reserve(str(submission), 0.34, "updated reason")

    reserved = budget.get_reserved()
    assert len(reserved) == 1
    assert Path(reserved[0]["path"]) == submission.resolve()
    assert reserved[0]["cv_score"] == 0.34
    assert reserved[0]["reason"] == "updated reason"


def test_reserve_keeps_distinct_submission_paths(tmp_path):
    budget = SubmissionBudget(tmp_path, max_daily=2)
    first = tmp_path / "submissions" / "sub_a.csv"
    second = tmp_path / "submissions" / "sub_b.csv"
    first.parent.mkdir()
    first.write_text("ID,prediction\n1,0.1\n", encoding="utf-8")
    second.write_text("ID,prediction\n1,0.2\n", encoding="utf-8")

    budget.reserve(str(first), 0.12, "first")
    budget.reserve(str(second), 0.23, "second")

    reserved = budget.get_reserved()
    assert len(reserved) == 2
    assert {Path(item["path"]).name for item in reserved} == {"sub_a.csv", "sub_b.csv"}
