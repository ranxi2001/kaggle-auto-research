from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from kaggle_auto.submission.submitter import SubmissionBudget, Submitter


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


def test_api_failure_does_not_consume_budget(tmp_path):
    submission = tmp_path / "submissions" / "submission.csv"
    submission.parent.mkdir()
    submission.write_text("id,prediction\n1,0.5\n", encoding="utf-8")
    sample_submission = tmp_path / "sample_submission.csv"
    sample_submission.write_text("id,prediction\n1,0.0\n", encoding="utf-8")

    config = SimpleNamespace(
        submission=SimpleNamespace(max_daily=1, best_threshold=0.0),
        data=SimpleNamespace(sample_submission="sample_submission.csv"),
        competition=SimpleNamespace(name="skill-lift", metric_direction="maximize"),
    )
    submitter = Submitter(tmp_path, config)

    class FailingAPI:
        def submit(self, slug, path, message):
            raise RuntimeError("403 rules not accepted")

    submitter.api = FailingAPI()
    result = submitter.submit(submission, cv_score=0.5, force=True)

    assert not result["success"]
    assert result["errors"] == ["403 rules not accepted"]
    assert result["remaining_today"] == 1
    assert submitter.budget.today_count() == 0


def test_writeup_mode_does_not_call_submission_api(tmp_path):
    submission = tmp_path / "submissions" / "skill.zip"
    submission.parent.mkdir()
    with ZipFile(submission, "w") as archive:
        archive.writestr(
            "skills/example/SKILL.md",
            "---\nname: example\ndescription: Example skill.\n---\n",
        )

    config = SimpleNamespace(
        submission=SimpleNamespace(
            max_daily=1,
            best_threshold=0.0,
            mode="writeup",
            format="skill_zip",
        ),
        data=SimpleNamespace(sample_submission=""),
        competition=SimpleNamespace(name="skill-lift", metric_direction="maximize"),
    )
    submitter = Submitter(tmp_path, config)

    class UnexpectedAPI:
        def submit(self, slug, path, message):
            raise AssertionError("writeup mode must not call the competition submission API")

    submitter.api = UnexpectedAPI()
    result = submitter.submit(submission, cv_score=0.5, force=True)

    assert not result["success"]
    assert result["writeup_required"]
    assert result["writeup_url"] == "https://www.kaggle.com/competitions/skill-lift/writeups"
    assert result["remaining_today"] == 1
    assert submitter.budget.today_count() == 0


def test_maximize_threshold_compares_against_highest_prior_score(tmp_path):
    submission = tmp_path / "submissions" / "submission.csv"
    submission.parent.mkdir()
    submission.write_text("id,prediction\n1,0.5\n", encoding="utf-8")
    sample_submission = tmp_path / "sample_submission.csv"
    sample_submission.write_text("id,prediction\n1,0.0\n", encoding="utf-8")

    config = SimpleNamespace(
        submission=SimpleNamespace(max_daily=1, best_threshold=0.0, mode="api", format="csv"),
        data=SimpleNamespace(sample_submission="sample_submission.csv"),
        competition=SimpleNamespace(name="example", metric_direction="maximize"),
    )
    submitter = Submitter(tmp_path, config)
    submitter.tracker.record_submission(submission, cv_score=0.4)
    submitter.tracker.record_submission(submission, cv_score=0.6)

    class UnexpectedAPI:
        def submit(self, slug, path, message):
            raise AssertionError("a score below the best maximize score must be queued")

    submitter.api = UnexpectedAPI()
    result = submitter.submit(submission, cv_score=0.5)

    assert not result["success"]
    assert result["queued"]
    assert submitter.budget.today_count() == 0


def test_notebook_mode_does_not_call_submission_api(tmp_path):
    submission = tmp_path / "submissions" / "submission.csv"
    submission.parent.mkdir()
    submission.write_text("id,tvt\nwell_1,0.5\n", encoding="utf-8")
    sample_submission = tmp_path / "sample_submission.csv"
    sample_submission.write_text("id,tvt\nwell_1,0.0\n", encoding="utf-8")

    config = SimpleNamespace(
        submission=SimpleNamespace(
            max_daily=5,
            best_threshold=0.0,
            mode="notebook",
            format="csv",
        ),
        data=SimpleNamespace(sample_submission="sample_submission.csv"),
        competition=SimpleNamespace(name="grouped-regression", metric_direction="minimize"),
    )
    submitter = Submitter(tmp_path, config)

    class UnexpectedAPI:
        def submit(self, slug, path, message):
            raise AssertionError("notebook mode must not call the competition submission API")

    submitter.api = UnexpectedAPI()
    result = submitter.submit(submission, cv_score=1.0, force=True)

    assert not result["success"]
    assert result["notebook_required"]
    assert result["competition_url"].endswith("/competitions/grouped-regression")
    assert submitter.budget.today_count() == 0


def test_notebook_mode_flush_keeps_reserved_candidate(tmp_path):
    config = SimpleNamespace(
        submission=SimpleNamespace(
            max_daily=5,
            best_threshold=0.0,
            mode="notebook",
            format="csv",
        ),
        data=SimpleNamespace(sample_submission="sample_submission.csv"),
        competition=SimpleNamespace(name="grouped-regression", metric_direction="minimize"),
    )
    submitter = Submitter(tmp_path, config)
    candidate = tmp_path / "submissions" / "candidate.csv"
    submitter.budget.reserve(str(candidate), 1.0, "candidate")

    results = submitter.submit_reserved()

    assert results[0]["notebook_required"]
    assert len(submitter.budget.get_reserved()) == 1
