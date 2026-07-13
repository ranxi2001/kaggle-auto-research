from pathlib import Path
from zipfile import ZipFile

from kaggle_auto.submission.validator import SubmissionValidator


def validate(path: Path):
    return SubmissionValidator().validate_skill_zip(path)


def test_validates_static_skill_zip(tmp_path):
    submission = tmp_path / "skill.zip"
    with ZipFile(submission, "w") as archive:
        archive.writestr(
            "skills/fjsp-repair-scheduler/SKILL.md",
            "---\nname: fjsp-repair-scheduler\ndescription: Repair schedules.\n---\n",
        )
        archive.writestr(
            "skills/fjsp-repair-scheduler/scripts/repair.py",
            "print('ok')\n",
        )

    result = validate(submission)

    assert result.is_valid
    assert result.errors == []


def test_rejects_zip_without_skill_manifest(tmp_path):
    submission = tmp_path / "skill.zip"
    with ZipFile(submission, "w") as archive:
        archive.writestr("skills/example/scripts/run.py", "print('ok')\n")

    result = validate(submission)

    assert not result.is_valid
    assert "ZIP must contain at least one skills/<name>/SKILL.md" in result.errors


def test_rejects_unsafe_zip_path(tmp_path):
    submission = tmp_path / "skill.zip"
    with ZipFile(submission, "w") as archive:
        archive.writestr("skills/example/SKILL.md", "valid")
        archive.writestr("../escape.txt", "unsafe")

    result = validate(submission)

    assert not result.is_valid
    assert "Unsafe ZIP member path: ../escape.txt" in result.errors


def test_rejects_corrupt_zip(tmp_path):
    submission = tmp_path / "skill.zip"
    submission.write_bytes(b"not a zip")

    result = validate(submission)

    assert not result.is_valid
    assert result.errors == ["Submission is not a valid ZIP archive"]


def test_rejects_skill_manifest_without_frontmatter(tmp_path):
    submission = tmp_path / "skill.zip"
    with ZipFile(submission, "w") as archive:
        archive.writestr("skills/example/SKILL.md", "# Example\n")

    result = validate(submission)

    assert not result.is_valid
    assert "Skill manifest lacks YAML frontmatter: skills/example/SKILL.md" in result.errors


def test_csv_validator_does_not_apply_skill_zip_contract(tmp_path):
    submission = tmp_path / "agent.zip"
    with ZipFile(submission, "w") as archive:
        archive.writestr("agent.yaml", "name: example\n")
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,prediction\n1,0.0\n", encoding="utf-8")

    result = SubmissionValidator().validate(submission, sample)

    assert not result.is_valid
    assert all("skills/<name>/SKILL.md" not in error for error in result.errors)
