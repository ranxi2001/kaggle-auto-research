from kaggle_auto.config import (
    CompetitionConfig,
    DataConfig,
    SubmissionConfig,
    WorkspaceConfig,
    load_config,
    save_config,
)


def test_submission_mode_and_format_round_trip(tmp_path):
    config = WorkspaceConfig(
        competition=CompetitionConfig(name="skill-lift"),
        submission=SubmissionConfig(mode="writeup", format="skill_zip"),
    )

    save_config(config, tmp_path)
    loaded = load_config(tmp_path)

    assert loaded.submission.mode == "writeup"
    assert loaded.submission.format == "skill_zip"


def test_group_order_and_notebook_mode_round_trip(tmp_path):
    config = WorkspaceConfig(
        competition=CompetitionConfig(name="grouped-regression"),
        data=DataConfig(group_column="well_id", order_column="MD"),
        submission=SubmissionConfig(mode="notebook"),
    )

    save_config(config, tmp_path)
    loaded = load_config(tmp_path)

    assert loaded.data.group_column == "well_id"
    assert loaded.data.order_column == "MD"
    assert loaded.submission.mode == "notebook"
