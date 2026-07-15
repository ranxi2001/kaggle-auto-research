import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "workspaces"
    / "rogii-wellbore-geology-prediction"
    / "scripts"
    / "rogii_hmm_smoother.py"
)
SPEC = importlib.util.spec_from_file_location("rogii_hmm_smoother", SCRIPT_PATH)
assert SPEC and SPEC.loader
smoother = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoother
SPEC.loader.exec_module(smoother)


def synthetic_well(*, suffix_rows=15, missing_suffix=False):
    prefix_rows = 8
    rows = prefix_rows + suffix_rows
    md = np.arange(rows, dtype=float)
    z = -0.25 * md
    target = 100.0 + 0.10 * md - z
    tvt_input = target.copy()
    tvt_input[prefix_rows:] = np.nan
    gr = 40.0 + 8.0 * np.sin(target / 3.0)
    if missing_suffix:
        gr[prefix_rows:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": md,
            "Y": md * 0.5,
            "Z": z,
            "TVT": target,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    )
    typewell_tvt = np.arange(80.0, 130.25, 0.25)
    typewell = pd.DataFrame(
        {
            "TVT": typewell_tvt,
            "GR": 40.0 + 8.0 * np.sin(typewell_tvt / 3.0),
        }
    )
    return horizontal, typewell


def test_data_fingerprint_hashes_contents_not_only_metadata(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "well.csv"
    path.write_bytes(b"abc")
    original_stat = path.stat()
    before = smoother.data_fingerprint(raw)

    path.write_bytes(b"xyz")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    after = smoother.data_fingerprint(raw)

    assert path.stat().st_size == original_stat.st_size
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert before != after


def test_registration_blocks_are_deterministic_disjoint_21_row_partitions():
    gr = np.arange(50, dtype=float)
    gr[[2, 22, 47]] = np.nan

    first = smoother.build_registration_blocks(gr, block_size=21)
    second = smoother.build_registration_blocks(gr.copy(), block_size=21)

    assert [(block.start, block.stop, block.anchor) for block in first] == [
        (block.start, block.stop, block.anchor) for block in second
    ]
    assert [(block.start, block.stop) for block in first] == [(0, 21), (21, 42), (42, 50)]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.observed_indices, right.observed_indices)
    assert [len(block.observed_indices) for block in first] == [20, 20, 7]
    flattened = np.concatenate([block.observed_indices for block in first])
    assert flattened.tolist() == np.flatnonzero(np.isfinite(gr)).tolist()
    assert len(flattened) == len(np.unique(flattened))


def test_registration_emissions_do_not_depend_on_suffix_target():
    horizontal, typewell = synthetic_well(suffix_rows=9)
    center_tvt = np.full(9, horizontal.loc[7, "TVT_input"], dtype=float)
    offset_grid = np.array([-2.0, 0.0, 2.0])
    rates = np.array([-0.05, 0.0, 0.05])

    before, before_diagnostics = smoother.build_registration_emissions(
        horizontal, typewell, center_tvt, offset_grid, rates
    )
    changed = horizontal.copy()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 50_000.0
    after, after_diagnostics = smoother.build_registration_emissions(
        changed, typewell, center_tvt, offset_grid, rates
    )

    np.testing.assert_array_equal(before.rows, after.rows)
    np.testing.assert_array_equal(before.observed_counts, after.observed_counts)
    np.testing.assert_allclose(before.values, after.values)
    assert before_diagnostics == after_diagnostics


def test_hmm_candidates_and_corridor_audit_do_not_depend_on_suffix_target():
    horizontal, typewell = synthetic_well(suffix_rows=9)
    evaluation = horizontal["TVT_input"].isna()
    parent_tvt = horizontal.loc[evaluation, "TVT"].to_numpy(dtype=float) + 0.5
    config = smoother.HMMConfig(
        name="hmm_test",
        momentum=0.95,
        rate_noise=0.05,
        position_noise=0.5,
        start_sigma=1.5,
        initial_rate_sigma=0.1,
    )
    kwargs = {
        "parent_tvt": parent_tvt,
        "configs": (config,),
        "initial_half_width": 4.0,
        "expanded_half_width": 8.0,
        "edge_threshold": 1.0,
        "block_size": 5,
    }

    before, before_diagnostics = smoother.hmm_candidates(horizontal, typewell, **kwargs)
    changed = horizontal.copy()
    changed.loc[evaluation, "TVT"] = np.linspace(-1e9, 1e9, int(evaluation.sum()))
    after, after_diagnostics = smoother.hmm_candidates(changed, typewell, **kwargs)

    pd.testing.assert_frame_equal(before, after)
    assert before_diagnostics == after_diagnostics
    assert np.isfinite(before[["hmm_test", "hmm_test__std"]].to_numpy()).all()


def test_missing_suffix_gr_is_neutral_and_never_duplicated():
    horizontal, typewell = synthetic_well(suffix_rows=9, missing_suffix=True)
    center_tvt = np.full(9, horizontal.loc[7, "TVT_input"], dtype=float)
    emissions, diagnostics = smoother.build_registration_emissions(
        horizontal,
        typewell,
        center_tvt,
        np.array([-2.0, 0.0, 2.0]),
        np.array([-0.05, 0.0, 0.05]),
    )

    assert diagnostics["observed_gr_rows"] == 0
    assert diagnostics["missing_gr_rows"] == 9
    assert diagnostics["registration_blocks"] == 1
    assert diagnostics["neutral_blocks"] == 1
    assert emissions.observed_counts.tolist() == [0]
    assert emissions.blocks[0].observed_indices.size == 0
    np.testing.assert_array_equal(emissions.values, np.zeros_like(emissions.values))


def forward_backward_case(last_emission_offset=None):
    offset_grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    rates = np.array([-0.1, 0.0, 0.1])
    values = np.zeros((2, len(offset_grid), len(rates)), dtype=np.float32)
    observed_counts = np.array([0, 0], dtype=np.int64)
    if last_emission_offset is not None:
        values[1] = -20.0
        values[1, np.flatnonzero(offset_grid == last_emission_offset)[0], :] = 0.0
        observed_counts[1] = 1
    emissions = smoother.SparseEmissions(
        rows=np.array([0, 1], dtype=np.int64),
        values=values,
        observed_counts=observed_counts,
        blocks=(),
    )
    config = smoother.HMMConfig(
        name="test",
        momentum=0.95,
        rate_noise=0.05,
        position_noise=0.5,
        start_sigma=1.5,
        initial_rate_sigma=0.1,
    )
    result = smoother.forward_backward(
        emissions,
        anchor_md=np.array([1.0, 2.0]),
        center_u=np.array([100.0, 100.0]),
        offset_grid=offset_grid,
        rates=rates,
        initial_md=0.0,
        initial_center_u=100.0,
        initial_offset=0.0,
        initial_rate=0.0,
        config=config,
        position_edge_cells=1,
        rate_edge_cells=1,
    )
    return result


def test_future_emission_changes_early_smoothed_posterior():
    neutral = forward_backward_case()
    future_right = forward_backward_case(last_emission_offset=2.0)

    assert future_right.mean_offset[0] > neutral.mean_offset[0] + 0.05


def test_forward_backward_posterior_moments_are_finite():
    result = forward_backward_case(last_emission_offset=2.0)

    for values in (
        result.mean_offset,
        result.std_offset,
        result.mean_rate,
        result.std_rate,
        result.position_edge_mass,
        result.rate_edge_mass,
    ):
        assert np.isfinite(values).all()
    assert np.isfinite(result.log_likelihood)
    assert (result.std_offset >= 0).all()
    assert (result.std_rate >= 0).all()
    assert ((result.position_edge_mass >= 0) & (result.position_edge_mass <= 1)).all()
    assert ((result.rate_edge_mass >= 0) & (result.rate_edge_mass <= 1)).all()


def test_corridor_expansion_uses_only_posterior_boundary_mass():
    low_position_high_rate = smoother.ForwardBackwardResult(
        mean_offset=np.zeros(2),
        std_offset=np.ones(2),
        mean_rate=np.zeros(2),
        std_rate=np.ones(2),
        position_edge_mass=np.array([0.001, 0.005]),
        rate_edge_mass=np.array([0.95, 0.95]),
        log_likelihood=0.0,
    )
    high_position = smoother.ForwardBackwardResult(
        mean_offset=np.zeros(2),
        std_offset=np.ones(2),
        mean_rate=np.zeros(2),
        std_rate=np.ones(2),
        position_edge_mass=np.array([0.001, 0.02]),
        rate_edge_mass=np.zeros(2),
        log_likelihood=0.0,
    )

    assert not smoother.needs_corridor_expansion(low_position_high_rate, threshold=0.01)
    assert smoother.needs_corridor_expansion(high_position, threshold=0.01)


def test_non_selected_candidate_expands_shared_two_config_pipeline(monkeypatch):
    horizontal, typewell = synthetic_well(suffix_rows=9)
    evaluation = horizontal["TVT_input"].isna().to_numpy()
    parent_tvt = horizontal.loc[evaluation, "TVT"].to_numpy(dtype=float)
    selected, non_selected = smoother.HMM_CANDIDATES
    calls = []

    def fake_emissions(horizontal, typewell, center_tvt, offset_grid, rates, **kwargs):
        del horizontal, typewell, center_tvt, kwargs
        emissions = smoother.SparseEmissions(
            rows=np.array([0], dtype=np.int64),
            values=np.zeros((1, len(offset_grid), len(rates)), dtype=np.float32),
            observed_counts=np.array([1], dtype=np.int64),
            blocks=(),
        )
        return emissions, {"observed_gr_rows": 1}

    def fake_prepared(
        horizontal,
        center_tvt,
        emissions,
        offset_grid,
        rates,
        initial_rate,
        config,
        half_width,
    ):
        del center_tvt, emissions, offset_grid, rates, initial_rate
        calls.append((config.name, half_width))
        suffix_rows = int(horizontal["TVT_input"].isna().sum())
        edge_mass = 0.5 if config == non_selected and half_width == 4.0 else 0.0
        result = smoother.ForwardBackwardResult(
            mean_offset=np.array([0.0]),
            std_offset=np.array([1.0]),
            mean_rate=np.array([0.0]),
            std_rate=np.array([0.1]),
            position_edge_mass=np.array([edge_mass]),
            rate_edge_mass=np.array([0.0]),
            log_likelihood=0.0,
        )
        prediction = np.full(suffix_rows, half_width, dtype=float)
        return prediction, np.ones(suffix_rows), result, {"half_width": half_width}

    monkeypatch.setattr(smoother, "build_registration_emissions", fake_emissions)
    monkeypatch.setattr(smoother, "_prepared_smoother", fake_prepared)

    frame, diagnostics = smoother.hmm_candidates(
        horizontal,
        typewell,
        parent_tvt=parent_tvt,
        configs=(selected, non_selected),
        initial_half_width=4.0,
        expanded_half_width=8.0,
        edge_threshold=0.01,
    )

    assert calls == [
        (selected.name, 4.0),
        (non_selected.name, 4.0),
        (selected.name, 8.0),
        (non_selected.name, 8.0),
    ]
    assert diagnostics["expanded_corridor"] is True
    assert diagnostics["final_half_width"] == 8.0
    assert frame[selected.name].tolist() == [8.0] * int(evaluation.sum())


def test_visible_test_final_candidate_uses_shared_two_config_pipeline(tmp_path, monkeypatch):
    horizontal, typewell = synthetic_well(suffix_rows=3)
    test_root = tmp_path / "data" / "raw" / "test"
    test_root.mkdir(parents=True)
    well = "shared01"
    horizontal.to_csv(test_root / f"{well}__horizontal_well.csv", index=False)
    typewell.to_csv(test_root / f"{well}__typewell.csv", index=False)
    selected = smoother.HMM_CANDIDATES[0]
    calls = []

    def fake_candidates(horizontal, typewell, parent_tvt=None, configs=(), **kwargs):
        del typewell, parent_tvt, kwargs
        calls.append(tuple(config.name for config in configs))
        evaluation_indices = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
        frame = pd.DataFrame({"_row_index": evaluation_indices})
        candidate_diagnostics = {}
        for index, config in enumerate(smoother.HMM_CANDIDATES):
            frame[config.name] = np.full(len(evaluation_indices), 100.0 + index)
            candidate_diagnostics[config.name] = {
                "unresolved_position_boundary": False,
                "unresolved_rate_boundary": False,
            }
        return frame, {"expanded_corridor": True, "candidates": candidate_diagnostics}

    monkeypatch.setattr(smoother, "hmm_candidates", fake_candidates)

    predictions, failures, _ = smoother.predict_visible_test(tmp_path, selected.name)

    assert failures == []
    assert calls == [tuple(config.name for config in smoother.HMM_CANDIDATES)]
    assert predictions["prediction"].tolist() == [100.0] * 3


def single_step_emissions(offset_grid, rates):
    return smoother.SparseEmissions(
        rows=np.array([0], dtype=np.int64),
        values=np.zeros((1, len(offset_grid), len(rates)), dtype=np.float32),
        observed_counts=np.array([0], dtype=np.int64),
        blocks=(),
    )


def test_forward_backward_rejects_unreachable_center_jump():
    offset_grid = np.arange(-2.0, 3.0)
    rates = np.array([-0.1, 0.0, 0.1])
    config = smoother.HMMConfig("unreachable", 0.95, 0.01, 0.01, 1.0, 0.1)

    with pytest.raises(
        (ValueError, RuntimeError), match="unreachable|transition|posterior|support"
    ):
        smoother.forward_backward(
            single_step_emissions(offset_grid, rates),
            anchor_md=np.array([1.0]),
            center_u=np.array([1e6]),
            offset_grid=offset_grid,
            rates=rates,
            initial_md=0.0,
            initial_center_u=0.0,
            initial_offset=0.0,
            initial_rate=0.0,
            config=config,
            position_edge_cells=1,
            rate_edge_cells=1,
        )


def test_forward_backward_rejects_descending_offset_grid():
    offset_grid = np.arange(2.0, -3.0, -1.0)
    rates = np.array([-0.1, 0.0, 0.1])
    config = smoother.HMMConfig("descending", 0.95, 0.01, 0.1, 1.0, 0.1)

    with pytest.raises(ValueError, match="offset.*increas|increasing.*offset"):
        smoother.forward_backward(
            single_step_emissions(offset_grid, rates),
            anchor_md=np.array([1.0]),
            center_u=np.array([0.0]),
            offset_grid=offset_grid,
            rates=rates,
            initial_md=0.0,
            initial_center_u=0.0,
            initial_offset=0.0,
            initial_rate=0.0,
            config=config,
        )


def test_nested_candidate_selection_does_not_read_held_out_fold():
    summary = pd.DataFrame(
        [
            {
                "fold": fold,
                "well": f"w{fold}",
                "candidate": candidate,
                "rows": 1,
                "squared_error": error,
            }
            for fold, errors in {
                1: {"slow": 0.0, "flex": 10_000.0},
                2: {"slow": 4.0, "flex": 1.0},
                3: {"slow": 4.0, "flex": 1.0},
            }.items()
            for candidate, error in errors.items()
        ]
    )

    selections, _ = smoother.choose_nested_candidates(summary)
    fold_one = next(row for row in selections if row["fold"] == 1)
    assert fold_one["selected_candidate"] == "flex"

    changed = summary.copy()
    changed.loc[(changed["fold"] == 1) & (changed["candidate"] == "slow"), "squared_error"] = 1e12
    changed.loc[(changed["fold"] == 1) & (changed["candidate"] == "flex"), "squared_error"] = 0.0
    changed_selections, _ = smoother.choose_nested_candidates(changed)
    changed_fold_one = next(row for row in changed_selections if row["fold"] == 1)
    assert changed_fold_one["selected_candidate"] == fold_one["selected_candidate"]


def test_nested_candidate_selection_fails_closed_on_incomplete_grid():
    incomplete = pd.DataFrame(
        [
            {"fold": 1, "well": "w1", "candidate": "slow", "rows": 1, "squared_error": 1.0},
            {"fold": 1, "well": "w1", "candidate": "flex", "rows": 1, "squared_error": 2.0},
            {"fold": 2, "well": "w2", "candidate": "slow", "rows": 1, "squared_error": 1.0},
        ]
    )

    with pytest.raises(ValueError, match="candidate|grid|complete|missing"):
        smoother.choose_nested_candidates(incomplete)


@pytest.mark.parametrize(
    ("overall", "folds_better", "expected"),
    [
        (10.95, 4, "candidate_ready"),
        (10.950001, 4, "promising_continue"),
        (10.5, 3, "promising_continue"),
        (11.749999, 3, "promising_continue"),
        (11.75, 5, "exhausted"),
        (10.5, 2, "exhausted"),
    ],
)
def test_candidate_gate_uses_predeclared_score_and_fold_thresholds(overall, folds_better, expected):
    assert smoother.candidate_decision(overall, folds_better) == expected


@pytest.mark.parametrize(
    ("failure_kwarg", "expected"),
    [
        ("training_failures", "candidate_blocked_training_failures"),
        ("selected_boundary_wells", "candidate_blocked_boundaries"),
        ("final_boundary_wells", "candidate_blocked_boundaries"),
        ("test_failures", "candidate_blocked_test_failures"),
    ],
)
def test_candidate_ready_gate_fails_closed_on_fallbacks_or_boundaries(failure_kwarg, expected):
    assert smoother.candidate_decision(10.0, 5, **{failure_kwarg: 1}) == expected


def test_atomic_parquet_publish_never_exposes_partial_path(tmp_path):
    partial = tmp_path / "features.parquet.partial"
    final = tmp_path / "features.parquet"
    expected = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    expected.to_parquet(partial, index=False)

    smoother.atomic_publish_parquet(partial, final)

    assert final.exists()
    assert not partial.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(final), expected)


def test_atomic_parquet_publish_fails_closed_for_invalid_partial(tmp_path):
    partial = tmp_path / "features.parquet.partial"
    final = tmp_path / "features.parquet"
    partial.write_bytes(b"not a parquet file")

    with pytest.raises(Exception):
        smoother.atomic_publish_parquet(partial, final)

    assert not final.exists()


def test_failed_completion_audit_never_publishes_partial_parquet(tmp_path, monkeypatch):
    horizontal, typewell = synthetic_well(suffix_rows=3)
    train_root = tmp_path / "data" / "raw" / "train"
    train_root.mkdir(parents=True)
    well = "audit001"
    horizontal.to_csv(train_root / f"{well}__horizontal_well.csv", index=False)
    typewell.to_csv(train_root / f"{well}__typewell.csv", index=False)
    manifest = pd.DataFrame({"well": [well], "fold": [1]})

    class FailingParentReader:
        def read_well(self, row_group, read_well, evaluation_indices):
            del row_group, read_well
            return np.full(len(evaluation_indices), 102.0, dtype=float)

        def finish(self, expected_groups):
            assert expected_groups == 1
            raise RuntimeError("parent completion audit failed")

    def fake_hmm_candidates(horizontal, typewell, parent_tvt, configs):
        del typewell, parent_tvt
        evaluation_indices = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
        frame = pd.DataFrame({"_row_index": evaluation_indices})
        candidate_diagnostics = {}
        for config in configs:
            frame[config.name] = np.full(len(evaluation_indices), 102.0, dtype=np.float32)
            frame[f"{config.name}__std"] = np.ones(len(evaluation_indices), dtype=np.float32)
            candidate_diagnostics[config.name] = {
                "unresolved_position_boundary": False,
                "unresolved_rate_boundary": False,
            }
        return frame, {"candidates": candidate_diagnostics}

    monkeypatch.setattr(smoother, "ParentOOFReader", lambda workspace: FailingParentReader())
    monkeypatch.setattr(smoother, "hmm_candidates", fake_hmm_candidates)
    final = tmp_path / "data" / "features" / "v004.parquet"
    final.parent.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="completion audit failed"):
        smoother.build_training_candidates(tmp_path, manifest, final)

    partial = Path(f"{final}.partial")
    assert not final.exists()
    assert partial.exists()
    assert len(pd.read_parquet(partial)) == 3


@pytest.mark.parametrize("wells", [1, 5, 7])
def test_runtime_smoke_requires_exactly_six_wells(tmp_path, monkeypatch, wells):
    monkeypatch.setattr(smoother, "NUMBA_AVAILABLE", True)

    with pytest.raises(ValueError, match="6|six"):
        smoother.run_smoke(tmp_path, wells=wells)


def test_smoke_inventory_rejects_fewer_than_six_training_wells(tmp_path):
    train_root = tmp_path / "train"
    train_root.mkdir()
    for index in range(5):
        pd.DataFrame({"TVT_input": [1.0, np.nan]}).to_csv(
            train_root / f"short{index:03d}__horizontal_well.csv", index=False
        )

    with pytest.raises(ValueError, match="at least 6"):
        smoother._smoke_inventory(train_root)


def test_runtime_smoke_audits_quantile_wells_and_data_fingerprint(tmp_path, monkeypatch):
    train_root = tmp_path / "data" / "raw" / "train"
    train_root.mkdir(parents=True)
    for index in range(10):
        horizontal, typewell = synthetic_well(suffix_rows=index + 1)
        well = f"smoke{index:03d}"
        horizontal.to_csv(train_root / f"{well}__horizontal_well.csv", index=False)
        typewell.to_csv(train_root / f"{well}__typewell.csv", index=False)

    def fake_candidates(horizontal, typewell, parent_tvt=None, configs=(), **kwargs):
        del typewell, parent_tvt, configs, kwargs
        evaluation_indices = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
        return pd.DataFrame({"_row_index": evaluation_indices}), {
            "expanded_corridor": False,
            "final_half_width": smoother.INITIAL_HALF_WIDTH,
            "any_candidate_unresolved_position_boundary": False,
            "any_candidate_unresolved_rate_boundary": False,
        }

    monkeypatch.setattr(smoother, "NUMBA_AVAILABLE", True)
    monkeypatch.setattr(smoother, "hmm_candidates", fake_candidates)
    monkeypatch.setattr(smoother, "_peak_rss_gib", lambda: 0.1)

    result = smoother.run_smoke(tmp_path, wells=6)
    expected_wells = ["smoke000", "smoke001", "smoke003", "smoke005", "smoke007", "smoke009"]
    smoke_path, required = smoother._require_runtime_smoke(tmp_path)

    assert result["selected_wells"] == expected_wells
    assert result["data_fingerprint"] == smoother.data_fingerprint(train_root)
    assert result["data_fingerprint_algorithm"] == smoother.DATA_FINGERPRINT_ALGORITHM
    assert result["source_sha256"] == smoother.sha256_file(SCRIPT_PATH.resolve())
    assert result["source_sha256_end"] == result["source_sha256"]
    assert result["parent_source_sha256"] == smoother.sha256_file(smoother._parent_source_path())
    assert result["parent_source_sha256_end"] == result["parent_source_sha256"]
    assert required == result
    assert json.loads(smoke_path.read_text(encoding="utf-8")) == result
    assert not list(smoke_path.parent.glob("*.partial"))

    with pytest.raises(FileExistsError):
        smoother.run_smoke(tmp_path, wells=6)

    tampered = result.copy()
    tampered["selected_wells"] = list(reversed(expected_wells))
    smoke_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="selected|quantile|well"):
        smoother._require_runtime_smoke(tmp_path)

    tampered = result.copy()
    tampered["data_fingerprint"] = "wrong"
    smoke_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        smoother._require_runtime_smoke(tmp_path)

    tampered = result.copy()
    tampered["parent_source_sha256"] = "wrong"
    smoke_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        smoother._require_runtime_smoke(tmp_path)
