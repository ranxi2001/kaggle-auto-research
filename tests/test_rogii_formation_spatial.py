import importlib.util
import json
from pathlib import Path
import signal
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "workspaces"
    / "rogii-wellbore-geology-prediction"
    / "scripts"
    / "rogii_formation_spatial.py"
)
SPEC = importlib.util.spec_from_file_location("rogii_formation_spatial", SCRIPT_PATH)
assert SPEC and SPEC.loader
spatial = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spatial
SPEC.loader.exec_module(spatial)


def formation_value(name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    index = spatial.FORMATIONS.index(name) + 1
    return 100.0 * index + 1.5 * x - 0.75 * y


def write_donor(
    root: Path,
    well: str,
    *,
    x: float,
    y: float,
    complete: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = 5
    frame = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float),
            "X": np.full(rows, x, dtype=float),
            "Y": np.full(rows, y, dtype=float),
            "Z": -np.arange(rows, dtype=float),
            "GR": 50.0 + np.arange(rows, dtype=float),
            "TVT_input": 100.0 + np.arange(rows, dtype=float),
            "TVT": 100.0 + np.arange(rows, dtype=float),
        }
    )
    for formation in spatial.FORMATIONS:
        frame[formation] = formation_value(
            formation,
            frame["X"].to_numpy(dtype=float),
            frame["Y"].to_numpy(dtype=float),
        )
    if not complete:
        frame.loc[:, spatial.FORMATIONS[0]] = np.nan
    path = root / f"{well}__horizontal_well.csv"
    frame.to_csv(path, index=False)
    return path


def build_planar_donors(tmp_path: Path, count: int = 12):
    train_root = tmp_path / "data" / "raw" / "train"
    wells = []
    for index in range(count):
        well = f"donor{index:03d}"
        wells.append(well)
        write_donor(
            train_root,
            well,
            x=float(index % 4),
            y=float(index // 4),
        )
    return train_root, wells, spatial.build_donor_table(train_root, wells)


def synthetic_query_well(*, suffix_rows: int = 9) -> pd.DataFrame:
    prefix_rows = 851
    rows = prefix_rows + suffix_rows
    md = np.arange(rows, dtype=float)
    x = 0.25 + md / 2000.0
    y = 0.50 + md / 3000.0
    z = -0.2 * md
    tvt = -z + formation_value("ANCC", x, y) + 7.0
    tvt_input = tvt.copy()
    tvt_input[prefix_rows:] = np.nan
    return pd.DataFrame(
        {
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            "GR": 50.0 + np.sin(md / 20.0),
            "TVT_input": tvt_input,
            "TVT": tvt,
        }
    )


def write_training_well(
    root: Path,
    well: str,
    *,
    x: float,
    y: float,
    suffix_rows: int,
) -> Path:
    query = synthetic_query_well(suffix_rows=suffix_rows)
    query["X"] = x + query["MD"] / 100_000.0
    query["Y"] = y + query["MD"] / 120_000.0
    for formation in spatial.FORMATIONS:
        query[formation] = formation_value(
            formation,
            query["X"].to_numpy(dtype=float),
            query["Y"].to_numpy(dtype=float),
        )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{well}__horizontal_well.csv"
    query.loc[:, spatial.TRAIN_COLUMNS].to_csv(path, index=False)
    return path


def test_fold_donor_subsets_exclude_outer_and_inner_validation_wells(tmp_path):
    train_root, wells, catalog = build_planar_donors(tmp_path, count=14)
    del train_root
    outer_validation = {wells[0], wells[1]}
    inner_validation = {wells[2], wells[3]}
    allowed = [well for well in wells if well not in outer_validation | inner_validation]

    donors = spatial.subset_donor_table(catalog, allowed)

    assert set(donors.well_ids) == set(allowed)
    assert not set(donors.well_ids) & outer_validation
    assert not set(donors.well_ids) & inner_validation
    assert set(donors.requested_well_ids) == set(allowed)


def test_donor_contexts_create_all_single_and_pair_fold_exclusions(tmp_path):
    train_root, wells, _ = build_planar_donors(tmp_path, count=25)
    manifest = pd.DataFrame(
        {
            "well": wells,
            "fold": [(index % 5) + 1 for index in range(len(wells))],
        }
    )
    catalog = spatial.build_donor_catalog(train_root, wells)

    contexts, audits = spatial.donor_contexts(catalog, manifest, (1, 2, 3, 4, 5))

    assert len(contexts) == 15
    assert len(audits) == 15
    assert set(contexts) == {
        *((fold,) for fold in range(1, 6)),
        *((left, right) for left in range(1, 6) for right in range(left + 1, 6)),
    }
    fold_map = manifest.set_index("well")["fold"].to_dict()
    for excluded, donors in contexts.items():
        assert all(fold_map[well] not in excluded for well in donors.well_ids)
    pair_audit = next(row for row in audits if row["excluded_folds"] == [1, 3])
    assert pair_audit["expected_donor_folds"] == [2, 4, 5]
    assert pair_audit["actual_donor_folds"] == [2, 4, 5]
    assert pair_audit["donor_fold_violations"] == []


def test_query_and_outer_fold_map_to_union_exclusion_context(tmp_path):
    train_root, wells, _ = build_planar_donors(tmp_path, count=25)
    manifest = pd.DataFrame(
        {
            "well": wells,
            "fold": [(index % 5) + 1 for index in range(len(wells))],
        }
    )
    catalog = spatial.build_donor_catalog(train_root, wells)
    contexts, _ = spatial.donor_contexts(catalog, manifest, (1, 2, 3, 4, 5))
    fold_map = manifest.set_index("well")["fold"].to_dict()

    for query_fold in range(1, 6):
        for outer_fold in range(1, 6):
            key = spatial._context_key(query_fold, outer_fold)
            assert key == tuple(sorted({query_fold, outer_fold}))
            assert all(fold_map[well] not in key for well in contexts[key].well_ids)
        assert spatial._context_key(query_fold, query_fold) == (query_fold,)


def test_same_id_is_excluded_before_scaling_and_neighbour_search(tmp_path):
    _, wells, donors = build_planar_donors(tmp_path)
    query_well = wells[0]
    query = np.asarray([[1.25, 0.75]], dtype=float)

    _, diagnostics = spatial.query_local_planes(
        donors,
        query,
        query_well_id=query_well,
    )

    keep = np.asarray([well != query_well for well in donors.well_ids])
    expected_mean = donors.coordinates[keep].mean(axis=0)
    expected_scale = donors.coordinates[keep].std(axis=0, ddof=0)
    np.testing.assert_allclose(diagnostics["coordinate_mean"], expected_mean)
    np.testing.assert_allclose(diagnostics["coordinate_std_ddof0"], expected_scale)
    assert diagnostics["same_id_donors_excluded"] == 1
    assert diagnostics["query_well_present_after_exclusion"] is False
    assert diagnostics["eligible_donor_count"] == len(donors.well_ids) - 1


def test_centered_local_plane_recovers_planar_surfaces(tmp_path):
    _, _, donors = build_planar_donors(tmp_path)
    query = np.asarray([[0.2, 0.3], [1.7, 1.4], [2.8, 0.6]], dtype=float)

    prediction, diagnostics = spatial.query_local_planes(
        donors,
        query,
        query_well_id="not-a-donor",
    )

    expected = np.column_stack(
        [formation_value(formation, query[:, 0], query[:, 1]) for formation in spatial.FORMATIONS]
    )
    np.testing.assert_allclose(prediction, expected, rtol=0.0, atol=2e-7)
    assert diagnostics["same_id_donors_excluded"] == 0
    assert np.isfinite(prediction).all()


def test_local_plane_fails_closed_after_self_exclusion_leaves_fewer_than_k(tmp_path):
    _, wells, donors = build_planar_donors(tmp_path, count=spatial.K_NEIGHBORS)

    with pytest.raises(ValueError, match="Fewer than k donors after same-well exclusion"):
        spatial.query_local_planes(
            donors,
            np.asarray([[0.5, 0.5]]),
            query_well_id=wells[0],
        )


def test_local_plane_rejects_nonfinite_query_and_unsolved_system(tmp_path, monkeypatch):
    _, _, donors = build_planar_donors(tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        spatial.query_local_planes(
            donors,
            np.asarray([[np.nan, 0.0]]),
            query_well_id="query",
        )

    def fail_solve(*args, **kwargs):
        del args, kwargs
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(spatial.np.linalg, "solve", fail_solve)
    with pytest.raises(ValueError, match="could not be solved"):
        spatial.query_local_planes(
            donors,
            np.asarray([[0.5, 0.5]]),
            query_well_id="query",
        )


def fake_surface_predictions(rows: int) -> np.ndarray:
    row = np.arange(rows, dtype=float)
    return np.column_stack(
        [0.01 * (index + 1) * row + 100.0 * index for index in range(len(spatial.FORMATIONS))]
    )


def test_best6_selector_uses_only_fixed_prefix_windows_and_declared_tie_break(monkeypatch):
    horizontal = synthetic_query_well()[list(spatial.QUERY_COLUMNS)]
    surfaces = fake_surface_predictions(640 + int(horizontal["TVT_input"].isna().sum()))

    def fake_planes(donors, query_xy, *, query_well_id):
        del donors, query_well_id
        assert len(query_xy) == len(surfaces)
        return surfaces.copy(), {"same_id_donors_excluded": 0}

    monkeypatch.setattr(spatial, "query_local_planes", fake_planes)
    placeholder = spatial.DonorTable(
        well_ids=(),
        coordinates=np.empty((0, 2)),
        formations=np.empty((0, len(spatial.FORMATIONS))),
    )
    result = spatial.spatial_streams(horizontal, placeholder, query_well_id="query")

    visible = np.arange(851)
    calibration = visible[-640:-128]
    holdout = visible[-128:]
    final_window = visible[-512:]
    z = horizontal["Z"].to_numpy(dtype=float)
    tvt_input = horizontal["TVT_input"].to_numpy(dtype=float)
    expected_rmse = []
    for index, formation in enumerate(spatial.FORMATIONS):
        bias = np.median(tvt_input[calibration] + z[calibration] - surfaces[:512, index])
        prediction = -z[holdout] + surfaces[512:640, index] + bias
        expected_rmse.append(spatial.rmse(tvt_input[holdout], prediction))
        assert result.diagnostics["prospective_bias"][formation] == pytest.approx(bias)
        assert result.diagnostics["prospective_rmse"][formation] == pytest.approx(expected_rmse[-1])
    expected_index = int(np.argmin(expected_rmse))
    assert result.diagnostics["selected_formation"] == spatial.FORMATIONS[expected_index]
    expected_final_bias = np.median(
        tvt_input[final_window] + z[final_window] - surfaces[128:640, expected_index]
    )
    assert result.diagnostics["final_bias"] == pytest.approx(expected_final_bias)

    tied = np.repeat(surfaces[:, [0]], len(spatial.FORMATIONS), axis=1)

    def tied_planes(donors, query_xy, *, query_well_id):
        del donors, query_xy, query_well_id
        return tied.copy(), {"same_id_donors_excluded": 0}

    monkeypatch.setattr(spatial, "query_local_planes", tied_planes)
    tied_result = spatial.spatial_streams(horizontal, placeholder, query_well_id="query")
    assert tied_result.diagnostics["selected_formation"] == spatial.FORMATIONS[0]


def test_spatial_streams_reject_target_column_and_ignore_prefix_rows_before_fixed_window(
    monkeypatch,
):
    horizontal = synthetic_query_well()[list(spatial.QUERY_COLUMNS)]
    surfaces = fake_surface_predictions(640 + int(horizontal["TVT_input"].isna().sum()))

    def fake_planes(donors, query_xy, *, query_well_id):
        del donors, query_xy, query_well_id
        return surfaces.copy(), {"same_id_donors_excluded": 0}

    monkeypatch.setattr(spatial, "query_local_planes", fake_planes)
    donors = spatial.DonorTable(
        well_ids=(),
        coordinates=np.empty((0, 2)),
        formations=np.empty((0, len(spatial.FORMATIONS))),
    )
    before = spatial.spatial_streams(horizontal, donors, query_well_id="query")
    changed = horizontal.copy()
    changed.loc[:210, "TVT_input"] += 1e6
    after = spatial.spatial_streams(changed, donors, query_well_id="query")

    pd.testing.assert_frame_equal(before.frame, after.frame)
    assert before.diagnostics["prospective_rmse"] == after.diagnostics["prospective_rmse"]
    assert before.diagnostics["prospective_bias"] == after.diagnostics["prospective_bias"]
    assert before.diagnostics["final_bias"] == after.diagnostics["final_bias"]

    with_target = horizontal.assign(TVT=np.linspace(0.0, 1.0, len(horizontal)))
    with pytest.raises(ValueError, match="Unexpected query schema"):
        spatial.spatial_streams(with_target, donors, query_well_id="query")


def test_simplex_solver_known_interior_and_active_edge_solutions():
    interior = spatial.simplex_nnls_from_gram(
        np.eye(2),
        np.asarray([0.2, 0.3]),
    )
    np.testing.assert_allclose(interior, [0.5, 0.2, 0.3], rtol=0.0, atol=1e-12)

    active_edge = spatial.simplex_nnls_from_gram(
        np.diag([1.0, 4.0]),
        np.asarray([0.8, 2.8]),
    )
    np.testing.assert_allclose(active_edge, [0.0, 0.4, 0.6], rtol=0.0, atol=1e-12)
    clipped_and_renormalized = np.asarray([0.8, 0.7]) / 1.5
    assert not np.allclose(active_edge[1:], clipped_and_renormalized)


def test_simplex_fit_matches_dense_bruteforce_objective():
    streams = np.asarray(
        [
            [0.0, 1.0, 3.0],
            [1.0, -1.0, 2.0],
            [2.0, 4.0, -2.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=float,
    )
    target = np.asarray([1.7, 0.2, 1.3, 3.1], dtype=float)
    fit = spatial.fit_simplex_weights(streams, target)

    grid = np.linspace(0.0, 1.0, 1001)
    best = np.inf
    for w1 in grid:
        w2 = grid[grid <= 1.0 - w1]
        weights = np.column_stack([np.full(len(w2), 1.0 - w1) - w2, np.full(len(w2), w1), w2])
        residual = target[:, None] - streams @ weights.T
        best = min(best, float(np.min(np.sum(np.square(residual), axis=0))))

    assert fit.squared_error <= best + 1e-10
    assert fit.weights.sum() == pytest.approx(1.0)
    assert (fit.weights >= 0.0).all()
    assert fit.squared_error == pytest.approx(
        float(np.sum(np.square(target - streams @ fit.weights)))
    )


def nested_feature_frame() -> pd.DataFrame:
    fold = np.repeat(np.arange(1, 6), 3)
    row = np.arange(len(fold), dtype=float)
    parent = 2.0 + 0.20 * row + np.sin(row)
    ancc = -1.0 + 0.45 * row + np.cos(row / 2.0)
    best6 = 4.0 - 0.10 * row + np.sin(row / 3.0)
    target = 0.2 * parent + 0.3 * ancc + 0.5 * best6 + 0.01 * np.cos(row)
    frame = pd.DataFrame(
        {
            "_id": [f"well{value}_{index}" for index, value in enumerate(fold)],
            "_well_id": [f"well{value}" for value in fold],
            "_row_index": np.arange(len(fold)),
            "_target": target,
            "fold": fold,
            "parent_v006": parent,
            "sp_plane_ancc_k10": ancc,
            "sp_plane_best6_k10": best6,
        }
    )
    for outer in range(1, 6):
        frame[spatial.nested_stream_column(outer, "sp_plane_ancc_k10")] = ancc + 0.01 * outer * row
        frame[spatial.nested_stream_column(outer, "sp_plane_best6_k10")] = (
            best6 - 0.005 * outer * row
        )
    return frame


def fold_selection(selections, fold):
    return next(item for item in selections if int(item["fold"]) == fold)


def test_nested_weights_do_not_read_their_held_out_fold(tmp_path):
    before_path = tmp_path / "before.parquet"
    after_path = tmp_path / "after.parquet"
    before = nested_feature_frame()
    before.to_parquet(before_path, index=False)
    changed = before.copy()
    held_out = changed["fold"] == 1
    changed.loc[held_out, "_target"] += 1e9
    changed.loc[held_out, "parent_v006"] -= 1e8
    changed.loc[
        held_out,
        spatial.nested_stream_column(1, "sp_plane_ancc_k10"),
    ] += 2e8
    changed.loc[
        held_out,
        spatial.nested_stream_column(1, "sp_plane_best6_k10"),
    ] -= 3e8
    changed.to_parquet(after_path, index=False)

    before_selections, _ = spatial.nested_stack_weights(before_path)
    changed_selections, _ = spatial.nested_stack_weights(after_path)
    before_fold = fold_selection(before_selections, 1)
    changed_fold = fold_selection(changed_selections, 1)

    assert before_fold["meta_training_folds"] == [2, 3, 4, 5]
    assert before_fold["weights"] == changed_fold["weights"]
    assert before_fold["training_rows"] == changed_fold["training_rows"]
    assert before_fold["training_squared_error"] == changed_fold["training_squared_error"]


def test_nested_weight_fit_refuses_canonical_oof_without_outer_specific_streams(tmp_path):
    path = tmp_path / "canonical-only.parquet"
    frame = nested_feature_frame()
    nested = [column for column in frame if column.startswith("nested_outer_")]
    frame.drop(columns=nested).to_parquet(path, index=False)

    with pytest.raises(Exception, match="nested_outer|column|FieldRef|schema"):
        spatial.nested_stack_weights(path)


def write_parent_artifacts(
    workspace: Path,
    *,
    bad_id=False,
    bad_candidate=False,
    bad_target=False,
):
    model_dir = workspace / "models" / "v006"
    feature_dir = workspace / "data" / "features"
    model_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "_id": ["well1_851", "well1_852", "well2_851"],
            "_well_id": ["well1", "well1", "well2"],
            "_row_index": [851, 852, 851],
            "_target": [100.0, 101.0, 102.0],
            "fold": [1, 1, 2],
        }
    )
    if bad_id:
        rows.loc[0, "_id"] = "wrong_851"
    if bad_target:
        rows.loc[1, "_target"] = -999.0
    rows.to_parquet(model_dir / "oof_rows.parquet", index=False)
    np.save(model_dir / "oof_preds.npy", np.asarray([10.0, 11.0, 12.0]))
    pd.DataFrame({"placeholder": [1]}).to_parquet(feature_dir / "v003.parquet", index=False)
    selections = [
        {
            "fold": fold,
            "selected_candidate": (
                "wrong" if bad_candidate and fold == 3 else spatial.PARENT_CANDIDATE
            ),
        }
        for fold in range(1, 6)
    ]
    (model_dir / "cv_scores.json").write_text(
        json.dumps(
            {
                "fold_selections": selections,
                "final_candidate": spatial.PARENT_CANDIDATE,
            }
        ),
        encoding="utf-8",
    )
    return pd.DataFrame({"well": ["well1", "well2"], "fold": [1, 2]})


def test_parent_oof_reader_aligns_ids_rows_wells_and_folds(tmp_path):
    manifest = write_parent_artifacts(tmp_path)
    reader = spatial.ParentOOFReader(tmp_path, manifest)

    first = reader.read_well(
        "well1",
        np.asarray([851, 852]),
        expected_fold=1,
        expected_target=np.asarray([100.0, 101.0]),
    )
    second = reader.read_well(
        "well2",
        np.asarray([851]),
        expected_fold=2,
        expected_target=np.asarray([102.0]),
    )
    reader.finish()

    np.testing.assert_array_equal(first, [10.0, 11.0])
    np.testing.assert_array_equal(second, [12.0])


def test_parent_oof_reader_fails_on_id_or_locked_candidate_mismatch(tmp_path):
    bad_id_workspace = tmp_path / "bad-id"
    manifest = write_parent_artifacts(bad_id_workspace, bad_id=True)
    reader = spatial.ParentOOFReader(bad_id_workspace, manifest)
    with pytest.raises(ValueError, match="ID alignment"):
        reader.read_well(
            "well1",
            np.asarray([851, 852]),
            expected_fold=1,
            expected_target=np.asarray([100.0, 101.0]),
        )

    bad_candidate_workspace = tmp_path / "bad-candidate"
    manifest = write_parent_artifacts(bad_candidate_workspace, bad_candidate=True)
    with pytest.raises(ValueError, match="locked prediction stream"):
        spatial.ParentOOFReader(bad_candidate_workspace, manifest)


def test_parent_oof_reader_fails_on_corrupt_parent_target(tmp_path):
    manifest = write_parent_artifacts(tmp_path, bad_target=True)
    reader = spatial.ParentOOFReader(tmp_path, manifest)

    with pytest.raises(ValueError, match="target alignment"):
        reader.read_well(
            "well1",
            np.asarray([851, 852]),
            expected_fold=1,
            expected_target=np.asarray([100.0, 101.0]),
        )


def prepare_critical_parent_workspace(workspace: Path):
    train_root = workspace / "data" / "raw" / "train"
    train_root.mkdir(parents=True)
    (train_root / "parent__horizontal_well.csv").write_text(
        "TVT_input\n1.0\n",
        encoding="utf-8",
    )
    raw_root = workspace / "data" / "raw"
    (raw_root / "sample_submission.csv").write_text("id,tvt\nparent_1,0\n", encoding="utf-8")
    reports = workspace / "reports"
    reports.mkdir(parents=True)
    fold_path = reports / "canonical_outer_folds_v001.csv"
    fold_path.write_text("well,scored_rows,fold,splitter\n", encoding="utf-8")
    (reports / "v008_formation_spatial_plan.md").write_text("fixed plan\n", encoding="utf-8")
    (workspace / "config.yaml").write_text("competition: synthetic\n", encoding="utf-8")

    model_dir = workspace / "models" / "v006"
    feature_dir = workspace / "data" / "features"
    model_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    (model_dir / "cv_scores.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.pkl").write_bytes(b"model")
    (model_dir / "oof_preds.npy").write_bytes(b"oof")
    (model_dir / "oof_rows.parquet").write_bytes(b"rows")
    (model_dir / "test_preds.npy").write_bytes(b"test")
    (feature_dir / "v003.parquet").write_bytes(b"feature")
    run_payload = {
        "status": "completed",
        "source_sha256": spatial.sha256_file(spatial._parent_source_path()),
        "fold_manifest_sha256": spatial.sha256_file(fold_path),
        "data_fingerprint": spatial._legacy_parent_data_fingerprint(train_root),
        "feature_path": "data/features/v003.parquet",
        "final_candidate": spatial.PARENT_CANDIDATE,
    }
    run_path = model_dir / "run.json"
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")
    return fold_path, run_path, run_payload


def test_critical_hashes_include_and_validate_parent_run_contract(tmp_path):
    fold_path, run_path, valid = prepare_critical_parent_workspace(tmp_path)

    original = spatial._critical_input_hashes(tmp_path, fold_path)
    assert original["parent_run"] == spatial.sha256_file(run_path)

    with_note = {**valid, "audit_note": "content hash must change"}
    run_path.write_text(json.dumps(with_note), encoding="utf-8")
    changed = spatial._critical_input_hashes(tmp_path, fold_path)
    assert changed["parent_run"] != original["parent_run"]

    corruptions = {
        "status": "failed",
        "source_sha256": "wrong",
        "fold_manifest_sha256": "wrong",
        "data_fingerprint": "wrong",
        "feature_path": "data/features/wrong.parquet",
        "final_candidate": "wrong",
    }
    for field, value in corruptions.items():
        run_path.write_text(json.dumps({**valid, field: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            spatial._critical_input_hashes(tmp_path, fold_path)


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
def test_candidate_gate_uses_predeclared_score_and_fold_thresholds(
    overall,
    folds_better,
    expected,
):
    assert spatial.candidate_decision(overall, folds_better) == expected


@pytest.mark.parametrize(
    "failure_name",
    [
        "inference_failures",
        "nonfinite_failures",
        "donor_fold_violations",
        "self_donor_violations",
        "test_failures",
    ],
)
def test_candidate_gate_fails_closed_on_any_contract_violation(failure_name):
    assert spatial.candidate_decision(10.0, 5, **{failure_name: 1}) == "candidate_blocked_failures"


def prepare_smoke_workspace(workspace: Path, monkeypatch):
    train_root = workspace / "data" / "raw" / "train"
    for index in range(25):
        write_training_well(
            train_root,
            f"train{index:03d}",
            x=float(index % 5),
            y=float(index // 5),
            suffix_rows=index + 1,
        )
    manifest = spatial.assign_balanced_folds(spatial.scored_row_counts(train_root), 5)
    reports = workspace / "reports"
    reports.mkdir(parents=True)
    fold_path = reports / "canonical_outer_folds_v001.csv"
    manifest.to_csv(fold_path, index=False)
    (reports / "v008_formation_spatial_plan.md").write_text("fixed plan\n", encoding="utf-8")
    (workspace / "config.yaml").write_text("competition: synthetic\n", encoding="utf-8")

    test_root = workspace / "data" / "raw" / "test"
    test_root.mkdir(parents=True)
    synthetic_query_well(suffix_rows=2).loc[:, spatial.QUERY_COLUMNS].to_csv(
        test_root / "visible__horizontal_well.csv",
        index=False,
    )
    pd.DataFrame({"id": ["visible_851", "visible_852"], "tvt": [0.0, 0.0]}).to_csv(
        workspace / "data" / "raw" / "sample_submission.csv", index=False
    )

    model_dir = workspace / "models" / "v006"
    feature_dir = workspace / "data" / "features"
    model_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    (model_dir / "cv_scores.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.pkl").write_bytes(b"model")
    np.save(model_dir / "oof_preds.npy", np.asarray([1.0]))
    np.save(model_dir / "test_preds.npy", np.asarray([2.0, 3.0]))
    pd.DataFrame({"_id": ["placeholder"]}).to_parquet(
        model_dir / "oof_rows.parquet",
        index=False,
    )
    pd.DataFrame({"value": [1.0]}).to_parquet(feature_dir / "v003.parquet", index=False)
    (model_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "source_sha256": spatial.sha256_file(spatial._parent_source_path()),
                "fold_manifest_sha256": spatial.sha256_file(fold_path),
                "data_fingerprint": spatial._legacy_parent_data_fingerprint(train_root),
                "feature_path": "data/features/v003.parquet",
                "final_candidate": spatial.PARENT_CANDIDATE,
            }
        ),
        encoding="utf-8",
    )

    selected = []
    for row in manifest.sort_values(["scored_rows", "well"]).head(6).itertuples(index=False):
        selected.append((str(row.well), int(row.fold), int(row.scored_rows)))
    smoke_query_rows = sum(5 * (640 + suffix) for _, _, suffix in selected)
    full_query_rows = 5 * (int(manifest["scored_rows"].sum()) + 640 * len(manifest))
    monkeypatch.setattr(spatial, "SMOKE_SELECTION", tuple(selected))
    monkeypatch.setattr(spatial, "SMOKE_QUERY_ROWS", smoke_query_rows)
    monkeypatch.setattr(spatial, "FULL_NESTED_QUERY_ROWS", full_query_rows)
    monkeypatch.setattr(spatial, "_peak_rss_gib", lambda: 0.1)
    return manifest, selected


@pytest.mark.parametrize("wells", [1, 5, 7])
def test_runtime_smoke_requires_exactly_six_wells(tmp_path, wells):
    with pytest.raises(ValueError, match="exactly 6"):
        spatial.run_smoke(tmp_path, wells=wells)


def test_runtime_smoke_locks_exact_six_hashes_and_atomic_no_overwrite(tmp_path, monkeypatch):
    _, selected = prepare_smoke_workspace(tmp_path, monkeypatch)

    result = spatial.run_smoke(tmp_path, wells=6)
    smoke_path, required = spatial._require_runtime_smoke(tmp_path)

    assert result["target_read"] is False
    assert result["selected_wells"] == [
        {"well": well, "fold": fold, "suffix_rows": suffix} for well, fold, suffix in selected
    ]
    assert result["required_smoke_wells"] == 6
    assert result["unique_donor_contexts"] == 15
    assert result["hashes_stable"] is True
    assert result["input_sha256_start"] == result["input_sha256_end"]
    assert result["source_sha256"] == result["input_sha256_start"]["source"]
    assert result["source_sha256_end"] == result["input_sha256_end"]["source"]
    assert result["plan_sha256"] == result["input_sha256_start"]["plan"]
    assert result["plan_sha256_end"] == result["input_sha256_end"]["plan"]
    assert result["fold_manifest_sha256"] == result["input_sha256_start"]["fold_manifest"]
    assert result["fold_manifest_sha256_end"] == result["input_sha256_end"]["fold_manifest"]
    assert result["train_data_fingerprint"] == result["train_data_fingerprint_end"]
    assert result["test_data_fingerprint"] == result["test_data_fingerprint_end"]
    assert result["failures"] == []
    assert result["eligible_for_full_cv"] is True
    assert required == result
    assert json.loads(smoke_path.read_text(encoding="utf-8")) == result
    assert not Path(f"{smoke_path}.partial").exists()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        spatial.run_smoke(tmp_path, wells=6)

    tampered = result.copy()
    tampered["source_sha256"] = "tampered"
    smoke_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer satisfies"):
        spatial._require_runtime_smoke(tmp_path)

    smoke_path.write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "config.yaml").write_text("competition: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no longer satisfies"):
        spatial._require_runtime_smoke(tmp_path)


def test_visible_test_inference_is_target_safe_excludes_same_id_and_aligns_sample(
    tmp_path,
    monkeypatch,
):
    train_root, wells, donors = build_planar_donors(tmp_path)
    del train_root
    well = wells[0]
    test_root = tmp_path / "data" / "raw" / "test"
    test_root.mkdir(parents=True, exist_ok=True)
    horizontal = synthetic_query_well(suffix_rows=3).loc[:, spatial.QUERY_COLUMNS]
    horizontal.to_csv(test_root / f"{well}__horizontal_well.csv", index=False)

    suffix = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    sample = pd.DataFrame(
        {
            "id": [f"{well}_{suffix[2]}", f"{well}_{suffix[0]}", f"{well}_{suffix[1]}"],
            "tvt": 0.0,
        }
    )
    sample.to_csv(tmp_path / "data" / "raw" / "sample_submission.csv", index=False)
    model_dir = tmp_path / "models" / "v006"
    model_dir.mkdir(parents=True)
    np.save(model_dir / "test_preds.npy", np.asarray([125.0, 123.0, 124.0]))
    predictions, failures, diagnostics = spatial.predict_visible_test(
        tmp_path,
        donors,
        weights=(1.0, 0.0, 0.0),
    )

    assert failures == []
    assert predictions["_id"].tolist() == [f"{well}_{row}" for row in suffix]
    assert predictions["prediction"].tolist() == [123.0, 124.0, 125.0]
    assert diagnostics[0]["plane"]["same_id_donors_excluded"] == 1
    assert diagnostics[0]["plane"]["query_well_present_after_exclusion"] is False

    aligned = spatial._align_test_predictions(
        tmp_path,
        predictions.assign(prediction=[1.0, 2.0, 3.0]),
    )
    np.testing.assert_array_equal(aligned, [3.0, 1.0, 2.0])


def test_module_has_no_submit_command_or_kaggle_request_path():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "kaggle competitions submit" not in lowered
    assert "kaggleapi" not in lowered
    assert "competitionsubmission" not in lowered
    assert "def write_submission" not in lowered
    assert '"submissions"' not in lowered


def test_failed_parent_completion_audit_never_publishes_feature_artifact(
    tmp_path,
    monkeypatch,
):
    train_root = tmp_path / "data" / "raw" / "train"
    well = "audit001"
    horizontal_path = write_training_well(
        train_root,
        well,
        x=1.0,
        y=2.0,
        suffix_rows=2,
    )
    horizontal = pd.read_csv(horizontal_path)
    suffix = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    manifest = pd.DataFrame({"well": [well], "fold": [1]})
    placeholder = spatial.DonorTable(
        well_ids=(),
        coordinates=np.empty((0, 2)),
        formations=np.empty((0, len(spatial.FORMATIONS))),
        requested_well_ids=(well,),
    )

    raw_expected_target = horizontal.loc[suffix, "TVT"].to_numpy(dtype=float)

    class FailingParent:
        def read_well(
            self,
            read_well,
            evaluation_indices,
            expected_fold,
            expected_target,
        ):
            assert read_well == well
            assert expected_fold == 1
            np.testing.assert_array_equal(evaluation_indices, suffix)
            np.testing.assert_array_equal(expected_target, raw_expected_target)
            return np.full(len(evaluation_indices), 120.0)

        def finish(self):
            raise RuntimeError("parent completion audit failed")

    def fake_streams(query, donors, *, query_well_id):
        del query, donors
        assert query_well_id == well
        return spatial.SpatialPrediction(
            frame=pd.DataFrame(
                {
                    "_row_index": suffix,
                    "sp_plane_ancc_k10": np.full(len(suffix), 121.0),
                    "sp_plane_best6_k10": np.full(len(suffix), 122.0),
                }
            ),
            diagnostics={
                "selected_formation": "ANCC",
                "plane": {"query_well_present_after_exclusion": False},
            },
        )

    monkeypatch.setattr(spatial, "build_donor_catalog", lambda root, wells: placeholder)
    monkeypatch.setattr(
        spatial,
        "donor_contexts",
        lambda catalog, fold_manifest, folds: ({(1,): placeholder}, []),
    )
    monkeypatch.setattr(spatial, "ParentOOFReader", lambda workspace, folds: FailingParent())
    monkeypatch.setattr(spatial, "spatial_streams", fake_streams)
    final = tmp_path / "data" / "features" / "v999.parquet"
    final.parent.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="completion audit failed"):
        spatial.build_training_streams(tmp_path, manifest, final)

    partial = Path(f"{final}.partial")
    assert not final.exists()
    assert partial.exists()
    assert len(pd.read_parquet(partial)) == len(suffix)


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("stream build failed"),
        KeyboardInterrupt("operator stop"),
        spatial.TerminationRequested(f"received signal {signal.SIGTERM}"),
    ],
    ids=["exception", "keyboard-interrupt", "sigterm"],
)
def test_training_failure_atomically_publishes_failed_run_state(tmp_path, monkeypatch, failure):
    workspace = tmp_path / type(failure).__name__
    reports = workspace / "reports"
    reports.mkdir(parents=True)
    smoke_path = reports / "v008_formation_runtime_smoke.json"
    smoke_path.write_text("{}\n", encoding="utf-8")
    fold_path = reports / "canonical_outer_folds_v001.csv"
    fold_path.write_text("well,fold\n", encoding="utf-8")
    manifest = pd.DataFrame({"well": [f"well{fold}" for fold in range(1, 6)], "fold": range(1, 6)})
    hashes = {
        "source": "source-hash",
        "plan": "plan-hash",
        "parent_source": "parent-source-hash",
        "fold_manifest": "fold-hash",
    }
    observed_running = []
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def fail_after_running_state(workspace_arg, fold_manifest, feature_path):
        del fold_manifest, feature_path
        running_path = Path(workspace_arg) / "models" / ".v008.partial" / "run.json"
        observed_running.append(json.loads(running_path.read_text(encoding="utf-8")))
        if isinstance(failure, spatial.TerminationRequested):
            installed = signal.getsignal(signal.SIGTERM)
            assert installed is spatial._raise_on_sigterm
            installed(signal.SIGTERM, None)
        raise failure

    monkeypatch.setattr(
        spatial,
        "_require_runtime_smoke",
        lambda workspace_arg: (smoke_path, {"projected_full_hours": 0.1}),
    )
    monkeypatch.setattr(
        spatial,
        "ensure_fold_manifest",
        lambda workspace_arg, folds: (fold_path, manifest),
    )
    monkeypatch.setattr(spatial, "_critical_input_hashes", lambda *args: hashes.copy())
    monkeypatch.setattr(spatial, "data_fingerprint", lambda root: "data-hash")
    monkeypatch.setattr(
        spatial,
        "next_feature_path",
        lambda workspace_arg: Path(workspace_arg) / "data" / "features" / "v999.parquet",
    )
    monkeypatch.setattr(spatial, "git_commit", lambda root: "commit")
    monkeypatch.setattr(spatial, "git_dirty_paths", lambda root: [])
    monkeypatch.setattr(spatial, "build_training_streams", fail_after_running_state)

    with pytest.raises(type(failure), match=str(failure)):
        spatial.train(workspace)

    assert observed_running[0]["status"] == "running"
    model_dir = workspace / "models" / "v008"
    run_path = model_dir / "run.json"
    failed = json.loads(run_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error"] == f"{type(failure).__name__}: {failure}"
    assert failed["test_predictions_saved"] is False
    assert not (workspace / "models" / ".v008.partial").exists()
    assert not Path(f"{run_path}.partial").exists()
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm_handler
