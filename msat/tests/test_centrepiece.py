from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from msat.figures.centrepiece import (
    CEILING_DASH,
    PREVIEW_WARNING,
    Cell,
    FigureInputMissing,
    build_from_disk,
    ceiling_curve,
    ceiling_from_artifact,
    cells_from_recall,
    cells_from_scores,
    centrepiece,
    merge_cells,
    preview_report,
    synthetic_ceiling,
    synthetic_cells,
    write_centrepiece,
    x_of,
)
from msat.figures.svg import (
    Figure,
    Series,
    Sheet,
    fmt,
    is_well_formed,
    iter_texts,
    parse,
    polylines,
    rects,
    shaded_regions,
)


@pytest.fixture()
def sheet() -> Sheet:
    return centrepiece(synthetic_cells(), synthetic_ceiling(), scale="S")


@pytest.fixture()
def svg(sheet: Sheet) -> str:
    return sheet.render()


def test_the_svg_is_well_formed_xml(svg: str) -> None:
    assert is_well_formed(svg)


def test_the_root_is_an_svg_with_a_viewbox(svg: str) -> None:
    assert svg.lstrip().startswith("<svg")
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg
    assert "viewBox=" in svg


def test_writing_lands_a_file_and_reports_its_size(tmp_path: Path) -> None:
    rep = write_centrepiece(
        synthetic_cells(), synthetic_ceiling(), tmp_path / "fig_centrepiece.svg"
    )
    for entry in rep["scale_reports"].values():
        p = Path(entry["path"])
        assert p.exists()
        assert entry["n_bytes"] == p.stat().st_size > 0


def test_the_two_scales_get_two_named_files(tmp_path: Path) -> None:
    rep = write_centrepiece(
        synthetic_cells(), synthetic_ceiling(), tmp_path / "fig_centrepiece.svg"
    )
    names = {s: Path(r["path"]).name for s, r in rep["scale_reports"].items()}
    assert names == {
        "S": "fig_centrepiece_S.svg",
        "M": "fig_centrepiece_M.svg",
    }


def test_writing_creates_the_directory(tmp_path: Path) -> None:
    dest = tmp_path / "deep" / "nested" / "fig_centrepiece.svg"
    write_centrepiece(synthetic_cells(), synthetic_ceiling(), dest)
    assert dest.parent.is_dir()


def test_there_is_no_shaded_region_anywhere(svg: str) -> None:
    assert shaded_regions(svg) == 0


def test_the_only_rects_are_legend_boxes(svg: str) -> None:
    found = rects(svg)
    assert found, "there should be one legend box per panel"
    assert all("legend-box" in r for r in found)


def test_square_markers_are_paths_not_rects(svg: str) -> None:
    assert '<path d="M' in svg


def test_the_ceiling_is_dashed(svg: str) -> None:
    assert f'stroke-dasharray="{CEILING_DASH}"' in svg


def test_the_ceiling_is_present_and_labelled_as_a_bound(sheet: Sheet) -> None:
    for panel in sheet.panels:
        labels = [s.label for s in panel.series]
        assert "ceiling(K)" in labels
        assert panel.annotations == [] or any("BOUND" in a for a in panel.annotations)
    assert any("BOUND" in a for a in sheet.panels[-1].annotations)


def test_a_figure_that_did_shade_would_be_detected() -> None:
    bad = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<rect x="10" y="10" width="200" height="120" fill="#ccc"/>'
        '<rect x="300" y="4" width="90" height="40" class="legend-box"/>'
        "</svg>"
    )
    assert shaded_regions(bad) == 1, (
        "a filled band must be counted; otherwise the zero above is unfalsifiable"
    )
    assert shaded_regions(bad) != 0


def test_the_legend_box_is_not_mistaken_for_a_band() -> None:
    only_legend = '<svg xmlns="x"><rect class="legend-box" width="9" height="9"/></svg>'
    assert shaded_regions(only_legend) == 0


def test_each_panel_draws_one_line_per_budget_level(sheet: Sheet) -> None:
    bs = sorted({c.b for c in synthetic_cells() if c.scale == "S"})
    for panel in sheet.panels:
        budget_lines = [s for s in panel.series if s.label.startswith("B=")]
        assert len(budget_lines) == len(bs), (
            "a panel that averaged over B has collapsed the K x B plane"
        )


def test_no_budget_line_covers_the_whole_b_range(svg: str) -> None:
    for points in polylines(svg):
        n = len(points.split())
        assert n >= 2


def test_the_budget_lines_are_distinguishable(sheet: Sheet) -> None:
    for panel in sheet.panels:
        colours = {s.colour for s in panel.series if s.label.startswith("B=")}
        assert len(colours) == len([s for s in panel.series if s.label.startswith("B=")])


def test_both_panels_share_a_y_range(sheet: Sheet) -> None:
    assert sheet.panels[0].y_range == sheet.panels[1].y_range


def test_the_panels_are_named_for_their_metric(sheet: Sheet) -> None:
    titles = [p.panel_title for p in sheet.panels]
    assert titles == ["accuracy", "recall@K"]


def test_the_x_axis_is_log2_not_raw_k() -> None:
    assert x_of(1) == 0.0
    assert x_of(2) == 1.0
    assert x_of(32) == 5.0


def test_log2_is_exact_for_the_power_of_two_grid() -> None:
    for k in (1, 2, 4, 8, 16, 32, 64):
        assert x_of(k) == float(k.bit_length() - 1)


def test_the_axis_label_says_log2(sheet: Sheet) -> None:
    assert all("log2" in p.xlabel for p in sheet.panels)


def test_the_scale_is_stated_on_the_figure(svg: str) -> None:
    assert "log2" in " ".join(iter_texts(svg))


def test_corpus_fraction_is_annotated_rather_than_axised(sheet: Sheet) -> None:
    joined = " ".join(sheet.panels[-1].annotations)
    assert "corpus_fraction" in joined
    assert "not a third axis" in joined


def test_the_annotation_reports_a_range_not_a_point() -> None:
    joined = " ".join(centrepiece(synthetic_cells(), synthetic_ceiling()).panels[-1].annotations)
    m = re.search(r"corpus_fraction ([\d.]+)-([\d.]+)", joined)
    assert m is not None
    lo, hi = float(m.group(1)), float(m.group(2))
    assert lo < hi


def test_the_interaction_is_named_so_the_read_is_predictable(sheet: Sheet) -> None:
    assert any("crossing" in a for a in sheet.panels[-1].annotations)


def test_the_metric_name_is_printed_on_the_figure() -> None:
    fig = centrepiece(synthetic_cells(), synthetic_ceiling(), metric_name="official")
    texts = " ".join(iter_texts(fig.render()))
    assert "official" in texts


def test_the_metric_name_reaches_the_report() -> None:
    fig = centrepiece(synthetic_cells(), synthetic_ceiling(), metric_name="em")
    assert fig.panels[0].subtitle.startswith("accuracy = em")


def test_without_a_metric_name_no_subtitle_claims_one() -> None:
    fig = centrepiece(synthetic_cells(), synthetic_ceiling())
    assert "accuracy =" not in fig.panels[0].subtitle


def test_the_ceiling_is_only_drawn_for_ks_it_was_given() -> None:
    curve = ceiling_curve([1, 2, 4, 8, 16, 32], {1: 0.5, 2: 0.9})
    assert [p[0] for p in curve] == [0.0, 1.0]


def test_an_empty_ceiling_yields_no_ceiling_series() -> None:
    fig = centrepiece(synthetic_cells(), {})
    for panel in fig.panels:
        assert all(s.dash != CEILING_DASH for s in panel.series)


def test_ceiling_values_outside_zero_one_are_refused() -> None:
    with pytest.raises(FigureInputMissing, match="outside"):
        ceiling_from_artifact({"scales": {"S": {"ceiling": {"1": 1.4}}}}, "S")
    with pytest.raises(FigureInputMissing, match="outside"):
        ceiling_from_artifact({"scales": {"S": {"ceiling": {"1": -0.2}}}}, "S")


def test_a_nan_ceiling_is_refused() -> None:
    with pytest.raises(FigureInputMissing, match="NaN"):
        ceiling_from_artifact({"scales": {"S": {"ceiling": {"1": "nan"}}}}, "S")


def test_the_endpoints_of_the_probability_range_are_accepted() -> None:
    got = ceiling_from_artifact({"scales": {"S": {"ceiling": {"1": 0.0, "4": 1.0}}}}, "S")
    assert got == {1: 0.0, 4: 1.0}


def test_a_non_numeric_ceiling_is_refused_not_coerced() -> None:
    with pytest.raises(FigureInputMissing):
        ceiling_from_artifact({"scales": {"S": {"ceiling": {"1": "n/a"}}}}, "S")


def test_a_missing_scale_is_a_named_error() -> None:
    with pytest.raises(FigureInputMissing, match="no scale"):
        ceiling_from_artifact({"scales": {"M": {"ceiling": {"1": 0.5}}}}, "S")


def test_the_ceiling_keys_are_strings_in_json_and_ints_after_loading() -> None:
    doc = json.loads('{"scales": {"S": {"ceiling": {"1": 0.5, "4": 0.9}}}}')
    got = ceiling_from_artifact(doc, "S")
    assert got == {1: 0.5, 4: 0.9}
    assert all(isinstance(k, int) for k in got)


def test_a_figure_with_no_cells_is_refused() -> None:
    with pytest.raises(ValueError, match="no cells"):
        centrepiece([], synthetic_ceiling())


def test_rendering_is_byte_stable() -> None:
    a = centrepiece(synthetic_cells(), synthetic_ceiling()).render()
    b = centrepiece(synthetic_cells(), synthetic_ceiling()).render()
    assert a == b


def test_fixed_precision_never_emits_float_noise() -> None:
    assert fmt(0.30000000000000004) == "0.3"
    assert fmt(1.0) == "1"
    assert fmt(0.0) == "0"
    assert fmt(1e-05) == "0"


def test_labels_are_xml_escaped(tmp_path: Path) -> None:
    fig = Figure(
        title="K & B <interaction>",
        series=[Series(label="a & b", points=[(0.0, 0.1), (1.0, 0.2)])],
    )
    svg = fig.render()
    assert is_well_formed(svg), "an unescaped ampersand makes the file unparseable"
    assert "&amp;" in svg


def test_the_preview_warning_survives_escaping() -> None:
    assert "&" not in PREVIEW_WARNING or "&amp;" in PREVIEW_WARNING
    assert "NOT RESULTS" in PREVIEW_WARNING


def test_the_preview_stamps_itself_in_the_report(tmp_path: Path) -> None:
    rep = preview_report(tmp_path)
    assert rep["is_preview"] is True
    assert rep["source"] == "synthetic"
    assert rep["warning"] == PREVIEW_WARNING


def test_the_preview_stamps_itself_on_the_figure(tmp_path: Path) -> None:
    rep = preview_report(tmp_path)
    for entry in rep["scale_reports"].values():
        on_disk = Path(entry["path"]).read_text(encoding="utf-8")
        assert PREVIEW_WARNING in on_disk, "the stamp must be in the FILE, not the report"
        assert PREVIEW_WARNING in " ".join(iter_texts(on_disk))


def test_the_preview_does_not_claim_a_metric_it_does_not_have(tmp_path: Path) -> None:
    rep = preview_report(tmp_path)
    assert rep["metric"] is None
    on_disk = Path(rep["scale_reports"]["S"]["path"]).read_text(encoding="utf-8")
    assert "accuracy =" not in on_disk


def test_a_data_figure_is_not_marked_as_a_preview(tmp_path: Path) -> None:
    rep = write_centrepiece(
        synthetic_cells(), synthetic_ceiling(), tmp_path / "fig_centrepiece.svg"
    )
    for entry in rep["scale_reports"].values():
        assert all(PREVIEW_WARNING not in a for a in entry["annotations"])


class _FakeRow:
    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def test_scores_aggregate_to_one_cell_per_k_b_scale() -> None:
    rows = [
        _FakeRow(metric="em", value=1.0, k=4, b=128.0, scale="S", qid="q1", answerable=True),
        _FakeRow(metric="em", value=0.0, k=4, b=128.0, scale="S", qid="q2", answerable=True),
        _FakeRow(metric="em", value=1.0, k=8, b=128.0, scale="S", qid="q3", answerable=True),
    ]
    cells = cells_from_scores(rows, metric="em")
    by_k = {c.k: c for c in cells}
    assert by_k[4].accuracy == pytest.approx(0.5)
    assert by_k[4].n_questions == 2
    assert by_k[8].accuracy == 1.0


def test_scores_ignore_rows_from_other_metrics() -> None:
    rows = [
        _FakeRow(metric="em", value=1.0, k=4, b=128.0, scale="S", qid="q1", answerable=True),
        _FakeRow(metric="official", value=0.0, k=4, b=128.0, scale="S", qid="q2", answerable=True),
    ]
    assert cells_from_scores(rows, metric="em")[0].accuracy == 1.0
    assert cells_from_scores(rows, metric="official")[0].accuracy == 0.0


def test_abstentions_are_excluded_from_the_accuracy_mean() -> None:
    rows = [
        _FakeRow(metric="em", value=1.0, k=4, b=128.0, scale="S", qid="q1", answerable=True),
        _FakeRow(metric="em", value=0.0, k=4, b=128.0, scale="S", qid="abs", answerable=False),
    ]
    cells = cells_from_scores(rows, metric="em")
    assert cells[0].accuracy == 1.0
    assert cells[0].n_questions == 1


def test_a_duplicate_qid_is_an_error_not_a_silent_reweighting() -> None:
    rows = [
        _FakeRow(metric="em", value=1.0, k=4, b=128.0, scale="S", qid="q1", answerable=True),
        _FakeRow(metric="em", value=0.0, k=4, b=128.0, scale="S", qid="q1", answerable=True),
    ]
    with pytest.raises(ValueError, match="duplicate qid"):
        cells_from_scores(rows, metric="em")


def test_null_values_are_dropped_not_treated_as_zero() -> None:
    rows = [
        _FakeRow(metric="em", value=None, k=4, b=128.0, scale="S", qid="q1", answerable=True),
        _FakeRow(metric="em", value=1.0, k=4, b=128.0, scale="S", qid="q2", answerable=True),
    ]
    assert cells_from_scores(rows, metric="em")[0].accuracy == 1.0


def test_recall_uses_the_summary_mean_not_a_mean_of_means() -> None:

    class S:
        def __init__(self, k: int, b: float, scale: str, vals: list[float]) -> None:
            self.k, self.b, self.scale = k, b, scale
            self.answerable = [object()] * len(vals)
            self._vals = vals

        @property
        def mean_recall(self) -> float:
            return sum(self._vals) / len(self._vals)

    cells = cells_from_recall([S(4, 128.0, "S", [1.0, 1.0, 1.0, 0.0])])
    assert cells[0].recall == 0.75
    assert cells[0].n_questions == 4


def test_merging_keeps_a_cell_that_has_only_one_metric() -> None:
    acc = [Cell(k=4, b=128.0, accuracy=0.5, recall=None, scale="S")]
    rec = [Cell(k=4, b=128.0, accuracy=None, recall=0.8, scale="S")]
    merged = merge_cells(acc, rec)
    assert len(merged) == 1
    assert merged[0].accuracy == 0.5
    assert merged[0].recall == 0.8


def test_merging_does_not_overwrite_a_value_with_none() -> None:
    acc = [Cell(k=4, b=128.0, accuracy=0.5, recall=None, scale="S")]
    rec = [Cell(k=4, b=128.0, accuracy=None, recall=0.8, scale="S")]
    assert merge_cells(rec, acc)[0].accuracy == 0.5


def test_merging_unions_cells_rather_than_intersecting() -> None:
    a = [Cell(k=4, b=128.0, accuracy=0.5, recall=None, scale="S")]
    b = [Cell(k=8, b=128.0, accuracy=None, recall=0.8, scale="S")]
    assert len(merge_cells(a, b)) == 2


def test_building_with_no_inputs_refuses_and_names_both(tmp_path: Path) -> None:
    with pytest.raises(FigureInputMissing) as exc:
        build_from_disk(
            scores_path=tmp_path / "absent.parquet",
            ceiling_path=tmp_path / "absent.json",
            metric="official",
            out_dir=tmp_path,
        )
    msg = str(exc.value)
    assert "scores" in msg and "ceiling" in msg


def test_building_with_one_missing_input_names_that_one(tmp_path: Path) -> None:
    (tmp_path / "scores.parquet").write_bytes(b"")
    with pytest.raises(FigureInputMissing) as exc:
        build_from_disk(
            scores_path=tmp_path / "scores.parquet",
            ceiling_path=tmp_path / "absent.json",
            metric="official",
            out_dir=tmp_path,
        )
    msg = str(exc.value)
    assert "ceiling" in msg
    assert "scores:" not in msg


def test_the_refusal_names_the_flag_that_would_work(tmp_path: Path) -> None:
    with pytest.raises(FigureInputMissing, match="--preview"):
        build_from_disk(
            scores_path=tmp_path / "absent.parquet",
            ceiling_path=tmp_path / "ceiling.json",
            metric="official",
            out_dir=tmp_path,
        )


def test_the_committed_ceiling_artifact_loads() -> None:
    p = Path("results/ceiling.json")
    if not p.exists():  # pragma: no cover - artifact is committed
        pytest.skip("results/ceiling.json not present")
    doc = json.loads(p.read_text(encoding="utf-8"))
    for scale in doc["scales"]:
        got = ceiling_from_artifact(doc, scale)
        assert got, scale
        assert got[min(got)] < 1.0, "ceiling(K=1) below 1 is what makes it a bound"
        assert all(0.0 <= v <= 1.0 for v in got.values())
        ks = sorted(got)
        assert ks == sorted(ks) and got[ks[-1]] == pytest.approx(1.0)


def test_an_empty_sheet_refuses() -> None:
    with pytest.raises(ValueError, match="at least one panel"):
        Sheet(panels=[]).render()


def test_element_census_reports_what_was_drawn(svg: str) -> None:
    counts = parse(svg)
    assert counts["svg"] == 3
    assert counts["polyline"] == 8
