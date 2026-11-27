from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from msat.planning import ExecutablePlan, plan_from_manifest, verify_crossed_grid

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
MANIFEST = CONFIGS.parent / "results" / "manifest.json"


@pytest.fixture(scope="module")
def design() -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / "design.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def qids() -> dict[str, list[str]]:
    return {"S": [f"s_q{i}" for i in range(5)], "M": [f"m_q{i}" for i in range(5)]}


@pytest.fixture(scope="module")
def plan(design: dict[str, Any], qids: dict[str, list[str]]) -> ExecutablePlan:
    if not MANIFEST.exists():
        pytest.skip("run `msat build-manifest --out results/manifest.json` first")
    return plan_from_manifest(
        MANIFEST, qids, seed=int(design["seed"]), runs_per_cell=int(design["runs_per_cell"])
    )


def _report(plan: ExecutablePlan, design: dict[str, Any]) -> dict[str, Any]:
    return verify_crossed_grid(
        plan,
        k_grid=design["K_grid"],
        b_grid=[b["value"] for b in design["B_grid"]],
        scales=design["scales"],
        readers=design["readers"],
    )


def test_every_k_b_cell_is_present_for_every_scale_and_reader(plan, design) -> None:
    report = _report(plan, design)
    assert report["crossed"], "\n".join(report["problems"])


def test_the_plan_has_one_cell_per_k_b_scale_reader(plan, design) -> None:
    n = (
        len(design["K_grid"])
        * len(design["B_grid"])
        * len(design["scales"])
        * len(design["readers"])
    )
    assert len(plan.conditions) == n, (
        f"{len(plan.conditions)} conditions planned, expected {n}. A missing cell here is a "
        f"marginal sweep in disguise."
    )


def test_the_grid_is_not_two_marginal_sweeps(plan, design) -> None:
    pairs = {(c.condition.k, c.condition.b) for c in plan.conditions}
    b_values = {bb for _, bb in pairs}
    k_values = {kk for kk, _ in pairs}
    assert len(b_values) == len(design["B_grid"])
    assert len(k_values) == len(design["K_grid"])

    for kk in k_values:
        budgets = {bb for a, bb in pairs if a == kk}
        assert len(budgets) > 1, f"K={kk} varies no budget"
    for bb in b_values:
        depths = {a for a, b in pairs if b == bb}
        assert len(depths) > 1, f"B={bb} varies no depth"


def test_every_reader_sees_the_same_grid(plan, design) -> None:
    report = _report(plan, design)
    assert report["reader_balanced"], (
        "readers do not see identical (K,B) grids; reader is confounded with the "
        "design. A per-reader K cap creates this silently."
    )


def test_rejected_cells_are_carried_not_dropped(plan) -> None:
    assert len(plan.conditions) + len(plan.rejected) > 0
    assert all(0.0 <= c.degeneracy_fraction <= 1.0 for c in plan.rejected)


def test_the_schedule_covers_every_cell_question_and_replicate(plan, qids, design) -> None:
    expected = {
        (c.condition.key, q, r)
        for c in plan.conditions
        for q in qids.get(c.condition.scale, [])
        for r in range(design["runs_per_cell"])
    }
    actual = {(u.condition.key, u.qid, u.repetition) for u in plan.schedule}
    assert actual == expected
    assert len(actual) == len(plan.schedule), "the schedule contains duplicates"


def test_the_plan_carries_the_config_hash_and_seed(plan, design) -> None:
    assert len(plan.config_hash) == 64
    assert plan.seed == int(design["seed"])
    assert plan.repetition == int(design["runs_per_cell"])


def test_the_plan_is_deterministic(design, qids) -> None:
    a = plan_from_manifest(MANIFEST, qids, seed=int(design["seed"]), runs_per_cell=1)
    b = plan_from_manifest(MANIFEST, qids, seed=int(design["seed"]), runs_per_cell=1)
    assert [u.condition.key for u in a.schedule] == [u.condition.key for u in b.schedule]
    assert [u.qid for u in a.schedule] == [u.qid for u in b.schedule]


def test_the_manifest_itself_is_a_complete_crossing_on_disk(design) -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    accepted = [c for c in raw["cells"] if c["status"] == "accepted"]
    seen = {(c["K"], c["B"], c["scale"], c["reader"]) for c in accepted}
    expected = {
        (k, b["value"], s, r)
        for k in design["K_grid"]
        for b in design["B_grid"]
        for s in design["scales"]
        for r in design["readers"]
    }
    assert seen == expected
