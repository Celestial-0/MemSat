"""The human-labelled sample and judge-vs-human agreement."""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msat.arms.evidence import ABSTENTION_SUFFIX

SAMPLE_FRACTION = 0.10
DEFAULT_MIN_PER_CELL = 5

HUMAN_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "qid",
    "condition_key",
    "arm",
    "K",
    "B",
    "scale",
    "reader",
    "question",
    "answer",
    "gold",
    "is_abstention",
    "human_verdict",
    "human_note",
)

JUDGE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "qid",
    "condition_key",
    "judge_a",
    "verdict_a",
    "judge_b",
    "verdict_b",
    "em",
    "f1",
)


@dataclass(frozen=True, slots=True)
class SampleItem:
    sample_id: str
    qid: str
    condition_key: str
    arm: str
    k: int
    b: float
    scale: str
    reader: str
    question: str
    answer: str
    gold: str

    @property
    def is_abstention(self) -> bool:
        return self.qid.endswith(ABSTENTION_SUFFIX)

    def human_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "qid": self.qid,
            "condition_key": self.condition_key,
            "arm": self.arm,
            "K": self.k,
            "B": self.b,
            "scale": self.scale,
            "reader": self.reader,
            "question": self.question,
            "answer": self.answer,
            "gold": self.gold,
            "is_abstention": self.is_abstention,
            "human_verdict": "",
            "human_note": "",
        }


@dataclass
class SamplePlan:
    fraction: float = SAMPLE_FRACTION
    min_per_cell: int = DEFAULT_MIN_PER_CELL
    seed: int = 20261012
    items: list[SampleItem] = field(default_factory=list)
    per_cell: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def n_abstention(self) -> int:
        return sum(1 for i in self.items if i.is_abstention)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "min_per_cell": self.min_per_cell,
            "seed": self.seed,
            "n_sampled": self.n,
            "n_abstention": self.n_abstention,
            "n_cells": len(self.per_cell),
            "per_cell": dict(sorted(self.per_cell.items())),
            "blinding": (
                "human_sample.csv contains no judge id, verdict, EM or F1. A labeller "
                "shown a judge's answer agrees with it more often, and the resulting "
                "judge-vs-human figure measures the anchor rather than the judge."
            ),
        }


def allocate(
    n_items: int, *, fraction: float = SAMPLE_FRACTION, floor: int = DEFAULT_MIN_PER_CELL
) -> int:
    if n_items <= 0:
        return 0
    take = max(floor, round(n_items * fraction))
    return min(n_items, take)


def draw_sample(
    candidates: Sequence[Mapping[str, Any]],
    *,
    fraction: float = SAMPLE_FRACTION,
    min_per_cell: int = DEFAULT_MIN_PER_CELL,
    seed: int = 20261012,
) -> SamplePlan:
    plan = SamplePlan(fraction=fraction, min_per_cell=min_per_cell, seed=seed)
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        by_cell.setdefault(str(row["condition_key"]), []).append(row)

    n_total = len(candidates)
    for cell in sorted(by_cell):
        rows = sorted(by_cell[cell], key=lambda r: str(r["qid"]))
        take = allocate(len(rows), fraction=fraction, floor=min_per_cell)
        rng = random.Random(f"{seed}:{cell}")
        chosen = rng.sample(rows, take)
        plan.per_cell[cell] = take
        for row in sorted(chosen, key=lambda r: str(r["qid"])):
            plan.items.append(
                SampleItem(
                    sample_id=f"{cell}#{row['qid']}",
                    qid=str(row["qid"]),
                    condition_key=cell,
                    arm=str(row.get("arm", "")),
                    k=int(row.get("K", 0) or 0),
                    b=float(row.get("B", 0.0) or 0.0),
                    scale=str(row.get("scale", "")),
                    reader=str(row.get("reader", "")),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    gold=str(row.get("gold", "")),
                )
            )
    plan.items.sort(key=lambda i: i.sample_id)
    plan.n_total = n_total  # type: ignore[attr-defined]
    return plan


def write_human_csv(plan: SamplePlan, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(HUMAN_COLUMNS))
        w.writeheader()
        for item in plan.items:
            w.writerow(item.human_row())
    return path


def write_judge_side(
    plan: SamplePlan,
    verdicts: Mapping[tuple[str, str], Mapping[str, Any]],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(JUDGE_COLUMNS))
        w.writeheader()
        for item in plan.items:
            entry = dict(verdicts.get((item.condition_key, item.qid), {}))
            w.writerow(
                {
                    "sample_id": item.sample_id,
                    "qid": item.qid,
                    "condition_key": item.condition_key,
                    "judge_a": entry.get("judge_a", ""),
                    "verdict_a": entry.get("verdict_a", ""),
                    "judge_b": entry.get("judge_b", ""),
                    "verdict_b": entry.get("verdict_b", ""),
                    "em": entry.get("em", ""),
                    "f1": entry.get("f1", ""),
                }
            )
    return path


def read_human_csv(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["sample_id"]] = row
    return out


def judge_vs_human(
    labelled: Mapping[str, Mapping[str, Any]],
    judge_side: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    from msat.judges.verdicts import Verdict, as_verdict

    shared = sorted(set(labelled) & set(judge_side))
    stats: dict[str, dict[str, int]] = {}
    unlabelled: list[str] = []
    for sid in shared:
        raw = str(labelled[sid].get("human_verdict", "")).strip()
        if not raw:
            unlabelled.append(sid)
            continue
        try:
            human = as_verdict(raw)
        except ValueError:
            unlabelled.append(sid)
            continue
        jv = judge_side[sid]
        pair = f"{jv.get('judge_a', '?')} vs {jv.get('judge_b', '?')}"
        s = stats.setdefault(
            pair, {"n": 0, "both_correct": 0, "both_wrong": 0, "panel_only": 0, "human_only": 0}
        )
        s["n"] += 1
        ja = as_verdict(jv["verdict_a"]) if jv.get("verdict_a") else None
        jb = as_verdict(jv["verdict_b"]) if jv.get("verdict_b") else None
        panel_right = Verdict.CORRECT in (ja, jb)
        if panel_right and human is Verdict.CORRECT:
            s["both_correct"] += 1
        elif panel_right:
            s["panel_only"] += 1
        elif human is Verdict.CORRECT:
            s["human_only"] += 1
        else:
            s["both_wrong"] += 1

    out: dict[str, Any] = {
        "pairs": stats,
        "n_shared": len(shared),
        "n_unlabelled": len(unlabelled),
        "unlabelled_sample_ids": unlabelled[:20],
    }
    from msat.judges.kappa import CAUTIONARY_KAPPA

    for pair, s in stats.items():
        n = max(s["n"], 1)
        out["pairs"][pair] = {
            **s,
            "panel_correct_rate": (s["both_correct"] + s["panel_only"]) / n,
            "human_correct_rate": (s["both_correct"] + s["human_only"]) / n,
            "raw_agreement": (s["both_correct"] + s["both_wrong"]) / n,
            "exceeds_chance_by": (s["both_correct"] + s["both_wrong"]) / n - CAUTIONARY_KAPPA,
        }
    if unlabelled:
        out["note"] = (
            f"{len(unlabelled)} sampled item(s) have no human verdict yet. They are "
            f"excluded from the agreement counts and named, so a partial round is "
            f"visible rather than silently smaller."
        )
    return out
