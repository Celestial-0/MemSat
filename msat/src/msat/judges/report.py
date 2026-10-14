"""`results/judge_audit.json` — the STANDALONE artefact (PRD §R8.7).

R8.7 requires this artifact to exist "independent of whether the interaction is
found". That is the whole design constraint, and it is easy to violate by accident.

WHY STANDALONE MATTERS
----------------------
The most natural way to build a judge audit is as a section of the results file.
Then it is produced at analysis time, which means it is produced *if the analysis
runs*, which means it is produced *if the sweep produced something worth analysing*.
A zero interaction, a broken run, or a run abandoned at hour 100 would then produce
no judge audit - and "we found nothing" is exactly the case where an unreliable
panel is the reason and nobody would know.

So: separate file, separate command, written from the verdict records alone. It
does not read `scores.parquet`, does not read `conditions.parquet`, and does not
need a run to have finished.

THE FOUR THINGS IT MUST CARRY
-----------------------------
1. The panel's pairwise kappa, with the 0.118 bar from 2511.02246, and the
   agreement matrix - kappa alone says the panel is weak, the matrix says why.
2. The R8.6 two-pass replication: n_flips per judge, the net delta it is NOT to be
   confused with, and 2609.38021's 8-of-500 as the reference bound.
3. The noise floor, and the rule that the WORST judge sets it.
4. Whether the panel is usable at all - a single boolean a script can check before
   it reports a headline number.

It is a contribution in its own right (G13) and it survives a zero interaction.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msat.judges.audit import flip_verdict, format_flip_table, reference_bound
from msat.judges.kappa import CAUTIONARY_KAPPA, panel_agreement
from msat.judges.verdicts import ItemKey, Verdict

AUDIT_SCHEMA_VERSION = 1


@dataclass
class JudgeAudit:
    """Everything R8.7 asks for, assembled from the verdict records alone."""

    n_judges: int = 0
    n_items: int = 0
    panel: dict[str, Any] = field(default_factory=dict)
    replication: dict[str, Any] = field(default_factory=dict)
    human: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def panel_usable(self) -> bool:
        return bool(self.panel.get("panel_usable", False))

    @property
    def noise_floor(self) -> float | None:
        return self.replication.get("worst_flip_rate")

    def headline(self) -> dict[str, Any]:
        """The three numbers a reader should not have to hunt for."""
        return {
            "panel_usable": self.panel_usable,
            "kappa_min": self.panel.get("kappa_min"),
            "cautionary_bar": CAUTIONARY_KAPPA,
            "judge_noise_floor": self.noise_floor,
            "n_judges": self.n_judges,
            "n_items": self.n_items,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "requirement": "PRD R8.7 - judge audit as a STANDALONE artefact",
            "standalone": (
                "Built from verdict records only. It does not read scores.parquet or "
                "conditions.parquet and does not require a run to have finished, so a "
                "zero interaction still produces it - which is exactly the case where "
                "an unreliable panel is the reason and nobody would otherwise know."
            ),
            "headline": self.headline(),
            "panel_agreement": self.panel,
            "two_pass_replication": self.replication,
            "human_agreement": self.human,
            "reference_bound": reference_bound(),
            "warnings": self.warnings,
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


def _verdicts_by_judge(
    records: Sequence[Mapping[str, Any]], *, pass_index: int = 1
) -> dict[str, dict[ItemKey, Verdict]]:
    out: dict[str, dict[ItemKey, Verdict]] = {}
    for r in records:
        if int(r.get("pass_index", 1)) != pass_index:
            continue
        j = str(r["judge_id"])
        out.setdefault(j, {})[(str(r["condition_key"]), str(r["qid"]))] = Verdict(str(r["verdict"]))
    return out


def build_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    human: Mapping[str, Any] | None = None,
) -> JudgeAudit:
    """Assemble the audit from flat verdict records.

    `records` may contain several `pass_index` values; pass 1 drives the panel and
    later passes drive the replication. A single-pass record set is NOT an error -
    the replication block then says the protocol was not run, which is a fact the
    artifact has to be able to state.
    """
    audit = JudgeAudit()
    pass1 = _verdicts_by_judge(records, pass_index=1)
    audit.n_judges = len(pass1)
    items = {k for v in pass1.values() for k in v}
    audit.n_items = len(items)

    if len(pass1) < 2:
        audit.warnings.append(
            f"Only {len(pass1)} judge present. PRD R8.2 requires >= 2 judges on ALL "
            f"outputs; with one judge there is no panel and kappa does not exist."
        )
        audit.panel = {
            "judges": sorted(pass1),
            "n_pairs": 0,
            "panel_usable": False,
            "verdicts": [],
            "note": "kappa requires at least two judges; none was computed.",
        }
    else:
        audit.panel = panel_agreement(pass1)
        if not audit.panel["panel_usable"]:
            worst = audit.panel.get("kappa_min")
            audit.warnings.append(
                f"Panel is NOT usable: the weakest pair scores kappa={worst} against a "
                f"bar of {CAUTIONARY_KAPPA} (2511.02246). Agreement is no better than "
                f"chance-corrected and cannot support a headline number."
            )

    # ── R8.6 replication ──────────────────────────────────────────────────────
    passes_present = sorted({int(r.get("pass_index", 1)) for r in records})
    if len(passes_present) < 2:
        audit.replication = {
            "protocol": "R8.6 - two passes of the same deterministic system under the same judge",
            "status": "NOT RUN",
            "judges": [],
            "worst_flip_rate": None,
            "note": (
                "Only one pass is present in the verdict records, so the judge-noise "
                "floor is UNMEASURED. Without it, no result of this study can be "
                "distinguished from the judge's own instability."
            ),
        }
        audit.warnings.append(
            "R8.6 was not run: the records contain one pass. The judge-noise floor is "
            "unmeasured and no effect can be called a result."
        )
    else:
        from msat.judges.audit import two_pass_replication

        reports = []
        for judge in sorted(pass1):
            p2 = _verdicts_by_judge(records, pass_index=passes_present[1]).get(judge, {})
            reports.append(
                two_pass_replication(
                    pass1[judge], p2, judge_id=judge, pass1_judge=judge, pass2_judge=judge
                )
            )
        rates = [r.flip_rate for r in reports if r.n_items]
        audit.replication = {
            "protocol": "R8.6 - two passes of the same deterministic system under the same judge",
            "status": "RUN",
            "pass_indices_compared": passes_present[:2],
            "judges": [r.as_dict() for r in reports],
            "verdicts": [flip_verdict(r) for r in reports],
            "worst_flip_rate": round(max(rates), 6) if rates else None,
            "best_flip_rate": round(min(rates), 6) if rates else None,
            "noise_floor_rule": (
                "Any effect smaller than the WORST judge's flip rate is inside the "
                "judge's own noise. The floor is set by the worst judge, not the mean."
            ),
            "aggregate_is_not_the_headline": (
                "n_flips counts ROWS whose verdict changed. Flips in opposite directions "
                "cancel in the aggregate, so net_delta understates instability - 2609.38021 "
                "moved 4 in aggregate while flipping 8 rows."
            ),
        }

    audit.human = dict(human or {})
    if not audit.human:
        audit.human = {
            "status": "NOT RUN",
            "note": (
                "No human-labelled sample was supplied. Judge-vs-human agreement is "
                "R8.2's second half and is not optional: a panel can agree with itself "
                "while being consistently wrong."
            ),
        }
        audit.warnings.append(
            "R8.2's human-labelled 10% is missing. Judge-judge kappa measures whether "
            "the panel is self-consistent, not whether it is right."
        )
    return audit


def format_audit(audit: JudgeAudit) -> str:
    """A fixed-width summary for the CLI."""
    h = audit.headline()
    lines = [
        "",
        f"  judges              {h['n_judges']}",
        f"  items               {h['n_items']:,}",
        f"  panel usable        {'YES' if h['panel_usable'] else 'NO'}",
        f"  kappa min           {h['kappa_min']}  (bar {h['cautionary_bar']})",
        f"  judge-noise floor   {h['judge_noise_floor']}",
    ]
    if audit.replication.get("judges"):
        lines += ["", format_flip_table(audit.replication)]
    if audit.warnings:
        lines += ["", "  warnings"]
        lines += [f"    - {w}" for w in audit.warnings]
    return "\n".join(lines)
