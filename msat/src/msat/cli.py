"""The `msat` command-line surface (PRD §7).

Every subcommand named in PRD §7 is registered here **before any feature lands**,
so the surface is fixed early and `tests/test_cli.py` can assert it verbatim.
Commands owned by an unbuilt phase raise a *distinguishable* error
(`EXIT_NOT_IMPLEMENTED`) rather than a generic crash — "not built yet" and
"broken" are different states and must not look alike.

Argument convention (plan decision D-a): the PRD's positional form is canonical —
`msat analyze <run_id>` — and `--run <run_id>` is accepted as an alias so the
pre-existing Makefile targets keep working. Supplying both, or neither, is an
error.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, NoReturn, cast

import typer

from msat import __version__
from msat.constants import (
    EXIT_ASSERTION_FAILED,
    EXIT_BUDGET_UNSET,
    EXIT_CONFIG_INVALID,
    EXIT_DEGENERATE_CELL,
    EXIT_DRIFT,
    EXIT_FAILURE,
    EXIT_NO_GO,
    EXIT_NOT_IMPLEMENTED,
    EXIT_UNDERPOWERED,
    OFFLINE_COMMANDS,
    PHASE_OF_COMMAND,
)
from msat.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from msat.metrics.normalisation import Normaliser
    from msat.runner.ladder import LadderState, RungName
    from msat.store.responses import ResponseStore

app = typer.Typer(
    name="msat",
    help=(
        "MemSat experimental harness — retrieval depth K x context budget B.\n"
        "\n"
        "The primary result is the estimated log2(K)*log2(B) interaction term, "
        "tested against a pre-registered sign prediction. See PRD.md."
    ),
    add_completion=False,
    no_args_is_help=True,
    # Markup parsing is OFF deliberately. The repo's convention is to cite
    # requirements as `[R9.1]`; under `rich_markup_mode="rich"` a future
    # contributor copying that convention into a docstring turns it into a style
    # tag and crashes `--help` with a MarkupError. Requirement IDs are written
    # `PRD §R9.1` here instead. Drop this line only if someone also adds a
    # test that every docstring is markup-free.
    rich_markup_mode=None,
)

#: Exactly the command names PRD §7 specifies, plus the three the Makefile
#: targets require. Asserted by `tests/test_cli.py::test_all_subcommands_registered`.
PRD_SURFACE: Final[tuple[str, ...]] = (
    # --- PRD §7, verbatim ---
    "validate-config",
    "design-power",
    "ceiling",
    "build-plan",
    "run",
    "status",
    "score",
    "judge-audit",
    "analyze",
    "figures",
    "paper",
    # --- Required by Makefile targets but not listed in §7 ---
    # `data.lock` (PRD §R2.3) is a separate operation from `ceiling`; `verify`
    # (PRD §G1) and `replay` (PRD §R6.3) are the two offline guarantees the Makefile
    # exposes as first-class targets.
    "data-lock",
    "verify",
    "replay",
    # --- Added in P5 alongside the Arm D-max and hardware-delta work ---
    # `max-context` reports what each reader can actually be shown, so the VRAM
    # cap is visible BEFORE a run rather than discovered in a results table.
    # `hardware-delta` bounds the cross-hardware confound by measuring it, which
    # is required the moment any arm is generated on different hardware.
    "max-context",
    "hardware-delta",
    # `vram` computes a checkpoint's VRAM floor from its own config.json, which
    # is what replaces the hand-typed ceilings in readers.yaml (open decision D1).
    "vram",
)

log = get_logger("cli")


# ── helpers ────────────────────────────────────────────────────────────────────
def _fail(message: str, code: int = EXIT_FAILURE) -> NoReturn:
    """Print a user-facing error and abort with `code`."""
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _pending(command: str, requirement: str) -> None:
    """Abort with `EXIT_NOT_IMPLEMENTED` for a phase that has not been built.

    The message names the owning phase so the reader knows this is scheduled
    work, not a defect. See `IMPLEMENTATION_PLAN.md` §3.
    """
    phase = PHASE_OF_COMMAND.get(command, "?")
    typer.secho(
        f"`msat {command}` is not implemented yet.\n"
        f"  requirement : {requirement}\n"
        f"  build phase : {phase} (see IMPLEMENTATION_PLAN.md §3)\n"
        f"  exit code   : {EXIT_NOT_IMPLEMENTED}",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(EXIT_NOT_IMPLEMENTED)


def _resolve_run_id(positional: str | None, alias: str | None) -> str:
    """Reconcile the canonical positional `run_id` with the `--run` alias."""
    if positional and alias:
        _fail(f"run_id given twice: {positional!r} positionally and {alias!r} via --run")
    resolved = positional or alias
    if not resolved:
        _fail("a run_id is required (positional, or via --run)")
    return resolved


def _read_ladder(path: Path) -> LadderState:
    """Load the ladder state, defaulting to "nothing has passed".

    A missing file is not an error. It is the normal state of a run that has not
    started, and refusing to report status because the gate has never been
    climbed would make the tool useless exactly when someone needs it most.
    """
    from msat.runner.ladder import LadderState

    if not path.exists():
        return LadderState()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return LadderState(passed=set(raw.get("passed", [])), notes=dict(raw.get("notes", {})))
    except (ValueError, OSError) as e:
        _fail(f"{path} is unreadable ({e}); refusing to report a ladder state it cannot confirm")
        raise AssertionError("unreachable") from e  # pragma: no cover


def _assert_offline(command: str) -> None:
    """Guard rail for PRD §N3 commands.

    Phase 0 only records the intent; the real socket-blocking enforcement lands
    with the first offline implementation, and `tests/conftest.py` provides the
    blocking fixture from this phase onward so nothing regresses.
    """
    log.debug("offline command", extra={"command": command, "offline": True})


def normalise_scale_name(split: str) -> str:
    """Accept `M`, `m`, `longmemeval_m`, or `lme_m_cleaned.json` as a scale name.

    PRD §7 spells the argument `longmemeval_m` while the loader keys on `M`;
    both must work rather than one being a silent no-op.
    """
    token = split.strip().lower()
    if token in ("s", "m"):
        return token.upper()
    for scale in ("s", "m"):
        if f"longmemeval_{scale}" in token or f"lme_{scale}" in token:
            return scale.upper()
    return split


# ── P2 · configuration ─────────────────────────────────────────────────────────
@app.command("validate-config")
def validate_config(
    configs_dir: Annotated[
        Path,
        typer.Argument(help="Directory (or file) holding the config bundle."),
    ] = Path("configs"),
    show_hash: Annotated[
        bool, typer.Option("--hash", help="Print the resolved config_hash (PRD §R1.4).")
    ] = False,
) -> None:
    """Fail non-zero if any PRD §R1.1 / §R0.1 field is missing, or `readers` < 2.

    Reports EVERY problem in one pass rather than stopping at the first: a harness
    that stops turns a ten-minute fix into ten round-trips.

    Warnings do not change the exit code. Some decisions are genuinely blocked on
    the guide (reader models B6, budget ceiling B7) and a harness that refuses to
    compute anything until a human replies is not a harness, it is a mailbox.

    PRD §R1.5, §R7.5.
    """
    from msat.config import errors_only, validate

    root = configs_dir if configs_dir.is_dir() else configs_dir.parent
    problems = validate(root)
    errs = errors_only(problems)
    warns = [p for p in problems if p not in errs]

    for problem in problems:
        colour = typer.colors.RED if problem.severity.value == "error" else typer.colors.YELLOW
        typer.secho(str(problem), fg=colour, err=True)

    if show_hash and not errs:
        from msat.config import load_bundle

        bundle = load_bundle(root)
        typer.echo(f"config_hash  {bundle.config_hash}")
        for key, value in bundle.summary().items():
            typer.echo(f"  {key:<12} {value}")

    if errs:
        _fail(
            f"{len(errs)} blocking problem(s) in {root}. "
            "A retriever specified after the sweep invalidates every figure.",
            code=EXIT_CONFIG_INVALID,
        )
    typer.secho(
        f"config OK ({len(warns)} warning(s))" if warns else "config OK",
        fg=typer.colors.GREEN,
        err=True,
    )


@app.command("design-power")
def design_power(
    design_yaml: Annotated[
        Path,
        typer.Argument(help="Path to configs/design.yaml — the single grid source."),
    ] = Path("configs/design.yaml"),
    out: Annotated[Path, typer.Option("--out")] = Path("results"),
    sims: Annotated[int, typer.Option("--sims", help="Simulations per power point.")] = 30,
    seed: Annotated[int, typer.Option("--seed")] = 20261012,
    skip_sensitivity: Annotated[
        bool, typer.Option("--fast", help="Skip the judge-noise sensitivity sweep.")
    ] = False,
) -> None:
    """Compute the MDE for the K x B INTERACTION, before any API call.

    Emits `results/power.json`. This is the single highest-leverage command in the
    project: an underpowered interaction (risk A13) is the most likely way this
    design produces an uninterpretable null, and it is discoverable for free.

    Method: Monte-Carlo simulation of the DECLARED primary model (PRD §R9.1) with
    cluster-robust standard errors by question, bisecting on empirical power.
    Covariate distributions are fitted to the REAL corpus measured in P1.

    If the interaction is underpowered at the full grid, this PROPOSES a reduced
    design that preserves estimability — it never silently truncates (PRD §R0.3).

    Exit codes: 0 powered · 6 underpowered · 11 budget ceiling still unset (B7).
    """
    from msat.config import load_bundle
    from msat.power import run_power_analysis

    root = design_yaml.parent
    bundle = load_bundle(root)
    report = run_power_analysis(
        bundle,
        out_dir=out,
        n_sim=sims,
        seed=seed,
        sensitivity=not skip_sensitivity,
    )
    _emit_power_report(report, out)
    budget = report["budget_guard"]
    if not report["verdict"]["powered"]:
        _fail(
            "the K x B interaction is not estimable at this grid; "
            "`reduced_design_proposals` contains designs that preserve estimability. "
            "Nothing was truncated automatically (PRD §R0.3).",
            code=EXIT_UNDERPOWERED,
        )
    if budget["state"] != "set":
        _fail(
            f"estimated cost ${budget['estimate_usd']:,.0f} but budget.max_usd is "
            "unset (blocker B7). The hard cost guard cannot be enforced and no run "
            "may be launched. This estimate is the input to the budget conversation.",
            code=EXIT_BUDGET_UNSET,
        )


def _emit_power_report(report: dict[str, Any], out_dir: Path) -> None:
    """Print the decision-relevant slice of `power.json`. The rest is in the file."""
    verdict = report["verdict"]
    grid = report["grid"]
    inter = report["interaction"]

    typer.secho("\n=== design-power · the K x B interaction ===", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        f"  grid       {grid['K_grid']} x {len(grid['B_grid'])} B "
        f"x {len(grid['scales'])} scales x {len(grid['readers'])} readers"
    )
    typer.echo(f"  cells      {grid['n_cells']}   observations  {grid['n_observations']:,}")
    typer.echo(f"  config     {report['_provenance']['config_hash'][:16]}…")
    typer.echo(
        f"  prereg     {report['_provenance']['preregistered_sign']}"
        f"   hash {report['_provenance']['preregistration_hash'][:16]}…"
    )

    if inter.get("powered"):
        typer.secho(
            f"\n  MDE (interaction)   {inter['mde_log_odds']} log-odds",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"    at accuracy p=0.5  {inter['mde_pp_at_p50']} pp")
        typer.echo(f"    at accuracy p=0.3  {inter['mde_pp_at_p30']} pp")
        typer.echo(f"    SE                 {inter['se_log_odds']}")
    else:
        typer.secho("\n  UNDERPOWERED", fg=typer.colors.RED, bold=True)
        typer.echo(f"    {inter.get('note', '')}")
        typer.echo("    proposals that preserve estimability:")
        for prop in report.get("reduced_design_proposals", {}).get("proposals", []):
            flag = "  [violates R7.5]" if prop.get("violates_r75") else ""
            typer.echo(f"      - {prop['step']}{flag}")

    rows = report.get("judge_noise_sensitivity") or []
    if rows:
        typer.secho(
            "\n  judge-noise sensitivity — MDE for the interaction by grader quality:",
            fg=typer.colors.CYAN,
        )
        typer.echo(f"    {'flip':<7}{'atten':>8}{'MDE':>10}{'pp @ p=.5':>11}  interpretation")
        for r in rows:
            mde = r.get("mde_log_odds")
            mde_str = f"{mde:.4f}" if mde is not None else "n/a"
            pp_str = f"{r['mde_pp_at_p50']:.3f}" if mde is not None else "n/a"
            typer.echo(
                f"    {r['judge_flip']:<7.2f}{r['effect_attenuation']:>8.2f}"
                f"{mde_str:>10}{pp_str:>11}  {r['interpretation']}"
            )

    cost = report["cost"]
    compute = report.get("compute_budget", {})
    typer.secho("\n  cost: $0 — every reader and judge is LOCAL", fg=typer.colors.CYAN)
    typer.echo(f"    cells              {cost['n_cells']} ({cost['n_cells_priced']} priced)")
    typer.echo(f"    calls              {cost['n_calls']:,}")
    typer.echo(
        f"    cost_uncached      ${cost['cost_uncached_total_usd']:,.2f}   <- PRIMARY (R9.8)"
    )
    if cost["unpriced_readers"]:
        typer.secho(f"    UNPRICED readers   {cost['unpriced_readers']}", fg=typer.colors.RED)

    if compute.get("available"):
        typer.secho(
            "\n  COMPUTE budget — the real constraint with no paid API "
            "(throughput ASSUMED; benchmark before P5):",
            fg=typer.colors.CYAN,
        )
        for reader, row in sorted(compute["per_reader"].items()):
            typer.echo(
                f"    reader {reader:<10} {row['hours']:>9,.1f} h   ({row['calls']:,} calls)"
            )
        typer.echo(
            f"    judge panel        {compute['judge_hours']:>9,.1f} h"
            f"   ({100 * compute['judge_share_of_total']:.0f}% of total)"
        )
        typer.echo(f"    judge audit (R8.6) {compute['judge_audit_hours']:>8,.1f} h")
        typer.echo(
            f"    {'TOTAL':<20} {compute['total_hours']:>9,.1f} h"
            f"  = {compute['total_days']:.1f} days on ONE GPU"
        )
        colour = typer.colors.GREEN if compute["within_budget"] else typer.colors.RED
        state = "within budget" if compute["within_budget"] else "OVER BUDGET"
        typer.secho(
            f"    ceiling            {compute['max_wall_clock_hours']:,.0f} h"
            f"   -> {state} ({compute['multiple_of_budget']}x)",
            fg=colour,
        )

    ctx = report.get("context_feasibility", {})
    if ctx.get("n_infeasible"):
        typer.secho(
            f"\n  CONTEXT OVERFLOW — {ctx['n_infeasible']} cell(s) exceed their reader's "
            f"window. Policy: {ctx['policy']}",
            fg=typer.colors.RED,
        )
        for reader, rows in sorted(ctx["infeasible_cells"].items()):
            scales = sorted({r["scale"] for r in rows})
            worst = max(rows, key=lambda r: r["overflow_by"])
            # Report the CEILING that actually rejected the cell, not the declared
            # window. Printing `context_window` here showed 131,072 next to an
            # overflow computed against a 66,160 VRAM ceiling - two different
            # numbers side by side with no indication which one bound.
            typer.echo(
                f"    {reader:<10} ceiling {worst['ceiling']:>7,} "
                f"(blocked_by={worst['blocked_by']}, declared window "
                f"{worst['context_window']:,})"
            )
            typer.echo(
                f"    {'':<10} worst prompt {worst['prompt_tokens']:>7,} "
                f"(+{worst['overflow_by']:,})  scales={scales}"
            )
        typer.secho(
            "    These cells must be DROPPED or the grid trimmed — never truncated.",
            fg=typer.colors.RED,
        )

    budget = report["budget_guard"]
    if budget["state"] != "set":
        typer.secho(
            "    budget.max_usd     UNSET (B7) — guard NOT enforceable",
            fg=typer.colors.YELLOW,
        )
    else:
        colour = typer.colors.GREEN if budget["within_budget"] else typer.colors.RED
        typer.secho(
            f"    budget.max_usd     ${budget['max_usd']:,.2f} "
            f"({budget['multiple_of_budget']}x the estimate)",
            fg=colour,
        )

    typer.secho(
        f"\n  {verdict['headline']}",
        fg=typer.colors.YELLOW if not verdict["powered"] else typer.colors.GREEN,
    )
    typer.secho(f"\n  {verdict['binding_constraint']}\n", fg=typer.colors.BRIGHT_BLACK)
    typer.secho(f"wrote {out_dir / 'power.json'}", fg=typer.colors.GREEN)


# ── P1 · provenance (zero API cost) ────────────────────────────────────────────
@app.command("ceiling")
def ceiling(
    split: Annotated[
        str | None,
        typer.Argument(
            help="Optional split to PRINT (e.g. `longmemeval_m` or `M`). "
            "The artifact is always computed over every available scale."
        ),
    ] = None,
    by_type: Annotated[
        bool, typer.Option("--by-type", help="Print the per-`question_type` table.")
    ] = False,
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Regenerate and byte-compare vs committed (PRD §R2.4)."),
    ] = False,
    out_dir: Annotated[
        Path, typer.Option("--out", help="Output directory for committed artifacts.")
    ] = Path("results"),
) -> None:
    """Compute ceiling(K), partial(K), corpus feasibility, and K*.

    Emits `ceiling.json` + `corpus_fraction.json`. Streams via `ijson` — it must
    survive the 2.6 GB M split, and does so in ~25 MB of RSS.

    **Zero API calls.** The paper's only free result.

    The artifacts are always computed over *every* available scale and are
    independent of which split you name, so the command is idempotent (PRD §N5)
    and `--verify` is a clean full rebuild. The positional only selects what is
    printed.

    PRD §R2.1, §R2.2, §R2.3, §R2.5, §R2.7.
    """
    from msat.ceiling import (
        build_ceiling_multiscale,
        build_corpus_fraction_artifact,
        build_dataset_report,
        dump_canonical,
        verify_artifact,
    )
    from msat.datasets import iter_metadata, raw_path, resolve_data_dir

    data_dir = resolve_data_dir()
    scales = {s: list(iter_metadata(raw_path(s, data_dir))) for s in ("S", "M")}

    ceiling_artifact = build_ceiling_multiscale(scales)
    fraction_artifact = build_corpus_fraction_artifact(scales)
    report_artifact = build_dataset_report(scales)

    ceiling_path = out_dir / "ceiling.json"
    fraction_path = out_dir / "corpus_fraction.json"
    report_path = out_dir / "dataset_report.json"

    if verify:
        _assert_offline("ceiling")
        problems: list[str] = []
        for path, artifact in (
            (ceiling_path, ceiling_artifact),
            (fraction_path, fraction_artifact),
            (report_path, report_artifact),
        ):
            ok, detail = verify_artifact(artifact, path)
            log.info("ceiling verify", extra={"artifact": path.name, "ok": ok})
            typer.secho(
                f"  {'OK  ' if ok else 'DRIFT'}  {path.name}: {detail}",
                fg=typer.colors.GREEN if ok else typer.colors.RED,
                err=True,
            )
            if not ok:
                problems.append(detail)
        if problems:
            _fail(
                f"{len(problems)} committed artifact(s) drifted from a regeneration "
                f"(PRD §R2.4). Re-run `msat ceiling --out {out_dir}` if the change is intended.",
                code=EXIT_DRIFT,
            )
        typer.secho(
            "ceiling: all committed artifacts reproduce byte-for-byte",
            fg=typer.colors.GREEN,
            err=True,
        )
        return

    dump_canonical(ceiling_artifact, ceiling_path)
    dump_canonical(fraction_artifact, fraction_path)
    dump_canonical(report_artifact, report_path)
    log.info(
        "ceiling computed",
        extra={"scales": sorted(scales), "out": str(out_dir)},
    )

    for path in (ceiling_path, fraction_path, report_path):
        typer.secho(f"wrote {path} ({path.stat().st_size} bytes)", fg=typer.colors.GREEN)

    shown = [normalise_scale_name(split)] if split else sorted(scales)
    for scale in shown:
        if scale not in scales:
            _fail(f"unknown split {split!r}; available: {sorted(scales)}")
        _print_ceiling_table(scale, ceiling_artifact["scales"][scale])
        if by_type:
            _print_by_type(scale, ceiling_artifact["scales"][scale])
        _print_corpus_fraction(scale, fraction_artifact["scales"][scale])


def _print_ceiling_table(scale: str, block: dict[str, Any]) -> None:
    ceiling = block["ceiling"]
    partial = block["partial"]
    survivors = block["survivors_at"]
    degenerate = block["degeneracy_fraction"]
    corpus = block["corpus_size"]
    typer.secho(
        f"\n[{scale}]  {block['n_answerable']} answerable, "
        f"{block['n_abstention']} abstention  ·  "
        f"{corpus['n_sessions_mean']:.1f} sessions/question "
        f"(min {corpus['n_sessions_min']}, max {corpus['n_sessions_max']})",
        fg=typer.colors.CYAN,
    )
    header = f"{'K':>4} {'ceiling':>9} {'partial':>9} {'surv>=K':>9} {'degenerate':>11}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for key in sorted(ceiling, key=int):
        k = int(key)
        typer.echo(
            f"{k:>4} {100 * ceiling[key]:8.2f}% {100 * partial[key]:8.2f}% "
            f"{100 * survivors[key]:8.2f}% {100 * degenerate[key]:10.2f}%"
        )
    kstar = block["analytic_saturation_k"]
    typer.secho(
        f"\n  K* = min{{K : ceiling(K) >= 0.995}} = {kstar}"
        f"   ·   beyond it, no retriever can add evidence"
        if kstar is not None
        else "\n  K* not reached on this grid",
        fg=typer.colors.YELLOW,
    )


def _print_by_type(scale: str, block: dict[str, Any]) -> None:
    typer.secho(f"\n[{scale}] by question_type:", fg=typer.colors.CYAN)
    typer.echo(
        f"{'type':<28}{'n':>5}{'mean_gold':>10}{'c(1)':>8}{'c(2)':>8}{'c(4)':>8}  classification"
    )
    typer.echo("-" * 100)
    for ability, row in sorted(block["by_type"].items(), key=lambda kv: -kv[1]["n"]):
        probe = row["probe"]
        typer.echo(
            f"{ability:<28}{row['n']:>5}{row['mean_gold_sessions']:>10.2f}"
            f"{100 * probe.get('1', 0):7.1f}%{100 * probe.get('2', 0):7.1f}%"
            f"{100 * probe.get('4', 0):7.1f}%  {row['classification']}"
        )
    typer.secho(
        "  retriever-independent: ceiling(1)=100% — any K-effect is purely reader-side.\n"
        "  retriever-limited:     ceiling(1)=0%   — K=1 is structurally impossible.",
        fg=typer.colors.BRIGHT_BLACK,
    )


def _print_corpus_fraction(scale: str, block: dict[str, Any]) -> None:
    typer.secho(f"\n[{scale}] corpus fraction  K / |corpus|  (gap G12):", fg=typer.colors.CYAN)
    typer.echo(f"{'K':>4}{'p10':>9}{'p50':>9}{'p90':>9}{'full_ctx':>11}{'K>corpus':>11}")
    typer.echo("-" * 53)
    for key in sorted(block["per_k"], key=int):
        row = block["per_k"][key]
        q = row["quantiles"]
        typer.echo(
            f"{int(key):>4}{q['p10']:9.3f}{q['p50']:9.3f}{q['p90']:9.3f}"
            f"{100 * row['frac_full_context']:10.1f}%{100 * row['frac_k_gt_corpus']:10.1f}%"
        )
    typer.secho(
        "  full_ctx = fraction where K/|corpus| == 1.0, i.e. the condition has silently\n"
        "  become full-context. The same K is a different system at different scales.",
        fg=typer.colors.BRIGHT_BLACK,
    )


@app.command("data-lock")
def data_lock(
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Check raw files against the committed data.lock."),
    ] = False,
    out_dir: Annotated[Path, typer.Option("--out")] = Path("results"),
    lock_path: Annotated[Path, typer.Option("--lock", help="Path to data.lock.")] = Path(
        "data.lock"
    ),
) -> None:
    """Write SHA-256 + byte size + row count for every raw benchmark file.

    PRD §R2.3. Streams, so the 2.6 GB M split is never resident. Files that are
    absent are recorded with `present: false` rather than omitted — `5_filler_sess`
    is currently missing, and that is a P7 blocker, not a footnote.
    """
    from msat.datalock import build_lock, dump_lock, verify_lock

    if verify:
        _assert_offline("data-lock")
        problems = verify_lock(lock_path)
        for problem in problems:
            typer.secho(f"  DRIFT  {problem}", fg=typer.colors.RED, err=True)
        if problems:
            _fail(
                f"{len(problems)} raw input(s) do not match data.lock. "
                "Every committed artifact derived from them is no longer reproducible.",
                code=EXIT_DRIFT,
            )
        typer.secho(f"data.lock verified: {lock_path}", fg=typer.colors.GREEN, err=True)
        return

    lock = build_lock()
    dump_lock(lock, lock_path)
    log.info("data lock written", extra={"lock": str(lock_path)})
    for split, entry in sorted(lock["files"].items()):
        if entry.get("present"):
            typer.secho(
                f"  locked  {split:<8} {entry['filename']:<22} "
                f"{entry['bytes'] / 1e6:9.1f} MB  rows={entry.get('rows')}  "
                f"sha256={entry['sha256'][:16]}…"
            )
        else:
            typer.secho(f"  MISSING {split:<8} {entry['filename']}", fg=typer.colors.YELLOW)
    typer.secho(f"wrote {lock_path}", fg=typer.colors.GREEN)


# ── P3/P4 · planning and execution ─────────────────────────────────────────────
@app.command("build-plan")
def build_plan(
    run_spec: Annotated[
        Path | None,
        typer.Argument(help="Run-spec YAML; defaults to configs/design.yaml."),
    ] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    corpus: Annotated[
        str, typer.Option("--corpus", help="synthetic | longmemeval")
    ] = "longmemeval",
    limit: Annotated[int | None, typer.Option("--limit", help="Cap questions per scale.")] = None,
) -> None:
    """Enumerate conditions, apply the K-guard and B-guard, emit a manifest.

    **Rejects** degenerate cells before any API call is made (PRD §R3.2), and
    refuses to launch a design whose planned cost exceeds `budget.max_usd`
    (PRD §N8).

    It then builds the EXECUTABLE plan from the manifest it just wrote and asks
    `planning.verify_crossed_grid` whether arms A and G really are one crossed
    grid (PRD §R4.4). The manifest proves the grid was *written down*; the plan
    proves it would be *executed*, and only the second is the requirement. A cell
    rejected on one scale leaves a hole the config file knows nothing about, and a
    hole that nobody was told about is the silent-truncated factorial R0.3 forbids.

    Exit codes: unchanged — 0 clean, 3 (`EXIT_DEGENERATE_CELL`) when any cell was
    rejected or the crossing is incomplete. A crossing gap and a rejected cell are
    the same class of defect: a declared condition that will not run.

    PRD §R3.2, §R3.3, §R3.4, §R4.4.
    """
    from msat.config import load_bundle
    from msat.manifest import build_manifest, stats_from_corpus, write_manifest
    from msat.planning import plan_from_manifest, verify_crossed_grid, write_plan

    root = Path(run_spec).parent if run_spec else Path("configs")
    if not (root / "design.yaml").exists():
        raise typer.BadParameter(f"no design.yaml in {root}")
    bundle = load_bundle(root)
    cpt = float(bundle.pricing.get("token_estimates", {}).get("chars_per_token", 4.0))

    corpus_map: dict[str, list[Any]] = {}
    raw_counts: dict[str, list[tuple[int, int]]] = {}

    if corpus == "synthetic":
        from msat.selftest import synthetic_corpus

        corpus_map, raw_counts = synthetic_corpus(cpt)
    else:
        from msat.datasets.longmemeval import iter_metadata, raw_path

        for scale in bundle.design.scales:
            split = scale.upper()  # splits are named S / M, not lowercase
            stats: list[Any] = []
            counts: list[tuple[int, int]] = []
            # Metadata-only: `n_sessions` and `n_chars` are exact without holding a
            # single session's text, so this streams the 2.6 GB M split on a laptop.
            for q in iter_metadata(raw_path(split)):
                stats.append(
                    stats_from_corpus(
                        q.qid, q.corpus.n_sessions, q.corpus.n_chars, chars_per_token=cpt
                    )
                )
                counts.append((q.corpus.n_sessions, q.corpus.n_chars))
                if limit and len(stats) >= limit:
                    break
            corpus_map[scale] = stats
            raw_counts[scale] = counts

    manifest = build_manifest(bundle, corpus_map, chars_per_token=cpt, raw_counts=raw_counts)
    dest = Path(out) if out else root / "manifest.json"
    write_manifest(manifest, dest)

    typer.echo(f"  config_hash   {manifest.config_hash}")
    typer.echo(f"  cells         {len(manifest.cells)}")
    typer.echo(
        f"  accepted      {len(manifest.accepted)}   rejected {len(manifest.rejected)}"
        f"   (degeneracy: {len(manifest.fully_degenerate)})"
    )
    log.info(
        "manifest built",
        extra={"config_hash": manifest.config_hash, "n_cells": len(manifest.cells)},
    )
    for cell in manifest.rejected:
        typer.secho(
            f"    REJECTED  {cell.condition.scale}/{cell.condition.reader} "
            f"K={cell.condition.k} B={int(cell.condition.b)}   {cell.reason}",
            fg=typer.colors.RED,
        )
        if cell.example_reason:
            typer.echo(f"      e.g. {cell.example_reason}")

    partial = [c for c in manifest.accepted if c.n_degenerate]
    if partial:
        worst = max(partial, key=lambda c: c.degeneracy_fraction)
        typer.secho(
            f"\n  {len(partial)} accepted cell(s) are degenerate for SOME questions. "
            f"Worst: {worst.condition.scale} K={worst.condition.k} at "
            f"{worst.degeneracy_fraction:.0%}. Those questions are excluded from "
            f"paired comparisons at P4.3.",
            fg=typer.colors.YELLOW,
        )

    typer.echo(f"\n  wrote {dest}")

    # ── R4.4 · is the grid actually CROSSED, or two marginal sweeps? ───────────
    # The manifest says which conditions are VALID. Only the executable plan says
    # which conditions will RUN, and a plan that quietly omits a cell is
    # indistinguishable from one built from a smaller grid — which is the whole
    # claim. `qids_by_scale` comes from the corpus stats already read above, so
    # the schedule is the one this grid would really execute.
    qids_by_scale = {
        scale: [str(s.question_id) for s in stats] for scale, stats in corpus_map.items()
    }
    executable = plan_from_manifest(
        dest,
        qids_by_scale,
        seed=int(bundle.design.seed),
        runs_per_cell=int(bundle.design.runs_per_cell),
    )
    plan_path = write_plan(executable, dest.with_name(f"{dest.stem}.plan.json"))
    crossed = verify_crossed_grid(
        executable,
        k_grid=bundle.design.K_grid,
        b_grid=bundle.design.b_values,
        scales=bundle.design.scales,
        readers=bundle.design.readers,
    )
    crossed_path = dest.with_name("crossed_grid.json")
    crossed_path.write_text(json.dumps(crossed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info(
        "crossed grid verified",
        extra={
            "config_hash": manifest.config_hash,
            "crossed": crossed["crossed"],
            "n_planned_cells": crossed["n_planned_cells"],
        },
    )

    typer.echo(f"  plan          {executable.n_units:,} unit(s) -> {plan_path.name}")
    typer.echo(
        f"  crossed       {'YES' if crossed['crossed'] else 'NO'}  "
        f"({crossed['n_planned_cells']}/{crossed['n_expected_cells']} cells, "
        f"reader_balanced={crossed['reader_balanced']})   R4.4"
    )
    for problem in crossed["problems"]:
        typer.secho(f"    NOT CROSSED  {problem}", fg=typer.colors.RED)
    typer.echo(f"  wrote {plan_path}")
    typer.echo(f"  wrote {crossed_path}")

    if manifest.rejected:
        typer.secho(
            "  A rejected cell is not one that runs badly - it is a condition already "
            "satisfied before the manipulation is applied. Excluding it is the point.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=EXIT_DEGENERATE_CELL)
    if not crossed["crossed"]:
        typer.secho(
            f"  {crossed['n_expected_cells'] - crossed['n_planned_cells']} declared "
            "(K, B) cell(s) are absent from the plan. Arms A and G must be ONE "
            "crossed grid (PRD §R4.4): a pair of marginal sweeps produces no "
            "interaction term, which is the paper's headline claim.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=EXIT_DEGENERATE_CELL)


@app.command("run")
def run_experiment(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Resume from the write-ahead log.")
    ] = False,
    run: Annotated[
        str | None, typer.Option("--run", help="Alias for the positional run_id.")
    ] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs"),
    plan: Annotated[Path | None, typer.Option("--plan", help="A built manifest.json.")] = None,
    results_dir: Annotated[Path, typer.Option("--results")] = Path("results"),
    rung: Annotated[
        str | None,
        typer.Option("--rung", help="negative | baselines | full. Overrides the ladder."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Build and validate; generate nothing.")
    ] = False,
    questions: Annotated[
        Path | None, typer.Option("--questions", help="JSON: qid -> answer_session_ids.")
    ] = None,
    gold: Annotated[
        Path | None,
        typer.Option(
            "--gold",
            help="JSON: qid -> gold answer. Required by the Arm E rung, which scores "
            "the control against something.",
        ),
    ] = None,
    arm_e_pool: Annotated[
        Path | None,
        typer.Option(
            "--arm-e-pool",
            help="JSON cross-question memory pool for the negative control: "
            "{qid: {question, memories: [...]}}. Required by the Arm E rung.",
        ),
    ] = None,
    reader_id: Annotated[
        str | None,
        typer.Option(
            "--reader",
            help="Restrict the run to one reader family. Default: the whole panel, "
            "because a pipeline that flatters itself on the primary may not flatter "
            "itself on the others.",
        ),
    ] = None,
) -> None:
    """Execute the crossed (K, B) grid: randomized order, WAL-backed, resumable.

    Never sequential by K — provider drift must not be able to masquerade as a
    curve (PRD §R5.1).

    THE ORDER OF THE CHECKS BELOW IS THE POINT
    --------------------------------------------
    1. the arm registry builds            (are there even renderers?)
    2. the ladder rung is entered         (may this rung run at all?)
    3. Arm E's gate has PASSED            (does the pipeline collapse correctly?)
    4. the plan is crossed                (will the interaction be estimable?)
    5. every cell fits its reader         (will it OOM, or be silently truncated?)
    6. Arm E actually RUNS                (draw -> render -> generate -> evaluate)

    Each is cheap and each can refuse a run that would otherwise burn the budget.
    Checking them in a different order means the expensive one runs first, and the
    expensive one is the run.

    Step 5 is PRD §R3.2 and it used to be missing entirely: nothing compared a
    cell's prompt against the reader's *servable* ceiling, so a deep cell on a
    small reader was dispatched and OOM'd at hour 60 instead of being rejected at
    the planning stage. Overflow is a REJECTED CELL, never a truncated prompt —
    `retrieval.yaml` sets `never_truncate_silently: true`.

    Step 6 is the reason the ladder exists. Arm E is the gate on the whole
    experiment: it draws memories from OTHER questions, so a correct reader scores
    near zero, and a control that does NOT collapse means the pipeline is flattering
    its own results. That has to be measured, not asserted.

    `--dry-run` performs steps 1-5 and stops. It is the only way to find out that
    the design is sound before scheduling the thing that costs 292 hours.

    PRD §R5, §R3.2, §R4.3.
    """
    from msat.arms.registry import RegistryError, build_registry, declared_crossed_arms
    from msat.config import load_bundle
    from msat.runner.ladder import LADDER, LadderOrderViolationError, check_order

    resolved = _resolve_run_id(run_id, run)
    run_dir = results_dir / resolved
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. the registry ──────────────────────────────────────────────────────
    bundle = load_bundle(config)
    try:
        registry, reg_report = build_registry(bundle)
    except RegistryError as e:
        _fail(str(e))
        return
    typer.echo(f"  arms        {', '.join(sorted(registry.arms))}")
    typer.echo(f"  prompts     {reg_report['prompt_arms']}")
    typer.echo(f"  config-only {reg_report['config_arms']}  (manipulated in the memory set)")

    # ── 2. the ladder ────────────────────────────────────────────────────────
    state = _read_ladder(run_dir / "ladder.json")
    chosen = _resolve_rung(rung, state)
    try:
        check_order(state, cast("RungName", chosen))
    except LadderOrderViolationError as e:
        typer.secho(f"\n  {e}", fg=typer.colors.YELLOW)
        _gate_exit(
            _evaluate_gate(
                run_dir,
                context=f"{resolved}/ladder",
                run_id=resolved,
                config_hash=bundle.config_hash,
                notes=[
                    f"Ladder order violation, not a measured result: {e}",
                    "No cell has been executed, so branch (iv) is NOT EVALUATED rather "
                    "than failed. This is a pre-run refusal, not a bounded negative.",
                ],
            ),
            message="the ladder refuses this run",
        )
    typer.echo(f"  ladder      entering rung {chosen!r}")

    # ── 3. Arm E's gate ──────────────────────────────────────────────────────
    e_thresholds = _arm_e_thresholds(bundle)
    if e_thresholds is None:
        typer.secho(
            "\n  arm E's pass thresholds are NULL, so the negative control cannot be "
            "evaluated and `proceed` is False.\n"
            "  This is the gate working, not a defect. Set "
            "`pass_threshold_accuracy_below` and\n"
            "  `pass_threshold_recall_below` in configs/prompts/arm_e.j2 from a GPU "
            "pilot before launching.",
            fg=typer.colors.YELLOW,
        )
        if not dry_run:
            _gate_exit(
                _evaluate_gate(
                    run_dir,
                    context=f"{resolved}/arm_e_thresholds",
                    run_id=resolved,
                    config_hash=bundle.config_hash,
                    notes=[
                        "Arm E's pass thresholds are null, so the negative control "
                        "cannot be EVALUATED. `null` does not mean 'loosely' - it "
                        "means the gate cannot be passed, and the gate gates "
                        "(PRD §R4.3).",
                    ],
                ),
                message="arm E cannot be evaluated, so no rung may proceed",
            )
    else:
        typer.echo(f"  arm E       thresholds set {e_thresholds}")

    # ── 4. the plan, and whether it is genuinely CROSSED (R4.4) ──────────────
    manifest_path = Path(plan) if plan else results_dir / "manifest.json"
    if not manifest_path.exists():
        _fail(f"no manifest at {manifest_path}; run `msat build-plan` first")
    crossed_arms = declared_crossed_arms(bundle)
    if not reg_report["crossed"]["agree"]:
        _fail(
            f"arms.yaml declares crossed_grid {crossed_arms} but the code has "
            f"{reg_report['crossed']['hard_coded']}. One of them is stale."
        )
    typer.echo(f"  crossed     {', '.join(sorted(crossed_arms))} (R4.4)")

    qids = _load_gold_sessions(questions) if questions else {}
    executable = _plan_from(manifest_path, bundle, qids)
    grid_report, n_interaction_cells, degenerate_cells = _crossing(executable, bundle)
    _write_json(run_dir / "crossed_grid.json", grid_report)
    typer.echo(
        f"  grid        {grid_report['n_planned_cells']}/{grid_report['n_expected_cells']} "
        f"cells crossed, {n_interaction_cells} (K,B) cell(s) estimable"
    )
    for problem in grid_report["problems"]:
        typer.secho(f"    NOT CROSSED  {problem}", fg=typer.colors.RED)

    # ── 5. R3.2 · every planned cell against its reader's SERVABLE ceiling ────
    panel = _panel(bundle)
    capacity = _capacity_preflight(bundle, executable, panel)
    _write_json(run_dir / "capacity.json", capacity)
    typer.echo(
        f"  capacity    {capacity['n_accepted']}/{capacity['n_checked']} cell(s) fit "
        f"their reader; {capacity['n_rejected']} rejected (R3.2)"
    )
    for reader_id_, row in sorted(capacity["per_reader"].items()):
        typer.echo(
            f"    {reader_id_:<8} ceiling {row['ceiling_tokens']:>7,}  "
            f"bound by {row['binding_constraint']}"
        )
    for rejected in capacity["rejected"][:10]:
        typer.secho(
            f"    REJECTED  {rejected['label']}  prompt {rejected['prompt_tokens']:,} "
            f"> ceiling {rejected['ceiling_tokens']:,} "
            f"(+{rejected['overflow_by']:,}, bound by {rejected['binding_constraint']})",
            fg=typer.colors.RED,
        )
    if capacity["n_rejected"] > 10:
        typer.echo(f"    ... and {capacity['n_rejected'] - 10} more, in capacity.json")

    # ── 6. the P5 GO/NO-GO gate, from `gate.py` ───────────────────────────────
    # Before any generation, so a run that cannot answer its own question never
    # spends the budget finding out.
    preflight_verdict = _evaluate_gate(
        run_dir,
        context=f"{resolved}/preflight",
        run_id=resolved,
        config_hash=bundle.config_hash,
        notes=[
            "Pre-run gate. Branches (i)-(iii) are empty because no output has been "
            "generated yet; branch (iv) is a property of the DESIGN and is decided "
            "here, before a single token is spent.",
        ],
        accuracy_by_cell={},
        recall_by_k={},
        stratified=None,
        degenerate_cells=degenerate_cells,
        n_interaction_cells=n_interaction_cells,
    )
    if not preflight_verdict.proceed:
        _gate_exit(preflight_verdict, message="the grid is not a crossed grid (R4.4)")

    if dry_run:
        typer.secho(
            "\n  DRY RUN: registry, ladder, arm E gate, the crossed grid and the R3.2 "
            "capacity preflight all checked. Nothing was generated.",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"  rungs available: {', '.join(r.name for r in LADDER)}")
        typer.echo(f"  wrote {run_dir / 'go_no_go.json'}")
        return

    # ── 7. Arm E RUNS. This is the rung, and it is the gate on the experiment. ──
    if chosen == "negative":
        _run_arm_e_rung(
            run_dir,
            bundle=bundle,
            registry=registry,
            panel=panel,
            executable=executable,
            questions=Path(questions) if questions else None,
            gold=Path(gold) if gold else None,
            pool=Path(arm_e_pool) if arm_e_pool else None,
            reader_id=reader_id,
            run_id=resolved,
        )
        typer.secho(
            "\n  ARM E COLLAPSED. The pipeline does not score without the right "
            "evidence,\n  which is the precondition for every other arm meaning "
            "anything.",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.echo("  next: `msat run " + resolved + " --rung baselines`")
        typer.echo(f"  wrote {run_dir / 'go_no_go.json'}")
        typer.echo(f"  wrote {run_dir / 'arm_e_control.json'}")
        return

    _fail(
        "generation beyond the Arm E rung needs a backend. Everything up to it — "
        "registry, ladder, arm E gate, crossed grid, R3.2 capacity, the negative "
        "control — is validated and written to this run directory; connect a "
        "vLLM/transformers backend and the executor takes over from here."
    )


def _resolve_rung(rung: str | None, state: LadderState) -> str:
    """Which rung to enter, refusing an unknown name rather than defaulting.

    Defaulting an unrecognised `--rung` to `full` would let a typo skip the ladder,
    which is the one thing the ladder exists to prevent.
    """
    from msat.runner.ladder import LADDER

    names = [r.name for r in LADDER]
    if rung is None:
        nxt = state.first_unpassed()
        return nxt.name if nxt else names[-1]
    if rung not in names:
        _fail(f"unknown rung {rung!r}; the ladder has {names}")
    return rung


def _arm_e_thresholds(bundle: Any) -> dict[str, float] | None:
    """Arm E's numeric thresholds, or None when unset. `null` means NOT EVALUABLE."""
    from msat.arms.negative import ControlConfig

    source = (bundle.prompts or {}).get("prompts/arm_e.j2") or (bundle.prompts or {}).get(
        "arm_e.j2", ""
    )
    if not source:
        return None
    cfg = ControlConfig.from_template(source)
    if not cfg.thresholds_measurable:
        return None
    # `thresholds_measurable` is what guarantees these are not None; the float()
    # casts keep that guarantee visible to a type checker instead of asserting it.
    return {
        "accuracy_below": float(cfg.pass_threshold_accuracy_below or 0.0),
        "recall_below": float(cfg.pass_threshold_recall_below or 0.0),
    }


# ── writing artefacts ──────────────────────────────────────────────────────────
def _write_json(path: Path, payload: Any) -> Path:
    """One JSON artefact, written the same way everywhere.

    `gate.py`'s own docstring says the verdict must be *written down, not printed*
    — a gate that only prints lives in a scrollback buffer, and the argument about
    whether it passed should not be reconstructible from memory three weeks later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ── P5 · the GO / NO-GO gate (the ONE implementation, in `gate.py`) ───────────
def _evaluate_gate(
    out_dir: Path,
    *,
    context: str,
    run_id: str,
    config_hash: str,
    notes: Sequence[str] = (),
    **inputs: Any,
) -> Any:
    """Evaluate `gate.gate()` and write `go_no_go.json`. Returns the verdict.

    Three call sites used to raise `EXIT_NO_GO` from inline `if` statements, so
    the four-branch verdict lived in three slightly different heads and the
    `BOUNDED_NEGATIVE`-vs-`GO` decision was made three times. It is made once,
    here, by the module that implements it, and the exit code follows
    `GateVerdict.proceed` rather than a second opinion.

    The four inputs default to EMPTY, never to a passing value. A branch with no
    data fails and says so, and `gate()` is explicit that an empty degenerate list
    over an unchecked grid is not estimability — so a caller that has only a design
    fact to report still gets the honest verdict rather than an accidental GO.

    `notes` is appended rather than passed through because `gate()` takes no notes
    argument; a note is how a caller says WHICH of the four branches this call is
    about, and a verdict that cannot say that is a verdict nobody can act on.
    """
    from msat.gate import gate

    verdict = gate(
        accuracy_by_cell=inputs.pop("accuracy_by_cell", {}),
        recall_by_k=inputs.pop("recall_by_k", {}),
        stratified=inputs.pop("stratified", None),
        degenerate_cells=inputs.pop("degenerate_cells", []),
        judge_floor_pp=inputs.pop("judge_floor_pp", 0.0),
        n_questions=inputs.pop("n_questions", 0),
        **inputs,
    )
    verdict.notes.extend(notes)
    _write_json(out_dir / "go_no_go.json", {"context": context, **verdict.as_dict()})
    log.info(
        "p5 gate evaluated",
        extra={
            "run_id": run_id,
            "config_hash": config_hash,
            "gate_context": context,
            "outcome": str(verdict.outcome),
            "passed_branches": verdict.passed_branches,
        },
    )
    typer.echo(f"  gate         {verdict.outcome}  ({context})")
    for branch in verdict.branches:
        mark = "PASS" if branch.passed else "fail"
        colour = typer.colors.GREEN if branch.passed else typer.colors.RED
        typer.secho(f"    {mark}  {branch.branch}", fg=colour)
    return verdict


def _gate_exit(verdict: Any, *, message: str) -> NoReturn:
    """Print the four branches and exit `EXIT_NO_GO`.

    One place decides the exit code from a verdict, so `EXIT_NO_GO` cannot mean
    three different things depending on which command raised it.
    """
    typer.secho(f"\n  NO-GO: {message}", fg=typer.colors.RED, bold=True)
    for branch in verdict.branches:
        typer.secho(f"    {branch.branch}: {branch.detail}", fg=typer.colors.RED)
    for note in verdict.notes:
        typer.secho(f"    note: {note}", fg=typer.colors.YELLOW)
    typer.secho(
        "  A BOUNDED_NEGATIVE is a result, not a failure — and it is decided with the "
        "guide IN ADVANCE,\n  not discovered after the budget is spent.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=EXIT_NO_GO)


# ── R3.2 · the servable ceiling, enforced before dispatch ──────────────────────
@dataclass(frozen=True, slots=True)
class CapacityVerdict:
    """One prompt's fit against ONE reader's SERVABLE ceiling (PRD §R3.2).

    A rejection, never a truncation. `retrieval.yaml` sets
    `never_truncate_silently: true` and `on_overflow: reject_cell`, and the reason
    is that a prompt cut to fit still contains its own answer while appearing not
    to — the accuracy curve that follows measures nothing at all.

    The ceiling is `min(declared window, VRAM)`, and `binding_constraint` says
    which one bound. Both are reported because every family in `readers.yaml`
    declares 131k-262k and can serve 2.7k-36k on an 8 GB card: a reader holding
    only the number, without the label, checks the wrong ceiling and finds nothing
    wrong right up until the allocation fails.
    """

    reader: str
    label: str
    prompt_tokens: int
    ceiling_tokens: int
    vram_ceiling_tokens: int
    declared_window: int
    binding_constraint: str
    accepted: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "reader": self.reader,
            "label": self.label,
            "prompt_tokens": self.prompt_tokens,
            "ceiling_tokens": self.ceiling_tokens,
            "vram_ceiling_tokens": self.vram_ceiling_tokens,
            "declared_context_window": self.declared_window,
            "binding_constraint": self.binding_constraint,
            "accepted": self.accepted,
            "overflow_by": max(0, self.prompt_tokens - self.ceiling_tokens),
            "reason": self.reason,
        }


def _check_capacity(reader: Any, prompt_tokens: int, *, label: str = "") -> CapacityVerdict:
    """`LocalReader.check_capacity` as a REPORT rather than an exception.

    `check_capacity` raises, which is right for a library and wrong for a loop
    over a thousand cells: the run has to keep going and name every cell it
    refused. The raise is caught here and the numbers the message carries are
    recorded, so a rejection in `capacity.json` is legible without the original
    traceback.
    """
    from msat.providers.local import ContextOverflowError

    try:
        limit = reader.check_capacity(prompt_tokens)
    except ContextOverflowError as exc:
        return CapacityVerdict(
            reader=reader.id,
            label=label,
            prompt_tokens=prompt_tokens,
            ceiling_tokens=reader.serving_limit_tokens,
            vram_ceiling_tokens=reader.ceiling_tokens(),
            declared_window=reader.declared_context_window,
            binding_constraint=reader.binding_constraint,
            accepted=False,
            reason=str(exc),
        )
    return CapacityVerdict(
        reader=reader.id,
        label=label,
        prompt_tokens=prompt_tokens,
        ceiling_tokens=limit,
        vram_ceiling_tokens=reader.ceiling_tokens(),
        declared_window=reader.declared_context_window,
        binding_constraint=reader.binding_constraint,
        accepted=True,
        reason=f"{prompt_tokens:,} prompt tokens fit the {limit:,}-token servable ceiling",
    )


def capacity_report(
    verdicts: Sequence[CapacityVerdict],
    *,
    coverage: dict[str, Any] | None = None,
    headroom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every rejection, with its prompt size, its ceiling, and what bound it."""
    per_reader: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        row = per_reader.setdefault(
            v.reader,
            {
                "n_checked": 0,
                "n_accepted": 0,
                "n_rejected": 0,
                "ceiling_tokens": v.ceiling_tokens,
                "vram_ceiling_tokens": v.vram_ceiling_tokens,
                "declared_context_window": v.declared_window,
                "binding_constraint": v.binding_constraint,
            },
        )
        row["n_checked"] += 1
        row["n_accepted" if v.accepted else "n_rejected"] += 1
    return {
        "requirement": "PRD §R3.2",
        "policy": "reject_cell — never truncate silently (retrieval.yaml context_policy)",
        "on_overflow": "reject_cell",
        "never_truncate_silently": True,
        "n_checked": len(verdicts),
        "n_accepted": sum(1 for v in verdicts if v.accepted),
        "n_rejected": sum(1 for v in verdicts if not v.accepted),
        "rejected": [v.as_dict() for v in verdicts if not v.accepted],
        "per_reader": per_reader,
        "panel_coverage": coverage,
        "headroom_sensitivity": headroom,
        "note": (
            "A rejected cell is reported, not repaired. Trimming the prompt leaves it "
            "able to contain its own answer while appearing not to, and the curve that "
            "follows measures nothing (PRD §R0.3 — propose, never truncate)."
        ),
    }


def _panel(bundle: Any) -> list[Any]:
    """Every declared family as a `LocalReader`, with its ceiling COMPUTED.

    Built from the same `readers.yaml` the config hash covers, so the panel the
    run dispatches to is the panel that was hashed. `readers_from_config` refuses
    a single-family panel (R7.5 needs a pair for R9.12 to mean anything).
    """
    from msat.providers.local import readers_from_config

    try:
        return readers_from_config(bundle.raw["readers.yaml"])
    except (ValueError, KeyError) as exc:
        _fail(f"the reader panel cannot be built: {exc}")
        raise AssertionError("unreachable") from exc  # pragma: no cover


def _capacity_preflight(bundle: Any, plan: Any, panel: Sequence[Any]) -> dict[str, Any]:
    """R3.2 over every PLANNED cell, before anything is dispatched.

    The prompt size here is the DECLARED model, because a planned cell has not been
    rendered yet; at dispatch time the RENDERED prompt is measured instead. Both go
    through `LocalReader.check_capacity`, so there is one ceiling and one reject
    rule.

    `design._prompt_tokens` is imported under its private name deliberately. It is
    the single implementation of the declared prompt-size model, and a second copy
    here is exactly how `design-power`'s feasibility numbers and the harness's
    stop-dispatching numbers come to disagree about which cells fit.
    """
    from msat.design import _prompt_tokens as declared_prompt_tokens
    from msat.providers.local import panel_coverage

    pricing = bundle.pricing or {}
    estimates = pricing.get("token_estimates", {})
    overhead = float(pricing.get("prompt_overhead_tokens", 500))
    chars_per_token = float(estimates.get("chars_per_token", 4.0))
    by_id = {r.id: r for r in panel}

    verdicts: list[CapacityVerdict] = []
    for cell in plan.cells:
        reader = by_id.get(cell.condition.reader)
        if reader is None:
            continue
        anchors = estimates.get(cell.condition.scale) or estimates.get("S") or {}
        if not anchors:
            continue
        prompt = int(
            declared_prompt_tokens(
                cell.k,
                cell.b,
                anchors,
                overhead,
                b_mode=cell.condition.b_mode,
                corpus_tokens=anchors.get("corpus_tokens"),
            )
        )
        verdicts.append(
            _check_capacity(
                reader,
                prompt,
                label=f"{cell.condition.scale}/{cell.condition.reader}/K={cell.k}/B={int(cell.b)}",
            )
        )

    coverage = panel_coverage(
        list(panel),
        [(c.k, c.b, c.condition.scale) for c in plan.cells],
    )
    # D11: `headroom_gb` is a guess about fragmentation, not a physical constant,
    # and on 8 GB it moves qwen from 20,889 to 2,682 tokens. Reported rather than
    # resolved, because resolving it here would be picking a number nobody decided.
    headroom = {r.id: r.ceiling_sensitivity() for r in panel}
    report = capacity_report(verdicts, coverage=coverage, headroom=headroom)
    report["prompt_tokens_model"] = (
        f"declared design model, +{overhead:,.0f} tokens overhead, chars_per_token="
        f"{chars_per_token} (ASSUMED). The RENDERED prompt is measured instead at "
        f"dispatch time."
    )
    report["headroom_gb_is_open_decision"] = (
        "D11. Every ceiling above moves with it: qwen's servable depth on 8 GB is "
        "20,889 tokens at 0.8 GB of headroom and 2,682 at the 3.0 GB default. The "
        "number nobody decided is the number that decides which cells run."
    )
    return report


# ── the plan, and whether it is really CROSSED (PRD §R4.4) ────────────────────
def _plan_from(manifest_path: Path, bundle: Any, qids: dict[str, list[str]]) -> Any:
    """The executable plan for a manifest already on disk."""
    from msat.planning import plan_from_manifest

    return plan_from_manifest(
        manifest_path,
        qids,
        seed=int(bundle.design.seed),
        runs_per_cell=int(bundle.design.runs_per_cell),
    )


def _crossing(plan: Any, bundle: Any) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    """`verify_crossed_grid`, plus the two numbers the gate needs from it.

    `n_interaction_cells` counts the `(K, B)` cells that are actually PRESENT for
    every `(scale, reader)` block. It is the difference between branch (iv)
    passing and branch (iv) never having run: `gate()` refuses to GO on an
    unevaluated estimability check, which is the one thing it exists to refuse.

    `degenerate_cells` is the gap list, shaped for the gate. A `missing` entry is
    a cell the design declared and the plan will not execute — the R4.4 hole that
    a per-reader K cap creates silently.
    """
    from msat.planning import verify_crossed_grid

    design = bundle.design
    report = verify_crossed_grid(
        plan,
        k_grid=design.K_grid,
        b_grid=design.b_values,
        scales=design.scales,
        readers=design.readers,
    )
    per_block = len(design.K_grid) * len(design.b_values)
    n_interaction_cells = per_block * sum(
        1 for b in report["per_scale_reader"].values() if b["complete"]
    )
    degenerate: list[dict[str, Any]] = []
    for block, row in sorted(report["per_scale_reader"].items()):
        for k, b in row["missing"]:
            degenerate.append({"scale_reader": block, "K": k, "B": b, "reason": "missing cell"})
        for k, b in row["extra"]:
            degenerate.append({"scale_reader": block, "K": k, "B": b, "reason": "undeclared cell"})
    return report, n_interaction_cells, degenerate


def _load_gold_sessions(path: Path) -> dict[str, list[str]]:
    """`qid -> answer_session_ids`, read from `--questions`.

    SESSION ids, never chunk ids. Gold is a property of a session and a memory is
    a chunk of one, so a mapping keyed by chunk id would exclude the wrong thing
    and let real evidence into the negative control.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _fail(f"{path} must be a JSON object mapping qid -> [answer_session_ids].")
    return {str(k): [str(s) for s in v] for k, v in raw.items()}  # type: ignore[union-attr]


# ── Arm E · the negative control, ACTUALLY RUN (PRD §R4.3 / §G5) ──────────────
def load_control_pool(path: Path) -> tuple[Any, dict[str, str]]:
    """Arm E's cross-question memory pool, from JSON.

        {"q_target": {"question": "...", "memories": [
             {"memory_id": "...", "session_id": "...", "text": "...", "n_tokens": 120}
         ]}, ...}

    Gold is NOT in here and must not be: it is `answer_session_ids`, passed
    separately, and `draw_control_memories` derives the exclusion from it. A pool
    that carried gold would let the control quietly keep the answer, which is the
    one thing the negative control exists to detect.

    Supplied as a file rather than built in-process so `msat run` does not have to
    re-stream the 2.6 GB M split to check whether the pipeline is flattering
    itself. The pool is what the write path (`memory.write`) already produced.
    """
    from msat.arms.negative import PooledMemory
    from msat.arms.render import Memory

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _fail(f"{path} must be a JSON object mapping qid -> {{question, memories}}.")
    by_question: dict[str, list[Any]] = {}
    question_text: dict[str, str] = {}
    for qid, entry in raw.items():
        if not isinstance(entry, dict) or "memories" not in entry:
            _fail(f"{path}: entry {qid!r} must be an object with a `memories` list.")
        question_text[str(qid)] = str(entry.get("question", ""))
        memories = []
        for raw_memory in entry["memories"]:  # type: ignore[index]
            memories.append(
                Memory(
                    memory_id=str(raw_memory["memory_id"]),
                    session_id=str(raw_memory["session_id"]),
                    source_question_id=str(qid),
                    text=str(raw_memory.get("text", "")),
                    n_tokens=int(raw_memory.get("n_tokens", 0)),
                )
            )
        by_question[str(qid)] = memories
    return PooledMemory(by_question=by_question), question_text


def _arm_e_cells(bundle: Any, scale: str) -> list[Any]:
    """Arm E's cells: the SHALLOWEST declared `(K, B)`, once per reader.

    The control has to detect a pipeline that flatters itself, and it detects that
    at any cell. Running it at the cheapest one is the ladder's whole ordering
    principle applied honestly: Arm E exists because it is the cheapest way to
    discover the run is measuring nothing.
    """
    from msat.store.responses import ConditionKey

    k0 = int(bundle.design.K_grid[0])
    b0 = bundle.design.B_grid[0]
    return [
        ConditionKey("E", k0, float(b0.value), str(b0.mode), scale, reader)
        for reader in bundle.design.readers
    ]


def run_negative_control(
    *,
    bundle: Any,
    arm: Any,
    cells: Sequence[Any],
    readers: Sequence[Any],
    pool: Any,
    question_text: dict[str, str],
    gold_by_question: dict[str, list[str]],
    gold_answers: dict[str, str],
    generate: Callable[[Any, Any], Any],
    max_questions: int | None = None,
) -> dict[str, Any]:
    """Draw -> render -> check capacity -> generate -> score -> evaluate.

    The whole arm, end to end, with `generate` as the single injected seam. That
    is the same seam `runner/executor.Generator` uses, for the same reason: the
    arm must be checkable without a GPU, or the check only happens on the
    execution box.

    Every refusal is RECORDED rather than raised. An overflowing prompt and an
    under-filled control are both "this cell cannot be run as declared", and both
    must be visible in the report instead of ending the arm at the first one.
    """
    from msat.arms.negative import ControlConfig, evaluate, go_no_go, render_control
    from msat.metrics.normalisation import token_f1
    from msat.providers.base import BackendUnavailableError

    config = ControlConfig.from_template(
        (bundle.prompts or {}).get("prompts/arm_e.j2") or (bundle.prompts or {}).get("arm_e.j2", "")
    )
    chars_per_token = float(
        (bundle.pricing or {}).get("token_estimates", {}).get("chars_per_token", 4.0)
    )
    gold_sets = {q: frozenset(v) for q, v in gold_by_question.items()}
    # Only questions that DECLARE GOLD are usable: recall is undefined without it
    # (there is nothing to be a fraction of) and there is no answer to score the
    # control against. Selecting the first pool entry alphabetically instead would
    # silently pick a donor and report a control with n=0 as INCONCLUSIVE, which
    # reads like a threshold problem rather than an input problem.
    qids = [q for q in sorted(gold_sets) if gold_sets[q] and q in pool.by_question]
    if max_questions is not None:
        qids = qids[:max_questions]
    if not qids:
        _fail(
            "no question has BOTH declared gold sessions and a pool entry, so Arm E "
            "cannot be count-matched against anything. --questions and --arm-e-pool "
            "must cover at least one common qid that declares `answer_session_ids`."
        )

    normaliser = _load_normaliser()
    verdicts: list[CapacityVerdict] = []
    refusals: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    scores: list[float] = []
    recalls: list[float] = []
    backend_error: dict[str, Any] | None = None
    n_generated = 0
    n_unattributed = 0

    for cell in cells:
        reader = next((r for r in readers if r.id == cell.reader), None)
        if reader is None:
            continue
        for qid in qids:
            gold = gold_sets[qid]
            label = f"E/{cell.scale}/{reader.id}/K={cell.k}/B={int(cell.b)}/{qid}"
            try:
                rendered = render_control(
                    arm,
                    question_text.get(qid, ""),
                    pool,
                    qid,
                    gold,
                    cell,
                    config,
                    budget_tokens=int(cell.b),
                    qid=qid,
                    gold_by_question=gold_sets,
                )
            except ValueError as exc:
                # Under-filled control: fewer donors than K, so the control would
                # collapse for the WRONG reason and the arm would "pass" while
                # measuring nothing.
                refusals.append(
                    {
                        "label": label,
                        "kind": "underfilled_control",
                        "reason": str(exc),
                        "prompt_tokens": None,
                        "ceiling_tokens": reader.serving_limit_tokens,
                        "binding_constraint": reader.binding_constraint,
                    }
                )
                continue

            # ── R3.2, before dispatch ─────────────────────────────────────────
            verdict = _check_capacity(
                reader,
                int(len(rendered.prompt) / chars_per_token) + 1,
                label=label,
            )
            verdicts.append(verdict)
            if not verdict.accepted:
                refusals.append({**verdict.as_dict(), "kind": "context_overflow"})
                continue

            # ── the one seam this harness cannot cross without a GPU ─────────
            # Everything above this line — draw, render, R3.2, the request body —
            # is the production path and runs on any machine. Only the token
            # generation needs an engine, and it is reported as exactly that: a
            # missing backend is NOT a scientific verdict, and letting an
            # ImportError escape here would present a missing optional dependency
            # as a crashed experiment.
            try:
                result = generate(reader, _generation_request(bundle, reader, rendered))
            except (BackendUnavailableError, ImportError, OSError) as exc:
                backend_error = {
                    "available": False,
                    "reader": reader.id,
                    "backend": str(getattr(reader, "backend_name", "transformers")),
                    "error": type(exc).__name__,
                    "detail": str(exc),
                    "n_generated_before_failure": n_generated,
                    "remedy": (
                        "Arm E is generated by a LOCAL engine; there is no API reader "
                        "in this design. Install `bitsandbytes` for the 4-bit path "
                        "(uv sync --extra bits) or run vLLM as a service "
                        "(make vllm), then re-run. `msat validate-config` reports "
                        "backend availability without loading weights."
                    ),
                }
                break  # a dead engine does not come back mid-loop

            n_generated += 1
            hits = sorted(set(rendered.session_ids) & gold)
            draws.append(
                {
                    "label": label,
                    "n_drawn": rendered.n_memories,
                    "count_matched_to_K": rendered.n_memories == cell.k,
                    "session_ids": list(rendered.session_ids),
                    "gold_sessions_drawn": hits,
                }
            )
            if gold:
                recalls.append(len(hits) / len(gold))
            answer = gold_answers.get(qid)
            if answer is None:
                n_unattributed += 1
                continue
            scores.append(token_f1(result.text, answer, normaliser))

    accuracy = sum(scores) / len(scores) if scores else None
    recall = sum(recalls) / len(recalls) if recalls else None
    outcome = evaluate(accuracy, recall, config, n_questions=len(scores))
    report: dict[str, Any] = {
        **go_no_go(outcome),
        "arm": "E",
        "n_cells": len(cells),
        "n_questions_considered": len(qids),
        "n_generated": n_generated,
        "n_refused": len(refusals),
        "n_scored": len(scores),
        "backend": backend_error
        or {
            "available": True,
            "backend": str(getattr(readers[0], "backend_name", "transformers"))
            if readers
            else None,
        },
        "diagnostics": {
            # `evaluate()` returns one INCONCLUSIVE reason for three different
            # causes, and only one of them is null thresholds. A reader told "the
            # thresholds are null" when the real problem is a gold file missing the
            # question would go and edit the wrong file.
            "accuracy_is_none_because": (
                "no usable control output was scored" if accuracy is None else None
            ),
            "recall_is_none_because": (
                "no drawn control carried a gold session, so recall is undefined"
                if recall is None
                else None
            ),
            "n_unattributed": n_unattributed,
            "refusal_kinds": sorted({str(r.get("kind")) for r in refusals}),
            "thresholds_measurable": config.thresholds_measurable,
        },
        "capacity": capacity_report(verdicts),
        "refusals": refusals,
        "control": {
            "count_matched": all(d["count_matched_to_K"] for d in draws) if draws else None,
            "mean_recall_of_gold": recall,
            "mean_token_f1": accuracy,
            "draws": draws,
        },
        "why_it_matters": (
            "Arm E draws memories from OTHER questions, count-matched to the Arm A "
            "condition so only the CONTENT differs. A correct reader scores near "
            "zero. If it does not, the pipeline is not respecting the condition and "
            "every other number in this project is suspect (PRD §R4.3 / §G5)."
        ),
    }
    return report


def _generation_request(bundle: Any, reader: Any, rendered: Any) -> Any:
    """One `GenerationRequest`, from the DECODING config rather than a default.

    Every field here is a confound that `decoding.yaml` exists to pin (R7.1-R7.4),
    so reading them is the point; a default written at the call site would be a
    second definition that agrees with the config until someone edits it.
    """
    from msat.providers.base import GenerationRequest

    decoding = bundle.decoding
    return GenerationRequest(
        model=reader.model,
        prompt=rendered.user,
        system=rendered.system,
        max_tokens=int(decoding.max_tokens),
        temperature=float(decoding.temperature),
        top_p=float(decoding.top_p),
        stop=tuple(str(s) for s in decoding.stop_sequences),
        seed=int(decoding.provider_seed),
        request_id=rendered.arm,
    )


def _require_arm_e_inputs(
    *, questions: Path | None, gold: Path | None, pool: Path | None
) -> tuple[Path, Path, Path]:
    """Every Arm E input, or a list of every one that is missing.

    Reported TOGETHER rather than one at a time. A user invoking the negative
    control for the first time is missing all three, and an error that names only
    the first costs a round-trip per round-trip — which at 3am on day one is the
    difference between a ten-second fix and an hour.

    The backend is named in the same breath because it is also true and also
    blocking: even with all three files, nothing is generated until a local engine
    is loadable. Saying only half of it sends the reader to fix the wrong thing.
    """
    missing: list[str] = []
    if questions is None:
        missing.append(
            "--questions (JSON: qid -> answer_session_ids. The control excludes "
            "gold by SESSION id, and gold is derived from `answer_session_ids`, so "
            "there is no default.)"
        )
    if gold is None:
        missing.append(
            "--gold (JSON: qid -> gold answer. The control's accuracy threshold is "
            "a number, and a number needs something to be measured against.)"
        )
    if pool is None:
        missing.append(
            "--arm-e-pool (JSON cross-question memory pool: {qid: {question, "
            "memories}}. A negative control is memories from OTHER questions, and "
            "this harness will not invent them.)"
        )
    if not missing:
        assert questions is not None and gold is not None and pool is not None
        return questions, gold, pool

    _fail(
        "the Arm E rung is missing "
        + ("input" if len(missing) == 1 else "inputs")
        + ":\n    - "
        + "\n    - ".join(missing)
        + "\n  Generation also needs a backend: even with all three, nothing is "
        "dispatched until a local engine (transformers / vLLM / onnxruntime-genai) "
        "is loadable."
    )
    raise AssertionError("unreachable")  # pragma: no cover


def _run_arm_e_rung(
    run_dir: Path,
    *,
    bundle: Any,
    registry: Any,
    panel: Sequence[Any],
    executable: Any,
    questions: Path | None,
    gold: Path | None,
    pool: Path | None,
    reader_id: str | None,
    run_id: str,
) -> Any:
    """Execute the negative control and gate the run on it.

    This is the ladder's first rung and the gate on the whole experiment. It runs
    BEFORE any other arm, on purpose: it is the cheapest possible moment to
    discover the pipeline is measuring nothing, and it costs one small arm.
    """
    from msat.runner.ladder import LADDER

    questions, gold, pool = _require_arm_e_inputs(questions=questions, gold=gold, pool=pool)

    chosen_readers = [r for r in panel if reader_id is None or r.id == reader_id]
    if not chosen_readers:
        _fail(f"unknown reader {reader_id!r}; the panel is {[r.id for r in panel]}")

    rung = next(r for r in LADDER if r.name == "negative")
    cells = _arm_e_cells(bundle, rung.scales[0])
    control_arm = registry.get("E")
    control_pool, question_text = load_control_pool(pool)

    report = run_negative_control(
        bundle=bundle,
        arm=control_arm,
        cells=cells,
        readers=chosen_readers,
        pool=control_pool,
        question_text=question_text,
        gold_by_question=_load_gold_sessions(questions),
        gold_answers=_load_gold(gold),
        generate=lambda reader, request: reader.generate(request),
        max_questions=None if rung.n_questions < 0 else rung.n_questions,
    )
    _write_json(run_dir / "arm_e_control.json", report)
    log.info(
        "arm e control evaluated",
        extra={
            "run_id": run_id,
            "config_hash": bundle.config_hash,
            "arm": "E",
            "verdict": report["verdict"],
            "accuracy": report.get("accuracy"),
            "recall": report.get("recall"),
            "backend_available": bool(report["backend"]["available"]),
        },
    )
    typer.echo(f"  wrote {run_dir / 'arm_e_control.json'}")

    # A missing backend is NOT a verdict, and it is checked BEFORE the verdict is
    # printed. `evaluate()` returns one INCONCLUSIVE reason for three different
    # causes, so printing "the thresholds are null" when the real problem is an
    # uninstalled optional dependency sends someone to edit the wrong file — and a
    # broken dependency must never be readable as a scientific result.
    backend = report["backend"]
    if not backend["available"]:
        _fail(
            f"Arm E was drawn, rendered, capacity-checked and ready to dispatch, but "
            f"reader {backend['reader']!r} has no loadable {backend['backend']} "
            f"backend: {backend['error']}: {backend['detail']}\n"
            f"  {backend['remedy']}\n"
            f"  Nothing was simulated and no verdict is claimed. "
            f"{run_dir / 'arm_e_control.json'} records what was built before the seam.",
            code=EXIT_FAILURE,
        )

    typer.secho(
        f"\n  arm E       {report['verdict']}   accuracy={report['accuracy']}  "
        f"recall={report['recall']}  n={report['n_questions']}",
        fg=typer.colors.GREEN if report["verdict"] == "PASS" else typer.colors.RED,
        bold=True,
    )
    typer.echo(f"  {report['reason']}")
    if report["n_refused"]:
        typer.secho(
            f"  {report['n_refused']} control unit(s) were REFUSED and are named in "
            f"arm_e_control.json — a refused cell is reported, never truncated.",
            fg=typer.colors.YELLOW,
        )

    # The gate re-decides, so `go_no_go.json` carries the negative control's
    # verdict rather than leaving it in a sibling file nobody opens.
    _, n_interaction_cells, degenerate_cells = _crossing(executable, bundle)
    verdict = _evaluate_gate(
        run_dir,
        context=f"{run_id}/arm_e",
        run_id=run_id,
        config_hash=bundle.config_hash,
        notes=[
            f"Arm E negative control: {report['verdict']} — {report['reason']}",
            "INCONCLUSIVE gates exactly like FAIL: proceeding on 'we don't know' is "
            "how a broken pipeline gets twelve days of GPU time.",
        ],
        accuracy_by_cell={},
        recall_by_k={},
        stratified=None,
        degenerate_cells=degenerate_cells,
        n_interaction_cells=n_interaction_cells if report["verdict"] == "PASS" else 0,
    )
    if not report["proceed"]:
        typer.secho(
            "\n  !!! ARM E DID NOT COLLAPSE !!!\n"
            "  The negative control draws memories from OTHER questions, so a correct\n"
            "  reader scores near zero. A control that does not collapse means the\n"
            "  pipeline is not respecting the condition it claims to manipulate — and\n"
            "  that flatters every other arm measured through the same path.\n"
            f"  {report['next']}",
            fg=typer.colors.RED,
            bold=True,
            err=True,
        )
        _gate_exit(verdict, message="the negative control did not collapse (PRD §R4.3)")
    return report


# ── R8.5 · latency, from the store ─────────────────────────────────────────────
def _latency_samples(store: Any) -> tuple[list[Any], dict[str, list[Any]], int]:
    """Every stored response as a `LatencySample`, plus a per-reader split.

    A record whose decomposition is impossible (`ttft_ms` below the queue wait)
    is COUNTED and reported rather than allowed to raise. `LatencySample` rejects
    it deliberately — it is a bug in the client — and a reporting command that
    crashes on one bad record has stopped reporting on the other 189,000.
    """
    from msat.metrics.latency import LatencySample

    total: list[Any] = []
    by_reader: dict[str, list[Any]] = {}
    n_impossible = 0
    for record in store.iter_all():
        try:
            sample = LatencySample(
                latency_ms=record.latency_ms,
                queue_wait_ms=record.queue_wait_ms,
                ttft_ms=record.ttft_ms,
            )
        except ValueError:
            n_impossible += 1
            continue
        total.append(sample)
        by_reader.setdefault(str(record.condition.get("reader", "unknown")), []).append(sample)
    return total, by_reader, n_impossible


def _ttft_samples(samples: Sequence[Any]) -> list[Any]:
    """TTFT as its OWN sample stream (PRD §R8.5).

    Reported separately from total latency because they measure different things:
    `ttft_ms` is queue wait -> first token, `latency_ms` is queue wait -> last
    token. Quoting one as the other is how a p50 of 200 ms and a p99 of 4 s gets
    read as server slowness when it is the generator's own tail.
    """
    from msat.metrics.latency import LatencySample

    return [
        LatencySample(
            latency_ms=s.ttft_ms,
            queue_wait_ms=s.queue_wait_ms,
            ttft_ms=s.ttft_ms,
        )
        for s in samples
        if s.ttft_ms is not None and s.ttft_ms > 0
    ]


def latency_artifact(store: Any) -> dict[str, Any]:
    """R8.5's report: medians, IQR and quantiles; TTFT separate from total.

    There is no bootstrap CI on the mean anywhere in this artifact, and there is
    none available to add: `metrics/latency.py` reports the shape rather than an
    interval whose width is set by the tail.
    """
    from msat.metrics.latency import latency_report, summarise_latency

    total, by_reader, n_impossible = _latency_samples(store)
    ttft = _ttft_samples(total)
    return {
        "requirement": "PRD §R8.5",
        "total_latency": latency_report(total, by_label=dict(by_reader)),
        "ttft": summarise_latency(ttft, label="ttft").as_dict(),
        "ttft_is_separate_because": (
            "ttft_ms is queue_wait -> first token and latency_ms is queue_wait -> last "
            "token. They measure different things, so they are summarised separately; "
            "the total is a property of the generator's tail, the TTFT is a property "
            "of the reader (PRD §R5.3)."
        ),
        "n_impossible_records": n_impossible,
        "n_latency_samples": len(total),
        "no_bootstrap_ci": (
            "Deliberately absent. A bootstrap CI on a heavy-tailed mean reports "
            "uncertainty about a statistic nobody should quote (PRD §R8.5)."
        ),
    }


def _print_latency(artifact: dict[str, Any]) -> None:
    """The R8.5 table: median, IQR and quantiles, for total and TTFT apart."""
    overall = artifact["total_latency"]["overall"]
    ttft = artifact["ttft"]
    typer.secho(
        "\n  latency (R8.5) — medians, IQR and quantiles; never a bootstrap CI",
        fg=typer.colors.CYAN,
    )
    typer.echo(f"    {'series':<10}{'n':>7}{'p50':>10}{'IQR':>10}{'p90':>10}{'p99':>11}   mean")
    for label, block in (("total", overall), ("ttft", ttft)):
        q = block["quantiles_ms"]
        typer.echo(
            f"    {label:<10}{block['n']:>7}{q.get('p50', 0):>10,.0f}"
            f"{block['iqr_ms']:>10,.0f}{q.get('p90', 0):>10,.0f}{q.get('p99', 0):>11,.0f}"
            f"{block['mean_ms']:>10,.0f}"
        )
    if ttft["n"] == 0:
        typer.secho(
            "    no TTFT was recorded on any response, so the reader's own latency to "
            "first token is unreported rather than inferred.",
            fg=typer.colors.YELLOW,
        )
    if not ttft["p99_reliable"]:
        typer.secho(
            f"    p99 rests on n={ttft['n']}; below ~100 calls it is one or two "
            f"observations and reads as a precise number.",
            fg=typer.colors.YELLOW,
        )
    for reader, block in sorted(artifact["total_latency"].get("by_reader", {}).items()):
        typer.echo(
            f"      {reader:<8} p50 {block['quantiles_ms'].get('p50', 0):>9,.0f} ms   "
            f"IQR {block['iqr_ms']:>9,.0f} ms   n={block['n']}"
        )
    queue = artifact["total_latency"]["queue_wait"]
    if queue["recorded"] and not queue["recording_concurrency"]:
        typer.secho(
            f"    queue wait is {100 * queue['share_of_latency']:.0f}% of total latency, "
            f"so `total` describes THIS HARNESS, not the reader. Use `queue_excluded_*` "
            f"to compare readers.",
            fg=typer.colors.YELLOW,
        )
    if artifact["n_impossible_records"]:
        typer.secho(
            f"    {artifact['n_impossible_records']} record(s) declare a TTFT below "
            f"their queue wait and were EXCLUDED rather than repaired.",
            fg=typer.colors.YELLOW,
        )


@app.command("status")
def status(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    results_dir: Annotated[
        Path, typer.Option("--results", help="Parent of the per-run directory.")
    ] = Path("results"),
    ladder: Annotated[
        bool, typer.Option("--ladder/--no-ladder", help="Show the smoke ladder state.")
    ] = True,
) -> None:
    """Print the per-cell completion ledger and spend to date (PRD §R5.4).

    Also prints where the run sits on the smoke ladder, because "the run is 40%
    done" is only interesting next to "and the rung that authorised it passed".

    The three numbers that must be read together:

        stored  responses in the content-addressed store  - the RECORD
        done    units the ledger credits                    - the ACCOUNT
        wal     units the write-ahead log claims            - may LAG, never lead

    A WAL entry with no store entry means a crash between append and commit. It
    is safe to resume; it is NOT safe to report as done. The store is the
    authority, which is why `stored` is printed first.

    PRD §R5.4.
    """
    from msat.runner.executor import RunConfig, status_report
    from msat.runner.ladder import LADDER

    resolved = _resolve_run_id(run_id, run)
    _assert_offline("status")
    run_dir = results_dir / resolved
    if not run_dir.is_dir():
        _fail(f"no run directory at {run_dir}. Build one with `msat build-plan`.")

    report = status_report(RunConfig(run_dir=run_dir))

    typer.echo(f"\n  run       {resolved}")
    typer.echo(f"  dir       {report['run_dir']}")
    typer.echo(f"  stored    {report['n_stored_responses']:,} responses   <- the record")
    typer.echo(
        f"  ledger    {report['n_done']:,} / {report['n_planned']:,} done"
        f"   across {report['n_cells']:,} cells"
    )
    if report["n_failed"]:
        typer.secho(f"  failed    {report['n_failed']:,}", fg=typer.colors.RED)
    if report["n_excluded"]:
        typer.secho(
            f"  excluded  {report['n_excluded']:,} question(s) - degenerate at this "
            f"K, dropped from paired comparisons",
            fg=typer.colors.YELLOW,
        )

    # A store/ledger disagreement is the one thing that must not scroll past.
    if report["n_done"] > report["n_stored_responses"]:
        typer.secho(
            f"\n  INCONSISTENT: the ledger credits {report['n_done'] - report['n_stored_responses']:,} "
            f"more unit(s) than the store holds. The ledger is ahead of the record. "
            f"Run `msat replay {resolved}` to rebuild it from the store.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=EXIT_ASSERTION_FAILED)

    cells = report["cells"]
    if cells:
        typer.echo("\n  cell                                     planned  done  failed")
        for c in sorted(cells, key=lambda x: tuple(str(p) for p in x["key"]))[:20]:
            # The key is (arm, K, B, b_mode, scale, reader). Unpacked rather than
            # read by name: a record that lost a field should still print, and a
            # silent `?` column would hide that it did.
            arm, k, b, _b_mode, scale, reader = c["key"]
            label = f"    {arm} {scale} {reader} K={k} B={b}"
            typer.echo(
                label.ljust(42)
                + f"{c.get('n_planned', 0):>7} {c.get('n_done', 0):>5} {c.get('n_failed', 0):>7}"
            )
        if len(cells) > 20:
            typer.echo(f"    ... and {len(cells) - 20} more cells")

    if not ladder:
        return
    state_path = run_dir / "ladder.json"
    state = _read_ladder(state_path)
    nxt = state.first_unpassed()
    typer.echo("\n  ladder")
    for rung, passed in zip(LADDER, [r.name in state.passed for r in LADDER], strict=True):
        mark = typer.style("pass", fg=typer.colors.GREEN) if passed else "todo"
        typer.echo(f"    {mark:<22} {rung.arms}  ({rung.description})")
    if nxt is not None:
        typer.secho(
            f"    next: {nxt.name} - {nxt.description}. The full grid is blocked until "
            f"it passes; that is the ladder working, not a bug.",
            fg=typer.colors.YELLOW,
        )
    typer.echo()


@app.command("replay")
def replay(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
) -> None:
    """Reproduce a run's metrics from the content-addressed store.

    Promises **metric-identical** output, never byte-identical: pyarrow/duckdb
    drift and hash-seed ordering make a byte guarantee falsifiable (PRD §A12).

    PRD §R6.3.
    """
    _resolve_run_id(run_id, run)
    _assert_offline("replay")
    _pending("replay", "R6.3")


# ── P6 · metrics and judges ────────────────────────────────────────────────────
@app.command("score")
def score(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    metric: Annotated[
        str,
        typer.Option("--metric", help="One of: em, f1, judge, official, all."),
    ] = "all",
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    results_dir: Annotated[
        Path, typer.Option("--results", help="Parent of the per-run directories.")
    ] = Path("results"),
    gold: Annotated[
        Path | None,
        typer.Option("--gold", help="JSON: qid -> gold answer. Required."),
    ] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs"),
) -> None:
    """Re-score a completed run. Reads Storage only — never calls a model.

    This is PRD §G4 in one command: metrics can be iterated in minutes with no
    re-inference.

    WHAT IS WRITTEN
    ---------------
    `scores.parquet`        one row per (condition, qid, metric). NEVER an aggregate —
                           R8.3, because the metric code will change and a stored
                           aggregate cannot be re-derived under a new one.
    `disagreements.parquet` per-item rows where judges differ, WITH their content.
    `latency.json`          medians, IQR and quantiles from the SAME records, with
                           TTFT summarised separately from total (R8.5).

    Normalisation comes from `configs/prompts/normalisation.yaml` (R1.2), not from
    this code. A `lower()` written here would be a second definition, and the two
    would agree right up until someone edited the config.

    R8.1 wants EM, token-F1 and the official evaluator SIDE BY SIDE. The official
    evaluator is supplied via `--official`; without it the run says so, because a
    two-metric table is indistinguishable from a three-metric one in the artifact.

    PRD §R8.1, §R8.4, §R8.5, §G4.
    """
    from msat.metrics.score import (
        build_disagreements,
        score_store,
        write_disagreements,
        write_scores,
    )
    from msat.store.responses import ResponseStore

    resolved = _resolve_run_id(run_id, run)
    _assert_offline("score")
    run_dir = results_dir / resolved
    if not run_dir.is_dir():
        _fail(f"no run directory at {run_dir}.")
    if gold is None:
        _fail(
            "--gold is required. Scoring without gold answers produces a table with "
            "no way to check a single row, and a metric code path that never runs."
        )
    gold_map = _load_gold(Path(gold))
    store = ResponseStore(run_dir / "responses")
    if len(store) == 0:
        _fail(f"{run_dir / 'responses'} holds no responses; nothing to score.")

    metrics = {"em", "f1"} if metric in ("all", "judge") else {metric}
    unknown = metrics - {"em", "f1"}
    if unknown:
        _fail(f"unknown metric(s) {sorted(unknown)}; use em, f1, judge or all")
    official = _load_official(results_dir, resolved)

    rows, report = score_store(store, gold_map, _load_normaliser(), official=official)
    dest = run_dir / "scores.parquet"
    write_scores(rows, dest)
    report.path = str(dest)
    report.n_bytes = dest.stat().st_size

    typer.echo(f"\n  scored      {report.n_attributable:,} item(s) -> {report.n_rows:,} rows")
    typer.echo(f"  metrics     {', '.join(report.metrics_written)}")
    typer.echo(f"  abstention  {report.n_abstention:,} scored NULL, not 0.0 (R8.4)")
    if report.n_unattributable:
        typer.secho(
            f"  UNATTRIBUTED {report.n_unattributable} response(s) carry no qid and were "
            f"counted, not scored. They cannot be joined to a question.",
            fg=typer.colors.YELLOW,
        )
    for metric_name, m in sorted(report.means.items()):
        mean = "n/a" if m["mean"] is None else f"{m['mean']:.4f}"
        typer.echo(
            f"  {metric_name:<10}{mean:>8}   over {m['n_scored']:,} scored "
            f"({m['n_abstention_excluded']:,} abstention excluded)"
        )
    for note in report.notes:
        typer.secho(f"  note: {note}", fg=typer.colors.YELLOW)
    typer.echo(f"\n  wrote {dest}")

    # ── R8.5 · latency, reduced from the SAME records ────────────────────────
    # `store/responses.py` has carried `latency_ms` / `ttft_ms` / `queue_wait_ms`
    # since P4 and nothing read them: the run's timing data was stored, summarised
    # by nobody, and a bootstrap CI on the mean was the only thing anyone would
    # have written by hand. Median, IQR and quantiles are what R8.5 asks for.
    latency = latency_artifact(store)
    latency_dest = _write_json(run_dir / "latency.json", latency)
    _print_latency(latency)
    typer.echo(f"  wrote {latency_dest}")

    # R8.3 — per-item disagreements, WITH their content. Written only when judge
    # verdicts exist, and the absence is stated rather than leaving a missing file
    # to be interpreted as "no disagreements".
    verdict_path = run_dir / "verdicts.json"
    if verdict_path.exists():
        from msat.judges.verdicts import Verdict

        records = json.loads(verdict_path.read_text(encoding="utf-8"))
        by_judge: dict[str, dict[tuple[str, str], Verdict]] = {}
        texts: dict[tuple[str, str], str] = {}
        for r in records:
            if int(r.get("pass_index", 1)) != 1:
                continue
            key = (str(r["condition_key"]), str(r["qid"]))
            by_judge.setdefault(str(r["judge_id"]), {})[key] = Verdict(str(r["verdict"]))
            if r.get("cache_key"):
                rec = store.get(str(r["cache_key"]))
                if rec is not None:
                    texts[key] = rec.text
        dis = build_disagreements(by_judge, texts, gold_map, _load_normaliser())
        dis_dest = run_dir / "disagreements.parquet"
        write_disagreements(dis, dis_dest)
        typer.echo(f"  disagreements {len(dis):,} row(s) -> {dis_dest.name}")
        if len(by_judge) < 2:
            typer.secho(
                f"  only {len(by_judge)} judge present, so there can be no disagreement "
                f"row. R8.2 requires >= 2 judges on ALL outputs.",
                fg=typer.colors.YELLOW,
            )
    else:
        typer.secho(
            f"  no {verdict_path.name}; disagreements.parquet was not written. A missing "
            f"file is not evidence that the judges agreed.",
            fg=typer.colors.YELLOW,
        )


@app.command("judge-audit")
def judge_audit(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    verdicts_path: Annotated[
        Path | None,
        typer.Option("--verdicts", help="JSON list of verdict records. Required."),
    ] = None,
    out: Annotated[Path, typer.Option("--out")] = Path("results/judge_audit.json"),
    human: Annotated[Path | None, typer.Option("--human", help="Filled human_sample.csv.")] = None,
) -> None:
    """Two passes of the same deterministic system under the same judge.

    Reports the verdict-flip count. Replicates 2609.38021's protocol, whose
    LongMemEval-S result (479 vs 475 / 500, 8 flipped rows) is the reference bound —
    our judge-noise floor is measured, not speculative.

    STANDALONE BY CONSTRUCTION (R8.7)
    ---------------------------------
    Reads verdict records only. It does not read `scores.parquet` or
    `conditions.parquet` and does not need a run to have finished, so a ZERO
    INTERACTION still produces an audit — which is exactly the case where an
    unreliable panel is the reason and nobody would otherwise know.

    `n_flips` counts ROWS whose verdict changed, not the aggregate difference.
    Flips in opposite directions cancel: 2609.38021 moved 4 in aggregate while
    flipping 8 rows.

    PRD §R8.6, §R8.7.
    """
    from msat.judges.report import build_audit, format_audit

    resolved = _resolve_run_id(run_id, run)
    _assert_offline("judge-audit")
    if verdicts_path is None or not Path(verdicts_path).exists():
        _fail(
            "--verdicts is required: the audit is built from verdict records alone, "
            "which is what makes it standalone."
        )
    records = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    human_data = _load_human_agreement(Path(human)) if human else None

    audit = build_audit(records, human=human_data)
    dest = audit.write(Path(out))
    typer.echo(format_audit(audit))
    typer.echo(f"\n  wrote {dest}")

    # ── the P5 gate, from `gate.py` ──────────────────────────────────────────
    # The panel's usability is decided here and nowhere else, and the exit code
    # follows `GateVerdict.proceed` rather than a second `if audit.panel_usable`.
    # The audit's flip rate becomes the judge's noise FLOOR, which is what branch
    # (i) is measured against — a 6pp grid spread means nothing if the panel itself
    # moves 8pp on identical outputs.
    floor = audit.noise_floor or 0.0
    gate_verdict = _evaluate_gate(
        dest.parent,
        context=f"{resolved}/judge_audit",
        run_id=resolved,
        config_hash="",
        notes=[
            f"Judge panel: kappa_min={audit.panel.get('kappa_min')} against a bar of "
            f"{audit.panel.get('cautionary_bar')}, {audit.n_judges} judge(s), "
            f"{audit.n_items:,} item(s).",
            "Branches (i)-(iii) are empty because this command reads verdict records "
            "only (R8.7): it is a STANDALONE artefact and does not need scores.",
        ],
        accuracy_by_cell={},
        recall_by_k={},
        stratified=None,
        degenerate_cells=(
            [] if audit.panel_usable else [{"judge_pair": p} for p in _unusable_pairs(audit)]
        ),
        n_interaction_cells=0,
        judge_floor_pp=100.0 * floor,
    )
    if not audit.panel_usable:
        _gate_exit(gate_verdict, message="the judge panel is not usable (PRD §R8.2)")


def _unusable_pairs(audit: Any) -> list[str]:
    """Which judge pairs failed kappa, named rather than counted.

    `kappa_min` alone says the panel is weak; the pair says why, and the reason is
    the thing someone has to go and fix. `panel_agreement` has already rendered
    each pair's verdict in words, so that is read rather than re-derived here.
    """
    panel = getattr(audit, "panel", {}) or {}
    return [
        str(entry.get("pair", "?"))
        for entry in panel.get("verdicts", []) or []
        if "BELOW" in str(entry.get("verdict", "")) or "UNDEFINED" in str(entry.get("verdict", ""))
    ]


def _load_gold(path: Path) -> dict[str, str]:
    """qid -> gold answer.

    JSON, and a flat mapping only. A gold file that also carried the judge's
    verdicts would put them in the same place as the answers, and R8.2's point is
    that the human labeller never sees a verdict.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _fail(f"{path} must be a JSON object mapping qid -> gold answer.")
    return {str(k): str(v) for k, v in raw.items()}


def _load_official(results_dir: Path, run_id: str) -> dict[str, float] | None:
    """qid -> official-evaluator score, if the evaluator has been run.

    Optional because the official evaluator is a model-based SERVICE and this
    command is offline by contract. Its absence is reported by `score_store` rather
    than being silently tolerated.
    """
    p = results_dir / run_id / "official.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in raw.items()}


def _load_human_agreement(path: Path) -> dict[str, Any]:
    """Judge-vs-human agreement from a filled labeller file plus the judge side."""
    from msat.judges.human_sample import judge_vs_human, read_human_csv

    labelled = read_human_csv(Path(path))
    judge_side_path = Path(path).with_name("judge_side.csv")
    if not judge_side_path.exists():
        _fail(
            f"{judge_side_path} is missing. The labeller file and the judge's view are "
            f"written separately on purpose - showing a labeller the judge's verdicts "
            f"makes them agree with it more often, and the resulting agreement figure "
            f"measures the anchor rather than the judge."
        )
    judge_side = read_human_csv(judge_side_path)
    out = dict(judge_vs_human(labelled, judge_side))
    out["status"] = "RUN"
    return out


# ── P9/P10 · analysis, figures, paper ──────────────────────────────────────────
@app.command("analyze")
def analyze(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Assert no network access (PRD §N3).")
    ] = True,
    check: Annotated[
        bool, typer.Option("--check", help="Recompute every metric from cache (PRD §G1).")
    ] = False,
) -> None:
    """Fit the primary model and write `results.md`. Interaction term first.

    The single declared specification (PRD §R9.1):
    `correct ~ log2(K)*log2(B) + log2(K) + log2(B) + corpus_fraction + gold_rank
    + ability + reader_strength + (1|question) + (1|model)`

    PRD §R9.
    """
    resolved = _resolve_run_id(run_id, run)
    _assert_offline("analyze")
    log.debug("analyze requested", extra={"run_id": resolved, "check": check})
    _pending("analyze", "R9.1-R9.12")


@app.command("figures")
def figures(
    out_dir: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("figures"),
    scores: Annotated[
        Path,
        typer.Option("--scores", help="scores.parquet from `msat score`."),
    ] = Path("results/scores.parquet"),
    ceiling: Annotated[
        Path, typer.Option("--ceiling", help="ceiling.json from `msat ceiling`.")
    ] = Path("results/ceiling.json"),
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            help=(
                "Which score column the accuracy panel plots. Required, not "
                "defaulted: `em` and `official` differ (R8.1) and the axis says "
                "'accuracy', so the number behind it must be named."
            ),
        ),
    ] = "official",
    scale: Annotated[
        list[str] | None,
        typer.Option("--scale", help="Restrict to one or more scales. Default: all."),
    ] = None,
    preview: Annotated[
        bool,
        typer.Option(
            "--preview",
            help=(
                "Render the synthetic fixture, stamped 'NOT RESULTS' in the "
                "figure itself. For checking the layout, never for a claim."
            ),
        ),
    ] = False,
) -> None:
    """Regenerate every figure from the artifacts a run wrote.

    Diagrams are code, never hand-drawn SVG (PRD §R10.2). `fig_centrepiece` is
    emitted as **SVG built from arithmetic** in `msat.figures.svg` rather than
    through a plotting library — an earlier draft of this docstring claimed
    "Plotly -> kaleido", which was wrong and stayed wrong until P7, which is the
    drift this project's own bug register keeps turning up.

    Emits SVG, not PDF. No converter is available in this environment, and SVG is
    the source format rather than a degraded copy of it.

    With no run behind it this command **refuses**. `--preview` is the only path
    that renders synthetic values, and it stamps them in the figure.
    """
    _assert_offline("figures")
    from msat.figures.centrepiece import (
        FigureInputMissing,
        build_from_disk,
        preview_report,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if preview:
            report = preview_report(out_dir)
        else:
            report = build_from_disk(
                scores_path=scores,
                ceiling_path=ceiling,
                metric=metric,
                out_dir=out_dir,
                scales=scale,
            )
    except FigureInputMissing as exc:
        _fail(str(exc), EXIT_CONFIG_INVALID)

    typer.echo(json.dumps(report, indent=2, default=str))
    if report.get("is_preview"):
        typer.secho(
            f"warning: {report['warning']} — do not cite these files.",
            fg=typer.colors.YELLOW,
            err=True,
        )


@app.command("paper")
def paper(
    out_dir: Annotated[Path, typer.Option("--out")] = Path("../paper/build"),
) -> None:
    """Build the PDF from `results.md` alone, refusing occupied framings.

    PRD §R10.3, §R10.5, §R10.6.
    """
    _assert_offline("paper")
    _pending("paper", "R10.3, R10.5, R10.6")


@app.command("verify")
def verify(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
) -> None:
    """Recompute every metric from cached responses with the network disabled.

    PRD §G1, §N3.
    """
    _resolve_run_id(run_id, run)
    _assert_offline("verify")
    _pending("verify", "G1, R6.3")


@app.command("max-context")
def max_context(
    config: Annotated[Path, typer.Option("--config", help="Config directory.")] = Path("configs"),
    out: Annotated[Path | None, typer.Option("--out", help="Write the plan as JSON.")] = None,
) -> None:
    """How much of each conversation each reader can actually be shown.

    The VRAM-capped ceiling (`full_context_max`) is the largest suffix of the
    conversation that fits. It is **not** `full_context` and is reported under its
    own name and its own `b_mode`, because a cut corpus and a whole one are
    different conditions.

    Exists so the cap is visible BEFORE a run rather than discovered in a results
    table, and so the decision about renting larger hardware is made on numbers
    instead of on a hunch.
    """
    from msat.arms.max_context import max_context_plan
    from msat.config import load_bundle
    from msat.power import _reader_limit

    bundle = load_bundle(config)
    cpt = float(bundle.pricing.get("token_estimates", {}).get("chars_per_token", 4.0))
    reserve = int(bundle.decoding.max_tokens)

    by_scale: dict[str, dict[str, Any]] = {}
    for scale in bundle.design.scales:
        sc = bundle.scales.get(scale)
        limits = {f.id: _reader_limit(f).ceiling for f in bundle.readers.families}
        by_scale[scale] = max_context_plan(
            corpus_tokens=int(sc.chars_per_question / cpt),
            sessions_median=sc.sessions_median,
            limits=limits,
            reserve_tokens=reserve,
        )

    for scale_id, block in by_scale.items():
        typer.echo(
            f"\n  scale {scale_id}: {block['corpus_tokens']:,} tokens, "
            f"{block['sessions_median']:.0f} sessions "
            f"({block['mean_session_tokens']:,.0f} tok/session)"
        )
        typer.echo(
            f"    {'reader':<8}{'ceiling':>10}{'sessions':>10}{'kept tok':>11}"
            f"{'corpus':>9}   full_context"
        )
        for reader, r in block["per_reader"].items():
            full = "yes" if r["full_context"] else f"NO, short {r['shortfall']:,}"
            typer.echo(
                f"    {reader:<8}{r['ceiling']:>10,}{r['n_sessions_kept']:>10,}"
                f"{r['kept_tokens']:>11,}{r['corpus_fraction']:>8.1%}   {full}"
            )
        if block["arm_d_infeasible"]:
            typer.secho(
                "    Arm D (true full context) is INFEASIBLE on this hardware. "
                "`full_context_max` is the substitute.",
                fg=typer.colors.YELLOW,
            )
        typer.secho(
            f"    {block['note']}",
            fg=typer.colors.YELLOW,
        )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(by_scale, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"\n  wrote {out}")


@app.command("hardware-delta")
def hardware_delta(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    against: Annotated[
        Path | None,
        typer.Option("--against", help="A second run's store to compare with."),
    ] = None,
    results_dir: Annotated[
        Path, typer.Option("--results", help="Parent of the per-run directories.")
    ] = Path("results"),
    probe: Annotated[
        bool, typer.Option("--probe/--compare", help="Plan the probe, or diff two stores.")
    ] = True,
    n: Annotated[int, typer.Option("--n", help="Probe size in cells.")] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 20261012,
) -> None:
    """Bound the cross-hardware confound by measuring it.

    Response cache keys include the model but NOT the hardware, so a baseline run
    on a rented 80 GB card lands in the same store as everything else. At
    temperature 0 the two boxes should agree; this reports by how much they do
    not, judged against the MDE rather than an invented threshold.

    `--probe` plans the re-run (which cells, how many prompt tokens). `--compare`
    diffs two stores. Comparing a store to a copy of itself is the harness's own
    check: it must report exactly zero drift.
    """
    from msat.hardware.delta import compare_stores, confound_verdict, probe_plan

    resolved = _resolve_run_id(run_id, run)
    _assert_offline("hardware-delta")
    run_dir = results_dir / resolved
    if not run_dir.is_dir():
        _fail(f"no run directory at {run_dir}.")
    store = ResponseStore(run_dir / "responses")

    if probe and against is None:
        plan = probe_plan(store, n=n, seed=seed)
        typer.echo("\n  probe plan")
        typer.echo(f"    cells          {plan['n_keys']:,} ({plan['n_resolved']:,} resolved)")
        typer.echo(f"    prompt tokens  {plan['prompt_tokens']:,}")
        typer.echo(f"    seed           {plan['seed']}")
        typer.secho(f"    {plan['note']}", fg=typer.colors.YELLOW)
        typer.echo("\n  re-run those exact cache keys on the second box, then:")
        typer.echo(f"    msat hardware-delta {resolved} --against <run_id> --compare")
        return

    if against is None:
        _fail("--compare requires --against <run_id>")

    other_dir = results_dir / against
    if not other_dir.is_dir():
        _fail(f"no comparison run directory at {other_dir}.")
    other_store = ResponseStore(other_dir / "responses")

    keys = sorted(r.key for r in store.iter_all())
    gold = _gold_by_key(results_dir, resolved)
    report = compare_stores(
        store,
        other_store,
        keys,
        gold_by_key=gold,
        norm=_load_normaliser(),
        label_a=resolved,
        label_b=against.name,
    )
    mde = _interaction_mde(results_dir)
    verdict = confound_verdict(
        report,
        mde_log_odds=mde.get("mde_log_odds"),
        mde_pp_at_p50=mde.get("mde_pp_at_p50"),
    )
    typer.echo(json.dumps(verdict, indent=2, sort_keys=True))


def _load_normaliser() -> Normaliser:
    """The R1.2 normalisation, from the SAME file the run used.

    Read from `configs/prompts/normalisation.yaml` rather than a copy, because the
    harness scores the run's responses and must score them by the run's own
    definition. A delta measured under a different normaliser is not a hardware
    delta.
    """
    import yaml

    from msat.metrics.normalisation import Normaliser

    spec = yaml.safe_load(Path("configs/prompts/normalisation.yaml").read_text(encoding="utf-8"))
    return Normaliser.from_config(spec)


def _interaction_mde(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "power.json"
    if not path.exists():
        return {}
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    interaction = raw.get("interaction", {})
    return interaction if isinstance(interaction, dict) else {}


def _gold_by_key(results_dir: Path, run_id: str) -> dict[str, str]:
    """Cache key -> gold answer, read from the run's own store.

    Derived from stored responses rather than the dataset so the harness needs no
    dataset access and cannot score against a different gold set than the run did.
    """
    path = results_dir / run_id / "gold.json"
    if not path.exists():
        _fail(
            f"{path} is missing. The delta is scored, not eyeballed: without a "
            f"key->gold map the harness can only count text differences, which says "
            f"nothing about whether the RESULTS moved."
        )
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


@app.command("vram")
def vram(
    model: Annotated[Path, typer.Argument(help="Local checkpoint directory (no network access).")],
    tokens: Annotated[int, typer.Option("--tokens", help="Context length to plan for.")] = 4096,
    quant: Annotated[
        str | None, typer.Option("--quant", help="Weight bits: 4, 8, 16, 32. Default: dtype.")
    ] = None,
    card: Annotated[
        float | None, typer.Option("--card", help="Also report the ceiling for this many GB.")
    ] = None,
) -> None:
    """How much VRAM does this checkpoint actually need? Read from `config.json`.

    Exists because every VRAM figure in this project was, until now, a NUMBER
    SOMEONE TYPED. The ceilings in `readers.yaml` were estimated by hand and only
    then checked against the corpus - which is how Arm D's infeasibility was found,
    but only after the fact and only for models already chosen.

        KV bytes/token = 2 x n_layers x n_kv_heads x head_dim x dtype_bytes

    The `num_key_value_heads` term is the GQA factor and it dominates the answer: a
    model with 8 KV heads uses 8x less KV cache than its parameter count suggests.
    For a mixture-of-experts model the KV cost depends only on the ATTENTION layers,
    not on the 30B parameters - which is why a 30B-A3B judge may fit comfortably
    where a 30B dense model would not.

    `weights + KV` is the FLOOR. Activation memory, the CUDA context and allocator
    fragmentation are in the headroom and are listed as omitted.
    """
    from msat.hardware.vram import (
        ConfigMissingError,
        kv_ceiling_tokens,
        load_config,
        plan_vram,
        shape_from_config,
    )

    try:
        shape = shape_from_config(load_config(Path(model)), name=Path(model).name)
    except ConfigMissingError as e:
        _fail(str(e))
        return

    quant_bits = float(quant) if quant else None
    plan = plan_vram(shape, n_tokens=tokens, quant_bits=quant_bits)

    typer.echo(f"\n  model        {shape.name}")
    typer.echo(
        f"  layers       {shape.n_layers}   kv_heads {shape.n_kv_heads}"
        f"   head_dim {shape.head_dim}   GQA factor {shape.gqa_factor}x"
    )
    typer.echo(f"  dtype        {shape.dtype}")
    if shape.is_moe:
        typer.echo(
            f"  MoE          {shape.n_experts} experts"
            f"{f', {shape.n_active_experts} active' if shape.n_active_experts else ''}"
            f"  - KV depends on attention layers only, NOT on the {shape.n_params / 1e9:.0f}B parameters"
        )
    typer.echo(f"  KV per token {shape.kv_bytes_per_token():,} bytes")
    typer.echo(
        f"  params       {shape.n_params / 1e9:.1f}B"
        + ("" if shape.n_params_is_exact else "   (ESTIMATED from config - not an exact count)")
    )
    typer.echo("")
    typer.echo(
        f"  at {tokens:,} tokens:  weights {plan.weights_gb:.1f} GB"
        f" + KV {plan.kv_gb:.1f} GB + headroom {plan.headroom_gb:.1f} GB"
        f" = {plan.total_gb:.1f} GB"
    )
    typer.echo(f"  smallest standard card: {plan.smallest_card()}")
    typer.secho(f"  omitted: {'; '.join(plan.omitted)}", fg=typer.colors.YELLOW)

    if card:
        ceiling = kv_ceiling_tokens(shape, card, quant_bits=quant_bits)
        typer.echo(
            f"\n  ceiling on a {card:g} GB card: {ceiling:,} prompt tokens"
            f"  (vs the hand-declared ceilings in readers.yaml)"
        )
    typer.echo()


# ── top-level callbacks ────────────────────────────────────────────────────────
def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"msat {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    """Root callback: configure structured logging once, before any command runs."""
    configure_logging()


def main(argv: list[str] | None = None) -> Any:
    """Console-script entry point (`msat = "msat:main"`)."""
    configure_logging()
    if argv is not None:
        return app(args=argv, standalone_mode=False)
    return app()


def registered_commands() -> tuple[str, ...]:
    """Every command name registered on the Typer app, sorted.

    Reads Typer's own registry rather than the built Click command, so this
    stays typed and does not reach into `click.Group` — an implementation
    detail Typer does not promise to expose.

    The single fact `tests/test_cli.py` asserts the surface against.
    """
    names = {info.name for info in app.registered_commands if info.name}
    names |= {info.name for info in app.registered_groups if info.name}
    return tuple(sorted(names))


__all__ = [
    "OFFLINE_COMMANDS",
    "PRD_SURFACE",
    "app",
    "main",
    "main_callback",
    "registered_commands",
]
