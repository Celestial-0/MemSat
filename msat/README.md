# msat — MemSat Experimental Harness

Measures how answer accuracy and evidence recall change as an LLM agent retrieves
**K** memories under context budget **B**, on a benchmark where the analytical
recall ceiling is known to bind at **K = 6**.

```text
A(K, B) = R(K; E, X, M) × U(gold | K, B, X, M, P, order)
```

`K` and `B` are swept as a **crossed factorial**. `E` (embedder), `X` (chunking),
`M` (similarity), `P` (prompt), and `order` (gold position) are pinned, named, and
hashed into every artifact — because an unnamed constant is a confound.

The primary result is the estimated `log2(K) × log2(B)` **interaction term**,
tested against a pre-registered sign prediction.

## Status

| Phase | What | State |
|---|---|---|
| **P0** | Toolchain, CLI surface, exit codes, structured logging | ✅ **done** |
| **P1** | `ceiling.json` + `corpus_fraction.json` + `data.lock` + `dataset_report.json` | ✅ **done** |
| **P2** | Config freeze + **`design-power`** + pre-registration | ✅ **done** |
| P3–P10 | memory · executor · arms · judges · analysis · paper | ⬜ |

**No scientific *experiment* has run yet.** The requirement contract — ~76 numbered
requirements across 11 groups — is in **[`PRD.md`](../Research/Final/PRD.md)**, and the
build order with per-phase deliverables and gates is in
**[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)**.

### The zero-cost result (P1)

Computed from benchmark metadata alone, **zero API calls**, and committed:

| | S | M |
|---|---|---|
| sessions per conversation | mean **47.7** (max 62) | mean **475.3** (max 490) |
| `ceiling(K)` at K=1 / 2 / 3 / 4 / 6 | **36.17 / 84.89 / 93.19 / 97.02 / 100%** | *identical* |
| `K*` | **6** | **6** |
| `degeneracy_fraction[64]` | **100%** — K=64 is full-context | 0% |

`ceiling(K)` is identical at two corpus scales, so it is a property of *evidence
structure*, not corpus size. And K=64 on S is not a retrieval condition at all —
the same grid coordinate is a different system on each scale, which is why
`corpus_fraction` is a covariate in every model from P9 onward.

```bash
make ceiling     # recompute all three artifacts, relock, then byte-compare
```

`make ceiling` ends in `--verify`, which regenerates and **byte-compares** against
the committed files and exits `5` on drift. The 2.74 GB M split streams at
**25 MB peak RSS**.

### The design, and its power (P2)

`msat design-power` answers whether the `log2(K) × log2(B)` interaction is
detectable **before** a single API call is spent.

```bash
uv run msat validate-config configs/ --hash
uv run msat design-power configs/design.yaml
```

| | |
|---|---|
| Grid | K ∈ {1,2,4,8,16,32,64} × 3 B × 2 scales × 2 readers = **84 cells** |
| Observations | 126,000, fully paired (every question at every cell) |
| **Interaction MDE** | **0.0050 log-odds** — 0.12 pp at accuracy 0.5 |
| **Cost estimate** | **$26,959** uncached (primary) · 126,000 calls |
| Pre-registered sign | **≈ 0** — utilisation depends on budget only once length is fixed |
| Exit code | **10** — `budget.max_usd` unset (blocker **B7**), guard not enforceable |

**Measurement, not sample size, is the binding constraint.** The MDE degrades
**2.1×** (0.0053 → 0.0110) as grader misclassification rises from 0% to 12% — the
territory implied by κ = 0.118 and by 2609.38021 flipping 8 of 500 verdicts between
two passes of the *same* system under the *same* judge. More cells would not help;
a better grader would.

⚠️ The cost is an **estimate** — `configs/pricing/latest.yaml` is marked
`_verified_against_invoice: false`. Both it and `budget.max_usd` need the guide.

### Where each phase runs

| Phase | Host | Why |
|---|---|---|
| P1, P2 | anywhere | zero API, zero GPU |
| P3 (embedder), P8 (local reader family B) | **execution env — RTX 4060 Laptop, sm_89, 8 GB VRAM, 32 GB RAM** | needs a working `import torch` |
| P7 (Arm B leakage test), P9 (analysis) | **execution env** | PRD §R9.10's classifier needs `scikit-learn` |
| P4, P5, P6, P10 | anywhere | no torch / sklearn import |

The authoring host has Windows Smart App Control enabled, which blocks
`torch_python.dll`, two `sklearn` extension modules, and `statsmodels`' Kalman
smoother (`WinError 4551`). That is a machine security policy, **not** a
dependency problem — the wheels resolved correctly and install cleanly. Run
`make app-control` to re-diagnose on any host.

One consequence to respect in code: `import statsmodels.api` fails on such a host.
Import the specific submodule (`statsmodels.genmod.generalized_estimating_equations`)
instead, which is what `design.py` and the P9 analysis will do.

## Install

**`uv` is the only supported package manager.** The machine default interpreter is
Python 3.10 (3.14 is also installed) and the project requires `>=3.12,<3.13`.

```bash
# one-time, if uv is not already on PATH
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

make doctor          # verify uv + a 3.12 interpreter, or print the fix
uv python install 3.12
uv sync              # or: make setup
uv run msat --help
```

Never `pip install`, never `python -m venv`, never a bare `python` or `pytest` on
the PATH. Every `make` target shells out to `uv run`.

```bash
uv add <pkg>         # add a dependency (writes pyproject + uv.lock)
uv lock --check      # the lockfile must be current
uv sync --frozen     # CI: install must not re-resolve
uv sync --extra glmm # optional PyMC/ArviZ for an exact nested GLMM
```

## CLI surface

All 12 commands from PRD §7 plus `data-lock`, `verify`, and `replay`:

```text
msat validate-config configs/       # non-zero on any missing field; ≥2 readers
msat design-power configs/design.yaml
msat ceiling longmemeval_m --by-type
msat ceiling --verify               # byte-compare vs committed
msat data-lock                      # SHA-256 + sizes + row counts per raw file
msat build-plan run.yaml            # REJECTS degenerate cells
msat run <run_id> --resume
msat status <run_id>
msat replay <run_id>                # metric-identical from cache
msat score <run_id> --metric em|f1|judge
msat judge-audit <run_id>
msat analyze <run_id>               # interaction term first
msat figures
msat paper
msat verify <run_id>
```

The surface is **frozen before any feature lands**: `tests/test_cli.py` asserts it
verbatim. An unbuilt command exits `9` (`EXIT_NOT_IMPLEMENTED`) and names its
owning phase — "not built yet" and "broken" must not look alike.

Each failure mode has a distinct exit code, because a silently-degenerate
condition must be impossible (PRD §G3):

| Code | Meaning | Code | Meaning |
|---|---|---|---|
| 0 | success | 5 | artifact drift (R2.4) |
| 1 | generic failure | 6 | interaction underpowered (R0.3) |
| 2 | config invalid (R1.5, R7.5) | 7 | GO/NO-GO failed all branches |
| 3 | degenerate (K,B) cell (R3.2) | 8 | scientific assertion failed |
| 4 | planned cost over budget (N8) | 9 | not implemented in this phase |
| | | 10 | Arm B label leakage detected (R9.10) |

## Conventions

- **`uv` only.** `uv.lock` is committed and never hand-edited.
- **`config_hash`** — SHA-256 of the resolved config bundle, stamped on every
  artifact row, manifest, and response file (PRD §G2).
- **Condition key** — `(run_id, arm, K, B, B_mode, scale, reader, seed, question_id)`.
- **Offline by default** (PRD §N3) — `verify`, `analyze`, `figures`, `paper`,
  `score`, `status`, and `replay` never touch the network. Enforced in tests by a
  socket-blocking fixture that is itself asserted to block.
- **Structured JSON logs** (PRD §N6) — one object per line carrying `run_id`,
  `config_hash`, and `(arm, K, B, scale, reader, question_id)`.
- **Idempotent** (PRD §N5) — every command is safe to re-run.

## Why the name

The package is `msat`; the *concept* remains an **experimental harness** — the
standard term in this literature (`lm-evaluation-harness` being the canonical
artifact). `harness` was the `uv init` placeholder and collided with that
project; `msat` matches the repo name already specified in
`Research/context/12_harness_architecture.md` and `99_final_report.md`.

## Documentation

| Document | Contents |
|---|---|
| [`PRD.md`](../Research/Final/PRD.md) | requirement contract, module layout, CLI surface, acceptance criteria |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | 11 build phases, deliverables, tests, exit gates, requirement traceability |
| [`CHANGELOG.md`](CHANGELOG.md) | what changed, and which audit findings are still open |
| [`../Research/Final/Architecture.md`](../Research/Final/Architecture.md) | build phases P0–P10, critical path, risk register |
| [`../Research/Final/Final_Idea.md`](../Research/Final/Final_Idea.md) | the contribution and the six-tuple identity |
| [`../Research/context/09_gaps.md`](../Research/context/09_gaps.md) | gaps G0–G14, including what NOT to claim |
