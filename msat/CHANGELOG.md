# Changelog

All notable changes to `msat`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The requirement contract lives in `../Research/Final/PRD.md`; the build order
lives in `IMPLEMENTATION_PLAN.md`.

## [Unreleased]

### P2c — Serving backend abstraction + dependency cleanup

**Answers "what's a better way to run 189,000 local generations?", and corrects
two things I previously got wrong.**

#### ⚠️ Correction: B8 and B9 were never real blockages

Earlier today I reported that Smart App Control blocks `torch` and `scikit-learn`,
and concluded P3/P7/P8/P9 had to run in a separate environment. **That diagnosis
was wrong.** Re-checked 2026-10-12:

- **All 38 of 38 torch DLLs now load.** `import torch` succeeds.
- `sklearn.metrics.roc_auc_score`, `sklearn.linear_model`, `sklearn.ensemble` and
  `statsmodels.api` all import fine.
- Smart App Control is **still On** — the policy never changed.

So the block was **cloud file reputation**, not policy. Files that had just been
downloaded scored badly; the score improved as they were fetched and seen more
widely, and the block lifted on its own. The lesson is recorded in
`scripts/check_app_control.py`: `WinError 4551` means "this file's reputation is
currently bad", and re-running the diagnostic an hour later is cheaper than
changing a machine's security settings.

**Every phase now runs anywhere with a working CUDA device.** The reason for a
separate box is hardware only — this host has no GPU.

#### Backend choice

| | verdict |
|---|---|
| **Ollama** | no — wrong tool for 189k short generations (no continuous batching), and a moving server that decides its own `num_ctx` and sampling defaults |
| **LangChain / LangGraph** | no — PRD §3 non-goals say *"no adaptive or agentic retrieval; K and B are set by the experiment"*. A graph framework is the wrong abstraction for a design whose defining property is that the model never controls the flow. It also breaks `R4.1` purity and fights the WAL/content-addressed-store requirements. |
| **vLLM** | **the right answer to the throughput problem — as a SERVICE, not a dependency** |

vLLM cannot be a dependency here, for three independent reasons:

1. PyPI publishes **no Windows wheel**.
2. vLLM 0.31 requires `transformers>=5.10.4,<5.18`; this project pins `>=5.18`.
3. vLLM 0.31 pins **`torch==2.13.0` exactly**; this project requires `torch>=2.14.1`
   from the explicit `pytorch-cu132` index. An exact pin cannot be relaxed.

Declaring it as an extra made `uv lock` fail **for every install, including the
base one** — so it is reached over an OpenAI-compatible HTTP surface instead.
That turned out to be the better architecture anyway: the executor can then
resume independently of the engine, which is what PRD §R5.2 (`kill -9` → identical
results) wants regardless.

#### Added

- **`providers/`** — one interface, three backends, selected by config.
  - `base.py` — `GenerationRequest` / `GenerationResult` / `LocalBackend`, plus
    `canonical_body()`, the exact bytes that get hashed and sent. Adding a field to
    `GenerationRequest` must be a deliberate act that changes the cache key.
  - `transformers_backend.py` — always available, the reproducible baseline.
  - `vllm_backend.py` — HTTP client for a vLLM server, with the two Windows/torch
    conflicts stated together so a half-fixed setup doesn't waste an afternoon.
  - `ort_backend.py` — onnxruntime-genai, the Windows-capable option.
  - `health()` returns an actionable string instead of an `ImportError`, because
    "No module named vllm" is not a diagnosis.
- **`scripts/bench_backend.py`** — measures prefill/decode throughput on **real
  corpus prompt sizes** (including the ~120k-token M/K64 cell that dominates the
  schedule) and re-projects the 314 h estimate from the measurement. This is what
  should decide the backend, rather than a number quoted in a README.
- **`analysis/leakage.py`** — PRD §R9.10's Arm B leakage assertion in numpy,
  replacing the `scikit-learn` dependency that existed solely for it. AUROC is the
  Mann-Whitney U statistic with ties at 0.5. The detector reports a **shuffled-label
  null** beside every result, because a leakage detector that reads 0.9 on random
  labels is not detecting leakage and there is no other way for a reader to tell.
- `make vllm` — starts a vLLM server in its own environment.
- Extras: `ort`, `bits` (4-bit for the `transformers` backend).

#### Removed

- **`scikit-learn` as a direct dependency.** It was declared for R9.10 alone, was
  never imported by any module, and R9.10 is now numpy. ⚠️ It is *still installed*
  transitively via `sentence-transformers`, which requires it — so this removes the
  dependency declaration and our use of it, not the package from the venv.
- The in-process vLLM engine mode, which could never run in this project anyway
  (see the torch pin conflict above).

#### Fixed

- **A real bug in the ORT backend**, caught by ruff: the decode loop referenced an
  undefined `new_token` instead of the `token_id` it had just read. It would have
  raised `NameError` on the first real generation.
- `backend_availability()` had a dead first loop that set every backend available
  and could never fail. It now probes each backend's own `health()`, which is the
  difference between "the module imported" and "the engine can start".

### P2b — All-local reader panel (team decision, 2026-10-12)

**The paid API plan is removed.** Three local ~9B families replace the two paid
readers: **Gemma, Qwen 3, Phi-4 mini**. Every reader *and every judge* is local.
Dollar cost of the main grid: **$0**.

That does not make the budget irrelevant — it moves it. Two things changed as a
consequence, and both are new blockers.

#### Measured consequences

| | before (2 paid readers) | now (3 local) |
|---|---|---|
| Cells | 84 | **126** |
| Calls | 126,000 | **189,000** |
| Dollar cost | $26,959 (unverified prices) | **$0** |
| **Wall-clock, one RTX 4060** | not modelled | **314.5 h ~ 13.1 days** |
| Reader time | — | 139.5 h (46.5 h x 3 families) |
| **Judge panel + R8.6 audit** | — | **175 h — 56% of the total** |

#### B10 (NEW, BLOCKING) — 15 cells are infeasible, and it is a *scientific* constraint

The M-scale K=64 cell is a **~120k-token prompt** (measured in P1: 4.84M
characters per M question). Reader context windows:

| reader | window | largest cell prompt | infeasible cells |
|---|---|---|---|
| gemma | 128,000 | 120,000 | **0** |
| qwen | 32,768 | 120,000 | **6** — M at K=32, 64 |
| phi | 16,384 | 120,000 | **9** — M at K=16, 32, 64 |

PRD §R3.2 forbids truncating the prompt to fit, so these cells are **invalid**,
not slow. They must be dropped or the M grid trimmed. This is now computed at
design time by `context_feasibility` rather than discovered during P5, which is
the entire reason P2 exists.

Options: enable Qwen's YaRN context extension (recovers 6 cells); drop M for the
short-window readers (kills G10); cap M's K grid at 8 for them; or run M on gemma
alone (loses R7.5 at those cells). **Needs a decision before P5.**

#### B11 (NEW, BLOCKING) — the compute budget is real and its assumptions are unverified

314.5 h rests on **assumed** throughput (8,000 prefill tok/s, 300 aggregate decode
tok/s at 4-bit, batch 8). Nothing has been benchmarked. If real throughput is 3x
worse this is 39 days on one GPU, which does not fit an 8-week schedule with a
paper still to write.

Two things make it worse before it makes it better:

- **Judges are 56% of the compute.** PRD §R8.2 wants >= 2 judges on *all* outputs.
  With a paid API that was a rounding error; with local models it is 175 h.
- **4-bit requires `bitsandbytes`, which is not in `pyproject.toml`.**

Recommended: benchmark 100 real M-scale calls before P5 and re-run
`design-power`. The estimate is only as good as that number.

#### B6 — resolved

Reader models are chosen. Three families *exceed* R7.5's "two families" and, at
similar size (0.72-0.80 `reader_strength`), give R7.6's covariate real variance —
which is what R7.5 actually asks for. A flat panel would test seed noise, not
reader-strength dependence, so `validate-config` now warns on one.

#### B12 (NEW, BLOCKING) — exact checkpoints are not pinned

`configs/readers.yaml` carries `UNCONFIRMED` for every `version` and `released`.
**`msat validate-config` exits 2 until these are filled in** — deliberately an
ERROR, not a warning. A warning is a thing you ship straight past, and an unnamed
model makes every number in the paper unattributable, which is the exact failure
this config layer exists to prevent.

To resolve, on the execution machine:

```
python -c "from huggingface_hub import model_info as m; i=m('<repo>'); print(i.id, i.sha, i.createdAt)"
```

Tags like `main` are not enough — tags get repointed, which silently changes the
weights under a fixed `config_hash`.

#### Removed

- All `anthropic` / `openai` readers and every paid price from
  `configs/pricing/latest.yaml` (a test asserts nothing non-local survives).
- Every API key from `.env.example`.
- The `budget.max_usd` framing — the conversation is now **wall-clock**, enforced
  as `max_wall_clock_hours` (PRD §N4 still applies; only the currency changed).

#### Added

- `estimate_compute()` — wall-clock GPU hours, per reader and for the judge panel,
  with assumptions surfaced for challenge.
- `context_feasibility()` — which cells exceed which reader's window, under the
  `reject_cell` policy from PRD §R3.2.
- `ContextPolicy` and `Quantisation` config models; `exact_checkpoint_confirmed`
  as a hard-failing validator.
- Local judge panel (`judges_local`) spanning two families, deliberately not
  including the primary reader as its own judge.
- Tests: the shipped bundle is now *asserted to fail* on unconfirmed checkpoints,
  and a function-scoped `complete_configs` fixture lets every other validator test
  run against a loadable bundle. Suite: **168 passing**.

#### Fixed

A test corrupting the shared `complete_configs` fixture poisoned every test that
ran after it. Session-scoped fixtures that tests deliberately mutate are a trap —
the symptom is a baffling pile of unrelated failures. It is function-scoped now,
with a comment saying why.

### P2 — Config freeze + design-power

**The single highest-leverage phase, and it cost $0.** `msat design-power` answers,
before a single API call, whether the headline interaction is estimable.

#### The answer

| | |
|---|---|
| Grid | K ∈ {1,2,4,8,16,32,64} × 3 B × 2 scales × 2 readers = **84 cells, 126,000 paired observations** |
| **Interaction MDE** | **0.0050 log-odds** — 0.12 pp at accuracy 0.5, 0.10 pp at 0.3 |
| Interaction SE | 0.0018 (matches `(z_a/2 + z_b) * SE = 2.80 * 0.0018`) |
| **Cost estimate** | **$26,959** uncached (primary, R9.8) · $2,186 cache-adjusted · 126,000 calls |
| Verdict | **Powered.** No reduced design needed. |
| Preregistered sign | `approximately_zero`, hash `2d8af1a6e245e17b...` |
| Exit code | **10** — `EXIT_BUDGET_UNSET`. Correct: blocker **B7** is live. |

**The binding constraint is measurement, not sample size.** The MDE degrades from
0.0053 to 0.0110 log-odds (**2.1x**) as grader misclassification rises from 0% to
12% — the territory implied by kappa = 0.118 (2511.02246) and by 2609.38021
flipping 8 of 500 verdicts between two passes of the *same* deterministic system
under the *same* judge. Buying more cells would not help; buying a better grader
would. That is why the judge-reliability audit (R8.7) is a contribution in its own
right rather than a footnote.

#### Added

- **Config bundle** (`src/msat/config.py`) — strict pydantic models,
  `extra="forbid"` throughout, plus a structural null-scan of `retrieval.yaml`
  (PRD §R1.1: no field may be null). `validate-config` reports **every** problem in
  one pass and separates `error` (non-zero exit) from `warn` (loud, non-fatal).
- **`configs/`** — `design.yaml` (the single grid, R0.1), `retrieval.yaml` (E, X, M,
  unit, tie-break, tokenizer), `decoding.yaml`, `readers.yaml` (two families),
  `budget.yaml`, `arms.yaml` (A–G plus `crossed_grid`), `budget_guard.yaml`,
  `pricing/latest.yaml`, and `prompts/` (7 templates + the abstention rule and the
  answer-normalisation function, both declarative and in config per R1.2).
- **`configs/preregistration.yaml`** — the pre-registered sign prediction, its
  three readings, why ≈0 was chosen, what is explicitly **not** claimed, and the
  **enumerated** secondary family that Holm-Bonferroni will correct over (R9.7).
  It gets its own hash, separate from the bundle, so "registered in advance" stays
  verifiable after later edits (R0.4).
- **`src/msat/design.py`** — the crossed grid, a cluster-robust estimator, the
  Monte-Carlo power engine, and the cost model.
- **`src/msat/power.py`** — orchestration; emits `results/power.json` (35 KB).
- **`msat validate-config` and `msat design-power`** are live commands.
- **Tests** — `test_config.py`, `test_design_power.py`. Suite: **160 passing**.

#### The estimator is cross-checked, not assumed

`design.py` hand-rolls the sandwich estimator because a power sweep needs ~450 fits
at 126k rows and `statsmodels.GEE` cannot do that in reasonable time. A fast
implementation that has never been compared to the reference library is a
liability, not an optimisation, so `test_estimator_matches_statsmodels_gee`
asserts agreement with `statsmodels.genmod.GEE` — **worst SE ratio 5.7e-06**.

It is emphatically **not** `MixedLM`, which PRD §6.1 records as silently accepting
a 0/1 outcome with no error.

#### Four bugs the work surfaced — each would have produced a green wrong answer

1. **The bisection hit its tolerance floor, not the answer.** It searched a fixed
   `[0, 1.5]` bracket with an absolute tolerance of 0.02. The true MDE is ~0.005,
   so the search stopped at `1.5 / 128 = 0.0117` — and returned that *identically*
   for every noise level. The bracket is now derived from the measured SE and the
   tolerance is relative. Pinned by `test_mde_is_not_an_artefact_of_the_search_bracket`.
2. **Judge noise was modelled as within-cluster variance, which does nothing.**
   Cluster-robust SEs are designed to stay valid under arbitrary within-cluster
   correlation, so inflating it barely moves the SE — the sweep showed an MDE
   identical to four decimals at every noise level. Judge noise is
   **misclassification**: it attenuates the signal. Re-modelled, the sweep responds
   (2.1x degradation) and the verdict became data-driven instead of asserted.
3. **The cost was double-counted.** Labelling the crossed grid `("A", "G")`
   produced 168 cells where PRD §R4.4 says A and G are **one** grid — the same
   (K, B) cell is simultaneously "sweep K" and "sweep B". That silently doubled the
   number going into the budget conversation, for no scientific reason.
4. **Cost returned $0** because reader family ids (`family_a`) are not pricing keys
   (`claude-opus-5-5`). A lookup failure that yields $0 looks like a triumph.
   Unmapped readers are now counted and reported, so it cannot hide again.

Two further corrections the tests forced: `mde_for_stratum` was returning a
normal tail **probability** rather than an effect size (and was therefore
*rising* with n); and the design matrix needed **centred** predictors — uncentred
`log2(B)` spans 12..16, so a 0.25-per-log2-unit main effect put the baseline near
98% accuracy before any effect was planted.

#### Carried forward

- **B7 — `budget.max_usd` is unset.** `design-power` emits the estimate and exits
  **10**, because the hard cost guard cannot be enforced and no run may be
  launched. **$26,959 is the number for the budget conversation**, and it is an
  estimate: `configs/pricing/latest.yaml` is marked
  `_verified_against_invoice: false`.
- **B6 — reader models are placeholders** (`claude-opus-5-5`, `Qwen/Qwen3-8B`).
  Structurally complete and hashed, but `pending_guide_signoff: true`; reported as
  a warning so it is impossible to miss and impossible to ignore.
- **Pricing is unverified.** Every cost figure above inherits that caveat.
- `statsmodels.api` cannot be imported on the authoring host (Smart App Control
  blocks `_kalman_smoother`), so the analysis layer must import the specific
  `statsmodels.genmod.*` submodule. **`sklearn` is also blocked**, which matters:
  PRD §R9.10's leakage classifier needs it, so **P7 and P9 must run in the
  execution environment** alongside P3 and P8.

### P1 — Provenance: ceiling, corpus fraction, data.lock

**The paper's only zero-API result, shipped as committed, regenerable,
byte-verifiable artifacts.** Risk A1 (`answer_session_ids` unresolved) is closed
by measurement.

#### Measured against the real data

| Quantity | S | M |
|---|---|---|
| records | 500 (470 answerable / 30 abstention) | 500 (470 / 30) |
| sessions per question | min 38, max 62, **mean 47.7** | min 460, max 490, **mean 475.3** |
| `ceiling(K)` at K=1/2/3/4/6 | **36.17 / 84.89 / 93.19 / 97.02 / 100%** | *identical* |
| `K*` | **6** | **6** |
| `survivors_at[64]` | **0.00%** | 100.00% |
| `degeneracy_fraction[64]` | **100.00%** | 0.00% |

Both figures reproduce `Final_Idea.md` Appendix A exactly. Three consequences,
now measured rather than assumed:

1. **Scale invariance holds.** `ceiling(K)` is identical at ~47.7 and ~475.3
   sessions per conversation, and the gold-session distribution is identical
   (`{1:176, 2:250, 3:41, 4:19, 5:11, 6:3}`). It is a property of evidence
   structure, not corpus size.
2. **The K=64 defect is reproduced.** On S, K=64 is full-context for **100%** of
   questions — it is not a retrieval condition at all. On M the same K is a
   legitimate RAG point. Same grid coordinate, different system (gap G12).
3. **Streaming works.** The 2.74 GB M split is processed in 11.3 s at a
   **25 MB peak RSS** — 0.9% of file size — against a 1.5 GB ceiling.

#### Added

- **Streaming loader** (`src/msat/datasets/`). Two modes over one file:
  `iter_metadata()` is memory-flat (ijson event walk, session text discarded
  without being accumulated) and backs every ceiling statistic; `iter_questions()`
  materialises one full record at a time for P3's write path.
- **`RAM_CEILING_BYTES = 1.5 GiB`** (`datasets/longmemeval.py`) — a *number*, not
  "under a budget". PRD §N2 requires it and `tests/test_streaming.py` asserts
  against it; a ceiling without a number is an unfailable test.
- **Ceiling module** (`src/msat/ceiling.py`). `ceiling(K)`, `partial(K)`, `K*`,
  `survivors_at[K]`, `degeneracy_fraction[K]`, per-`question_type`
  stratification with the retriever-independent / retriever-limited
  classification, and the per-`(scale, K)` corpus-fraction distribution.
- **`data.lock`** (`src/msat/datalock.py`) — SHA-256 + byte size + row count per
  raw file, streamed in 1 MiB chunks. Contains **no timestamp**, because a lock
  that changes on every run proves nothing.
- **`msat ceiling` / `msat data-lock`** are live commands, not stubs.
- **`dataset_report.json`** — loader integrity receipt, kept separate from
  `ceiling.json` so a loader change cannot silently rewrite a committed number.
- **Tests** — `test_ceiling.py`, `test_streaming.py`, `test_schema.py`,
  `test_data_lock.py`. Suite is now **108 passing, 1 skipped** (the skip is the
  2.74 GB RAM test, marked `slow`; it passes under `make test-all` in 11.5 s).

#### Key design decisions

- **`CEILING_GRID` is a superset, not a rival, of the design grid.** PRD §R0.1
  makes `configs/design.yaml` the single source of the *design* grid. The ceiling
  publishes the bound at more K than the experiment sweeps, and
  `test_ceiling.py::test_ceiling_grid_is_a_superset_of_any_design_grid` asserts
  that relationship the moment `design.yaml` exists — which is what stops this
  becoming a second, competing grid.
- **The artifact is multi-scale and order-independent.** `msat ceiling <split>`
  always computes every available scale, so the Makefile sequence cannot leave
  `ceiling.json` holding whichever split ran last, re-running is idempotent
  (PRD §N5), and `--verify` is a clean full rebuild.
- **`survivors_at[K]`, not `degenerate_at[K]`** (PRD §R2.5). The legacy name
  describes the complement and invites exactly the misreading this project
  prevents. Every artifact carries a `_key_semantics` block so the meaning
  travels with the number.
- **No token counts in P1.** Tokenising needs the tokenizer pinned in
  `retrieval.yaml`, which lands in Phase 2. Exact character counts are emitted; an
  *estimated* token count presented as a measurement is precisely the quiet fudge
  this project exists to avoid.

#### Two dataset properties measured, and the invariants written to match

- **Gold is declared twice and they agree — on answerable items only.** Measured
  over S and oracle: **0 mismatches across all 470 answerable records**, and
  **21 of 30 abstention records** mismatch. That is correct, not damage: an
  abstention question asks about information the corpus does not contain, so
  `answer_session_ids` names the sessions that *would* hold the answer and no
  turn can carry `has_answer: true`. `validate_record` therefore gates the
  cross-check on `answerable` — enforcing it everywhere would reject 21 valid
  records; skipping it everywhere would hide a real lost-flag bug.
- **Abstention split is exactly 470/30**, asserted against the data. Every recall
  denominator in the paper depends on it.

#### Fixed

- Three stale `[V]` provenance tags in `Research/context/architecture/` are
  corrected with dated notes that preserve the original claim rather than
  deleting it: `dependencies = []` (×2), `harness/src/harness/` + "Makefile and
  tests do not exist", and "`data.lock` neither exists". Claims that remain true
  — `configs/` and `pricing/` do not exist, no Dockerfile, no CI — are untouched.
- `msat ceiling` crashed on `typer.echo(..., fg=...)`, which has no `fg`
  parameter. It now uses `typer.secho`.
- `iter_questions` had a `NameError` on `sids` in a default argument.

#### Carried forward

- **🔴 `5_filler_sess` is still missing** — `data.lock` records it as
  `present: false`. It is Arm B's only permitted padding source, so **P7 is
  blocked** until it is fetched. Flagged in week 1 for exactly this reason.
- **Token counting** requires the tokenizer pinned in `configs/retrieval.yaml`
  (P2). `tokens(corpus)` is needed by the **B-guard** (P3), so the tokenizer must
  be pinned before P3 starts.
- **Disk budget** (PRD §R6.4) is still unquantified. P1 now supplies its input:
  M-scale corpus averages **4.84 M characters** per question (~1.2 M tokens at
  ~4 chars/token). P4 enforces the budget.

### P0 — Toolchain & repository scaffold

Nothing scientific ships in this phase. It makes every later phase possible and
fixes three discrepancies found during the Phase 0 audit.

#### Status

| Exit-gate item | State |
|---|---|
| `uv sync --frozen` succeeds | ✅ 90 packages installed, 133 resolved |
| `uv run msat --help` lists the full surface | ✅ 15 commands |
| `ruff check` / `ruff format --check` / `mypy src/msat` | ✅ clean |
| `pytest` | ✅ 56 passed |
| `make env` shows a CUDA device | ⏸️ deferred — GPU phases run in a separate environment (below) |
| `make` itself runnable | ⚠️ not installed; chocolatey needs elevation. Every recipe is `uv run …`, so nothing is lost. |

#### Added

- **`uv` toolchain provisioning.** `uv` 0.13.0 + uv-managed CPython 3.12.15.
  Every `make` target and every invocation goes through `uv run`; `pip` and
  bare `python` are no longer viable paths. `make doctor` verifies both and
  prints the exact recovery command when `uv` is absent.
- **CLI surface frozen** (`src/msat/cli.py`). All 12 commands from PRD §7 plus
  `data-lock`, `verify`, and `replay` — the three the Makefile exposes but §7
  omits — are registered before any feature lands, so a missing command is a
  build failure rather than a week-4 discovery. `tests/test_cli.py` asserts the
  surface verbatim against `PRD_SURFACE`.
- **Phase-owned stubs** (`src/msat/constants.py`). An unbuilt command exits
  `EXIT_NOT_IMPLEMENTED` (9) and names its owning phase and the requirement IDs
  it serves. "Not built yet" and "broken" are different states and must not
  produce the same exit code.
- **Distinct exit codes per failure mode** (PRD §G3): config invalid (2),
  degenerate cell (3), budget exceeded (4), artifact drift (5), underpowered
  design (6), GO/NO-GO failure (7), assertion failure (8). A silently-degenerate
  condition must be impossible; a generic `1` would hide which one it was.
- **Structured JSON logging** (`src/msat/logging.py`, PRD §N6). One JSON object
  per line carrying `run_id`, `config_hash`, and the condition key
  `(arm, K, B, scale, reader, question_id)`. Paths and collections are coerced;
  an unserialisable field cannot truncate a log line mid-run.
- **Network-blocking test fixture** (`tests/conftest.py`). `block_network()`
  raises on socket construction. Asserted against itself in two separate tests —
  one proving it blocks, one proving teardown restores `socket.socket` — so the
  §N3 tests cannot go vacuous if the fixture breaks.
- **Application Control diagnostic** (`scripts/check_app_control.py`, `make
  app-control`). Loads every DLL in `torch/lib` individually and reports which
  are refused, separating a *machine policy* problem from a *dependency* problem.
- **Repository artifacts**: `LICENSE` (MIT, with an explicit note that data
  licences are separate), `CITATION.cff`, this changelog.

#### Changed

- **`pyproject.toml`**: version `1.0.0` → `0.1.0`, matching the fact that no
  phase has shipped a feature yet. Comment corrected from "81 packages" to 133
  to match `uv lock`'s actual resolution and PRD §6.1. Added the `selftest`
  marker, enabled ruff's `BLE` rule so the deliberate broad `except` sites can
  declare themselves with `# noqa: BLE001`, and widened
  `build-system.requires` to `<0.14.0` so the backend tracks the installed uv.
- **`scripts/env_report.py`**: no longer **crashes** when torch is unimportable.
  It previously caught only `ImportError`, but an Application Control refusal
  raises `OSError` — so the preflight died with a traceback at exactly the
  moment its output mattered most. It now distinguishes `MISSING` from
  `BLOCKED by Application Control`, and prints the remediation.
- **`Makefile`**: added `doctor`, `selftest`, `validate-config`, `design-power`,
  `build-plan`, `score`, `judge-audit`, `data-lock`, `replay`, `verify`,
  `app-control`. Fixed the §7-vs-Makefile signature mismatch — the positional
  form (`msat analyze <run_id>`) is now canonical and `--run` is an accepted
  alias, so both the PRD's spelling and the pre-existing targets work.

#### Fixed

- **Discrepancy D1/D2 — toolchain absent.** `uv` was not installed and the
  machine default interpreter is Python 3.10 (3.14 is also present), while the
  project requires `>=3.12,<3.13`.
- **Discrepancy D3 — CLI surface drift.** The Makefile called
  `msat analyze --run <id>`; PRD §7 specifies `msat analyze <id>`. Both now work;
  supplying both is an error rather than a silent precedence rule.
- **Discrepancy D9 — package count.** `pyproject.toml` claimed 81 packages;
  `uv lock --check` resolves **133**, matching PRD §6.1.
- **`env_report.py` crashed on `OSError`** rather than reporting it (above).

### 🔴 Blocker found in P0 — RESOLVED by deferral

**Windows Smart App Control blocks `torch` on the *authoring* machine, so no GPU work
is possible here. GPU-dependent phases will be executed in a separate environment.**

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\.venv\Lib\site-packages\torch\lib\shm.dll"
```

Measured with `make app-control`:

- **36 of 38** torch DLLs load fine — including *every* CUDA runtime
  (`cudart64_13.dll`, `c10_cuda.dll`, `torch_cuda.dll`, `cublas64_13.dll`,
  `cudnn64_9.dll`).
- Only **`shm.dll`** (DataLoader shared memory, Windows-only) and
  **`torch_python.dll`** (the Python extension module) are refused.
- `torch_python.dll` is required, so `import torch` fails outright.
- **Signature status is not the discriminator.** `c10.dll` is equally unsigned
  and loads fine; `cudart64_13.dll` is NVIDIA-signed and also loads. Smart App
  Control is applying **cloud file reputation per file**.
- **Mark-of-the-Web is absent** on all of them, so `Unblock-File` cannot help.
- Reproducible across retries; not transient.

**Resolution (team decision, 2026-10-12): the GPU phases execute in a separate
environment** — NVIDIA **RTX 4060 Laptop (sm_89, 8 GB VRAM)** with **32 GB system
RAM**. This machine is the authoring host.

| Phase | Runs where |
|---|---|
| P1, P2 | anywhere — zero API, zero GPU |
| P3 (embedder), P8 (local reader family B) | **execution environment** |
| P4–P7, P9, P10 | anywhere — none import torch |

The 8 GB VRAM cap in PRD §6.1 still binds for full-GPU residency; the 32 GB system
RAM makes CPU offload viable for P8's local reader if a larger model is wanted.
`msat[glmm]`'s PyMC path is CPU-only and unaffected. `scripts/check_app_control.py`
stays in the repo because the authoring host may be re-provisioned, and
re-diagnosing this costs one command.

### Audit findings carried forward

Recorded in `IMPLEMENTATION_PLAN.md` §2.3; these are *not* fixed by this phase.

- **D4** `5_filler_sess` is missing — Arm B has no legal padding source. Blocks
  P7; must be resolved in P1.
- **D5** PRD §R2.6 may be stale: `partial_credit_table` in
  `context/scripts/ceiling_analysis.py` appears to honour its `grid` argument
  already. P1 verifies rather than fixing a possible non-bug.
- **D6** `degenerate_at[K]` must be renamed `survivors_at[K]` (PRD §R2.5), and
  corpus-wide percentiles are forbidden as a guard basis (PRD §R3.1). P1.
- **D7** Benchmarks live in the `Research` git submodule, not `msat/data/`. P1
  defines one documented resolution strategy.
- **D8** Disk budget is unquantified (PRD §R6.4) while M-scale responses reach
  ~1.2M tokens. P1 estimates, P4 enforces.
