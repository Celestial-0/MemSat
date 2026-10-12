# msat — MemSat Experimental Harness

Measures how answer accuracy and evidence recall change as an LLM agent retrieves
**K** memories under context budget **B**, on a benchmark where the analytical
recall ceiling is known to bind at **K = 6**.

```
A(K, B) = R(K; E, X, M) × U(gold | K, B, X, M, P, order)
```

`K` and `B` are swept as a **crossed factorial**. `E` (embedder), `X` (chunking),
`M` (similarity), `P` (prompt), and `order` (gold position) are pinned, named, and
hashed into every artifact — because an unnamed constant is a confound.

The primary result is the estimated `log2(K) × log2(B)` **interaction term**,
tested against a pre-registered sign prediction.

## Status

**Not yet implemented.** This is a scaffold. The full requirement contract —
~60 numbered requirements across 11 groups — is in **[`PRD.md`](PRD.md)**, and the
build direction is in [`../Research/Architecture.md`](../Research/Architecture.md).

Start with the zero-API-cost phases:

| Phase | What | Cost |
|---|---|---|
| P1 | `ceiling.json` + `corpus_fraction.json` + `data.lock` | zero API |
| P2 | `retrieval.yaml`, `prompts/`, `design.yaml`, **`design-power`** | zero API |

`design-power` is the highest-leverage command in the project: it tells you
whether the interaction is estimable at the intended grid **before** you spend
anything. See PRD §R0.

## Install

```bash
uv sync
uv run msat
```

Python ≥3.12. Dependencies are pinned with constraints per PRD §N1.

## Why the name

The package is `msat`; the *concept* remains an **experimental harness** — the
standard term in this literature (`lm-evaluation-harness` being the canonical
artifact). `harness` was the `uv init` placeholder and collided with that
project; `msat` matches the repo name already specified in
`Research/context/12_harness_architecture.md` and `99_final_report.md`.

## Documentation

| Document | Contents |
|---|---|
| [`PRD.md`](PRD.md) | requirement contract, module layout, CLI surface, acceptance criteria |
| [`../Research/Final_Idea.md`](../Research/Final_Idea.md) | the contribution and the six-tuple identity |
| [`../Research/Architecture.md`](../Research/Architecture.md) | build phases P0–P10, critical path, risk register |
| [`../Research/context/09_gaps.md`](../Research/context/09_gaps.md) | gaps G0–G14, including what NOT to claim |
