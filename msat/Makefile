# msat — MemSat experimental harness
# Make targets map to PRD.md requirement IDs. Every target that touches the
# network says so; `verify`, `analyze`, `figures`, `paper` must never (PRD §N3).

PY      := uv run
PYTHON  := uv run python

.DEFAULT_GOAL := help
.PHONY: help setup ceiling verify figures paper test lint typecheck \
        design-power replay status clean distclean chrome env

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────────
setup: ## Create the venv and install all deps (torch cu132 is ~2.5 GB)
	uv sync
	@echo ""
	@echo "Next: verify the GPU is visible ->  make env"

chrome: ## One-time: fetch the Chrome build kaleido 1.x needs for static export
	$(PYTHON) -c "import kaleido; kaleido.get_chrome_sync()"
	@echo "Chrome ready. Required by `make figures` (PRD §R10.2)."

env: ## Print resolved versions + GPU/CUDA visibility  [R1][R7.6]
	@$(PYTHON) scripts/env_report.py

# ── P1 · Provenance (zero API cost) ──────────────────────────────────────────
ceiling: ## Recompute ceiling.json + corpus_fraction.json + data.lock  [R2]
	$(PY) ceiling longmemeval_m --by-type --out results/
	$(PY) ceiling longmemeval_s --by-type --out results/
	$(PY) ceiling --verify

design-power: ## MDE for the K x B interaction, BEFORE any API call  [R0.2]
	$(PY) design-power configs/design.yaml

# ── Experiment ───────────────────────────────────────────────────────────────
run: ## Execute the crossed grid (RESUMABLE).  make run RUN=<run_id>
	$(PY) run $(RUN) --resume

status: ## Per-cell completion ledger  [R5.4]
	$(PY) status $(RUN)

# ── Analysis (offline: reads Store only) ─────────────────────────────────────
verify: ## Recompute every metric from cached responses, no network  [G1][N3]
	$(PY) analyze --run $(RUN) --offline --check

analyze: ## -> results.md. Interaction term first.  [R9.1]
	$(PY) analyze --run $(RUN)

figures: ## Regenerate every figure (Plotly -> kaleido -> PDF/SVG)  [R10.2]
	$(PY) figures --out figures/

paper: ## Build the PDF from results.md alone  [R10.3]
	$(PY) paper --out ../paper/build

replay: ## Metric-identical reproduction from the content-addressed store  [R6.3]
	$(PY) replay $(RUN)

# ── Quality ──────────────────────────────────────────────────────────────────
test: ## Fast tests only (excludes @pytest.mark.slow / .network)
	$(PY) pytest -m "not slow and not network"

test-all: ## Everything, including the full grid and live API  [§5]
	$(PY) pytest

lint: ## ruff check + format check
	$(PY) ruff check .
	$(PY) ruff format --check .

typecheck: ## mypy, strict on our own modules
	$(PY) mypy src/msat

# ── Hygiene ──────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts
	rm -rf figures/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: ## clean + remove the venv (requires a full `make setup` after)
	$(MAKE) clean
	rm -rf .venv/
