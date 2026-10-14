# msat — MemSat experimental harness
#
# Make targets map to PRD.md requirement IDs. Every target that touches the
# network says so; `verify`, `analyze`, `figures`, `paper`, `score`, `status`
# must never (PRD §N3).
#
# EVERY recipe goes through `uv run`. Never invoke `python` or `pytest` directly:
# the machine default interpreter is 3.10 and 3.14 is also installed, and the
# project requires >=3.12,<3.13 provisioned by uv.

PY      := uv run
PYTHON  := uv run python

.DEFAULT_GOAL := help
.PHONY: help setup doctor ceiling verify figures paper test lint typecheck \
        design-power replay status clean distclean chrome env app-control backends \
        bench vllm validate-config build-plan score judge-audit data-lock selftest test-all

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────
doctor: ## Verify the uv toolchain exists and the project Python is 3.12
	@command -v uv >/dev/null 2>&1 || { \
	  echo "uv NOT FOUND."; \
	  echo "  install : powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""; \
	  echo "  then    : uv python install 3.12 && uv sync"; \
	  exit 1; }
	@uv --version
	@uv python find 3.12 >/dev/null 2>&1 || uv python install 3.12
	@$(PYTHON) -c "import sys; \
	  assert (3,12) <= sys.version_info < (3,13), f'need >=3.12,<3.13, got {sys.version}'; \
	  print('python', '.'.join(map(str, sys.version_info[:3])), 'OK')"

setup: ## Create the venv and install all deps (torch cu132 is ~2.5 GB)
	$(PY) sync --frozen
	@echo ""
	@echo "Next: verify the GPU is visible ->  make env"

chrome: ## One-time: fetch the Chrome build kaleido 1.x needs for static export
	$(PYTHON) -c "import kaleido; kaleido.get_chrome_sync()"
	@echo "Chrome ready. Required by `make figures` (PRD §R10.2)."

env: ## Print resolved versions + GPU/CUDA visibility  [R1][R7.6]
	@$(PYTHON) scripts/env_report.py

app-control: ## List the DLLs Windows Application Control refuses to load  [P0]
	@$(PYTHON) scripts/check_app_control.py || true

backends: ## Report which local inference backends can run here, and why not
	@$(PYTHON) -c "import json; from msat.providers import backend_availability; \
	[print(f'  {k:<20} ready={v.get(\"ready\")}  {v.get(\"reason\",\"\")}') \
	 for k,v in backend_availability().items()]"

bench: ## Measure local inference throughput on REAL prompt sizes  [B11]
	$(PYTHON) scripts/bench_backend.py --backends $(or $(BACKENDS),transformers) \
	  --prompts $(or $(N),20) --out results/backend_bench.json

# vLLM runs in its OWN environment. It cannot be an extra here: no Windows wheel,
# it needs transformers<5.18 (we pin >=5.18), and it pins torch==2.13.0 exactly
# (we require >=2.14.1 from the cu132 index). An extra would break `uv lock` for
# every install including the base one. So: a separate venv, reached over HTTP.
vllm: ## Start a vLLM server in its own environment (Linux / WSL2 only)
	@echo "Creating an isolated vLLM environment in .venv-vllm ..."
	uv venv .venv-vllm --python 3.12
	uv pip install --python .venv-vllm "vllm>=0.31,<0.32"
	@echo ""
	@echo "Start it with:"
	@echo "  .venv-vllm/Scripts/python -m vllm.entrypoints.openai.api_server \\"
	@echo "      --model <repo> --served-model-name <repo> --port 8000"
	@echo ""
	@echo "Then set serving.backend: vllm in configs/readers.yaml and point"
	@echo "MSAT_VLLM_URL at it (default http://127.0.0.1:8000/v1)."
	@echo "Benchmark first:  make bench BACKENDS=vllm"

# ── P2 · Config freeze (zero API cost) ─────────────────────────────────────────
validate-config: ## Fail non-zero on any missing field; require >=2 readers  [R1.5,R7.5]
	$(PY) validate-config configs/

design-power: ## MDE for the K x B INTERACTION, BEFORE any API call  [R0.2]
	$(PY) design-power configs/design.yaml

# ── P1 · Provenance (zero API cost) ────────────────────────────────────────────
data-lock: ## Write SHA-256 + sizes + row counts for every raw file  [R2.3]
	$(PY) data-lock --out results/

# `msat ceiling` is multi-scale and order-independent (PRD §N5), so one call
# covers both splits — calling it twice would only re-stream 2.7 GB for nothing.
ceiling: ## Recompute + verify ceiling.json, corpus_fraction.json, dataset_report.json  [R2]
	$(PY) ceiling --out results/
	$(PY) data-lock --out results/
	$(PY) ceiling --verify --out results/

# ── P3/P4 · Plan and run ───────────────────────────────────────────────────────
build-plan: ## Enumerate conditions; REJECT degenerate (K,B) cells  [R3.2]
	$(PY) build-plan $(SPEC)

run: ## Execute the crossed grid (RESUMABLE).  make run RUN=<run_id>
	$(PY) run $(RUN) --resume

status: ## Per-cell completion ledger  [R5.4]
	$(PY) status $(RUN)

# ── P6/P9/P10 · Metrics, analysis, paper (all offline) ─────────────────────────
score: ## Re-score from cache; never calls a model  [G4]. make score RUN=<id> METRIC=em
	$(PY) score $(RUN) --metric $(or $(METRIC),all)

judge-audit: ## Two-pass same-judge replication; report the verdict-flip count  [R8.6]
	$(PY) judge-audit $(RUN)

verify: ## Recompute every metric from cached responses, no network  [G1][N3]
	$(PY) verify $(RUN)

analyze: ## -> results.md. Interaction term first.  [R9.1]
	$(PY) analyze $(RUN)

figures: ## Regenerate every figure (Plotly -> kaleido -> PDF/SVG)  [R10.2]
	$(PY) figures --out figures/

paper: ## Build the PDF from results.md alone  [R10.3]
	$(PY) paper --out ../paper/build

replay: ## Metric-identical reproduction from the content-addressed store  [R6.3]
	$(PY) replay $(RUN)

# ── Quality ────────────────────────────────────────────────────────────────────
selftest: ## P3 gate: write -> guard -> manifest on a synthetic fixture, offline
	$(PY) python -m msat.selftest

test: ## Fast tests only (excludes @pytest.mark.slow / .network)
	$(PY) pytest -m "not slow and not network"

test-all: ## Everything, including the full grid and live API  [§5]
	$(PY) pytest

lint: ## ruff check + format check
	$(PY) ruff check .
	$(PY) ruff format --check .

typecheck: ## mypy, strict on our own modules
	$(PY) mypy src/msat

# ── Hygiene ────────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts
	rm -rf figures/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: ## clean + remove the venv (requires a full `make setup` after)
	$(MAKE) clean
	rm -rf .venv/
