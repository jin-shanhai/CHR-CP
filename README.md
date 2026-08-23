# CHR-CP

Confidence-Gated Hierarchical Routing with Cache-Preserved Switching.

CHR-CP is a multi-agent LLM routing framework for API-based heterogeneous model pools. It routes easy tasks to cheaper tiers, escalates uncertain steps to stronger tiers, and preserves handoff context and prompt-cache value when switching across model tiers or providers.

## Core Ideas

- **VC2 uncertainty**: combines verbalized confidence with lightweight consistency checks, without relying on logprobs.
- **CA2R routing**: adapts STAY / BRANCH / ESCALATE thresholds using budget and cache-health signals.
- **Cache-preserved switching**: uses stable prompts, compressed handoff context, and structured verifier-corrector handoffs.
- **Cost-aware multi-agent execution**: records tier usage, API cost, latency, routing decisions, and evaluation traces.

## Project Layout

```text
chr_cp/
  clients/       # provider clients and unified CompletionResponse
  confidence/    # VC2 verbalized + consistency uncertainty estimators
  prompts/       # stable prefix, role templates, distillation, handoff
  routing/       # L1/L2/L3 routing, budget, cache history, orchestrator
  benchmarks/    # MATH, AIME, HumanEval, MMLU, GPQA, GSM8K loaders
  utils/         # cost tracking, text similarity, AST utilities
experiments/     # experiment runners and phase controls
tests/           # unit tests, diagnostics, result analyzers
paper_assets/    # paper figures, result summary, and submission assets
configs/         # model pool and pricing configuration
```

## Setup

```bash
conda create -n chrcp python=3.11 -y
conda activate chrcp
pip install -r requirements.txt
```

Configure API keys in your environment or local `.env` file before running live experiments.

## Quick Checks

```bash
python -m pytest tests
python paper_assets/generate_figures.py
```

## Paper Figures

The current project figures are generated from `paper_assets/data/results_summary.json`:

```bash
python paper_assets/generate_figures.py
```

Open `paper_assets/figure_contact_sheet.html` to preview the generated SVG figures. PNG exports are also stored in `paper_assets/figures_png/` when exported with a headless browser.

## IPCCC submission assets

The IPCCC manuscript and its figures are under `paper_latex/ipccc2026_ieee/`.
The completed four-model evaluation artifact is under `paper_assets/results/`;
it contains item-level traces, repeated-run aggregates, metadata, and the
submission table/figure. The summary under `paper_assets/data/` is retained for
the project's original CHR-CP figures.

To compile the manuscript locally:

    cd paper_latex/ipccc2026_ieee
    latexmk -pdf -interaction=nonstopmode main.tex
