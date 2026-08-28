# Reproducibility repository: LLM agents in public-good experiments

This is the minimal companion code repository for the main corrected
Oechssler--Reischmann--Sofianos (ORS 2022) public-good experiments. It contains
the game and mechanism definitions, agent/parser/client code, the sweep driver,
the exact main design manifest, metric computation, and small verification
scripts. It intentionally excludes collected outputs, logs, plots, generated
configs, API ledgers, private credentials, and unrelated pilot code.

## Layout

- `src/games/`, `src/mechanisms/`: executable game and mechanism rules.
- `src/agents.py`, `src/llm.py`, `src/runner.py`: decision parsing, model
  client, and experiment runner.
- `configs/ors_2022_main.yaml`: five-repetition ORS design with VCM, SCCM,
  CCM, and fixed-player conditions.
- `scripts/run_ro_open_models_sweep.py`: offline vLLM sweep orchestration.
- `scripts/simple_new_metrics.py`: the paper's metric framework.
- `scripts/verify_*.py`: rule and normalized-welfare checks.
- `data/`: pointer/instructions for the companion Hugging Face data release.

## Setup

Install with `uv sync --locked`. The lockfile records the environment used by
the project. The sweep additionally requires compatible local model snapshots
under `model_snapshots/` (or `MODEL_SNAPSHOT_ROOT`) and a vLLM installation
compatible with the selected hardware. No credentials or endpoints are stored
in this repository.

## Reproduce analysis

Place the released data files in `data/` or set `LLM_PGG_OUTPUTS` to a local
directory containing the analysis inputs, then run:

```bash
uv run python scripts/simple_new_metrics.py
```

The exact released observations are the authoritative input for the paper.
The metric script writes derived results below the selected output directory;
generated results are not tracked here.

## Re-run inference

After installing the required model snapshots and starting the local vLLM
servers, render or run the configured sweep:

```bash
uv run python scripts/run_ro_open_models_sweep.py \
  --manifest configs/ors_2022_main.yaml --render-configs-only
```

The full sweep is computationally expensive. The manifest preserves the fixed
game rules, prompt treatment, seeds, repetitions, chain count, period count,
and fixed-player design; do not alter these when claiming replication.

## Verification

```bash
uv run python scripts/verify_normalized_welfare.py
uv run python scripts/verify_ro_2022_mechanisms.py
```

Model-provider snapshots, serving templates, hardware, and external API
versions can affect generated decisions. Therefore inference reruns are a
replication aid, while the released decision tables support exact reanalysis
of the paper's reported runs.

Release: 2026-08-28.
