# MCPSIM benchmark worker — PC3

## Files and folders

- `assignment.json`: Deepseek V3.1, GPT-5.4 mini, Qwen 3.8 27B; 15 repeats per physics and input method, 810 workflows.
- `run.py`: runtime checks, model preflight, parallel execution, progress and result export.
- `frozen_inputs/`: cards, meshes, five-stage templates and Card Only / Document Top3 / KG contexts.
- `frozen_postprocess/`: current sixth-stage prompt and corresponding contexts.
- `retrieval_assets/`: KG2 without `IS_VALID_INPUT_TO`, API embeddings, selected document chunks, rankings and token counts. KG includes API first-paragraph and input-role descriptions.
- `experiments/`: stage worker, self-repair, retained-process postprocessing, server storage and fixed reference fields.
- `common/`, `codegen/`, `*_agent/`: shared dependencies of the stage worker.
- `conda-linux-64.explicit.txt`, `runtime_fingerprint.json`, `requirements.txt`: pinned execution environment.
- `outputs/three_arm_20261117/pc3/`: generated results; not committed.

The six cases are heat sink, tensile specimen, rubber bellows, thermoelastic I-beam, Joule-heating busbar, and piezoelectric cantilever. Each model runs 6 cases × 3 methods × 15 repeats = 270 workflows.

## Setup

Use Ubuntu or WSL2 on Linux x86-64 with Miniforge installed. From a fresh clone:

```bash
conda create -n mcpsim-worker --file conda-linux-64.explicit.txt
conda activate mcpsim-worker
python -m pip install -r requirements.txt
python -B run.py check
python -B run.py prepare
```

If the pinned environment is already installed, activate it and run `check`. Required scientific versions include DOLFINx 0.10.0, Basix 0.10.0 and UFL 2025.2.1. The controller uses one MPI rank; do not launch it through `mpirun`.

## Run

Set keys in the terminal, never in tracked files:

```bash
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
export OPENROUTER_API_KEY
printf '\n'
```

PC3 also requires an OpenAI API key for GPT-5.4 mini:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
```

```bash
python -u -B run.py run --workers 12 --per-model 4 2>&1 | tee worker.log
```

Keep the terminal open or use `tmux`. For a smaller machine, choose `--workers 6 --per-model 2` on the first run. Resume with the same settings. Only run this assignment on its named PC; keep any earlier worker stopped.

Sampling uses temperature 0 and low reasoning, falling back only to an accepted enabled reasoning setting. The preflight records the actual configuration. Web search and tool calls are disabled. Each stage allows an initial attempt and up to five current-stage repairs. API-returned reasoning, usage, costs, errors and timing are saved; private reasoning is not reconstructed.

The sixth stage extracts coordinates and values in the same process as the accepted solve. The server writes NPZ and metadata. Reference comparison is performed afterward and is never feedback for code repair. Execution success and reference agreement are reported separately. The established relative-error threshold is 1e-5, or 2e-3 for hyperelasticity; temperature uses the reference temperature-increment normalization.

## Progress and results

```bash
python -B run.py status
python -B run.py export
```

`summary.json` under the output folder provides model/method/physics counts, stage-six outcomes and reference agreement. `export` creates `pc3_three_arm_20261117_results.zip`; transfer it to PC1 after completion. Successful API response time, failed-request time and full workflow time are separate; internal provider waits remain part of response latency.

```bash
python -B run.py stop
```

Stop prevents new submissions but may wait for an in-flight request. After it stops, remove the two STOP files under the campaign output and its `postprocess/` subfolder to resume. Recorded responses are reused; an unreceipted submission is not silently repeated. Do not edit frozen inputs or move an initialized bundle before resuming.
