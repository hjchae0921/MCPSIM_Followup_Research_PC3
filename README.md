# MCPSIM_Followup_Research_PC3

This bundle runs PC3 models only: Deepseek V3.1, MiniMax M2.7, Deepseek V4 Flash-0731. It contains fixed cards, mesh files, five-stage templates, pre-retrieved API contexts, stage-six export code infrastructure, and reference NPZ files. No API keys are included. Model calls have no web-search or tool access.

## Files

| Path | Contents |
|---|---|
| `run.py` | Environment check, input preparation, execution, status and result export |
| `assignment.json` | Models assigned exclusively to this PC |
| `frozen_inputs/` | Six meshes, completed cards, stage templates and retrieved API contexts |
| `experiments/` | Experiment runners, postprocessing and reference solutions |
| `common/`, `codegen/`, `*_agent/` | Shared implementation dependencies |
| `conda-linux-64.explicit.txt`, `requirements.txt` | Scientific runtime and Python client dependencies |
| `bundle_manifest.json` | Input and implementation checksums |
| `outputs/` | Local run artifacts; excluded from Git |

## Setup (Ubuntu or Windows WSL2, Linux x86-64)

Install Miniforge if it is not already available. Clone the repository and open its directory:

```bash
git clone https://github.com/hjchae0921/MCPSIM_Followup_Research_PC3.git
cd MCPSIM_Followup_Research_PC3
```

Create the scientific runtime:

```bash
conda create -n mcpsim-worker --file conda-linux-64.explicit.txt
conda activate mcpsim-worker
python -m pip install -r requirements.txt
python -B run.py check
```

The explicit Conda package list reproduces PC1's scientific runtime. The check requires DOLFINx 0.10.0, Basix 0.10.0, UFL 2025.2.1, FFCx 0.10.0, the recorded PETSc/NumPy/SciPy versions, MUMPS and a single MPI rank. It does not perform any API request. Native Windows Python is not supported; use WSL2.

Set the API key locally without placing it in this directory:

```bash
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
export OPENROUTER_API_KEY
printf '\n'
python -B run.py prepare
python -u -B run.py run --workers 4 2>&1 | tee worker.log
```

`prepare` verifies the package, relocates only the mesh path in each mesh card and prints the planned counts. The source cards are retained under `frozen_inputs`. The run performs the same model capability check, current-stage self-repair (at most five corrections), and input formats as PC1. Sampling is temperature 0 with low reasoning or the lowest accepted fallback. Provider responses and reasoning returned by the API are recorded. Postprocessing is generated after solve and executes in the retained Python process; reference values are never provided to the coder.

Use `--workers 2` if memory is limited. Keep the terminal open or use `tmux`. Multiple PCs do not remove shared provider/API rate limits. Do not run the same station bundle concurrently on another PC.

## Progress and result transfer

From a second terminal with the same environment:

```bash
python -B run.py status
```

Results are under `outputs/pc3/runs` (stages 1-5) and `outputs/pc3/postprocess/items` (stage six and reference comparison). Original evaluation thresholds remain unchanged. For a clean result snapshot, wait until the run finishes:

```bash
python -B run.py export
```

Return the generated `pc3_results.zip` to PC1. It contains prompts, generated code, traces, API-returned usage/reasoning, timing, NPZ fields and reference-comparison reports. No credentials are stored by this runner. Cache directories are omitted from the archive.

`python -B run.py stop` stops further submissions and requests runtime interruption. It does not forcibly cancel an in-flight model request. To resume after it stops, remove STOP files under `outputs` and run the same command; recorded receipts are reused and unreceipted requests are not silently resubmitted. Do not edit fixed input files or rename/move an initialized bundle before resuming.
