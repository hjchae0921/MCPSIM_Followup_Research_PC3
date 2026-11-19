# MCPSIM timestamp worker — PC3

Assigned models: GPT-5 mini (low), GLM 4.7 Flash (low), GLM 5.3 Flash (low). Each model runs six physics cases × three methods × 15 independent workflows (270 workflows). This station has **810 workflows**.

All three station assignments are listed in `station_assignments.json`. PC1 runs Claude Haiku 4.5 and Gemma 4 31B; PC2 runs MiniMax M3.1 Flash and Qwen 3.8 27B; PC3 runs GPT-5 mini, GLM 4.7 Flash and GLM 5.3 Flash.

## Environment and keys

Use Ubuntu/WSL2 Linux x86-64, one MPI rank, and the pinned environment:

```bash
conda create -n mcpsim-worker --file conda-linux-64.explicit.txt
conda activate mcpsim-worker
python -m pip install -r requirements.txt
python -B run.py check
```

Set `OPENROUTER_API_KEY` in the environment or point `OPENROUTER_API_KEY_FILE` at a local key file. PC3 additionally needs `OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` for **GPT-5 mini (`gpt-5-mini`)**. The launcher can also read `openrouter_api_key.txt` and `openai_api.txt` from the checkout or its parent. Never commit keys.

## Start and resume

Stop any earlier worker using this checkout before updating it. Existing older campaign outputs stay untouched; the new campaign has a separate output directory.

```bash
python -u -B launch.py --background --workers 6 --per-model 2
python -B run.py status
```

Use the activated pinned Python interpreter for the launcher. Keep the same concurrency values on resume. `worker.log` stores the controller output; `launcher.log` records launcher errors. The launcher runs checks, prepares immutable inputs, executes, organizes the terminal results, and exports a ZIP. It detaches from the terminal. Windows sleep or WSL shutdown still interrupts computation.

## Cards and prompts

The six benchmark cards omit empty internal-state-variable entries. Stages 2, 3 and 5 use the current prompts without internal-state preparation or update instructions. Unknown-field initial conditions and load stepping remain. Card-only, document RAG and KG share these cards and prompts; the supplied API context is the only method difference.

## Identity and raw records

Experiment paths use `YYYY_MM_DD_HH_MM_SS_microseconds` in Korea time (UTC+9). The timestamp is captured **inside the workflow execution thread at actual execution start**. The exact same instant is written as `started_at` and used for the directory basename. Queue preparation and submission allocate neither an experiment timestamp nor an execution directory. Resuming retains the original start time. The task registry separately records which jobs have actually started. There is no numbered experiment-directory identity. Repetition counts, stage numbers, repair calls and attempt counters remain valid.

The worker creates timestamp identities directly. There is **no runtime content filter, label rejection or text replacement**. Requests, responses, errors and API-returned reasoning are preserved as returned; missing provider fields stay null. Historical workflows are not imported or rewritten.

All model calls request **low reasoning**. A failed low-reasoning preflight blocks the station; there is no fallback to another effort or disabled reasoning. Claude Haiku 4.5 and GPT-5 mini omit temperature; the remaining assigned models use temperature 0. Accepted configurations are recorded in `preflight` and `protocol.json`. OpenRouter may map its unified low effort to provider-native controls: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

## Results and stop

Outputs: `outputs/three_arm_state_free_actual_start_20261119/pc3/`. Raw workflows are under `records/model/method/physics/timestamp`. Stage-six processing uses the same accepted solver process. Reference agreement never feeds code repair; execution success and reference agreement are distinct metrics.

```bash
python -B run.py organize
python -B run.py export
python -B run.py stop
```

Organized results use `organized_results/model/KG|Card|RAG/physics/Success|Fail/timestamp`. Success means execution success. `index.csv` lists timestamp identities. All organization is copied from the raw results. Export includes this organized view. After a stop, remove the campaign's `STOP` and `postprocess/STOP` files before resuming. Unreceipted submissions are not silently repeated.

## Select another station from the same source

All station assignments are published in this repository. To use this source for PC1, run `python -u -B launch.py --station pc1 --background --workers 6 --per-model 2`. Inspect that station with `python -B run.py status --station pc1`. The default remains the station in `assignment.json`. Use a dedicated checkout for each running station.

## Actual-start release and existing campaigns

The new campaign is `three_arm_state_free_actual_start_20261119`. Registration-time campaign records are never renamed or rewritten. This release does not resume or migrate that older campaign. Stop a running worker before updating its checkout. Updating this source does not itself start a paid experiment.

## Offline timestamp regression tests

Run `python -B tests/test_actual_start.py -v`. These tests make no API calls. They verify that prepare/report do not assign experiment start times; worker-thread start time exactly determines the KST directory name; parallel starts persist atomically; resume retains the first actual start; legacy registration-time records are not edited; clock collisions cannot fabricate times; and completed jobs are not restarted.
