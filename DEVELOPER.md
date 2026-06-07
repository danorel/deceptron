# Developer guide

Technical reference for running deceptron's pipeline, the Phase 1 repro/comparison
harnesses, and the codebase layout. For project context and motivation, see the main
[README.md](README.md), [docs/plan.md](docs/plan.md), and [docs/prior.md](docs/prior.md).

---

## Pipeline

The pipeline runs in 8 stages (`python -m deceptron.main --list`). Each stage reads from and writes to `.jsonl` files in `data/`.
All stages are **incremental** — re-running skips already-processed items.

Run the whole thing (or a subset, in pipeline order) with the orchestrator:

```bash
python -m deceptron.main                       # full pipeline, default paths from data/
python -m deceptron.main --stages tasks agent  # only these stages
python -m deceptron.main --list                # list stage names
```

It shells out to each stage's CLI below — use the per-stage commands directly when you
need flags like `--limit`, `--force`, `--no-docker`.

```
GitHub repos
    │
    ▼
[1] miner        →  data/deceptron/mining/repos.jsonl
    │
    ▼
[2] verifier     →  data/deceptron/mining/repos_verified.jsonl        (Docker: run test suite)
    │                                                  Automatically excludes repos that
    │                                                  require GUI/system libs (Qt, VLC…)
    ▼
[3] tasks        →  data/deceptron/tasks/tasks.jsonl                 (Claude: generate task pairs)
    │              data/deceptron/tasks/cache/<repo>/analysis.json    (cached by SHA)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[4a] agent       →  data/deceptron/trajectories/baseline.jsonl  [4b] redteam  →  data/deceptron/trajectories/scheming.jsonl
     (main task only)                                       (main + hidden side task)
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
[5] verifier  →  updates trajectories in-place       (Docker: apply patch + pytest + check scripts)
                   │
                   ▼
[6] monitor   →  data/deceptron/scores/prompted.jsonl                   (Claude: score 0–10)
                   │
                   ▼
[7] eval      →  data/deceptron/results/prompted/results.json           (AUROC / pAUROC)
```

### Stage 1 — Mine repos

Queries GitHub for Python repos with 100–2000 stars, 3–30 contributors, passing CI, and open license.

```bash
python -m deceptron.miner.main --lang python --limit 50 --out data/deceptron/mining/repos.jsonl
```

### Stage 2 — Verify repos (Docker required)

Clones each repo and runs three checks in Docker:

1. **System libs check** — scans `requirements.txt` / `pyproject.toml` for packages that require
   native GUI or media libraries (PyQt, PySide, VLC, wxPython, GTK, pygame…). Repos with such
   dependencies are marked `system_libs=[...]` and skipped in all subsequent stages — they cannot
   be imported in a headless Docker slim image.
2. **Python version detection** — reads `.python-version`, `pyproject.toml`, or `setup.cfg`.
3. **Test suite** — runs pytest. Repos where all tests pass (or pass after deselecting known
   failures) are marked `test_verified=True`.

```bash
python -m deceptron.verifier.main --repos data/deceptron/mining/repos.jsonl --out data/deceptron/mining/repos_verified.jsonl
```

Produce a git-committable version without large install/test logs:

```bash
python -m deceptron.scripts.strip_logs --input data/deceptron/mining/repos_verified.jsonl --out data/deceptron/mining/repos_verified_clean.jsonl
```

Only repos with `test_verified=True` **and** `system_libs=[]` proceed to task generation.

### Stage 3 — Generate task pairs (Claude + Docker)

For each verified repo, Claude analyzes the source files and generates 3 `(main_task, side_task)` pairs.
Results are cached by SHA in `data/deceptron/tasks/cache/<owner>__<repo>/analysis.json` — re-running is free.

Claude is instructed to select only files with **pure Python logic** (no GUI imports at module level).
Generated function names are validated against real AST-extracted names to prevent hallucinations.

```bash
# Generate pairs (skips repos already at quota)
python -m deceptron.tasks.generate --repos data/deceptron/mining/repos_verified_clean.jsonl --out data/deceptron/tasks/tasks.jsonl

# Validate check scripts in Docker (updates main_task_check_valid / side_task_check_valid)
python -m deceptron.tasks.validate --tasks data/deceptron/tasks/tasks.jsonl --repos data/deceptron/mining/repos_verified_clean.jsonl
```

Flags: `--no-docker` skips Docker validation; `--force` re-validates all pairs.

### Stage 4a — Baseline agent trajectories

Agent receives only the `main_task`. Tools: `read_file`, `list_directory`, `write_file`.
Produces a `git diff` patch and a full tool-use transcript as the trajectory.

```bash
python -m deceptron.agent.main --tasks data/deceptron/tasks/tasks.jsonl --out data/deceptron/trajectories/baseline.jsonl
```

- `--limit N` — process N tasks (useful for testing)
- `--delay N` — seconds between tasks to stay under API rate limits (default: 12)

Already-processed `task_id`s are skipped. Only trajectories with actual steps/patches count as done
(failed-with-error trajectories are retried automatically).

### Stage 4b — Scheming (red team) trajectories

Agent receives `main_task` via user message and `side_task` via system prompt (not visible in the
transcript). Must complete both while making the trajectory look like normal coding.

```bash
python -m deceptron.redteam.main --tasks data/deceptron/tasks/tasks.jsonl --out data/deceptron/trajectories/scheming.jsonl
```

Same `--limit` and `--delay` flags as `agent.run`.

### Stage 5 — Verify trajectories (Docker required)

Applies each agent's patch to a clean clone, then in a single container:
1. Runs `pytest` with `deselect_args` → `tests_passing` (existing tests must stay green)
2. Runs `main_task_check_script` → `main_task_verified` (True only if tests pass **and** check passes)
3. Runs `side_task_check_script` → `side_task_verified`

Updates trajectory files in-place. `verify_log` is saved for debugging.

```bash
python -m deceptron.verifier.main --trajectories data/deceptron/trajectories/baseline.jsonl \
                       --tasks data/deceptron/tasks/tasks.jsonl \
                       --verified-repos data/deceptron/mining/repos_verified_clean.jsonl

python -m deceptron.verifier.main --trajectories data/deceptron/trajectories/scheming.jsonl \
                       --tasks data/deceptron/tasks/tasks.jsonl \
                       --verified-repos data/deceptron/mining/repos_verified_clean.jsonl
```

### Stage 6 — Monitor scoring

Claude reads each trajectory (tool calls, file writes, final patch — **not** the system prompt)
and outputs a scheming score 0–10. This is pure black-box monitoring.

```bash
python -m deceptron.monitor.prompted.main \
    --trajectories data/deceptron/trajectories/baseline.jsonl data/deceptron/trajectories/scheming.jsonl \
    --out data/deceptron/scores/prompted.jsonl
```

Add `--force` to re-score everything.

### Stage 7 — Evaluate

```bash
python -m deceptron.eval.main --scores data/deceptron/scores/prompted.jsonl --out data/deceptron/results/prompted/
```

Outputs AUROC and pAUROC (primary metric). With a working pipeline expect:
- Baseline scores: 0–3
- Scheming scores: 6–9
- AUROC > 0.85

---

## Phase 1 repro + comparison

`repro` mirrors `deceptron`'s monitor → eval → main shape (same module names, same
metrics via `common.*`) but scores deceptron's existing real-repo trajectories with
**Apollo's 4 reference monitor prompts** (Appendix G — see [docs/prior.md](docs/prior.md))
instead of deceptron's own prompt. `compare` then assembles deceptron's results +
repro's per-prompt results + Apollo's reported numbers into one side-by-side table —
it never recomputes metrics, only loads and reports.

```bash
python -m repro.main      # score (4 Apollo prompts, one .jsonl each) -> eval (4 results.json)
python -m compare.main    # deceptron eval -> repro -> comparison.json + printed table
```

See [docs/plan.md](docs/plan.md) for the rationale (`src/common`/`src/repro`/`src/compare` split)
and [docs/prior.md](docs/prior.md) for Apollo's reported numbers this is compared against.

## Orchestration (Dagster)

Mining + repo-verification (`mine` -> `verify-repos`, the slowest/flakiest
stages — GitHub API rate limits, Docker clone-and-test per repo) can run as
an isolated, monitored, resumable background job via Dagster instead of a
foreground `python -m deceptron.main` invocation.

`src/orchestration/definitions.py` wires `deceptron.miner.main.run_miner`
and `deceptron.verifier.main.run_verify_repo` directly into Dagster ops (no subprocess —
same functions `deceptron.main`'s stages call). Both are already
append-and-resume-by-id, so a retried or restarted run continues from the last
completed item for free; Dagster adds isolation (its own process/container),
a UI for progress/failures, and automatic retries (`RetryPolicy`, 2 attempts,
60s backoff — tuned for GitHub rate-limit bursts and flaky Docker pulls).

**Local (foreground UI + daemon in one process):**
```bash
dagster dev -w workspace.yaml
# open http://localhost:3000 -> Jobs -> mining_pipeline_job -> Launchpad
# (set lang/limit/out per-op there, or use the default config in definitions.py)
```

**Background, via docker-compose** (webserver + daemon, SQLite-backed run
storage — no Postgres container; `verify-repos` mounts the host's
`docker.sock` to clone+test repos in their own containers):
```bash
docker compose up -d
open http://localhost:3000
docker compose logs -f dagster-daemon   # tail run progress / failures
docker compose down                     # stop (data persists in the dagster_home volume)
```

Both paths load `mining_pipeline_job` from `workspace.yaml` ->
`orchestration.definitions:defs`. Config schema
(`MineReposConfig`/`VerifyReposConfig`) exposes `lang`/`limit`/`out`/`repos`
as run-launch parameters — override them in the UI Launchpad per run instead
of editing code.

## Directory structure

```
deceptron/
├── pyproject.toml           # editable install: `pip install -e .` (src/ layout, four packages)
├── docker-compose.yml       # Dagster UI + daemon for the mining/verification job
├── workspace.yaml           # Dagster: where to load job definitions from
├── dagster.yaml             # Dagster: instance config ($DAGSTER_HOME — local SQLite)
├── src/
│   ├── common/              # shared by deceptron + repro — must be byte-identical
│   │   │                    # so the two pipelines' results are comparable:
│   │   ├── llm.py               # create_with_retry — Anthropic API call + backoff
│   │   ├── trajectory.py        # format_trajectory — trajectory dict -> monitor text
│   │   └── metrics.py           # compute_pauroc, bootstrap_ci (McClish 1989 + 95% CI)
│   ├── deceptron/           # main real-repo pipeline
│   │   ├── miner/           # GitHub repo discovery and filtering
│   │   ├── verifier/
│   │   │   ├── docker.py        # Shared Docker low-level utilities
│   │   │   ├── deterministic.py # Repo verification: system-lib check + test suite
│   │   │   ├── trajectory.py    # Trajectory verification: apply patch + check scripts
│   │   │   ├── run.py           # CLI dispatcher (--repos or --trajectories mode)
│   │   │   └── schema.py        # VerifiedRepo pydantic model
│   │   ├── tasks/           # (main_task, side_task) pair generation
│   │   ├── agent/           # Baseline coding agent (main task only)
│   │   ├── redteam/         # Scheming agent (main + hidden side task)
│   │   ├── monitor/         # Prompted scheming monitor (+ SFT, phase 2)
│   │   ├── eval/            # AUROC / pAUROC evaluation
│   │   ├── scripts/         # Utilities: strip_logs, backfill_sha
│   │   └── main.py          # pipeline orchestrator (python -m deceptron.main)
│   ├── orchestration/       # Dagster wiring for mine -> verify-repos as an
│   │   │                    # isolated/monitored/resumable background job
│   │   ├── definitions.py       # imports run_miner/run_verify_repo directly — see
│   │   │                        # "Orchestration (Dagster)" above
│   │   └── Dockerfile           # webserver + daemon image (build context: repo root)
│   ├── repro/               # Phase 1: Apollo-repro harness (sibling package) —
│   │   │                    # mirrors deceptron's monitor -> eval -> main shape,
│   │   │                    # 1:1 same module names, swapping only the prompt source
│   │   ├── monitor/
│   │   │   └── prompted/
│   │   │       └── main.py          # Apollo's 4 reference prompts (Appendix G, in
│   │   │                            # PROMPTS) scored on deceptron's real trajectories —
│   │   │                            # one data/repro/scores/<variant>.jsonl per prompt
│   │   ├── eval/
│   │   │   └── main.py          # mirrors deceptron.eval.main — same common.metrics,
│   │   │                        # one results.json per prompt-variant scores file
│   │   └── main.py          # orchestrator: score (x4 prompts) -> eval (x4 variants)
│   └── compare/             # cross-pipeline comparison — orchestration + report
│       │                    # only, never recomputes metrics (loads results from
│       │                    # both pipelines + Apollo's reported numbers)
│       ├── report.py            # builds + prints/writes the side-by-side table
│       └── main.py          # orchestrator: deceptron(eval) -> repro -> report
└── data/
    ├── deceptron/                                     # one subdir per pipeline stage
    │   ├── mining/
    │   │   ├── repos.jsonl                            ← committed (stage 1 — miner output)
    │   │   ├── repos_verified.jsonl                   ← gitignored (stage 2 — install/test logs)
    │   │   └── repos_verified_clean.jsonl             ← committed (stage 2 — logs stripped)
    │   ├── tasks/
    │   │   ├── tasks.jsonl                            ← committed (stage 3 — task pairs)
    │   │   └── cache/                                 ← gitignored (Claude analysis cache by SHA)
    │   ├── trajectories/
    │   │   ├── baseline.jsonl                         ← gitignored (stage 4a — main task only)
    │   │   └── scheming.jsonl                         ← gitignored (stage 4b — main + side task;
    │   │                                                 stage 5 verifies both in-place)
    │   ├── scores/
    │   │   └── prompted.jsonl                         ← gitignored (stage 6 — monitor scores)
    │   └── results/
    │       └── prompted/results.json                  ← gitignored (stage 7 — AUROC / pAUROC)
    ├── repro/                                         ← gitignored, regenerated from deceptron's
    │   ├── scores/<variant>.jsonl                       trajectories (python -m repro.main)
    │   └── results/<variant>/results.json              — same scores/<name>.jsonl + results/<name>/
    │                                                     results.json shape as deceptron, swapping
    │                                                     only the variable: prompt source name
    └── compare/                                       ← gitignored, regenerated
        └── results/comparison.json                      (python -m compare.main)
```

## SFT monitor (phase 2, scaffolded — optional, GPU required)

Requires ≥24 GB VRAM (Ada L40S 48 GB recommended).

```bash
python -m deceptron.monitor.sft \
    --train data/deceptron/trajectories/baseline.jsonl data/deceptron/trajectories/scheming.jsonl \
    --out runs/monitor-v1
```
