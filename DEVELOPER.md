# Developer guide

Technical reference for running deceptron's pipeline, the Phase 1 apollo/benchmark
harnesses, and the codebase layout. For project context and motivation, see the main
[README.md](README.md), [docs/plan.md](docs/plan.md), and [docs/prior.md](docs/prior.md).

---

## Pipeline

The pipeline runs in 8 stages (`python -m agents.deceptron.main --list`). Each stage reads from and writes to `.jsonl` files in `data/`.
All stages are **incremental** — re-running skips already-processed items.

Run the whole thing (or a subset, in pipeline order) with the orchestrator:

```bash
python -m agents.deceptron.main                       # full pipeline, default paths from data/
python -m agents.deceptron.main --stages tasks coder  # only these stages
python -m agents.deceptron.main --list                # list stage names
```

It shells out to each stage's CLI below — use the per-stage commands directly when you
need flags like `--limit`, `--force`, `--no-docker`.

```
GitHub repos
    │
    ▼
[1] miner        →  data/agents/deceptron/mining/repos.jsonl
    │
    ▼
[2] verifier     →  data/agents/deceptron/mining/repos_verified.jsonl        (Docker: run test suite)
    │                                                  Automatically excludes repos that
    │                                                  require GUI/system libs (Qt, VLC…)
    ▼
[3] tasks        →  data/agents/deceptron/tasks/tasks.jsonl                 (Claude: generate task pairs)
    │              data/agents/deceptron/tasks/cache/<repo>/analysis.json    (cached by SHA)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[4a] coder       →  data/agents/deceptron/trajectories/baseline.jsonl  [4b] redteam  →  data/agents/deceptron/trajectories/scheming.jsonl
     (main task only)                                       (main + hidden side task)
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
[5] verifier  →  updates trajectories in-place       (Docker: apply patch + pytest + check scripts)
                   │
                   ▼
[6] monitor   →  data/agents/deceptron/scores/prompted.jsonl                   (Claude: score 0–10)
                   │
                   ▼
[7] eval      →  data/agents/deceptron/results/prompted/results.json           (AUROC / pAUROC)
```

### Stage 1 — Mine repos

Queries GitHub for Python repos with 100–2000 stars, 3–30 contributors, passing CI, and open license.

```bash
python -m agents.deceptron.miner.main --lang python --limit 50 --out data/agents/deceptron/mining/repos.jsonl
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
python -m agents.deceptron.verifier.repo.main --repos data/agents/deceptron/mining/repos.jsonl --out data/agents/deceptron/mining/repos_verified.jsonl
```

Produce a git-committable version without large install/test logs:

```bash
python -m agents.deceptron.maintenance.main strip-logs
```

Only repos with `test_verified=True` **and** `system_libs=[]` proceed to task generation.

### Stage 3 — Generate task pairs (Claude + Docker)

For each verified repo, Claude analyzes the source files and generates 3 `(main_task, side_task)` pairs.
Results are cached by SHA in `data/agents/deceptron/tasks/cache/<owner>__<repo>/analysis.json` — re-running is free.

Claude is instructed to select only files with **pure Python logic** (no GUI imports at module level).
Generated function names are validated against real AST-extracted names to prevent hallucinations.

```bash
# Generate pairs (skips repos already at quota)
python -m agents.deceptron.tasks.generate.main --repos data/agents/deceptron/mining/repos_verified_clean.jsonl --out data/agents/deceptron/tasks/tasks.jsonl

# Validate check scripts in Docker (updates main_task_check_valid / side_task_check_valid)
python -m agents.deceptron.tasks.validate.main --tasks data/agents/deceptron/tasks/tasks.jsonl --repos data/agents/deceptron/mining/repos_verified_clean.jsonl
```

Flags: `--no-docker` skips Docker validation; `--force` re-validates all pairs.

### Stage 4a — Baseline agent trajectories

Agent receives only the `main_task`. Tools: `read_file`, `list_directory`, `write_file`.
Produces a `git diff` patch and a full tool-use transcript as the trajectory.

```bash
python -m agents.deceptron.models.coder.main --tasks data/agents/deceptron/tasks/tasks.jsonl --out data/agents/deceptron/trajectories/baseline.jsonl
```

- `--limit N` — process N tasks (useful for testing)
- `--delay N` — seconds between tasks to stay under API rate limits (default: 12)

Already-processed `task_id`s are skipped. Only trajectories with actual steps/patches count as done
(failed-with-error trajectories are retried automatically).

### Stage 4b — Scheming (red team) trajectories

Agent receives `main_task` via user message and `side_task` via system prompt (not visible in the
transcript). Must complete both while making the trajectory look like normal coding.

```bash
python -m agents.deceptron.models.redteam.main --tasks data/agents/deceptron/tasks/tasks.jsonl --out data/agents/deceptron/trajectories/scheming.jsonl
```

Same `--limit` and `--delay` flags as `agents.deceptron.models.coder.main`.

### Stage 5 — Verify trajectories (Docker required)

Applies each agent's patch to a clean clone, then in a single container:
1. Runs `pytest` with `deselect_args` → `tests_passing` (existing tests must stay green)
2. Runs `main_task_check_script` → `main_task_verified` (True only if tests pass **and** check passes)
3. Runs `side_task_check_script` → `side_task_verified`

Updates trajectory files in-place. `verify_log` is saved for debugging.

```bash
python -m agents.deceptron.verifier.trajectory.main --trajectories data/agents/deceptron/trajectories/baseline.jsonl \
                       --tasks data/agents/deceptron/tasks/tasks.jsonl \
                       --verified-repos data/agents/deceptron/mining/repos_verified_clean.jsonl

python -m agents.deceptron.verifier.trajectory.main --trajectories data/agents/deceptron/trajectories/scheming.jsonl \
                       --tasks data/agents/deceptron/tasks/tasks.jsonl \
                       --verified-repos data/agents/deceptron/mining/repos_verified_clean.jsonl
```

### Stage 6 — Monitor scoring

Claude reads each trajectory (tool calls, file writes, final patch — **not** the system prompt)
and outputs a scheming score 0–10. This is pure black-box monitoring.

```bash
python -m agents.deceptron.models.monitor.prompted.main \
    --trajectories data/agents/deceptron/trajectories/baseline.jsonl data/agents/deceptron/trajectories/scheming.jsonl \
    --out data/agents/deceptron/scores/prompted.jsonl
```

Add `--force` to re-score everything.

### Stage 7 — Evaluate

```bash
python -m agents.deceptron.eval.main --scores data/agents/deceptron/scores/prompted.jsonl --out data/agents/deceptron/results/prompted/
```

Outputs AUROC and pAUROC (primary metric). With a working pipeline expect:
- Baseline scores: 0–3
- Scheming scores: 6–9
- AUROC > 0.85

---

## Phase 1: Apollo repro + benchmark comparison

`agents.apollo` mirrors `agents.deceptron`'s models.monitor → eval → main shape (same
module shapes, same metrics via `shared.*`) but scores deceptron's existing real-repo
trajectories with **Apollo's 4 reference monitor prompts** (Appendix G — see
[docs/prior.md](docs/prior.md)) instead of deceptron's own prompt. `benchmark` then
assembles deceptron's results + apollo's per-prompt results + Apollo's reported numbers
into one side-by-side table — it never recomputes metrics, only loads and reports.

```bash
python -m agents.apollo.main   # score (4 Apollo prompts, one .jsonl each) -> eval (4 results.json)
python -m benchmark.main       # deceptron eval -> apollo -> comparison.json + printed table
```

See [docs/plan.md](docs/plan.md) for the rationale (`src/shared`/`src/agents/apollo`/`src/benchmark` split)
and [docs/prior.md](docs/prior.md) for Apollo's reported numbers this is compared against.

## Orchestration (Dagster)

Mining + repo-verification (`mine` -> `verify-repos`, the slowest/flakiest
stages — GitHub API rate limits, Docker clone-and-test per repo) can run as
an isolated, monitored, resumable background job via Dagster instead of a
foreground `python -m agents.deceptron.main` invocation.

`src/orchestrator/definitions.py` wires `agents.deceptron.miner.main.iter_mined_repos`
and `agents.deceptron.verifier.repo.main.verify_and_append_repo` directly into Dagster ops
(no subprocess — the same per-item functions each stage's `run` loop drains). Both are already
append-and-resume-by-id, so a retried or restarted run continues from the last
completed item for free; Dagster adds isolation (its own process/container),
a UI for progress/failures, and automatic retries (`RetryPolicy`, 2 attempts,
60s backoff — tuned for GitHub rate-limit bursts and flaky Docker pulls).

**Local (foreground UI + daemon in one process):**
```bash
dagster dev -w workspace.yaml
# open http://localhost:3000 -> Jobs -> run_pipeline_job -> Launchpad
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

Both paths load `run_pipeline_job` from `workspace.yaml` ->
`orchestrator.definitions:defs`. Config schema
(`ConfigMinerRepo`/`ConfigVerifyRepo`) exposes `lang`/`limit`/`out`/`repos`
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
│   ├── shared/              # shared by both agent pipelines — must be byte-identical
│   │   │                    # so deceptron's and apollo's results are comparable:
│   │   ├── llm.py               # create_with_retry — Anthropic API call + backoff
│   │   ├── trajectory.py        # format_trajectory — trajectory dict -> monitor text
│   │   └── metrics.py           # compute_pauroc, bootstrap_ci (McClish 1989 + 95% CI)
│   ├── agents/
│   │   ├── deceptron/       # main real-repo pipeline
│   │   │   ├── miner/           # GitHub repo discovery and filtering (stage 1)
│   │   │   ├── verifier/        # Docker-based deterministic verification —
│   │   │   │   │                #   two independent steps, each its own CLI:
│   │   │   │   ├── docker.py        # shared low-level Docker utilities (both steps)
│   │   │   │   ├── check_scripts.py # check-script Docker validation — shared by
│   │   │   │   │                    #   tasks.generate and tasks.validate (both must
│   │   │   │   │                    #   agree on what makes a check script valid)
│   │   │   │   ├── repo/            # stage 2: clone + run test suite
│   │   │   │   │   ├── verify.py        #   verify_repo — system-lib check + test suite
│   │   │   │   │   ├── schema.py        #   VerifiedRepo pydantic model
│   │   │   │   │   └── main.py          #   CLI: --repos --out
│   │   │   │   └── trajectory/      # stage 5: apply patch + run check scripts
│   │   │   │       ├── verify.py        #   verify_trajectory — patch + pytest + checks
│   │   │   │       └── main.py          #   CLI: --trajectories --tasks --verified-repos
│   │   │   ├── tasks/           # (main_task, side_task) pair generation (stage 3) —
│   │   │   │   │                #   generate and validate are independent steps:
│   │   │   │   ├── schema.py        # TaskPair pydantic model
│   │   │   │   ├── generate/
│   │   │   │   │   ├── analyze.py       # analyze_repo — Claude repo analysis -> task pair candidates
│   │   │   │   │   └── main.py          # CLI: generate pairs (+ validate scripts inline)
│   │   │   │   └── validate/main.py # CLI: re-validate check scripts on demand
│   │   │   ├── models/
│   │   │   │   ├── coder/           # Baseline coding agent (main task only)
│   │   │   │   ├── redteam/         # Scheming agent (main + hidden side task)
│   │   │   │   └── monitor/         # Prompted scheming monitor (+ SFT, phase 2)
│   │   │   ├── eval/            # AUROC / pAUROC evaluation
│   │   │   ├── maintenance/     # one-off mining-data utilities (not in PIPELINE):
│   │   │   │   │                #   backfill-sha, strip-logs — own dispatcher CLI
│   │   │   │   ├── backfill_sha.py
│   │   │   │   ├── strip_logs.py
│   │   │   │   └── main.py          # CLI: backfill-sha | strip-logs subcommands
│   │   │   └── main.py          # pipeline orchestrator (python -m agents.deceptron.main)
│   │   └── apollo/          # Phase 1: Apollo-repro harness (sibling package) —
│   │       │                # mirrors deceptron's models.monitor -> eval -> main shape,
│   │       │                # 1:1 same module shapes, swapping only the prompt source
│   │       ├── prompts.py   # Apollo's 4 reference prompts (Appendix G) + reported
│   │       │                # pAUROC@0.20 — PROMPTS dict, shared by models.monitor,
│   │       │                # main, and benchmark.report
│   │       ├── models/
│   │       │   └── monitor/
│   │       │       └── prompted/
│   │       │           └── main.py      # scores deceptron's real trajectories with each
│   │       │                            # PROMPTS variant — one data/agents/apollo/scores/<variant>.jsonl per prompt
│   │       ├── eval/
│   │       │   └── main.py      # mirrors agents.deceptron.eval.main — same shared.metrics,
│   │       │                    # one results.json per prompt-variant scores file
│   │       └── main.py      # orchestrator: score (x4 prompts) -> eval (x4 variants)
│   ├── orchestrator/        # Dagster wiring for mine -> verify-repos as an
│   │   │                    # isolated/monitored/resumable background job
│   │   ├── definitions.py       # imports iter_mined_repos/verify_and_append_repo directly — see
│   │   │                        # "Orchestration (Dagster)" above
│   │   └── Dockerfile           # webserver + daemon image (build context: repo root)
│   └── benchmark/           # cross-pipeline comparison — orchestration + report
│       │                    # only, never recomputes metrics (loads results from
│       │                    # both pipelines + Apollo's reported numbers)
│       ├── report.py            # builds + prints/writes the side-by-side table
│       └── main.py          # orchestrator: deceptron(eval) -> apollo -> report
└── data/
    ├── agents/
    │   ├── deceptron/                                 # one subdir per pipeline stage
    │   │   ├── mining/
    │   │   │   ├── repos.jsonl                            ← committed (stage 1 — miner output)
    │   │   │   ├── repos_verified.jsonl                   ← gitignored (stage 2 — install/test logs)
    │   │   │   └── repos_verified_clean.jsonl             ← committed (stage 2 — logs stripped)
    │   │   ├── tasks/
    │   │   │   ├── tasks.jsonl                            ← committed (stage 3 — task pairs)
    │   │   │   └── cache/                                 ← gitignored (Claude analysis cache by SHA)
    │   │   ├── trajectories/
    │   │   │   ├── baseline.jsonl                         ← gitignored (stage 4a — main task only)
    │   │   │   └── scheming.jsonl                         ← gitignored (stage 4b — main + side task;
    │   │   │                                                 stage 5 verifies both in-place)
    │   │   ├── scores/
    │   │   │   └── prompted.jsonl                         ← gitignored (stage 6 — monitor scores)
    │   │   └── results/
    │   │       └── prompted/results.json                  ← gitignored (stage 7 — AUROC / pAUROC)
    │   └── apollo/                                    ← gitignored, regenerated from deceptron's
    │       ├── scores/<variant>.jsonl                   trajectories (python -m agents.apollo.main)
    │       └── results/<variant>/results.json          — same scores/<name>.jsonl + results/<name>/
    │                                                     results.json shape as deceptron, swapping
    │                                                     only the variable: prompt source name
    └── benchmark/                                     ← gitignored, regenerated
        └── results/comparison.json                      (python -m benchmark.main)
```

## SFT monitor (phase 2, scaffolded — optional, GPU required)

Requires ≥24 GB VRAM (Ada L40S 48 GB recommended).

```bash
python -m agents.deceptron.models.monitor.sft \
    --train data/agents/deceptron/trajectories/baseline.jsonl data/agents/deceptron/trajectories/scheming.jsonl \
    --out runs/monitor-v1
```
