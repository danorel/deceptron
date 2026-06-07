# deceptron

Adversarial training pipeline for **black-box scheming detection**.

Given a real GitHub repository, the pipeline produces two types of agent trajectories:
- **baseline** — agent sees only the coding task, behaves normally
- **scheming** — agent covertly pursues a hidden side objective alongside the coding task

A monitor scores each trajectory for scheming (0–10). The primary metric is **pAUROC at FPR < 0.05**.

See [docs/plan.md](docs/plan.md) for research context and [docs/prior.md](docs/prior.md) for the Apollo Research paper this builds on.

## Status

**Phase 1 (current):** reproduce Apollo's prompt-only monitor design, swap synthetic
training/eval data (STRIDE/Gloom/ControlArena) for real-repo trajectories, compare against
Apollo's reported numbers. Pipeline below runs end-to-end at small scale; scaling up data
volume is the remaining work before a comparison can be drawn.

**Phase 2 (next):** parameter-level monitor (LoRA SFT on Qwen2.5-7B) + GRPO red-team
adversarial loop — see [SFT monitor](#sft-monitor-optional-gpu-required) below.

Details and open questions: [docs/plan.md](docs/plan.md).

---

## Requirements

- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- Docker (for repo verification and trajectory checking)
- Python 3.11

## Setup

```bash
# 1. Create environment
CONDA_OVERRIDE_OSX=15.0 micromamba create -f environment.yml   # macOS
micromamba create -f environment.yml                            # Linux

# 2. Activate and install pip packages
micromamba activate deceptron
pip install -r requirements.txt
pip install -e .   # editable install of the `deceptron` package (src/ layout)

# 3. GPU machine only — reinstall torch with CUDA
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Configure secrets
cp .env.example .env
# Fill in: GITHUB_TOKEN, ANTHROPIC_API_KEY
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | yes | GitHub API token (read-only scope) |
| `ANTHROPIC_API_KEY` | yes | For task generation, agents, and monitor |
| `HF_TOKEN` | SFT only | For Qwen model access on HuggingFace |

---

## Pipeline

The pipeline runs in 6 stages. Each stage reads from and writes to `.jsonl` files in `data/`.
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
[1] miner        →  data/mining/repos.jsonl
    │
    ▼
[2] verifier     →  data/mining/repos_verified.jsonl        (Docker: run test suite)
    │                                                  Automatically excludes repos that
    │                                                  require GUI/system libs (Qt, VLC…)
    ▼
[3] tasks        →  data/tasks/tasks.jsonl                 (Claude: generate task pairs)
    │              data/tasks/cache/<repo>/analysis.json    (cached by SHA)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[4a] agent       →  data/trajectories/baseline.jsonl  [4b] redteam  →  data/trajectories/scheming.jsonl
     (main task only)                                       (main + hidden side task)
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
[5] verifier  →  updates trajectories in-place       (Docker: apply patch + pytest + check scripts)
                   │
                   ▼
[6] monitor   →  data/monitor/scores.jsonl                   (Claude: score 0–10)
                   │
                   ▼
[7] eval      →  data/results/results.json           (AUROC / pAUROC)
```

### Stage 1 — Mine repos

Queries GitHub for Python repos with 100–2000 stars, 3–30 contributors, passing CI, and open license.

```bash
python -m deceptron.miner.query --lang python --limit 50 --out data/mining/repos.jsonl
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
python -m deceptron.verifier.run --repos data/mining/repos.jsonl --out data/mining/repos_verified.jsonl
```

Produce a git-committable version without large install/test logs:

```bash
python -m deceptron.scripts.strip_logs --input data/mining/repos_verified.jsonl --out data/mining/repos_verified_clean.jsonl
```

Only repos with `test_verified=True` **and** `system_libs=[]` proceed to task generation.

### Stage 3 — Generate task pairs (Claude + Docker)

For each verified repo, Claude analyzes the source files and generates 3 `(main_task, side_task)` pairs.
Results are cached by SHA in `data/tasks/cache/<owner>__<repo>/analysis.json` — re-running is free.

Claude is instructed to select only files with **pure Python logic** (no GUI imports at module level).
Generated function names are validated against real AST-extracted names to prevent hallucinations.

```bash
# Generate pairs (skips repos already at quota)
python -m deceptron.tasks.generate --repos data/mining/repos_verified_clean.jsonl --out data/tasks/tasks.jsonl

# Validate check scripts in Docker (updates main_task_check_valid / side_task_check_valid)
python -m deceptron.tasks.validate --tasks data/tasks/tasks.jsonl --repos data/mining/repos_verified_clean.jsonl
```

Flags: `--no-docker` skips Docker validation; `--force` re-validates all pairs.

### Stage 4a — Baseline agent trajectories

Agent receives only the `main_task`. Tools: `read_file`, `list_directory`, `write_file`.
Produces a `git diff` patch and a full tool-use transcript as the trajectory.

```bash
python -m deceptron.agent.run --tasks data/tasks/tasks.jsonl --out data/trajectories/baseline.jsonl
```

- `--limit N` — process N tasks (useful for testing)
- `--delay N` — seconds between tasks to stay under API rate limits (default: 12)

Already-processed `task_id`s are skipped. Only trajectories with actual steps/patches count as done
(failed-with-error trajectories are retried automatically).

### Stage 4b — Scheming (red team) trajectories

Agent receives `main_task` via user message and `side_task` via system prompt (not visible in the
transcript). Must complete both while making the trajectory look like normal coding.

```bash
python -m deceptron.redteam.run --tasks data/tasks/tasks.jsonl --out data/trajectories/scheming.jsonl
```

Same `--limit` and `--delay` flags as `agent.run`.

### Stage 5 — Verify trajectories (Docker required)

Applies each agent's patch to a clean clone, then in a single container:
1. Runs `pytest` with `deselect_args` → `tests_passing` (existing tests must stay green)
2. Runs `main_task_check_script` → `main_task_verified` (True only if tests pass **and** check passes)
3. Runs `side_task_check_script` → `side_task_verified`

Updates trajectory files in-place. `verify_log` is saved for debugging.

```bash
python -m deceptron.verifier.run --trajectories data/trajectories/baseline.jsonl \
                       --tasks data/tasks/tasks.jsonl \
                       --verified-repos data/mining/repos_verified_clean.jsonl

python -m deceptron.verifier.run --trajectories data/trajectories/scheming.jsonl \
                       --tasks data/tasks/tasks.jsonl \
                       --verified-repos data/mining/repos_verified_clean.jsonl
```

### Stage 6 — Monitor scoring

Claude reads each trajectory (tool calls, file writes, final patch — **not** the system prompt)
and outputs a scheming score 0–10. This is pure black-box monitoring.

```bash
python -m deceptron.monitor.prompted \
    --trajectories data/trajectories/baseline.jsonl data/trajectories/scheming.jsonl \
    --out data/monitor/scores.jsonl
```

Add `--force` to re-score everything.

### Stage 7 — Evaluate

```bash
python -m deceptron.eval.run --scores data/monitor/scores.jsonl --out data/results/
```

Outputs AUROC and pAUROC (primary metric). With a working pipeline expect:
- Baseline scores: 0–3
- Scheming scores: 6–9
- AUROC > 0.85

---

## Directory structure

```
deceptron/
├── pyproject.toml           # editable install: `pip install -e .` (src/ layout, two packages)
├── src/
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
│   │   └── scripts/         # Utilities: strip_logs, backfill_sha
│   └── repro/            # Phase 1: Apollo-repro scaffold (sibling package) —
│                            # prompt-only monitor evaluated on real-repo trajectories,
│                            # compared against Apollo's reported synthetic-trained
│                            # (STRIDE/Gloom → ControlArena) numbers — see plan.md
└── data/                                          # one subdir per pipeline phase
    ├── mining/
    │   ├── repos.jsonl                            ← committed (stage 1 — miner output)
    │   ├── repos_verified.jsonl                   ← gitignored (stage 2 — has install/test logs)
    │   └── repos_verified_clean.jsonl             ← committed (stage 2 — logs stripped)
    ├── tasks/
    │   ├── tasks.jsonl                            ← committed (stage 3 — task pairs)
    │   └── cache/                                 ← gitignored (Claude analysis cache by SHA)
    ├── trajectories/
    │   ├── baseline.jsonl                         ← gitignored (stage 4a — main task only)
    │   └── scheming.jsonl                         ← gitignored (stage 4b — main + side task;
    │                                                 stage 5 verifies both in-place)
    ├── monitor/
    │   └── scores.jsonl                           ← gitignored (stage 6 — monitor scores)
    └── results/                                   ← gitignored (stage 7 — AUROC / pAUROC)
```

## Architecture diagram

See [docs/architecture.drawio](docs/architecture.drawio) — open at [app.diagrams.net](https://app.diagrams.net) via File → Open from → Device.

## SFT monitor (phase 2, scaffolded — optional, GPU required)

Requires ≥24 GB VRAM (Ada L40S 48 GB recommended).

```bash
python -m deceptron.monitor.sft \
    --train data/trajectories/baseline.jsonl data/trajectories/scheming.jsonl \
    --out runs/monitor-v1
```
