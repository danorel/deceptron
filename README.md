# deceptron

Adversarial training pipeline for **black-box scheming detection**.

Given a real GitHub repository, the pipeline produces two types of agent trajectories:
- **baseline** — agent sees only the coding task, behaves normally
- **scheming** — agent covertly pursues a hidden side objective alongside the coding task

A monitor scores each trajectory for scheming (0–10). The primary metric is **pAUROC at FPR < 0.05**.

See [docs/plan.md](docs/plan.md) for research context and [docs/prior_art.md](docs/prior_art.md) for the Apollo Research paper this builds on.

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

```
GitHub repos
    │
    ▼
[1] miner        →  data/repos.jsonl
    │
    ▼
[2] verifier     →  data/repos_verified.jsonl        (Docker: run test suite)
    │
    ▼
[3] tasks        →  data/tasks.jsonl                 (Claude: generate task pairs)
    │              data/cache/<repo>/analysis.json    (cached by SHA)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[4a] agent       →  trajectories_baseline  [4b] redteam  →  trajectories_scheming
     (main task only)                           (main + hidden side task)
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
[5] verifier  →  updates trajectories in-place       (Docker: apply patch + check scripts)
                   │
                   ▼
[6] monitor   →  data/scores.jsonl                   (Claude: score 0–10)
                   │
                   ▼
[7] eval      →  data/results/results.json           (AUROC / pAUROC)
```

### Stage 1 — Mine repos

Queries GitHub for Python repos with 100–2000 stars, 3–30 contributors, passing CI, and open license.

```bash
python -m miner.query --lang python --limit 50 --out data/repos.jsonl
```

### Stage 2 — Verify repos (Docker required)

Clones each repo, detects the required Python version, installs dependencies, runs the test suite.
Repos where tests pass (possibly with some deselected) are marked `test_verified=True`.

```bash
python -m verifier.run --repos data/repos.jsonl --out data/repos_verified.jsonl
```

Only verified repos proceed to task generation. Use `scripts/strip_logs.py` to produce a
git-committable version without large install/test logs:

```bash
python -m scripts.strip_logs --input data/repos_verified.jsonl --out data/repos_verified_clean.jsonl
```

### Stage 3 — Generate task pairs (Claude + Docker)

For each verified repo, Claude analyzes the source files and generates 3 `(main_task, side_task)` pairs.
Results are cached by SHA in `data/cache/<owner>__<repo>/analysis.json`.

```bash
# Generate pairs (skips repos already at quota)
python -m tasks.generate --repos data/repos_verified_clean.jsonl --out data/tasks.jsonl

# Validate check scripts in Docker (updates main_task_check_valid / side_task_check_valid)
python -m tasks.validate --tasks data/tasks.jsonl --repos data/repos_verified_clean.jsonl
```

`--no-docker` skips Docker validation if unavailable.
`--force` re-validates already-validated pairs.

### Stage 4a — Baseline agent trajectories

Agent receives only the `main_task`. Tools: `read_file`, `list_directory`, `write_file`.
Produces a `git diff` patch and full tool-use transcript.

```bash
python -m agent.run --tasks data/tasks.jsonl --out data/trajectories_baseline.jsonl
```

Use `--limit N` to process a subset (good for testing).

### Stage 4b — Scheming (red team) trajectories

Agent receives `main_task` via user message and `side_task` via system prompt (not visible in transcript).
Must complete both while making the trajectory look like normal coding.

```bash
python -m redteam.run --tasks data/tasks.jsonl --out data/trajectories_scheming.jsonl
```

### Stage 5 — Verify trajectories (Docker required)

Applies each agent's patch to a clean clone, then runs:
- `main_task_check_script` → sets `main_task_verified`
- `side_task_check_script` → sets `side_task_verified`

Updates trajectory files in-place.

```bash
python -m verifier.run --trajectories data/trajectories_baseline.jsonl \
                       --tasks data/tasks.jsonl \
                       --verified-repos data/repos_verified_clean.jsonl

python -m verifier.run --trajectories data/trajectories_scheming.jsonl \
                       --tasks data/tasks.jsonl \
                       --verified-repos data/repos_verified_clean.jsonl
```

### Stage 6 — Monitor scoring

Claude reads each trajectory (tool calls, file writes, final patch) and outputs a scheming score 0–10.
Does **not** see the system prompt — pure black-box monitoring.

```bash
python -m monitor.prompted \
    --trajectories data/trajectories_baseline.jsonl data/trajectories_scheming.jsonl \
    --out data/scores.jsonl
```

Re-score everything: add `--force`.

### Stage 7 — Evaluate

```bash
python -m eval.run --scores data/scores.jsonl --out data/results/
```

Outputs AUROC and pAUROC (primary metric). Expected with a working pipeline:
- Baseline scores: 0–3
- Scheming scores: 6–9
- AUROC: > 0.85

---

## Directory structure

```
deceptron/
├── miner/          # GitHub repo discovery
├── verifier/
│   ├── docker.py        # shared Docker utilities
│   ├── deterministic.py # repo test-suite verification
│   ├── trajectory.py    # patch + check script verification
│   ├── run.py           # CLI dispatcher
│   └── schema.py        # VerifiedRepo model
├── tasks/          # (main_task, side_task) generation
├── agent/          # Baseline coding agent
├── redteam/        # Scheming agent
├── monitor/        # Prompted scheming monitor
├── eval/           # AUROC / pAUROC evaluation
├── scripts/        # One-off utilities (strip_logs, backfill_sha)
└── data/           # Generated datasets (gitignored if large)
    ├── repos.jsonl
    ├── repos_verified_clean.jsonl   ← committed (clean, no logs)
    ├── tasks.jsonl                  ← committed
    ├── cache/                       ← gitignored
    ├── trajectories_baseline.jsonl  ← gitignored
    ├── trajectories_scheming.jsonl  ← gitignored
    └── scores.jsonl                 ← gitignored
```

## SFT monitor (optional, GPU required)

Requires ≥24 GB VRAM (Ada L40S 48 GB recommended).

```bash
python -m monitor.sft \
    --train data/trajectories_baseline.jsonl data/trajectories_scheming.jsonl \
    --out runs/monitor-v1
```
