# deceptron

Adversarial training pipeline for black-box scheming detection — built on real GitHub repositories with deterministic ground truth verification, no LLM judges.

## Project structure

```
deceptron/
├── pyproject.toml          # editable install (`pip install -e .`), src/ layout, four packages
├── src/
│   ├── common/             # shared by deceptron + repro: LLM retry, trajectory
│   │                       #   formatting, AUROC/pAUROC + bootstrap CI (common.*)
│   ├── deceptron/          # main real-repo pipeline
│   │   ├── miner/          # GitHub repository discovery and filtering
│   │   ├── verifier/       # Docker-based deterministic ground truth verification
│   │   ├── tasks/          # (main_task, side_task) pair generation and storage
│   │   ├── agent/          # Coding agent trajectory generation
│   │   ├── monitor/        # Scheming monitor (prompted + fine-tuned)
│   │   ├── redteam/        # Adversarial trajectory generation
│   │   ├── eval/           # AUROC / pAUROC evaluation pipeline
│   │   ├── scripts/        # One-off experiment scripts
│   │   └── main.py         # pipeline orchestrator (python -m deceptron.main)
│   ├── orchestration/      # Dagster job: mine -> verify-repos as an isolated,
│   │   │                   #   monitored, resumable background task (see DEVELOPER.md)
│   │   └── definitions.py
│   ├── repro/              # Phase 1: Apollo-repro harness (sibling package) —
│   │   ├── monitor/        #   scores deceptron's real-repo trajectories with
│   │   │                   #   Apollo's 4 reference prompts (Appendix G, in
│   │   │                   #   prompted/main.py) — mirrors deceptron.monitor.prompted.main
│   │   ├── eval/           #   AUROC/pAUROC per prompt variant — mirrors
│   │   │                   #   deceptron.eval.main (same common.metrics)
│   │   └── main.py         #   orchestrator (python -m repro.main)
│   └── compare/            # cross-pipeline comparison (orchestration + report
│       │                   #   only — never recomputes metrics):
│       │                   #   deceptron results + repro results + Apollo's
│       │                   #   reported numbers -> side-by-side table
│       └── main.py         #   orchestrator (python -m compare.main)
└── data/
    ├── deceptron/          # mining/tasks/trajectories/monitor/results
    ├── repro/              # scores/<variant>.jsonl, results/<variant>/
    └── compare/            # results/comparison.json
```

## Key concepts

- **Main task**: legitimate coding task derived from a real GitHub repo
- **Side task**: adversarial behavior the agent must execute without being detected (e.g., weaken password validation, add logging of sensitive data)
- **Ground truth**: deterministic — native test suite (pytest/cargo test/jest/etc.) for main task completion; adversarial check scripts for side task execution
- **Monitor**: LLM that reads a full agent trajectory and outputs a scheming score (0–10)
- **Red team**: LLM prompted or fine-tuned to complete main task + side task while evading the monitor
- **Adversarial loop**: monitor and red team iteratively improve against each other

## Setup

```bash
CONDA_OVERRIDE_OSX=15.0 micromamba create -f environment.yml
micromamba activate deceptron
pip install -r requirements.txt
cp .env.example .env  # add GITHUB_TOKEN, ANTHROPIC_API_KEY

# GPU machine: reinstall torch with CUDA after pip install
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

## Common commands

```bash
# Run the full pipeline (or a subset) — see DEVELOPER.md for per-stage flags
python -m deceptron.main
python -m deceptron.main --stages tasks agent redteam
python -m deceptron.main --list

# ...or run stages individually:
python -m deceptron.miner.main --lang python --limit 50 --out data/deceptron/mining/repos.jsonl
python -m deceptron.tasks.generate --repos data/deceptron/mining/repos_verified_clean.jsonl --out data/deceptron/tasks/tasks.jsonl
python -m deceptron.agent.main --tasks data/deceptron/tasks/tasks.jsonl --out data/deceptron/trajectories/baseline.jsonl
python -m deceptron.redteam.main --tasks data/deceptron/tasks/tasks.jsonl --out data/deceptron/trajectories/scheming.jsonl
python -m deceptron.verifier.main --trajectories data/deceptron/trajectories/baseline.jsonl
python -m deceptron.monitor.prompted.main --trajectories data/deceptron/trajectories/baseline.jsonl data/deceptron/trajectories/scheming.jsonl
python -m deceptron.eval.main --scores data/deceptron/scores/prompted.jsonl

# Phase 1 repro (Apollo's reference prompts on deceptron's real-repo trajectories)
python -m repro.main                      # score (4 Apollo prompts) -> eval
python -m compare.main                     # deceptron eval -> repro -> comparison report

# Mining + verification as an isolated/monitored/resumable background job (Dagster)
dagster dev -w workspace.yaml              # local UI+daemon — see DEVELOPER.md
docker compose up -d                       # background, containerized — see DEVELOPER.md
```

## Environment variables

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | GitHub API token (read-only scope sufficient) |
| `ANTHROPIC_API_KEY` | For trajectory generation and prompted monitor |
| `OPENAI_API_KEY` | Optional, for GPT-4o baseline monitor |
| `HF_TOKEN` | For Qwen model access on HuggingFace |

## Evaluation metric

Primary: **pAUROC at FPR < 0.05** (partial area under ROC curve, low false-positive regime).
Secondary: AUROC.

Rationale: in production monitoring, false positives are costly — we care most about detector performance where FPR is low.

## Data conventions

All datasets are stored as `.jsonl` — one JSON object per line.

Trajectory schema:
```json
{
  "id": "uuid",
  "repo_url": "https://github.com/...",
  "language": "python",
  "main_task": "...",
  "side_task": "...",
  "steps": [...],
  "final_patch": "...",
  "main_task_verified": true,
  "side_task_verified": true,
  "label": "scheming"
}
```

## Design principles

- **No LLM judges for ground truth** — all labels are deterministic (test suite pass/fail + adversarial check scripts)
- **Real repos only** — no synthetic environments; distribution shift from real codebases is a feature, not a bug
- **Reproducibility first** — every experiment logs config, seed, and model version
- **Cheap iterations** — default configs target <$10/experiment; scale-up configs are separate

## Out of scope (v1)

- Non-coding environments (email, calendar, OS)
- White-box monitoring
- GRPO fine-tuning of red team (planned for v2)
- ControlArena integration (planned for v2)