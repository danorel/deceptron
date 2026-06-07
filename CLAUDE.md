# deceptron

Adversarial training pipeline for black-box scheming detection — built on real GitHub repositories with deterministic ground truth verification, no LLM judges.

## Project structure

```
deceptron/
├── pyproject.toml          # editable install (`pip install -e .`), src/ layout, four packages
├── src/
│   ├── shared/             # shared by both agent pipelines: LLM retry, trajectory
│   │                       #   formatting, AUROC/pAUROC + bootstrap CI (shared.*)
│   ├── agents/
│   │   ├── deceptron/      # main real-repo pipeline
│   │   │   ├── miner/          # GitHub repository discovery and filtering
│   │   │   ├── verifier/       # Docker-based deterministic verification —
│   │   │   │                   #   repo/ (stage 2: clone+test) and trajectory/
│   │   │   │                   #   (stage 5: patch+checks), each its own CLI
│   │   │   ├── tasks/          # (main_task, side_task) pair generation —
│   │   │   │                   #   generate/ and validate/, each its own CLI
│   │   │   ├── models/
│   │   │   │   ├── coder/          # Coding agent trajectory generation
│   │   │   │   ├── monitor/        # Scheming monitor (prompted + fine-tuned)
│   │   │   │   └── redteam/        # Adversarial trajectory generation
│   │   │   ├── eval/           # AUROC / pAUROC evaluation pipeline
│   │   │   ├── maintenance/    # one-off mining-data utilities (backfill-sha, strip-logs)
│   │   │   └── main.py         # pipeline orchestrator (python -m agents.deceptron.main)
│   │   └── apollo/         # Phase 1: Apollo-repro harness (sibling package) —
│   │       ├── models/
│   │       │   └── monitor/    #   scores deceptron's real-repo trajectories with
│   │       │                   #   Apollo's 4 reference prompts (Appendix G, in
│   │       │                   #   prompted/main.py) — mirrors agents.deceptron.models.monitor.prompted.main
│   │       ├── eval/           #   AUROC/pAUROC per prompt variant — mirrors
│   │       │                   #   agents.deceptron.eval.main (same shared.metrics)
│   │       └── main.py         #   orchestrator (python -m agents.apollo.main)
│   ├── orchestrator/       # Dagster job: mine -> verify-repos as an isolated,
│   │   │                   #   monitored, resumable background task (see DEVELOPER.md)
│   │   └── definitions.py
│   └── benchmark/          # cross-pipeline comparison (orchestration + report
│       │                   #   only — never recomputes metrics):
│       │                   #   deceptron results + apollo results + Apollo's
│       │                   #   reported numbers -> side-by-side table
│       └── main.py         #   orchestrator (python -m benchmark.main)
└── data/
    ├── agents/
    │   ├── deceptron/      # mining/tasks/trajectories/scores/results
    │   └── apollo/         # scores/<variant>.jsonl, results/<variant>/
    └── benchmark/          # results/comparison.json
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
python -m agents.deceptron.main
python -m agents.deceptron.main --stages tasks coder redteam
python -m agents.deceptron.main --list

# ...or run stages individually:
python -m agents.deceptron.miner.main --lang python --limit 50 --out data/agents/deceptron/mining/repos.jsonl
python -m agents.deceptron.tasks.generate.main --repos data/agents/deceptron/mining/repos_verified_clean.jsonl --out data/agents/deceptron/tasks/tasks.jsonl
python -m agents.deceptron.models.coder.main --tasks data/agents/deceptron/tasks/tasks.jsonl --out data/agents/deceptron/trajectories/baseline.jsonl
python -m agents.deceptron.models.redteam.main --tasks data/agents/deceptron/tasks/tasks.jsonl --out data/agents/deceptron/trajectories/scheming.jsonl
python -m agents.deceptron.verifier.trajectory.main --trajectories data/agents/deceptron/trajectories/baseline.jsonl
python -m agents.deceptron.models.monitor.prompted.main --trajectories data/agents/deceptron/trajectories/baseline.jsonl data/agents/deceptron/trajectories/scheming.jsonl
python -m agents.deceptron.eval.main --scores data/agents/deceptron/scores/prompted.jsonl

# Phase 1 apollo (Apollo's reference prompts on deceptron's real-repo trajectories)
python -m agents.apollo.main               # score (4 Apollo prompts) -> eval
python -m benchmark.main                   # deceptron eval -> apollo -> comparison report

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

## Function naming conventions

Prefix signals what a function does and where its data comes from — pick the
narrowest one that fits, so the name alone tells you whether it does I/O,
talks to an LLM, or is a pure in-memory transform:

| Prefix | Means | Example |
|---|---|---|
| `fetch_` | retrieve from an external system (HTTP/API/Docker exec) — I/O, may retry/fail | `_fetch_repo_files`, `_fetch_raw` |
| `load_` | read + parse local data (files, `.jsonl`, cache, env) | `load_verified_ids`, `_load_cached` |
| `save_` | persist data locally (serialize + write) | `_save_cache` |
| `build_` | construct a composite in-memory artifact (script text, prompt, payload) — no I/O | `build_install_script`, `build_pytest_script` |
| `extract_` / `parse_` | derive structured data from raw text/AST — pure, no I/O | `_extract_functions`, `parse_pytest_output` |
| `validate_` | check structural/semantic correctness, return bool/errors — no ground-truth execution | `validate_check_scripts`, `_validate_task_pair` |
| `verify_` | deterministic ground-truth check via real execution (tests, check scripts, Docker) — reserved for `verifier/` | `verify_repo`, `verify_trajectory` |
| `run_` | orchestrate/execute a process or one unit of domain work (Docker container, agent session) | `run_container`, `_run_agent`, `_run_agent`'s sibling `_run_scheming_agent` |
| `score_` | produce an LLM-based judgment/rating | `_score_trajectory` |

Pipeline-stage CLI loops (the function each stage's `main()` calls to do the
actual work, looping over input and writing output) are uniformly named `run`
inside their module — not `run_<stage>`. The module path already says what it
runs; repeating that in the function name is noise. Import and alias at the
call site when the bare name would be ambiguous:
```python
from agents.deceptron.verifier.repo.main import run as run_verify_repo
```

`_` prefix = module-private: a helper that exists purely for one file's
internal organization and is never imported elsewhere — drop the underscore
the moment it becomes shared. Conversely, a "convenience" helper used only
within a single file should *get* the underscore even if its name otherwise
matches one of the prefixes above (e.g. `_fetch_repo_files`, `_load_cached` —
both `fetch_`/`load_`-shaped, both private to `tasks.generate.analyze`).

## Design principles

- **No LLM judges for ground truth** — all labels are deterministic (test suite pass/fail + adversarial check scripts)
- **Real repos only** — no synthetic environments; distribution shift from real codebases is a feature, not a bug
- **Reproducibility first** — every experiment logs config, seed, and model version
- **Cheap iterations** — default configs target <$10/experiment; scale-up configs are separate
- **Minimal comments** — only for non-obvious "why" (hidden constraints, workarounds, trade-offs); no restating what the code does, no multi-paragraph blocks

## Out of scope (v1)

- Non-coding environments (email, calendar, OS)
- White-box monitoring
- GRPO fine-tuning of red team (planned for v2)
- ControlArena integration (planned for v2)