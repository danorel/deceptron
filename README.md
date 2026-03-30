# deceptron

Adversarial training pipeline for black-box scheming detection.
See [docs/plan.md](docs/plan.md) for research context and timeline.

## Requirements

- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- Docker (for verifier)
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
# Edit .env and fill in: GITHUB_TOKEN, ANTHROPIC_API_KEY, HF_TOKEN
```

## Pipeline

Run each step in order. All outputs are `.jsonl` files in `data/`.

```bash
# Step 1 — Mine GitHub repos
python -m miner.query --lang python --limit 50 --out data/repos.jsonl

# Step 2 — Generate (main_task, side_task) pairs
python -m tasks.generate --repos data/repos.jsonl --out data/tasks.jsonl

# Step 3a — Baseline agent trajectories (main task only)
python -m agent.run --tasks data/tasks.jsonl --out data/trajectories_baseline.jsonl

# Step 3b — Red team trajectories (main task + side task)
python -m redteam.run --tasks data/tasks.jsonl --out data/trajectories_scheming.jsonl

# Step 4 — Verify trajectories (Docker required)
python -m verifier.run --trajectories data/trajectories_baseline.jsonl
python -m verifier.run --trajectories data/trajectories_scheming.jsonl

# Step 5 — Score with monitor
python -m monitor.prompted --trajectories data/trajectories_baseline.jsonl data/trajectories_scheming.jsonl --out data/scores.jsonl

# Step 6 — Evaluate
python -m eval.run --scores data/scores.jsonl --out data/results/

# Full adversarial iteration (wraps steps 3b–6)
python -m scripts.adversarial_iteration --iter 1
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | yes | GitHub API token (read-only scope) |
| `ANTHROPIC_API_KEY` | yes | For agent, redteam, and prompted monitor |
| `HF_TOKEN` | SFT only | For Qwen model access on HuggingFace |
| `OPENAI_API_KEY` | no | GPT-4o baseline monitor (optional) |

## SFT monitor (optional)

Requires a GPU with ≥24 GB VRAM (Ada L40S 48 GB recommended).

```bash
python -m monitor.sft --train data/trajectories_baseline.jsonl data/trajectories_scheming.jsonl --out runs/monitor-v1
```
