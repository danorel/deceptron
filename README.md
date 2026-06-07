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
Apollo's reported numbers. The pipeline runs end-to-end at small scale; scaling up data
volume is the remaining work before a comparison can be drawn.

**Phase 2 (next):** parameter-level monitor (LoRA SFT on Qwen2.5-7B) + GRPO red-team
adversarial loop.

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
pip install -e .   # editable install (src/ layout, five packages: common/deceptron/orchestration/repro/compare)

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

## Running the pipeline

For per-stage commands, flags, the Phase 1 repro/comparison harnesses, and the full
directory layout, see **[DEVELOPER.md](DEVELOPER.md)**. Quick start:

```bash
python -m deceptron.main      # full real-repo pipeline (mine -> ... -> monitor -> eval)
python -m repro.main          # Phase 1 repro: score real trajectories with Apollo's prompts
python -m compare.main        # cross-pipeline comparison report
```

Mining + repo verification (slowest, flakiest stages) can run as an isolated,
monitored, resumable background job via Dagster instead — see "Orchestration
(Dagster)" in [DEVELOPER.md](DEVELOPER.md):

```bash
dagster dev -w workspace.yaml   # local UI + daemon -> http://localhost:3000
docker compose up -d            # background, containerized
```

## Architecture diagram

See [docs/architecture.drawio](docs/architecture.drawio) — open at [app.diagrams.net](https://app.diagrams.net) via File → Open from → Device.
