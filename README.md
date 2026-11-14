# deceptron

> **Status: Paused (August 2026).** Development is on hold after the
> data-generation stage, pending compute funding for the training phase.
> The data pipeline below is functional at small scale; the monitor-training
> and red-team phases were designed but not implemented.

Data pipeline for **black-box scheming detection** on real GitHub repositories, with
deterministic ground truth (native test suites + check scripts, no LLM judges).

Given a real GitHub repository, the pipeline produces two types of agent trajectories:
- **baseline** — agent sees only the coding task, behaves normally
- **scheming** — agent covertly pursues a hidden side objective alongside the coding task

A monitor scores each trajectory for scheming (1–10). The target metric is
**pAUROC at FPR < 0.05**.

See [docs/plan.md](docs/plan.md) for research context and [docs/prior.md](docs/prior.md) for the Apollo Research paper this builds on.

## What works

Each stage runs from its own CLI and has been run at small scale. There is no automated
test suite for the pipeline.

| Component | Entry point |
|---|---|
| GitHub repo mining (stars, contributors, CI, license filters) | `python -m agents.deceptron.miner.main` |
| Docker-based repo verification (clone + native test suite) | `python -m agents.deceptron.verifier.repo.main` |
| `(main_task, side_task)` pair generation + check-script validation in Docker | `python -m agents.deceptron.tasks.generate.main`, `python -m agents.deceptron.tasks.validate.main` |
| Baseline coding-agent trajectories | `python -m agents.deceptron.models.coder.main` |
| Red-team (scheming) agent trajectories | `python -m agents.deceptron.models.redteam.main` |
| Trajectory verification (apply patch + tests + check scripts in Docker) | `python -m agents.deceptron.verifier.trajectory.main` |
| Prompted (black-box) monitor baseline + AUROC/pAUROC evaluation, run at small scale | `python -m agents.deceptron.models.monitor.prompted.main`, `python -m agents.deceptron.eval.main` |
| End-to-end pipeline runner (all stages above, in order) | `python -m agents.deceptron.main` |
| Dagster job for mining + repo verification (isolated, resumable) | `dagster dev -w workspace.yaml` → `src/orchestrator/definitions.py` |
| Switchable LLM backend: Anthropic or an open model via vLLM ([LLM.md](LLM.md)) | `LLM_PROVIDER` in `.env` |

**Partial:** the Apollo-style harness (`python -m agents.apollo.main`) scores trajectories
with Apollo's 4 reference monitor prompts, and `python -m benchmark.main` builds the
side-by-side comparison report. Both run, but only on a ~10-trajectory pilot set.

## Designed, not implemented

- Parameter-level monitor: LoRA SFT on Qwen2.5-7B-Instruct
- GRPO red-team adversarial loop
- Evaluation at target scale (N≈200 balanced) not run

The design for these is in [docs/plan.md](docs/plan.md).

## Resuming

The next step is to scale the data to ~100 verified baseline + ~100 verified scheming
trajectories, then train and evaluate the LoRA SFT monitor against the prompted baseline.
That and the GRPO red-team loop need GPU compute.

Open to collaboration or funding. Contact details are on my
[GitHub profile](https://github.com/danorel).

---

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker (for repo verification and trajectory checking)
- Python 3.11 (uv fetches it automatically)

## Setup

```bash
# 1. Install all deps + editable package into .venv
uv sync

# 2. Configure secrets
cp .env.example .env
# Fill in: GITHUB_TOKEN, ANTHROPIC_API_KEY

# 3. Run anything via uv run
uv run python -m agents.deceptron.main --list
```

The `sft` extra in `pyproject.toml` (`uv sync --extra sft`) is not implemented: it
reserves dependencies for the planned SFT monitor, and no code uses them yet.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | yes | GitHub API token (read-only scope) |
| `ANTHROPIC_API_KEY` | yes (default backend) | For task generation, agents, and monitor |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `OPENAI_API_KEY` | open-model backend only | Switch all stages to an OpenAI-compatible endpoint (vLLM) — see [LLM.md](LLM.md) |
| `HF_TOKEN` | vLLM server only | For Qwen model download when serving with vLLM ([LLM.md](LLM.md)); SFT use not implemented |

---

## Running the pipeline

For per-stage commands, flags, the apollo/benchmark harnesses, and the full
directory layout, see **[DEVELOPER.md](DEVELOPER.md)**. Quick start:

```bash
python -m agents.deceptron.main   # full real-repo pipeline (mine -> ... -> monitor -> eval)
python -m agents.apollo.main      # partial: score trajectories with Apollo's 4 prompts
python -m benchmark.main          # partial: cross-pipeline comparison report
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
