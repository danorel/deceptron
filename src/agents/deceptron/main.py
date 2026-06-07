"""Pipeline orchestrator — shells out to each stage's `python -m agents.deceptron.<module>`
in order: mine -> verify-repos -> tasks -> coder -> redteam -> verify-trajectories
-> monitor -> eval. `--stages` runs a subset; per-stage flags aren't exposed here,
run that module directly when you need them."""

from __future__ import annotations

import argparse
import subprocess
import sys

from agents.deceptron.miner.main import DEFAULT_OUT as MINED_REPOS
from agents.deceptron.verifier.repo.main import DEFAULT_VERIFY_REPOS_OUT as VERIFIED_REPOS

PY = sys.executable

PIPELINE: list[tuple[str, list[list[str]]]] = [
    ("mine", [
        [PY, "-m", "agents.deceptron.miner.main",
         "--lang", "python", "--limit", "50", "--out", MINED_REPOS],
    ]),
    ("verify-repos", [
        [PY, "-m", "agents.deceptron.verifier.repo.main",
         "--repos", MINED_REPOS,
         "--out", VERIFIED_REPOS],
    ]),
    ("tasks", [
        [PY, "-m", "agents.deceptron.tasks.generate.main",
         "--repos", "data/agents/deceptron/mining/repos_verified_clean.jsonl",
         "--out", "data/agents/deceptron/tasks/tasks.jsonl"],
        [PY, "-m", "agents.deceptron.tasks.validate.main",
         "--tasks", "data/agents/deceptron/tasks/tasks.jsonl",
         "--repos", "data/agents/deceptron/mining/repos_verified_clean.jsonl"],
    ]),
    ("coder", [
        [PY, "-m", "agents.deceptron.models.coder.main",
         "--tasks", "data/agents/deceptron/tasks/tasks.jsonl",
         "--out", "data/agents/deceptron/trajectories/baseline.jsonl"],
    ]),
    ("redteam", [
        [PY, "-m", "agents.deceptron.models.redteam.main",
         "--tasks", "data/agents/deceptron/tasks/tasks.jsonl",
         "--out", "data/agents/deceptron/trajectories/scheming.jsonl"],
    ]),
    ("verify-trajectories", [
        [PY, "-m", "agents.deceptron.verifier.trajectory.main",
         "--trajectories", "data/agents/deceptron/trajectories/baseline.jsonl",
         "--tasks", "data/agents/deceptron/tasks/tasks.jsonl",
         "--verified-repos", "data/agents/deceptron/mining/repos_verified_clean.jsonl"],
        [PY, "-m", "agents.deceptron.verifier.trajectory.main",
         "--trajectories", "data/agents/deceptron/trajectories/scheming.jsonl",
         "--tasks", "data/agents/deceptron/tasks/tasks.jsonl",
         "--verified-repos", "data/agents/deceptron/mining/repos_verified_clean.jsonl"],
    ]),
    ("monitor", [
        [PY, "-m", "agents.deceptron.models.monitor.prompted.main",
         "--trajectories", "data/agents/deceptron/trajectories/baseline.jsonl", "data/agents/deceptron/trajectories/scheming.jsonl",
         "--out", "data/agents/deceptron/scores/prompted.jsonl"],
    ]),
    ("eval", [
        [PY, "-m", "agents.deceptron.eval.main",
         "--scores", "data/agents/deceptron/scores/prompted.jsonl",
         "--out", "data/agents/deceptron/results/prompted/"],
    ]),
]

STAGE_NAMES = [name for name, _ in PIPELINE]


def run_stage(name: str, commands: list[list[str]]) -> None:
    for cmd in commands:
        print(f"[{name}] $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deceptron pipeline end to end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stages", nargs="+", choices=STAGE_NAMES, metavar="STAGE",
        help=f"run only these stages, in pipeline order (default: all). choices: {', '.join(STAGE_NAMES)}",
    )
    parser.add_argument("--list", action="store_true", help="list stage names and exit")
    args = parser.parse_args()

    if args.list:
        for name in STAGE_NAMES:
            print(name)
        return

    selected = set(args.stages) if args.stages else set(STAGE_NAMES)
    for name, commands in PIPELINE:
        if name in selected:
            run_stage(name, commands)


if __name__ == "__main__":
    main()
