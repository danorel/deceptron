"""Run the deceptron pipeline end to end (see README.md / docs/plan.md).

Each stage shells out to `python -m deceptron.<module>` using the data/
layout's default paths, in pipeline order:

    mine -> verify-repos -> tasks -> agent -> redteam
         -> verify-trajectories -> monitor -> eval

Usage:
    python -m deceptron.main                       # run the full pipeline
    python -m deceptron.main --stages tasks agent  # run a subset, in pipeline order
    python -m deceptron.main --list                # list stage names

Per-stage flags (--limit, --force, --no-docker, ...) aren't exposed here —
run that stage's module directly when you need them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

PY = sys.executable

PIPELINE: list[tuple[str, list[list[str]]]] = [
    ("mine", [
        [PY, "-m", "deceptron.miner.query",
         "--lang", "python", "--limit", "50", "--out", "data/mining/repos.jsonl"],
    ]),
    ("verify-repos", [
        [PY, "-m", "deceptron.verifier.run",
         "--repos", "data/mining/repos.jsonl",
         "--out", "data/mining/repos_verified.jsonl"],
    ]),
    ("tasks", [
        [PY, "-m", "deceptron.tasks.generate",
         "--repos", "data/mining/repos_verified_clean.jsonl",
         "--out", "data/tasks/tasks.jsonl"],
        [PY, "-m", "deceptron.tasks.validate",
         "--tasks", "data/tasks/tasks.jsonl",
         "--repos", "data/mining/repos_verified_clean.jsonl"],
    ]),
    ("agent", [
        [PY, "-m", "deceptron.agent.run",
         "--tasks", "data/tasks/tasks.jsonl",
         "--out", "data/trajectories/baseline.jsonl"],
    ]),
    ("redteam", [
        [PY, "-m", "deceptron.redteam.run",
         "--tasks", "data/tasks/tasks.jsonl",
         "--out", "data/trajectories/scheming.jsonl"],
    ]),
    ("verify-trajectories", [
        [PY, "-m", "deceptron.verifier.run",
         "--trajectories", "data/trajectories/baseline.jsonl",
         "--tasks", "data/tasks/tasks.jsonl",
         "--verified-repos", "data/mining/repos_verified_clean.jsonl"],
        [PY, "-m", "deceptron.verifier.run",
         "--trajectories", "data/trajectories/scheming.jsonl",
         "--tasks", "data/tasks/tasks.jsonl",
         "--verified-repos", "data/mining/repos_verified_clean.jsonl"],
    ]),
    ("monitor", [
        [PY, "-m", "deceptron.monitor.prompted",
         "--trajectories", "data/trajectories/baseline.jsonl", "data/trajectories/scheming.jsonl",
         "--out", "data/monitor/scores.jsonl"],
    ]),
    ("eval", [
        [PY, "-m", "deceptron.eval.run",
         "--scores", "data/monitor/scores.jsonl",
         "--out", "data/results/"],
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
