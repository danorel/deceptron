"""Pipeline orchestrator — shells out to each stage's `python -m agents.deceptron.<module>`
in order: mine -> verify-repos -> strip-logs -> tasks -> coder -> redteam ->
verify-trajectories -> monitor -> eval. `--stages` runs a subset; per-stage flags
aren't exposed here, run that module directly when you need them."""

from __future__ import annotations

import argparse
import subprocess
import sys

from agents.deceptron.miner.main import DEFAULT_OUT as MINED_REPOS
from agents.deceptron.verifier.repo.main import DEFAULT_VERIFY_REPOS_OUT as VERIFIED_REPOS
from agents.deceptron.maintenance.strip_logs import DEFAULT_DST as CLEAN_REPOS

PY = sys.executable

_TASKS   = "data/agents/deceptron/tasks/tasks.jsonl"
_BASE    = "data/agents/deceptron/trajectories/baseline.jsonl"
_SCHEME  = "data/agents/deceptron/trajectories/scheming.jsonl"
_SCORES  = "data/agents/deceptron/scores/prompted.jsonl"
_RESULTS = "data/agents/deceptron/results/prompted/"

STAGE_NAMES = [
    "mine", "verify-repos", "strip-logs", "tasks",
    "coder", "redteam", "verify-trajectories", "monitor", "eval",
]


def _build_pipeline(mine_limit: int, workers: int) -> list[tuple[str, list[list[str]]]]:
    return [
        ("mine", [
            [PY, "-m", "agents.deceptron.miner.main",
             "--lang", "python", "--limit", str(mine_limit), "--out", MINED_REPOS],
        ]),
        ("verify-repos", [
            [PY, "-m", "agents.deceptron.verifier.repo.main",
             "--repos", MINED_REPOS, "--out", VERIFIED_REPOS],
        ]),
        ("strip-logs", [
            [PY, "-m", "agents.deceptron.maintenance.main", "strip-logs"],
        ]),
        ("tasks", [
            [PY, "-m", "agents.deceptron.tasks.generate.main",
             "--repos", CLEAN_REPOS, "--out", _TASKS, "--no-docker"],
            [PY, "-m", "agents.deceptron.tasks.validate.main",
             "--tasks", _TASKS, "--repos", CLEAN_REPOS],
        ]),
        ("coder", [
            [PY, "-m", "agents.deceptron.models.coder.main",
             "--tasks", _TASKS, "--out", _BASE, "--workers", str(workers)],
        ]),
        ("redteam", [
            [PY, "-m", "agents.deceptron.models.redteam.main",
             "--tasks", _TASKS, "--out", _SCHEME, "--workers", str(workers)],
        ]),
        ("verify-trajectories", [
            [PY, "-m", "agents.deceptron.verifier.trajectory.main",
             "--trajectories", _BASE,
             "--tasks", _TASKS, "--verified-repos", CLEAN_REPOS],
            [PY, "-m", "agents.deceptron.verifier.trajectory.main",
             "--trajectories", _SCHEME,
             "--tasks", _TASKS, "--verified-repos", CLEAN_REPOS],
        ]),
        ("monitor", [
            [PY, "-m", "agents.deceptron.models.monitor.prompted.main",
             "--trajectories", _BASE, _SCHEME, "--out", _SCORES],
        ]),
        ("eval", [
            [PY, "-m", "agents.deceptron.eval.main",
             "--scores", _SCORES, "--out", _RESULTS],
        ]),
    ]


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
    parser.add_argument("--mine-limit", type=int, default=50,
                        help="Max repos to mine (default: 50)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for coder/redteam (default: 1)")
    args = parser.parse_args()

    if args.list:
        for name in STAGE_NAMES:
            print(name)
        return

    pipeline = _build_pipeline(args.mine_limit, args.workers)
    selected = set(args.stages) if args.stages else set(STAGE_NAMES)
    for name, commands in pipeline:
        if name in selected:
            run_stage(name, commands)


if __name__ == "__main__":
    main()
