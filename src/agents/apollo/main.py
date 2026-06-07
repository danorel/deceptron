"""Phase 1 apollo orchestrator — mirrors `agents.deceptron.main`'s shape: score
(Apollo's 4 reference prompts, one data/agents/apollo/scores/<variant>.jsonl each)
-> eval (one results.json per variant). Cross-pipeline comparison lives in
`src/benchmark/` — apollo only produces its own numbers."""

from __future__ import annotations

import argparse
import subprocess
import sys

from agents.apollo.prompts import PROMPTS

PY = sys.executable

PIPELINE: list[tuple[str, list[list[str]]]] = [
    ("score", [
        [PY, "-m", "agents.apollo.models.monitor.prompted.main",
         "--trajectories", "data/agents/deceptron/trajectories/baseline.jsonl", "data/agents/deceptron/trajectories/scheming.jsonl",
         "--out-dir", "data/agents/apollo/scores"],
    ]),
    ("eval", [
        [PY, "-m", "agents.apollo.eval.main",
         "--scores", f"data/agents/apollo/scores/{variant}.jsonl",
         "--out", f"data/agents/apollo/results/{variant}/"]
        for variant in PROMPTS
    ]),
]

STAGE_NAMES = [name for name, _ in PIPELINE]


def run_stage(name: str, commands: list[list[str]]) -> None:
    for cmd in commands:
        print(f"[{name}] $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the apollo pipeline end to end",
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
