"""Run the Phase 1 cross-pipeline comparison end to end (see docs/plan.md).

Calls deceptron and repro as black boxes — runs each pipeline's eval stage(s),
then builds the side-by-side comparison table from their independently-written
results files (never recomputes metrics itself, see compare.report):

    deceptron eval -> repro (score + eval) -> report

Usage:
    python -m compare.main                   # run all three stages
    python -m compare.main --stages report   # rerun report only (results already on disk)
    python -m compare.main --list            # list stage names
"""

from __future__ import annotations

import argparse
import subprocess
import sys

PY = sys.executable

PIPELINE: list[tuple[str, list[list[str]]]] = [
    ("deceptron", [
        [PY, "-m", "deceptron.main", "--stages", "eval"],
    ]),
    ("repro", [
        [PY, "-m", "repro.main"],
    ]),
    ("report", [
        [PY, "-m", "compare.report", "--out", "data/compare/results/"],
    ]),
]

STAGE_NAMES = [name for name, _ in PIPELINE]


def run_stage(name: str, commands: list[list[str]]) -> None:
    for cmd in commands:
        print(f"[{name}] $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 cross-pipeline comparison end to end",
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
