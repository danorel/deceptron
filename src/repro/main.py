"""Run the Phase 1 repro pipeline end to end (see docs/plan.md).

Mirrors `deceptron.main`'s shape — stages chained in pipeline order, each
shelling out to `python -m repro.<module>`:

    score (Apollo's 4 reference prompts) -> eval (per prompt variant)

`score` writes one data/repro/scores/<variant>.jsonl per Apollo reference
prompt; `eval` runs `repro.eval.main` once per variant file, writing
data/repro/results/<variant>/results.json. Final cross-pipeline comparison
against Apollo's reported numbers and deceptron's own results lives in
`src/compare/` (kept separate — repro itself only produces its own numbers).

Usage:
    python -m repro.main                  # run both stages
    python -m repro.main --stages eval     # rerun eval only (e.g. after manual rescoring)
    python -m repro.main --list           # list stage names
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from repro.monitor.prompted.main import PROMPTS

PY = sys.executable

PIPELINE: list[tuple[str, list[list[str]]]] = [
    ("score", [
        [PY, "-m", "repro.monitor.prompted.main",
         "--trajectories", "data/deceptron/trajectories/baseline.jsonl", "data/deceptron/trajectories/scheming.jsonl",
         "--out-dir", "data/repro/scores"],
    ]),
    ("eval", [
        [PY, "-m", "repro.eval.main",
         "--scores", f"data/repro/scores/{variant}.jsonl",
         "--out", f"data/repro/results/{variant}/"]
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
        description="Run the repro pipeline end to end",
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
