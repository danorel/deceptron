"""Build the Phase 1 comparison table: deceptron's prompted monitor (scored on
real-repo trajectories, deceptron's own prompt) vs. repro (Apollo's 4
reference prompts, scored on the *same* real-repo trajectories) vs. Apollo's
own reported numbers (synthetic-trained, ControlArena-evaluated).

Reads three independently-produced result sets — never recomputes metrics
itself — so the comparison can't silently diverge from what each pipeline
already reported (see common.metrics, used identically by agents.deceptron.eval.main
and agents.apollo.eval.main):
  - data/agents/deceptron/results/prompted/results.json              (agents.deceptron.eval.main output)
  - data/agents/apollo/results/<variant>/results.json  (agents.apollo.eval.main output, x4)
  - agents.apollo.prompts.PROMPTS                                       (Apollo's reported pAUROC@0.20, hardcoded from the paper)

Usage:
    python -m benchmark.report
    python -m benchmark.report --out data/benchmark/results/
"""

import argparse
import json
import sys
from pathlib import Path

from agents.apollo.prompts import PROMPTS


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deceptron-results", default="data/agents/deceptron/results/prompted/results.json",
                        help="agents.deceptron.eval.main output (deceptron's own prompted monitor)")
    parser.add_argument("--repro-results", default="data/agents/apollo/results",
                        help="directory of <variant>/results.json (agents.apollo.eval.main output)")
    parser.add_argument("--out", default=None, help="directory to write comparison.json (optional)")
    args = parser.parse_args()

    rows = []

    deceptron_results = _load_json(Path(args.deceptron_results))
    if deceptron_results:
        rows.append({
            "source": "deceptron",
            "monitor": "deceptron prompted (own prompt, real-repo data)",
            "n_total": deceptron_results["n_total"],
            "auroc": deceptron_results["auroc"],
            "pauroc_fpr05": deceptron_results["pauroc"],
            "pauroc_fpr20": deceptron_results["pauroc_apollo_cutoff"],
            "apollo_reported_pauroc_fpr20": None,
        })
    else:
        print(f"Warning: {args.deceptron_results} not found — run "
              f"`python -m agents.deceptron.main --stages eval` first", file=sys.stderr)

    repro_results = Path(args.repro_results)
    for variant, (_, reported_pauroc, opt_dataset) in PROMPTS.items():
        result = _load_json(repro_results / variant / "results.json")
        if not result:
            print(f"Warning: {repro_results / variant / 'results.json'} not found — "
                  f"run `python -m agents.apollo.main` first", file=sys.stderr)
            continue
        rows.append({
            "source": "repro",
            "monitor": f"repro / {variant} (Apollo prompt, real-repo data)",
            "n_total": result["n_total"],
            "auroc": result["auroc"],
            "pauroc_fpr05": result["pauroc"],
            "pauroc_fpr20": result["pauroc_apollo_cutoff"],
            "apollo_reported_pauroc_fpr20": reported_pauroc,
            "apollo_optimization_dataset": opt_dataset,
        })

    if not rows:
        sys.exit("No results found in either pipeline — nothing to compare")

    print(f"{'monitor':<48} {'n':>4}  {'AUROC':>8}  {'pAUROC@0.05':>12}  "
          f"{'pAUROC@0.20':>12}  {'Apollo reported':>15}")
    print("-" * 105)
    def _fmt(x):
        return f"{x:.3f}" if x is not None else "—"

    for r in rows:
        apollo = _fmt(r["apollo_reported_pauroc_fpr20"])
        print(f"{r['monitor']:<48} {r['n_total']:>4}  {_fmt(r['auroc']):>8}  "
              f"{_fmt(r['pauroc_fpr05']):>12}  {_fmt(r['pauroc_fpr20']):>12}  {apollo:>15}")

    print()
    print("-> 'pAUROC@0.20' is the column directly comparable to Apollo's reported numbers")
    print("   (their cutoff; deceptron's primary metric is pAUROC@0.05 — see docs/prior.md)")
    print("-> Apollo's in-distribution ceiling (real-ish data, optimized+evaluated together): "
          "0.81 [0.79, 0.83]")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "comparison.json"
        out_file.write_text(json.dumps(rows, indent=2))
        print(f"\nComparison written to {out_file}")


if __name__ == "__main__":
    main()
