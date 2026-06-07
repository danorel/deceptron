# Reflection

## Hypothesis

Formed before running anything: deterministic ground truth on real GitHub repositories,
combined with parameter-level monitor training (LoRA SFT on Qwen2.5-7B), should improve
calibration on out-of-distribution scheming over a prompt-only monitor trained on synthetic
data (Apollo's STRIDE/Gloom).

Motivation: coding agents are moving into high-stakes environments — unmonitored deception
there causes security failures with no way to catch them after the fact.

Gaps in [docs/prior.md](prior.md) (Apollo, Constitutional Black-Box Monitoring) that motivate this:
- monitors trained on synthetic data saturate early; pushing past saturation overfits to the
  synthetic distribution rather than real agent behavior
- optimization is prompt-only — prompt sweeps, no parameter-level fine-tuning of the monitor
- ControlArena is the sole realistic benchmark, limited environment diversity

How deceptron addresses each:
- training data from real GitHub repos — distribution shift from real codebases is a feature,
  not synthetic noise to overfit
- monitor trained at parameter level (LoRA SFT on Qwen2.5-7B), not prompt-only
- deterministic ground truth (test suites + check scripts), target is an adversarial loop
  (GRPO red-team vs. monitor), not one static benchmark

## What's built (~80h, alongside full-time ML job)

End-to-end data pipeline (https://github.com/danorel/deceptron):
- mines GitHub Python repos (100–2000 stars, 3–30 contributors, passing CI, open license)
- verifies each repo in Docker: runs test suite, drops repos needing GUI/system libs that
  can't run headless
- generates `(main_task, side_task)` pairs per repo via LLM (Claude Haiku), validated against
  AST-extracted function names to block hallucinated targets
- generates baseline trajectories (main task only) and scheming trajectories (hidden side
  task via system prompt, excluded from transcript)
- verifies trajectories in Docker: applies patch, runs test suites + deterministic check
  scripts for main and side tasks
- scores trajectories with a prompted monitor, computes pAUROC@FPR<0.05

Current state: both agents and the monitor are **prompted**. LoRA SFT monitor and GRPO
red-team are scaffolded as next phase, **not yet run** — no eval numbers testing the core
hypothesis yet.

## Biggest surprise: ground truth is the bottleneck, not monitoring

Expected the monitor to be the hard part. It isn't — sourcing verifiable repos and
trajectories is.

- most public repos carry system-level deps that break in Docker without manual fixing
- few repos clear the filter (100–2000 stars, 3–30 contributors, passing CI, …)
- of those, many lack tests or ship broken ones
- each repo must be cloned and run in Docker before knowing whether tests even pass —
  failures and undocumented system deps are common
- **>70% of candidate repos dropped** at this stage

Reframe: the binding constraint is not monitoring, it's extracting clean ground truth.
Every valid trajectory is expensive — directly caps eval size.

## What I'd do differently: scope down to one variable first

Scoped too much before getting any result. Faster path: isolate the synthetic-data gap
alone — swap in real repos but keep Apollo's prompt-only monitor design, run that one
experiment, read the result. One variable vs. Apollo's design would have been faster,
more comparable, more reliable than building the full adversarial loop up front.

## Open questions

1. **Core**: does a parameter-level monitor improve over prompt-only, and does it
   generalize better to out-of-distribution scheming? No results yet — open.
2. **CoT blind spot** (surfaced after the architecture was already set, via Korbak et al.
   on CoT monitorability): deceptron operates on actions and outcomes only, ignores CoT
   entirely. Treated this as a clean design choice (outcomes are verifiable, reasoning
   isn't) — but it's a blind spot. Future version: add CoT as input, test whether it adds
   signal or degrades under red-team pressure.
