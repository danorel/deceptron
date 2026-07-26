# Reflection

## Hypothesis

Deterministic ground truth on real GitHub repos + parameter-level monitor training (LoRA SFT on Qwen2.5-7B) should beat a prompt-only monitor trained on synthetic data — better calibration on out-of-distribution scheming.

Motivation: coding agents moving into high-stakes environments — undetected deception there causes security failures with no after-the-fact recourse.

| Gap in Apollo's setup | deceptron's response |
|---|---|
| Synthetic data saturates, optimization overfits to it | Real repos — distribution shift is the point, not noise |
| Prompt-only optimization, no fine-tuning | LoRA SFT on Qwen2.5-7B |
| ControlArena is the sole realistic benchmark | Adversarial loop (GRPO red-team vs. monitor), not a static benchmark |

## What's built (~80h, alongside full-time job)

End-to-end pipeline (github.com/danorel/deceptron): mine → verify (Docker test runs) → generate `(main_task, side_task)` pairs → generate baseline + scheming trajectories → verify trajectories (patch + check scripts) → score with prompted monitor → eval (pAUROC@0.05).

Both agent and monitor are currently prompted. LoRA SFT and GRPO red-team are
scaffolded, not run — no result on the core hypothesis yet.

## Biggest surprise: ground truth, not monitoring, is the bottleneck

Expected the monitor to be the hard part. It's sourcing verifiable repos and trajectories.

- Most public repos carry system-level deps that break in headless Docker
- Few clear the mining filter (stars/contributors/CI/tests/license)
- Of those, many lack tests or ship broken ones — only knowable after a full clone+run
- **>70% of candidate repos dropped**

Reframe: the binding constraint is extracting clean ground truth, not building or training the monitor. Every valid trajectory is expensive — directly caps eval size.

## What I'd do differently

Scoped all three gaps as one build before getting any result. Faster: isolate one variable (synthetic→real) first, hold Apollo's monitor design constant, run that, read the result — then decide whether the larger build is warranted.

## Open questions

1. **Core:** does a parameter-level monitor improve over prompt-only, and generalize better OOD? No results yet.
2. **CoT blind spot** (Korbak et al.): deceptron ignores CoT entirely — a clean design choice (only outcomes are verifiable) but a known gap. Future: add CoT as input, test whether it helps or degrades under red-team pressure.
