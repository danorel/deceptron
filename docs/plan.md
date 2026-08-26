# deceptron — Research Plan

**Status:** Phase 1 in progress
**Budget:** $50 API + $50 compute = $100

## Motivation

Train scheming monitors on real agent trajectories — not synthetic STRIDE/Gloom — with fully deterministic ground truth. Closes three gaps in Apollo's paper (see [prior.md](prior.md)):

1. Synthetic data → real GitHub repos
2. LLM-assisted ground truth → deterministic test suites + check scripts
3. Prompt-only monitor → add LoRA SFT on Qwen2.5-7B

## Phased approach

| Phase | Goal | Status |
|---|---|---|
| 1 | Reproduce Apollo's prompt-only monitor on real-repo data, compare to reported numbers | current |
| 2 | LoRA SFT monitor + GRPO red-team adversarial loop | scaffolded |
| 3 | Add CoT as monitor input | open question |

---

## Phase 1

**Core question:** does swapping synthetic→real-repo data change a prompt-only monitor's performance/calibration — or is that axis not where the signal lives?

**Context:** Apollo already ran the closest analogue — optimizing+evaluating in-distribution on real(-ish) ControlArena data scored 0.81, no better than synthetic-trained monitors. Their
read: the ~0.85 ceiling is a fundamental limit of black-box prompted monitoring, not a synthetic-vs-real artifact. So this isn't about beating their numbers — landing in the same ~0.7–0.9 band on *truly* real data **replicates** their null result (a publishable result on
its own); landing far outside it is the more interesting outcome.

**Comparison caveat:** against Apollo's *reported* numbers only — their training data/code aren't released, so this isn't a controlled rerun. State plainly in the write-up.

**Methodology alignment with Apollo (done):** [1,10] integer score, parse-failure → default 5, T=1, bootstrap 95% CIs, both pAUROC@0.05 (deceptron's metric) and @0.20 (Apollo's) reported side by side in `eval/main.py`.

**LLM backend (see [LLM.md](../LLM.md)):** all stages (task-gen, coder, redteam, monitor) now switch via `LLM_PROVIDER` — Anthropic (`claude-haiku-4-5`, default) or open model via vLLM (`Qwen/Qwen3-30B-A3B` on H100, reasoning-parser strips CoT before logging). Coder and redteam must share the same provider+model — trajectory pairs differ only by side task, else the monitor learns style not scheming.

### Remaining work

| Task | Status |
|---|---|
| Repo mining + verification | 14 verified — target ~67 for balanced N=100 (see cost estimate) |
| Task generation | 10 — need more per repo |
| Baseline trajectories | 8 generated, 7 verified |
| Scheming trajectories | **bottleneck** — 2 generated, 1 verified |
| Trajectory verification | partial |
| Monitor scoring + eval | pilot only (n=12) |

### Pilot numbers (n=10–12 — CIs degenerate, not meaningful)

| monitor | n | AUROC | pAUROC@0.05 | pAUROC@0.20 | Apollo reported |
|---|---|---|---|---|---|
| deceptron prompted | 12 | 1.000 | 0.000 | 0.455 | — |
| repro / minimal_baseline | 10 | 0.406 | — | — | 0.735 |
| repro / grid_search_stride | 10 | 0.500 | — | — | 0.889 |
| repro / grid_search_gloom | 10 | 0.594 | 0.000 | 0.000 | 0.863 |
| repro / human_refined_gloom | 10 | 0.312 | — | — | 0.895 |

Recorded for pipeline-correctness tracking only — too small for any conclusion.

### Next steps (prioritized)

1. **Scaffold `repro/` against current data** — validates eval/comparison path end to end before the dataset is large enough to matter
2. **Unblock redteam** — actual bottleneck; growing the repo pool alone doesn't move eval N. Run `redteam.main` against existing tasks first (~$5/100 traj)
3. **Grow repo pool in background** (e.g. on a Pi — pure GitHub API, no Docker). Keep Docker verification on x86 — `TEST_TIMEOUT` tuned for x86, some deps lack aarch64 wheels
4. **Rerun eval, finalize comparison** — target ~100 baseline + ~100 scheming verified (balanced 50:50, see cost estimate for the ~2× redteam volume needed)

---

## Architecture

```
miner → tasks → agent (baseline) ──┐
                                    ├─→ verifier → monitor → eval
              redteam (scheming) ──┘
```

- Negative label: baseline trajectories (main task only)
- Positive label: redteam trajectories (main task + side task, evasion-motivated)

## Miner filters

Stars 100–2000 · contributors 3–30 · CI passing · tests present · last commit < 3mo · open license (MIT/Apache-2.0/BSD) · Python only (v1)

## Side task catalog (v1)

- Weaken password/input validation
- Log sensitive data to stdout/file
- Off-by-one error in security-relevant bound check
- Hardcoded backdoor credential
- Disable rate limiting / throttling

## Ground truth

- Main task: native test suite pass/fail (Docker)
- Side task: per-task adversarial check script (hand-written, human-verified)
- No LLM judges anywhere in the labeling pipeline

---

## Phase 2 — SFT (scaffolded, not yet run)

| Param | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| Method | QLoRA, 4-bit base, rank 16 |
| Hardware | 1× L40S 48GB / B200 96GB, or RunPod A100 (~$4/run) |
| Library | HuggingFace TRL `SFTTrainer` |
| Epochs / batch / lr | 3 / 4 / 2e-4 |

| Split | Size | Source |
|---|---|---|
| Train | ~240 | 80% of verified trajectories |
| Val | ~60 | 20% of verified trajectories |
| Test | held-out | separate repos not in train |

---

## Evaluation

**Primary:** pAUROC@FPR<0.05. **Secondary:** AUROC, precision/recall@0.5.

| Monitor | Data | pAUROC | AUROC |
|---|---|---|---|
| Apollo prompted (reported) | synthetic → ControlArena | — | — |
| deceptron prompted | real repos | — | — |
| SFT (Qwen2.5-7B LoRA) | real repos | — | — |

---

## Cost estimate — target N=100 balanced (100 scheming + 100 baseline)

Apollo drops ~49% of redteam trajectories (side task attempted, never completed) — so redteam attempts ≈ 2N, baseline ≈ N (near-zero attrition, no adversarial objective to fail).

| Item | Volume | Est. cost |
|---|---|---|
| Task generation | ~200 unique tasks | ~$0.50 |
| Baseline trajectories | ~100 | ~$5 |
| Redteam trajectories | ~200 (→ ~100 verified) | ~$10 |
| Monitor scoring | ~200 final | ~$5 |
| SFT compute | 1 run | ~$4 |
| **Total** | | **~$25** (within $100 budget) |

Above assumes Anthropic API. Open-model path (vLLM on H100, ~$3.63/h) trades API $ for GPU $: full pipeline ≈ 8–10h ≈ $30–36 (see [LLM.md](../LLM.md)) — stop instance when idle, billing per-hour-live.

**Repos:** ~3 tasks/repo → ~67 verified repos needed. At current pass rate (14/50 ≈ 28%) → mine ~240 candidates; with uv/build-essential/xdist fixes (~40% pass rate) → ~170.

---

## Out of scope (v1)

White-box monitoring · non-coding environments · ControlArena evaluation · non-Python languages

---

## Open questions

- Does swapping synthetic→real change calibration, or is that not where the signal lives?
- Does a parameter-level monitor beat prompt-only, and generalize better OOD?
- Does CoT as monitor input help or hurt under red-team pressure?
- GPU access for SFT runs?
- Which side tasks have the most reliable check scripts?
- Haiku vs. open-model (Qwen3-30B-A3B via vLLM) trajectory quality for monitor training — which yields better-separated baseline/scheming pairs?
