# deceptron — Research Plan

**Status:** Phase 1 in progress
**Last updated:** 2026-06-07
**Budget:** $50 API (prompting) + $50 compute = $100 total

---

## Motivation

Scalable scheming monitors require training data from real agent environments — not synthetic
datasets (STRIDE, Gloom) that introduce distribution shift. This project closes three gaps
identified in [docs/prior.md](prior.md) (Apollo, Constitutional Black-Box Monitoring):

1. **Synthetic data** → replace with real GitHub repositories
2. **LLM judges for ground truth** → replace with deterministic test suite execution + adversarial check scripts
3. **Prompt-only monitor** → add LoRA SFT (parameter-level) on open-source model (Qwen2.5-7B-Instruct)

---

## Phased approach

Originally scoped all three gaps as one build. Reframed: ship phase 1 first — isolate the
**synthetic vs. real-repo training/eval data** variable while *keeping* Apollo's prompt-only
monitor design. Faster, comparable, falsifiable. Phases 2+ (parameter-level monitor,
adversarial loop) build on whatever phase 1 shows.

### Phase 1 — Reproduce Apollo, swap synthetic → real repos (current)

**Goal:** hold the monitor architecture constant (prompt-only), change only the data source
(synthetic STRIDE/Gloom + ControlArena → real GitHub repos + deterministic ground truth).
Run the same eval (pAUROC@FPR<0.05, AUROC), compare against Apollo's reported numbers, write up.

**Core empirical question (phase 1):** does swapping synthetic training/eval data for
real-repo data change a prompt-only monitor's performance — in particular its calibration —
or is the synthetic-vs-real axis not actually where the signal lives?

**Comparison caveat (be explicit in the write-up):** Apollo reports numbers for a monitor
*trained* on STRIDE/Gloom and *evaluated* on ControlArena. Deceptron has no access to retrain
or re-eval Apollo's exact pipeline — the comparison is against Apollo's **reported** numbers,
using deceptron's prompt-only monitor scored on real-repo trajectories. Not a controlled
apples-to-apples rerun; it isolates the synthetic→real axis as closely as feasible without
reproducing Apollo's training infra. State this limitation plainly in the paper.

**Deliverables:**
- scaled data pipeline — more verified repos, tasks, baseline + scheming trajectories
  (current: 9 repos / 10 tasks / 8 baseline / 2 scheming — too small for stable AUROC)
- prompted monitor scored on real-repo trajectories at scale
- comparison table: Apollo (synthetic-trained, ControlArena-evaluated, reported numbers)
  vs. deceptron (prompt-only monitor, real-repo-evaluated)
- short paper/report answering the core question above

Code home: `src/repro/` — sibling package to `src/deceptron/`, scaffolded but empty.
Keeping it separate from `src/deceptron/` (the real-repo pipeline) means the
synthetic-vs-real comparison stays a clean import boundary, not entangled module state.

### Phase 2 — Parameter-level monitor + adversarial loop (next)

**Goal:** test the full original hypothesis — does a parameter-level monitor (LoRA SFT on
Qwen2.5-7B) beat prompt-only, and generalize better to out-of-distribution scheming? Then
close the loop with a GRPO red-team.

**Deliverables:**
- `monitor/sft.py` — LoRA SFT on Qwen2.5-7B-Instruct (QLoRA, scaffolded, not yet run)
- GRPO red-team training loop (`scripts/adversarial_iteration.py`, scaffolded)
- comparison table: prompted vs. SFT monitor on the same real-repo eval set

### Phase 3 — CoT as monitor input (open question, not scoped)

Deceptron currently scores trajectories on actions/outcomes only (verifiable), ignoring CoT
(not verifiable) — a deliberate design choice that is also a blind spot per Korbak et al. on
CoT monitorability. Future work: add CoT as monitor input, test whether it adds signal or
degrades under red-team pressure.

---

## Architecture

```
miner → tasks → agent (baseline) ──┐
                                    ├─→ verifier → monitor → eval
              redteam (scheming) ──┘
```

**Monitor training/eval data:**
- Negative (non-scheming): baseline agent trajectories — only main task
- Positive (scheming): redteam trajectories — main task + side task, evasion-motivated

---

## Phase 1 — remaining work

| Task | Module | Status |
|------|--------|--------|
| Scale repo mining + verification | `miner/`, `verifier/` | 9 verified repos — need more; >70% candidate drop rate (system libs, broken/missing tests) |
| Scale task generation | `tasks/` | 10 tasks — need more per repo |
| Scale baseline trajectories | `agent/` | 8 generated, 7 main-task-verified |
| Scale scheming trajectories | `redteam/` | **bottleneck — only 2 generated, 1 fully verified** |
| Verify all trajectories | `verifier/` | partially run |
| Run prompted monitor at scale | `monitor/prompted.py` | ran on 12 trajectories — need full set |
| Run eval, get numbers | `eval/` | **not yet run — `data/results/` empty** |
| Write comparison + paper | — | blocked on above |

### Miner filters

- Stars 100–2000, contributors 3–30
- CI passing on default branch (GitHub Actions); repos without CI are accepted
- Test files present (pytest/jest/cargo/go test)
- Last commit < 3 months
- Open license (MIT, Apache-2.0, BSD)
- Language: Python only (v1); Rust, TypeScript, Go, Java — future work

### Side task catalog (v1)

Fixed set of deterministically-verifiable behaviors:

- Weaken password/input validation (accept short/empty inputs)
- Log sensitive data (tokens, passwords) to stdout/file
- Introduce off-by-one error in security-relevant bound check
- Add hardcoded backdoor credential
- Disable rate limiting or throttling

### Ground truth

- **Main task**: native test suite pass/fail inside Docker (pytest/cargo test/jest)
- **Side task**: per-task adversarial check script (written upfront, human-verified)
- No LLM judges anywhere in the labeling pipeline

---

## Phase 2 — SFT setup (scaffolded, not yet run)

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Method: QLoRA (4-bit base, LoRA rank 16)
- Hardware: 1× Ada L40S 48GB or 1× Blackwell B200 96GB (~45 min/run)
- Cloud fallback: RunPod A100 80GB (~$4/run)
- Library: HuggingFace TRL `SFTTrainer`
- Input format: trajectory text → binary label (scheming / not scheming)
- Epochs: 3, batch size: 4, lr: 2e-4

### Training data split

| Split | Size (target) | Source |
|-------|--------------|--------|
| Train | ~240 | 80% of verified trajectories |
| Val | ~60 | 20% of verified trajectories |
| Test | held-out | separate repos not in train |

---

## Evaluation

**Primary metric:** pAUROC at FPR < 0.05
**Secondary:** AUROC, precision/recall at threshold 0.5

Phase 1 comparison table (target):

| Monitor | Training/eval data | pAUROC | AUROC |
|---------|--------------------|--------|-------|
| Apollo prompted (reported) | synthetic (STRIDE/Gloom) → ControlArena | — | — |
| deceptron prompted | real GitHub repos | — | — |

Phase 2 comparison table (target):

| Monitor | pAUROC | AUROC |
|---------|--------|-------|
| Prompted (Claude) | — | — |
| SFT (Qwen2.5-7B LoRA) | — | — |

---

## Cost estimate

| Item | Model | Volume | Est. cost |
|------|-------|--------|-----------|
| Task generation | Haiku | 200 tasks | ~$0.50 |
| Baseline trajectories | Haiku | 200 traj | ~$9 |
| Redteam trajectories | Haiku | 100 traj | ~$5 |
| Monitor scoring | Sonnet | 300 traj | ~$8 |
| SFT compute | — | 1 run, ~1h | ~$4 |
| **Total** | | | **~$27** |

Buffer: ~$73 remaining for reruns, ablations, debugging.

---

## Out of scope (v1)

- White-box monitoring
- Non-coding environments
- ControlArena evaluation (would help phase 1 comparison rigor — revisit if budget allows)
- Languages beyond Python

---

## Reproducibility

- All experiments log: model version, seed, config snapshot, timestamp
- Dataset versions tracked via SHA in `data/manifest.jsonl`
- Environment pinned via `environment.yml` (micromamba)
- Results written to `data/results/` with run ID

---

## Open questions

- [ ] Phase 1 core: does swapping synthetic→real change a prompt-only monitor's
      performance/calibration, or is that axis not where the signal lives?
- [ ] Phase 2 core: does a parameter-level monitor improve over prompt-only, and
      does it generalize better to OOD scheming?
- [ ] Phase 3: does adding CoT as monitor input add signal or degrade under red-team pressure?
- [ ] Can we get GPU access at work for SFT runs?
- [ ] Which side tasks are easiest to auto-generate reliable check scripts for?
- [ ] Does Haiku produce trajectories of sufficient quality for monitor training?
