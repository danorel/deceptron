# Prior Work

## Constitutional Black-Box Monitoring for Scheming in LLM Agents

**Paper:** https://arxiv.org/abs/2603.00829
**Authors:** Storf, Barton-Cooper, Peters-Gill, Hobbhahn (Apollo Research / MATS) — ICML 2026

Studies prompted LLM monitors that detect scheming from observable agent behavior only (no chain-of-thought) — trained on synthetic data, evaluated on semi-realistic ControlArena trajectories.

**Scheming definition:** pursuing misaligned objectives *while* concealing them — both required; only clear violations (data exfiltration, sabotage, covert resource hijacking) counted.

---

## Datasets

| Dataset | Role | Type | N | Notes |
|---|---|---|---|---|
| STRIDE | Train/Val | Synthetic | 1,000 | LLM-generated, iteratively refined for realism |
| Gloom | Train/Val | Synthetic | 1,000 | Agent w/ hidden objective + simulated environment |
| ControlArena | Test | Semi-realistic | 7,500 | Real sandbox envs, 3 agent models × 5 settings |

All balanced 50:50 positive/negative. N≈1,000 chosen empirically — past that point pAUROC confidence intervals stop overlapping and monitor rankings stabilize.

---

## Ground truth methodology

- Side task assigned out-of-band, **stripped from the transcript** before scoring
- Completion checked via deterministic environment flags — **no LLM judges**
- Runs that fail to complete the side task are **excluded** (~49% of redteam trajectories)
— no reliable way to distinguish "tried and failed" from "never tried"

---

## Key results

**Score format** (drives deceptron's [1,10] design choice):

| Format | pAUROC |
|---|---|
| Binary, 1 sample | 0.624 |
| Binary, 5 samples averaged | 0.808 |
| **[1,10] integer** | **0.876** |
| [1,100] integer | 0.886 |

[1,10] chosen — near-equal to [1,100], far cheaper; deceptron already uses this format.

**Optimization saturates fast** (Fig 1):

| Strategy | Test pAUROC |
|---|---|
| No optimization | 0.724 |
| Prompt sweep (light) | **0.853** |
| Automated/GEPA (heavy) | 0.800 |
| SFT, open-model pilot (Kimi K2.6) | 0.907 — tied with prompt-sweep best |

Heavier optimization **overfits** to synthetic-distribution artifacts rather than learning robust real-world indicators.

**In-distribution ceiling check (§6)** — central pivot for deceptron: training+evaluating *directly* on real(-ish) ControlArena data → **pAUROC = 0.81 [0.79, 0.83]**, no better than
synthetic-trained-and-transferred monitors (0.85–0.91). The ~0.85 ceiling looks like a fundamental limit of prompted black-box monitoring — not a synthetic-vs-real artifact.

---

## Reference prompts (Appendix G — usable directly in `repro/`)

| Name | Test pAUROC |
|---|---|
| Minimal baseline | 0.735 |
| Grid-search (STRIDE) `eg_v3` | 0.889 |
| Grid-search (Gloom) `sonnet4.5_1` | 0.863 |
| Human-refined (Gloom) `minimal_sonnet_simon_v8` | **0.895** |

No single dominant prompt–model pair; recommends pairing any of the above with Claude Sonnet 4.5.

---

## What successful/failed monitors key on (qualitative, §6)

- **Catches:** unjustified code complexity, stated-vs-actual behavior mismatches, suspicious special-casing (backdoor-like), self-incriminating "blabbing" in comments/identifiers
- **Misses:** "weaponized transparency" (discloses suspicious actions framed as benign), scheming disguised as "helpful overreach"

---

## Open-source SFT pilot (Kimi K2.6, §5.4)

Run because closed-model fine-tuning APIs reject scheming-themed data via content screening. Same base model, same training pool, four optimization strategies — **all converge to ~0.90–0.91 pAUROC, including SFT**. Confirms the ceiling isn't an artifact of closed-model prompt optimization specifically.

LoRA recipe (relevant to deceptron's Phase 2): rank 32, all linear modules, LR 5e-7 — higher LRs caused mode collapse (predictions degenerate to score 1 or 10 within ~80 steps).

---

## What deceptron changes

| Gap | Apollo's approach | deceptron's approach |
|---|---|---|
| Synthetic data | STRIDE/Gloom + semi-realistic ControlArena | Real GitHub repos — true distribution shift |
| Ground truth | Deterministic via "privileged judge" LLMs (STRIDE/Gloom) / env flags (ControlArena) | Pure: native test suite + hand-written check scripts, **zero LLM in labeling** |
| Monitor optimization | Prompt sweeps; SFT only on open model (closed-model APIs block scheming data) | Plans LoRA SFT on Qwen2.5-7B — sidesteps the same content-screening issue |
| Benchmark | ControlArena only (5 settings) | Arbitrary real repos — broader, costlier to curate |
| Data release | Withheld (dual-use) | Open question — same tension applies |

---

## Critical constraint carried over

Side task must **never** appear in the trajectory text — agent told separately
(system prompt / out-of-band), monitor must detect it from behavior alone:
`agent/` (baseline) gets only `main_task`; `redteam/` gets both but logs only `main_task`.

---

## Metric note

Apollo integrates pAUROC over FPR ∈ [0, 0.20]; deceptron's primary cutoff is FPR < 0.05. Not directly comparable — `eval/main.py` reports both side by side so the Apollo column is always computed at Apollo's own cutoff.

---

## Code / data availability

- STRIDE, Gloom, monitor-optimization code: **not released** (dual-use risk)
- ControlArena: open source — github.com/UKGovernmentBEIS/control-arena
- Bloom (Gloom's parent framework): open source — github.com/safety-research/bloom
- Reference prompts (Appendix G) are the only directly-reusable artifact
