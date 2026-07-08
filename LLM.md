# LLM Backend Setup

The pipeline supports two backends, switched via `LLM_PROVIDER` in `.env`.
All stages (task-gen, coder, redteam, monitor) respect the same env vars.

---

## Option A — Anthropic (default)

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
# LLM_MODEL defaults to claude-haiku-4-5-20251001 per stage
```

No extra setup. Prompt caching enabled automatically.

---

## Option B — OpenAI-compatible (vLLM on Sesterce / any GPU cloud)

### 1. Pick an instance on Sesterce

| Instance | VRAM | Cost | Fits |
|---|---|---|---|
| L40S 2× | 96 GB | $1.94/h | Qwen2.5-Coder-32B fp16 ✓ (recommended) |
| A100 80G 1× | 80 GB | $1.49/h | Qwen2.5-Coder-32B int8 only |
| L40S 1× | 48 GB | $0.97/h | Qwen2.5-Coder-32B int4 only (quality loss) |

**Recommendation:** `L40S 2×` — Qwen2.5-Coder-32B in fp16 without quantization,
`--tensor-parallel-size 2`, good throughput for parallel agent calls.

### 2. Deploy vLLM on the instance

SSH into the instance, then:

```bash
pip install vllm

# Qwen2.5-Coder-32B — replace HF_TOKEN with your token
HF_TOKEN=hf_... vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8000
```

For smaller instances with int4 quantization:

```bash
HF_TOKEN=hf_... vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
  --quantization awq \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --host 0.0.0.0 \
  --port 8000
```

vLLM exposes an OpenAI-compatible API at `http://<host>:8000/v1`.
No API key required — vLLM does not validate it.

### 3. Configure the pipeline

```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://<sesterce-public-ip>:8000/v1
LLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
OPENAI_API_KEY=none
```

> **Note:** `OPENAI_API_KEY` must be set to any non-empty string — the openai
> library requires it even though vLLM ignores its value.

### 4. Verify the connection

```bash
curl http://<host>:8000/v1/models | python3 -m json.tool
# Should list Qwen/Qwen2.5-Coder-32B-Instruct

# Or via Python:
python3 -c "
import os; os.environ.update({
    'LLM_PROVIDER': 'openai',
    'LLM_BASE_URL': 'http://<host>:8000/v1',
    'LLM_MODEL': 'Qwen/Qwen2.5-Coder-32B-Instruct',
    'OPENAI_API_KEY': 'none',
})
import sys; sys.path.insert(0, 'src')
from shared.llm import make_client
client = make_client()
resp = client.chat(
    model='Qwen/Qwen2.5-Coder-32B-Instruct',
    system='You are a helpful assistant.',
    messages=[{'role': 'user', 'content': 'Reply with: OK'}],
    max_tokens=8,
)
print(resp.content)
"
```

### 5. Run the pipeline

```bash
# Full pipeline (uses LLM_MODEL for all stages)
python -m agents.deceptron.main --stages tasks coder redteam verify-trajectories monitor eval

# Or stages individually
python -m agents.deceptron.tasks.generate.main --repos data/agents/deceptron/mining/repos_verified_clean.jsonl
python -m agents.deceptron.models.coder.main --tasks data/agents/deceptron/tasks/tasks.jsonl
python -m agents.deceptron.models.redteam.main --tasks data/agents/deceptron/tasks/tasks.jsonl
python -m agents.deceptron.models.monitor.prompted.main \
    --trajectories data/agents/deceptron/trajectories/baseline.jsonl \
                   data/agents/deceptron/trajectories/scheming.jsonl
```

---

## Per-stage model override

`LLM_MODEL` is a global override for all stages. If you need different models
per stage, set `LLM_MODEL` before each command:

```bash
# Stronger model for task generation, lighter for coder/redteam
LLM_MODEL=Qwen/Qwen2.5-72B-Instruct \
    python -m agents.deceptron.tasks.generate.main ...

LLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
    python -m agents.deceptron.models.coder.main ...
```

---

## Model selection — benchmarks and tradeoffs

### TL;DR recommendation for this pipeline

| Stage | Recommended | Why |
|---|---|---|
| task-gen | `Qwen/Qwen2.5-72B-Instruct` | Complex JSON, instruction following |
| coder | `Qwen/Qwen2.5-72B-Instruct` | Agentic tool use (see note below) |
| redteam | `Qwen/Qwen2.5-72B-Instruct` | Same |
| monitor | `Qwen/Qwen2.5-72B-Instruct` | Reasoning quality matters for scoring |

> **Why not Qwen2.5-Coder-32B for agentic tasks?**
> "Coder" models are fine-tuned on isolated problems (Codeforces, LeetCode, short
> snippets) — not on multi-turn agentic loops with tool calling. In practice they
> underperform general-purpose 70B+ models on agentic benchmarks like SWE-bench.
> Use Coder variants only for single-shot code generation, not for agent loops.

### Benchmark comparison

| Model | HumanEval | LiveCodeBench | SWE-bench Verified | Agentic tool use |
|---|---|---|---|---|
| **Claude Haiku 4.5** (current pipeline) | ~88% | mid-pack | **73.3%** | strong |
| **Qwen2.5-Coder-32B-Instruct** | 92.7% | 69.6 | ~40–50% agentic | ⚠ struggles |
| **Qwen2.5-72B-Instruct** | ~87% | competitive | comparable to Haiku 4.5 | good |
| **Qwen3-32B** (newer) | 95%+ | ~70% | ~70% | better than 2.5 |
| GPT-4o | 90%+ | high | 49% (no scaffold) | strong |
| Claude Sonnet 4.6 | 93%+ | high | 72%+ | very strong |

**Key observations:**
- HumanEval is saturated — every frontier model scores 88–95%. Not useful for
  differentiating models on agentic tasks.
- LiveCodeBench (post-cutoff competitive problems) is more reliable than HumanEval
  because it cannot be memorized.
- SWE-bench Verified with agentic scaffolding is the best proxy for this pipeline —
  it tests multi-turn file editing with tool use, identical to what coder/redteam do.
- Qwen2.5-Coder-32B falls to **Tier D** in practical agentic benchmarks for
  "inventing APIs or hanging" — a real risk in our multi-turn agent loop.
- Qwen3 models (released 2025) significantly improve agentic performance over 2.5.

### If you want to save VRAM

Acceptable quality-cost tradeoffs on L40S 1× (48GB):

| Model | VRAM (fp16) | Notes |
|---|---|---|
| `Qwen/Qwen3-32B` | ~64GB → needs int8 (~32GB) | Better agentic than Qwen2.5-Coder-32B |
| `Qwen/Qwen2.5-32B-Instruct` | ~64GB → int8 (~32GB) | General, decent agentic |
| `Qwen/Qwen2.5-14B-Instruct` | ~28GB fp16 | Fits on 1×L40S, quality drop noticeable |

All models available on HuggingFace — set `HF_TOKEN` on the vLLM server if the
model is gated.

---

## Cost estimate (Sesterce L40S 2× at $1.94/h)

| What | Time | Cost |
|---|---|---|
| 136 task pairs (task-gen, cached) | ~1h | ~$2 |
| 136 baseline trajectories (coder) | ~3-4h | ~$6–8 |
| 136 scheming trajectories (redteam) | ~3-4h | ~$6–8 |
| 272 monitor scores | ~1h | ~$2 |
| **Full pipeline** | **~8-10h** | **~$16–20** |

Compared to ~$14 on Anthropic Haiku for coder+redteam alone (without monitor).
With $2000 in Sesterce credits: ~100 full pipeline runs.

---

## Switching back to Anthropic

```env
LLM_PROVIDER=anthropic
# Remove or comment out LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY
```
