# LLM Backend Setup

Two backends, switched via `LLM_PROVIDER` in `.env`. All stages (task-gen,
coder, redteam, monitor) share the same env vars.

## Option A — Anthropic (default)

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
# LLM_MODEL defaults to claude-haiku-4-5-20251001 per stage
```

No extra setup. Prompt caching automatic.

## Option B — Open model via vLLM (Sesterce / any GPU cloud)

The pipeline's `OpenAILLMClient` (`src/shared/llm.py`) already speaks the OpenAI
wire format. Use the shipped `vllm/vllm-openai` image; don't swap in SGLang/TGI.

**Recommended model:** `Qwen/Qwen3-30B-A3B` — MoE reasoning model, ~3B active
params/token, fits one H100 80GB fp16. A reasoning model is the right pick: it
schemes covertly in CoT and acts cleanly, and `--reasoning-parser` strips that
CoT from the logged trajectory — mirroring a genuinely dangerous model. The MoE
decodes ~5× faster single-stream than dense `Qwen3-32B` (only ~3B params read per
token vs 32B) at similar quality — measured iter 1→2, see `docs/log/`.

Alternative — `Qwen/Qwen3-32B` (dense): marginally stronger, but ~26 tok/s
single-stream (memory-bandwidth bound). Use only if 30B-A3B quality proves
insufficient. Either way, throughput for bulk comes from `--workers` batching,
not single-stream speed.

### Serve

The Sesterce H100 image runs vLLM on boot via root-owned `~/init.sh` — `cat` it
first to avoid a port-8000 collision (its default is a tiny `Qwen3-0.6B`; kill it:
`docker rm -f $(docker ps -q)`). To launch explicitly:

```bash
docker run -d --name vllm --runtime nvidia --gpus all --net=host --ipc=host \
  -e HF_TOKEN=hf_... \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-30B-A3B \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 --port 8000
```

**The three flags are mandatory:**
- `--enable-auto-tool-choice --tool-call-parser hermes` — without it vLLM never
  returns `tool_calls`; the agent dispatches nothing and every `final_patch` is
  empty. (`hermes` = Qwen's parser.)
- `--reasoning-parser qwen3` — routes CoT into `reasoning_content`, which the
  client discards. Enforces the "side task never in the transcript" constraint
  (docs/prior.md). Without it, `<think>…plan…</think>` leaks into the log and the
  monitor sees the scheme.

⚠ Reasoning + tool-calling is fragile in vLLM — smoke-test that `tool_calls`
still parse (see Verify) before any bulk run.

Weights ≈ 60 GB fp16 → ample KV headroom on 80 GB → push `--workers` high
(12–16); vLLM batches, aggregate throughput scales ~linearly. Watch KV cache % in
the logs, back off near 90%.

### Configure + connect

```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8000/v1   # via tunnel; see Security
LLM_MODEL=Qwen/Qwen3-30B-A3B
OPENAI_API_KEY=none                      # any non-empty string; vLLM ignores it
```

**Security:** `--net=host` exposes port 8000 with no auth — anyone who finds the
IP burns your credits. Don't use the public IP; tunnel instead:
```bash
ssh -N -L 8000:localhost:8000 sesterce@<host>
```

### Verify (also the tool-call smoke test)

```bash
curl http://localhost:8000/v1/models   # lists Qwen/Qwen3-30B-A3B
python -c "
import sys; sys.path.insert(0,'src')
from shared.llm import make_client
c = make_client()
r = c.chat(model='Qwen/Qwen3-30B-A3B', system='You are helpful.',
           messages=[{'role':'user','content':'Reply with: OK'}], max_tokens=8)
print(r.content)"
```

## Per-stage model override

`LLM_MODEL` overrides all stages globally. Set it inline for a one-stage change:

```bash
LLM_MODEL=Qwen/Qwen3-30B-A3B python -m agents.deceptron.models.coder.main ...
```

> **Coder and redteam MUST use the same model + scaffold.** Baseline (negative)
> and scheming (positive) trajectories differ *only* by the side task. Different
> models/scaffolds → the monitor learns style, not scheming → invalid eval.

## Cost (H100 1× @ ~$3.63/h)

Full pipeline (~136 pairs: task-gen + baseline + scheming + scoring) ≈ 8–10 h ≈
$30–36. Tunnel down / stop the instance when idle — billing is per-hour-live.

## Switch back to Anthropic

```env
LLM_PROVIDER=anthropic
# remove LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY
```
