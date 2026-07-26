# Session recap — 2026-07-26 20:00 — open-weights generator + run_tests

## Done
- **LLM.md rewritten** (~half the length): open-weights vLLM backend, `Qwen3-30B-A3B`
  recommended, the two mandatory flags (`--tool-call-parser hermes`,
  `--reasoning-parser qwen3`) explained, security (SSH tunnel, not public IP).
- **uv-only migration:** deps moved into `pyproject.toml` (+ `sft` extra),
  removed `requirements.txt` / `environment.yml`, `README.md` + `CLAUDE.md` setup
  updated, `uv.lock` added.
- **Generator adopted: `Qwen3-30B-A3B`** — ~131 tok/s single-stream (~5× over
  dense 32B). Fixes: `AGENT_MAX_TOKENS=8192` + empty-turn retry (reasoning was
  truncating turns → empty patch); `ast` syntax gate on `edit_file`/`write_file`
  (col-0 indentation bug). Coder + redteam.
- **`run_tests` affordance (`--enable-tests`)**: sandboxed `ContainerTestSession`
  (mounts repo, installs once, `docker exec` pytest); Docker precheck **aborts**
  the run if the daemon is down (avoids the thrash-into-garbage failure mode).
  Symmetric across coder + redteam. Validated: clean baseline self-verified
  (43 passed, converged in 5 turns).
- **Conventions:** experiment-log format + first log
  (`docs/log/2026-07-26_1906_qwen3-30b-a3b-generator.md`); session-recap format.

## In progress
- `run_tests` loop validated on **one** clean baseline only — not yet used at bulk.
- Trajectory verifier on the 4-line smoke set: `main=3/4` (the ✗ is the stale
  404 junk line, empty patch), `side=0/4` (expected — baselines have no side task).

## Next
1. **Quality gate (proposed, NOT built):** ast code-clean check in the trajectory
   verifier — reject NEW duplicate defs/imports (delta vs HEAD) so thrashed
   patches fail validation, not just correctness. Proven to separate thrash
   (dup import/def) from clean. Plus cheap trajectory-hygiene flags (repeated
   identical edits, run_tests errors, hit max_turns).
2. **Bulk generate on the H100** with `--enable-tests` for BOTH baseline + redteam
   (symmetric scaffold), `--workers 12+` (KV headroom is large).
3. **Attack 132/136 invalid task check scripts** — the real eval-N bottleneck,
   independent of generation.
4. Run trajectory verify → monitor → eval end to end.

## Notes / blockers
- **Stale shell env caused a 404:** exported `LLM_MODEL=Qwen/Qwen3-32B` overrode
  `.env` (`load_dotenv` doesn't override). `unset LLM_MODEL LLM_PROVIDER LLM_BASE_URL OPENAI_API_KEY` before runs.
- **`run_tests` needs Docker on the generation host** → run on the H100 (Linux,
  native Docker, colocated with vLLM). macOS Docker Desktop won't bind-mount
  `/var/folders` tmpdirs → tests can't see the code.
- **`main_task_verified: true` ≠ clean trajectory** — thrashed patches (duplicate
  code) still pass the end-behavior check. Hence the quality gate in Next #1.
- **Use a fresh `--out` per smoke** — the writer appends; old junk lines accumulate.
- **Left untouched (not this session's work):** `docs/plan.md`, `docs/prior.md`,
  `docs/reflection.md` (pre-existing uncommitted edits) and untracked `.claude/`
  config — deliberately not committed.
