# Session 2026-11-14 — mark project paused, align docs with code

## Done
- Inventory of implemented vs planned components (entry points smoke-tested via `--help`, Dagster defs load, `docker compose config` valid). No pipeline test suite exists.
- README: paused status (Aug 2026), "What works" / "Designed, not implemented" / "Resuming"; `sft` extra marked not implemented.
- All docs aligned with the code: CLAUDE.md, DEVELOPER.md, plan.md, reflection.md, LLM.md, prior.md, architecture.drawio, pyproject comment, Demo.ipynb (caveat on the pilot eval numbers; score range 1–10).
- Ported the autonomy window tool from bridge-jspace to `.claude/autonomy/` (12 tests pass via `uv run --with pytest pytest .claude/autonomy`).
- Stopped tracking `.claude/settings.local.json` (was committed despite `.gitignore`).

## In progress
- Branch `docs/paused`, not pushed. GitHub metadata (description, topics) and PR not done yet — commands handed to the user.

## Next
1. Push `docs/paused`, set repo description + topics, merge to `main`.
2. If resumed: filter eval labels by verification, drop orphan scores, record generator/monitor model per trajectory, then scale to N≈200 balanced.

## Notes / blockers
- Paused for lack of compute funding for SFT/GRPO; not archived.
- No open GitHub issues, so nothing to label "paused".
- Trajectory and score records don't store the model used — runs are not reproducible from data alone.
- Current `.env` points at the vLLM/Qwen backend; a rerun would not use Haiku.
