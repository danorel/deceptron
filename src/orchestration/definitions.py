"""Dagster wiring for the mining + repo-verification stages.

Wraps `deceptron.miner.main.run_miner` and `deceptron.verifier.main.run_verify_repo`
directly (no subprocess — same functions `deceptron.main`'s "mine" and
"verify-repos" stages call). Both are already incremental/append-and-resume
by id, so a crashed or restarted run picks up where it left off for free —
Dagster adds the missing pieces: isolation (its own process/container),
a UI to watch progress and failures, and automatic retries.

Run locally:
    dagster dev -w workspace.yaml

Or via docker-compose (webserver + daemon, see docker-compose.yml):
    docker compose up
"""

from dagster import Config, In, Nothing, OpExecutionContext, RetryPolicy, Definitions, job, op
from dotenv import load_dotenv

from deceptron.miner.main import DEFAULT_OUT as DEFAULT_MINER_OUT, run_miner
from deceptron.verifier.main import DEFAULT_VERIFY_REPOS_OUT, run_verify_repo

# Mirrors every CLI entrypoint's `load_dotenv()` — ops call run_miner/run_verify_repo
# directly (not via subprocess), so GITHUB_TOKEN etc. must be loaded here too.
load_dotenv()

# Repo verification clones + runs each repo's test suite in Docker — slow and
# occasionally flaky (network, registry pulls); mining hits the GitHub API,
# which rate-limits in bursts. Both are worth a couple of automatic retries
# before surfacing as a real failure in the UI.
_RETRY_POLICY = RetryPolicy(max_retries=2, delay=60)


class ConfigMinerRepo(Config):
    lang: str = "python"
    limit: int = 50
    out: str = DEFAULT_MINER_OUT


class ConfigVerifyRepo(Config):
    repos: str = DEFAULT_MINER_OUT
    out: str = DEFAULT_VERIFY_REPOS_OUT


@op(retry_policy=_RETRY_POLICY)
def run_miner_op(context: OpExecutionContext, config: ConfigMinerRepo) -> None:
    context.log.info(f"Mining up to {config.limit} {config.lang} repos -> {config.out}")
    run_miner(lang=config.lang, limit=config.limit, out=config.out)


@op(ins={"start": In(Nothing)}, retry_policy=_RETRY_POLICY)
def run_verify_repo_op(context: OpExecutionContext, config: ConfigVerifyRepo) -> None:
    context.log.info(f"Verifying repos from {config.repos} -> {config.out}")
    run_verify_repo(repos=config.repos, out=config.out)


@job(
    description=(
        "Mine GitHub repos, then verify each in Docker (clone + run test suite). "
        "Both stages are append/resume-by-id, so retries and re-launches after a "
        "crash continue from the last completed item rather than starting over."
    ),
)
def run_pipeline_job() -> None:
    run_verify_repo_op(start=run_miner_op())


defs = Definitions(jobs=[run_pipeline_job])
