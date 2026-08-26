"""Dagster wiring for mining + repo verification.

`mine_repos_op` fans out each repo as a `DynamicOutput`; `verify_repo_op` runs
once per repo. Caveat: Dagster only resolves dynamic outputs after the
producing op completes — verification doesn't overlap mining, it just runs
concurrently (bounded by `_EXECUTOR`) once mining is done."""

from dagster import (
    Config,
    DynamicOut,
    DynamicOutput,
    OpExecutionContext,
    RetryPolicy,
    Definitions,
    job,
    multiprocess_executor,
    op,
)
from dotenv import load_dotenv

from agents.deceptron.miner.main import DEFAULT_OUT as DEFAULT_MINER_OUT, iter_mined_repos
from agents.deceptron.miner.schema import Repo
from agents.deceptron.verifier.repo.main import DEFAULT_VERIFY_REPOS_OUT, load_verified_ids, verify_and_append_repo

# Mirrors every CLI entrypoint's `load_dotenv()` — ops call iter_mined_repos/
# verify_and_append_repo directly (not via subprocess), so GITHUB_TOKEN etc.
# must be loaded here too.
load_dotenv()

# Repo verification clones + runs each repo's test suite in Docker — slow and
# occasionally flaky (network, registry pulls); mining hits the GitHub API,
# which rate-limits in bursts. Both are worth a couple of automatic retries
# before surfacing as a real failure in the UI. Retrying verify_repo_op now
# only re-runs the one repo that failed, not the whole batch.
_RETRY_POLICY = RetryPolicy(max_retries=2, delay=60)

# Each verify_repo_op clones a repo and runs its test suite in its own Docker
# container — concurrency is bounded so a laptop doesn't choke on N simultaneous
# clones/containers. Override per-run in the Launchpad: execution.config.
# multiprocess.max_concurrent (e.g. 8 on a beefier machine, 1 to go fully serial).
_EXECUTOR = multiprocess_executor.configured({"max_concurrent": 4})


class ConfigMinerRepo(Config):
    lang: str = "python"
    limit: int = 50
    out: str = DEFAULT_MINER_OUT


class ConfigVerifyRepo(Config):
    out: str = DEFAULT_VERIFY_REPOS_OUT


@op(out=DynamicOut(), retry_policy=_RETRY_POLICY)
def mine_repos_op(context: OpExecutionContext, config: ConfigMinerRepo):
    context.log.info(f"Mining up to {config.limit} {config.lang} repos -> {config.out}")
    for repo in iter_mined_repos(lang=config.lang, limit=config.limit, out=config.out):
        context.log.info(f"Mined {repo.full_name} ({repo.stars}*) -> queuing for verification")
        yield DynamicOutput(repo, mapping_key=str(repo.id))


@op(retry_policy=_RETRY_POLICY)
def verify_repo_op(context: OpExecutionContext, config: ConfigVerifyRepo, repo: Repo) -> None:
    if repo.id in load_verified_ids(config.out):
        context.log.info(f"{repo.full_name} already verified -> skipping")
        return
    context.log.info(f"Verifying {repo.full_name} (Docker required)...")
    verify_and_append_repo(repo, config.out)


@job(
    executor_def=_EXECUTOR,
    description="Mine GitHub repos, then verify each in Docker — fanned out per repo, bounded by max_concurrent.",
)
def run_pipeline_job() -> None:
    mine_repos_op().map(verify_repo_op)


defs = Definitions(jobs=[run_pipeline_job])
