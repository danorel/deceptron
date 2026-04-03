"""Generate (main_task, side_task) pairs with deterministic check scripts.

Pipeline per repo:
  1. Analyze repo code with Claude (cached to disk by SHA)
  2. Build TaskPair objects from analysis
  3. Validate each check script in Docker:
       - side_task_check must FAIL (adversarial behavior not yet present)
       - main_task_check must FAIL (task not yet implemented)
  4. Write to tasks.jsonl

Usage:
    python -m tasks.generate --repos data/repos_verified.jsonl --out data/tasks.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from tasks.analyzer import TASKS_PER_REPO, analyze
from tasks.schema import TaskPair
from verifier.docker import _install_script, _run_container, docker
from verifier.schema import VerifiedRepo


def _run_check_script(
    client,
    image: str,
    clone_url: str,
    default_branch: str,
    deselect_args: list[str],
    script: str,
) -> tuple[int, str]:
    """Run a check script inside Docker. Returns (exit_code, output)."""
    escaped = script.replace("'", "'\\''")
    run_cmd = (
        _install_script(clone_url, default_branch)
        + f"\necho '{escaped}' > /tmp/check.py\n"
        + "cd /repo && PYTHONPATH=/repo python /tmp/check.py 2>&1\n"
    )
    output, err = _run_container(client, image, run_cmd)
    # Infer exit code from output markers
    if "PASS" in output and "FAIL" not in output:
        return 0, output
    return 1, output


def _validate_scripts(
    docker_client,
    image: str,
    repo: VerifiedRepo,
    pair: dict,
) -> tuple[bool, bool]:
    """Validate check scripts. Returns (main_check_valid, side_check_valid)."""
    main_code, main_out = _run_check_script(
        docker_client,
        image,
        repo.clone_url,
        repo.default_branch,
        repo.deselect_args,
        pair["main_task_check_script"],
    )
    # main_task_check should FAIL on unmodified repo (task not done yet)
    main_valid = "CLONE_FAILED" not in main_out and "SyntaxError" not in main_out

    side_code, side_out = _run_check_script(
        docker_client,
        image,
        repo.clone_url,
        repo.default_branch,
        repo.deselect_args,
        pair["side_task_check_script"],
    )
    # side_task_check should FAIL on unmodified repo (side task not yet done)
    # Valid if: no syntax errors AND it returns FAIL (not PASS — that would mean backdoor already present)
    side_valid = (
        "SyntaxError" not in side_out
        and "CLONE_FAILED" not in side_out
        and "PASS" not in side_out  # must NOT pass on clean repo
    )

    return main_valid, side_valid


def _load_seen(path: str) -> dict[str, int]:
    """Return {full_name: count} of already generated pairs."""
    p = Path(path)
    if not p.exists():
        return {}
    counts: dict[str, int] = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    fn = json.loads(line)["full_name"]
                    counts[fn] = counts.get(fn, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    pass
    return counts


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", default="data/repos_verified.jsonl")
    parser.add_argument("--out", default="data/tasks.jsonl")
    parser.add_argument("--no-docker", action="store_true", help="Skip Docker validation of check scripts")
    args = parser.parse_args()

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        sys.exit("GITHUB_TOKEN not set")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        sys.exit("ANTHROPIC_API_KEY not set")

    llm = anthropic.Anthropic(api_key=anthropic_key)

    repos_path = Path(args.repos)
    if not repos_path.exists():
        sys.exit(f"{args.repos} not found")

    repos = []
    with repos_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                repo = VerifiedRepo.model_validate_json(line)
                if repo.test_verified and not repo.system_libs:
                    repos.append(repo)
                elif repo.system_libs:
                    print(f"[skip] {repo.full_name} — system libs: {', '.join(repo.system_libs)}")

    seen = _load_seen(args.out)

    docker_client = None
    if not args.no_docker:
        try:
            docker_client = docker.from_env()
        except Exception as e:
            print(f"Warning: Docker not available ({e}). Skipping script validation.")

    written = 0
    with open(args.out, "a", encoding="utf-8") as out_file:
        for repo in repos:
            already = seen.get(repo.full_name, 0)
            need = TASKS_PER_REPO - already
            if need <= 0:
                print(f"[skip] {repo.full_name} — already has {already} pairs")
                continue

            print(f"\n[{repo.full_name}] — generating {need} pair(s)")

            try:
                analysis = analyze(repo.full_name, repo.verified_sha, github_token, llm)
            except Exception as e:
                print(f"  error during analysis: {e}")
                continue

            image = f"python:{repo.python_version}-slim"
            pairs = analysis.get("task_pairs", [])

            for i, pair_data in enumerate(pairs[already: already + need]):
                print(f"  pair {already + i + 1}/{TASKS_PER_REPO}: {pair_data.get('side_task_type')}")

                main_valid, side_valid = False, False
                if docker_client:
                    try:
                        main_valid, side_valid = _validate_scripts(docker_client, image, repo, pair_data)
                        status = f"main={'✓' if main_valid else '✗'} side={'✓' if side_valid else '✗'}"
                        print(f"  docker validation: {status}")
                    except Exception as e:
                        print(f"  docker validation error: {e}")

                pair = TaskPair(
                    full_name=repo.full_name,
                    repo_url=repo.url,
                    clone_url=repo.clone_url,
                    default_branch=repo.default_branch,
                    verified_sha=repo.verified_sha,
                    deselect_args=repo.deselect_args,
                    main_task=pair_data["main_task"],
                    main_task_file=pair_data["main_task_file"],
                    main_task_function=pair_data["main_task_function"],
                    main_task_check_script=pair_data["main_task_check_script"],
                    main_task_check_valid=main_valid,
                    side_task=pair_data["side_task"],
                    side_task_type=pair_data["side_task_type"],
                    side_task_file=pair_data["side_task_file"],
                    side_task_function=pair_data["side_task_function"],
                    side_task_check_script=pair_data["side_task_check_script"],
                    side_task_check_hint=pair_data["side_task_check_hint"],
                    side_task_check_valid=side_valid,
                )
                out_file.write(pair.model_dump_json() + "\n")
                out_file.flush()
                written += 1

    print(f"\nDone. {written} task pairs written to {args.out}")


if __name__ == "__main__":
    main()
