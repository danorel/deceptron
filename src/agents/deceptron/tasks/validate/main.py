"""Re-validate check scripts for task pairs that have not yet been verified.

Reads tasks.jsonl, runs Docker for pairs where main_task_check_valid=False
or side_task_check_valid=False, then rewrites the file in-place.

Usage:
    python -m agents.deceptron.tasks.validate.main --tasks data/agents/deceptron/tasks/tasks.jsonl
    python -m agents.deceptron.tasks.validate.main --tasks data/agents/deceptron/tasks/tasks.jsonl --force   # re-validate all
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents.deceptron.verifier.check_scripts import validate_check_scripts
from agents.deceptron.tasks.schema import TaskPair
from agents.deceptron.verifier.repo.schema import VerifiedRepo

try:
    import docker
    from docker.errors import DockerException
except ImportError:
    docker = None


def _load_verified_repos(repos_path: str) -> dict[str, VerifiedRepo]:
    p = Path(repos_path)
    if not p.exists():
        return {}
    result = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    repo = VerifiedRepo.model_validate_json(line)
                    result[repo.full_name] = repo
                except Exception:
                    pass
    return result


def run(tasks: str, repos: str, force: bool) -> None:
    tasks_path = Path(tasks)
    if not tasks_path.exists():
        sys.exit(f"{tasks} not found")

    if docker is None:
        sys.exit("docker package not installed")

    try:
        docker_client = docker.from_env()
    except Exception as e:
        sys.exit(f"Docker not available: {e}")

    verified_repos = _load_verified_repos(repos)

    pairs: list[TaskPair] = []
    with tasks_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(TaskPair.model_validate_json(line))

    needs_validation = [
        p for p in pairs
        if force or not p.main_task_check_valid or not p.side_task_check_valid
    ]
    print(f"Pairs: {len(pairs)} total, {len(needs_validation)} need validation")

    updated = 0
    for i, pair in enumerate(needs_validation, 1):
        repo = verified_repos.get(pair.full_name)
        if repo is None:
            print(f"  [{i}/{len(needs_validation)}] {pair.full_name} — repo not found in {repos}, skipping")
            continue

        image = f"python:{repo.python_version}-slim"
        print(f"  [{i}/{len(needs_validation)}] {pair.full_name} / {pair.main_task_function} ...", end=" ", flush=True)

        try:
            main_valid, side_valid = validate_check_scripts(docker_client, image, repo, {
                "main_task_check_script": pair.main_task_check_script,
                "side_task_check_script": pair.side_task_check_script,
            })
        except Exception as e:
            print(f"error: {e}")
            continue

        pair.main_task_check_valid = main_valid
        pair.side_task_check_valid = side_valid
        status = f"main={'✓' if main_valid else '✗'} side={'✓' if side_valid else '✗'}"
        print(status)
        updated += 1

    # Rewrite tasks.jsonl with updated pairs (preserve order)
    pair_index = {p.id: p for p in needs_validation}
    final = [pair_index.get(p.id, p) for p in pairs]

    with tasks_path.open("w", encoding="utf-8") as f:
        for p in final:
            f.write(p.model_dump_json() + "\n")

    print(f"\nDone. {updated} pairs updated, {len(final)} total written to {tasks}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/agents/deceptron/tasks/tasks.jsonl")
    parser.add_argument("--repos", default="data/agents/deceptron/mining/repos_verified_clean.jsonl")
    parser.add_argument("--force", action="store_true", help="Re-validate all pairs, not just unvalidated ones")
    args = parser.parse_args()
    run(args.tasks, args.repos, args.force)


if __name__ == "__main__":
    main()
