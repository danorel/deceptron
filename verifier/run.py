"""Verify repos or trajectories in Docker.

Repo mode:
    python -m verifier.run --repos data/repos.jsonl --out data/repos_verified.jsonl

Trajectory mode:
    python -m verifier.run --trajectories data/trajectories_baseline.jsonl \
                           --tasks data/tasks.jsonl \
                           --repos data/repos_verified_clean.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from miner.schema import Repo
from verifier.deterministic import verify_repo
from verifier.schema import VerifiedRepo
from verifier.trajectory import verify_trajectory


# ── Repo verification ──────────────────────────────────────────────────────────

def _load_seen_ids(path: str) -> set[int]:
    p = Path(path)
    if not p.exists():
        return set()
    seen = set()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return seen


def run_repos(args: argparse.Namespace) -> None:
    repos_path = Path(args.repos)
    if not repos_path.exists():
        sys.exit(f"Error: {args.repos} not found. Run miner first.")

    repos = []
    with repos_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                repos.append(Repo.model_validate_json(line))

    seen_ids = _load_seen_ids(args.out)
    if seen_ids:
        print(f"Resuming: {len(seen_ids)} repos already verified.")

    pending = [r for r in repos if r.id not in seen_ids]
    print(f"Verifying {len(pending)} repos (Docker required)...\n")

    with open(args.out, "a", encoding="utf-8") as out_file:
        for i, repo in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] {repo.full_name} — cloning + testing...")
            result = verify_repo(repo.clone_url, repo.default_branch)

            verified = VerifiedRepo(**repo.model_dump(), **result)
            out_file.write(verified.model_dump_json() + "\n")
            out_file.flush()

            if verified.test_verified:
                status = "✓ verified"
            elif verified.error == "timeout":
                status = "⏱ timeout"
            else:
                status = "✗ failed"

            detail = (
                f"{verified.tests_passed} passed, {verified.tests_failed} failed"
                if not verified.error
                else f"error: {verified.error}"
            )
            print(f"  {status} — {detail}")

    total = len(seen_ids) + len(pending)
    passed = sum(
        1 for r in [json.loads(l) for l in Path(args.out).read_text().splitlines() if l]
        if r.get("test_verified")
    )
    print(f"\nDone. {passed}/{total} repos verified.")


# ── Trajectory verification ────────────────────────────────────────────────────

def _load_tasks(path: str) -> dict[str, dict]:
    """Return task_id → task dict."""
    p = Path(path)
    if not p.exists():
        return {}
    tasks = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    t = json.loads(line)
                    tasks[t["id"]] = t
                except (json.JSONDecodeError, KeyError):
                    pass
    return tasks


def _load_python_versions(path: str) -> dict[str, str]:
    """Return full_name → python_version from repos file."""
    p = Path(path)
    if not p.exists():
        return {}
    versions = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    versions[r["full_name"]] = r.get("python_version", "3.11")
                except (json.JSONDecodeError, KeyError):
                    pass
    return versions


def run_trajectories(args: argparse.Namespace) -> None:
    traj_path = Path(args.trajectories)
    if not traj_path.exists():
        sys.exit(f"{args.trajectories} not found")

    tasks = _load_tasks(args.tasks)
    python_versions = _load_python_versions(args.repos)

    trajectories = []
    with traj_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    pending = [
        t for t in trajectories
        if not (t.get("main_task_verified") and t.get("side_task_verified"))
        and t.get("final_patch") is not None
        and t.get("error") != "git clone failed"
        and t.get("error") != "git apply failed"
    ]
    print(f"Trajectories: {len(trajectories)} total, {len(pending)} to verify\n")

    updated: dict[str, dict] = {}

    for i, traj in enumerate(pending, 1):
        task_id = traj.get("task_id", "")
        task = tasks.get(task_id)
        full_name = traj["full_name"]

        print(f"[{i}/{len(pending)}] {full_name} ...", end=" ", flush=True)

        if not task:
            print(f"✗ task {task_id} not found in {args.tasks}")
            continue

        python_version = python_versions.get(full_name, "3.11")
        result = verify_trajectory(
            clone_url=traj["clone_url"],
            branch=traj["default_branch"],
            sha=traj["verified_sha"],
            python_version=python_version,
            patch=traj.get("final_patch", ""),
            deselect_args=traj.get("deselect_args", []),
            main_check_script=task["main_task_check_script"],
            side_check_script=task["side_task_check_script"],
        )

        traj["main_task_verified"] = result["main_task_verified"]
        traj["side_task_verified"] = result["side_task_verified"]
        traj["tests_passing"] = result["tests_passing"]
        if result["error"]:
            traj["error"] = result["error"]

        tests_sym = "✓" if result["tests_passing"] else "✗"
        main_sym  = "✓" if result["main_task_verified"] else "✗"
        side_sym  = "✓" if result["side_task_verified"] else "✗"
        print(f"tests={tests_sym} main={main_sym} side={side_sym}"
              + (f" error={result['error']}" if result["error"] else ""))
        updated[traj["id"]] = traj

    # Rewrite file in-place
    final = [updated.get(t["id"], t) for t in trajectories]
    with traj_path.open("w", encoding="utf-8") as f:
        for t in final:
            f.write(json.dumps(t) + "\n")

    n_main = sum(1 for t in final if t.get("main_task_verified"))
    n_side = sum(1 for t in final if t.get("side_task_verified"))
    print(f"\nDone. main_verified={n_main}/{len(final)}, side_verified={n_side}/{len(final)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repos", help="Input repos.jsonl")
    group.add_argument("--trajectories", help="Input trajectories.jsonl")
    parser.add_argument("--out", help="Output file (repo mode only)")
    parser.add_argument("--tasks", default="data/tasks.jsonl", help="tasks.jsonl for check scripts")
    parser.add_argument("--verified-repos", default="data/repos_verified_clean.jsonl",
                        help="repos_verified.jsonl for python_version lookup (trajectory mode)")
    args = parser.parse_args()

    if args.trajectories:
        args.repos = args.verified_repos  # reuse _load_python_versions(args.repos)
        run_trajectories(args)
    else:
        if not args.out:
            sys.exit("--out required in repo mode")
        run_repos(args)


if __name__ == "__main__":
    main()
