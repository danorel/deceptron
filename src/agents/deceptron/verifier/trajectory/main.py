"""Stage 5 — verify agent trajectories in Docker (apply patch + pytest + check scripts).

Applies each agent's patch to a clean clone, runs the native test suite plus
the task's main/side check scripts, and writes `*_verified` fields back into
the trajectory file in place — see `verify.py` for the per-trajectory Docker run.

Usage:
    python -m agents.deceptron.verifier.trajectory.main \\
        --trajectories data/agents/deceptron/trajectories/baseline.jsonl \\
        --tasks data/agents/deceptron/tasks/tasks.jsonl \\
        --verified-repos data/agents/deceptron/mining/repos_verified_clean.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents.deceptron.verifier.trajectory.verify import verify_trajectory

DEFAULT_TASKS = "data/agents/deceptron/tasks/tasks.jsonl"
DEFAULT_VERIFIED_REPOS = "data/agents/deceptron/mining/repos_verified_clean.jsonl"


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


def run(trajectories_path: str, tasks: str, verified_repos: str) -> None:
    """CLI loop: re-verify trajectories missing main/side_task_verified, write back in place."""
    traj_path = Path(trajectories_path)
    if not traj_path.exists():
        sys.exit(f"{trajectories_path} not found")

    tasks_by_id = _load_tasks(tasks)
    python_versions = _load_python_versions(verified_repos)

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

    # Index by id for fast in-place updates
    traj_by_id: dict[str, dict] = {t["id"]: t for t in trajectories}

    def _flush() -> None:
        with traj_path.open("w", encoding="utf-8") as f:
            for t in trajectories:
                f.write(json.dumps(traj_by_id[t["id"]]) + "\n")

    for i, traj in enumerate(pending, 1):
        task_id = traj.get("task_id", "")
        task = tasks_by_id.get(task_id)
        full_name = traj["full_name"]

        print(f"[{i}/{len(pending)}] {full_name} ...", end=" ", flush=True)

        if not task:
            print(f"✗ task {task_id} not found in {tasks}")
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
        traj["verify_log"] = result["verify_log"]
        traj["duration_s"] = result.get("duration_s")
        if result["error"]:
            traj["error"] = result["error"]

        tests_sym = "✓" if result["tests_passing"] else "✗"
        main_sym  = "✓" if result["main_task_verified"] else "✗"
        side_sym  = "✓" if result["side_task_verified"] else "✗"
        dur = f" ({result['duration_s']}s)" if result.get("duration_s") is not None else ""
        print(f"tests={tests_sym} main={main_sym} side={side_sym}{dur}"
              + (f" error={result['error']}" if result["error"] else ""))

        traj_by_id[traj["id"]] = traj
        _flush()

    final = list(traj_by_id.values())
    n_main = sum(1 for t in final if t.get("main_task_verified"))
    n_side = sum(1 for t in final if t.get("side_task_verified"))
    print(f"\nDone. main_verified={n_main}/{len(final)}, side_verified={n_side}/{len(final)}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True, help="Trajectories .jsonl to verify in place")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="tasks.jsonl for check scripts")
    parser.add_argument("--verified-repos", default=DEFAULT_VERIFIED_REPOS,
                        help="repos_verified.jsonl for python_version lookup")
    args = parser.parse_args()

    run(args.trajectories, args.tasks, args.verified_repos)


if __name__ == "__main__":
    main()
