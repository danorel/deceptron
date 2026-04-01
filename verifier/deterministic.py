"""Deterministic repo verification: clone, detect Python version, run test suite.

This is the ground-truth check for whether a repo's tests pass on a clean checkout.
Used by the verifier pipeline before task generation.
"""

import re

import docker
from docker.errors import DockerException

from verifier.docker import (
    DEFAULT_PYTHON_VERSION,
    MAX_LOG_CHARS,
    _detect_python_version,
    _install_script,
    _parse_pytest_output,
    _run_container,
    _run_script,
    _split_logs,
)


def verify_repo(clone_url: str, default_branch: str) -> dict:
    """Clone repo, detect Python version, install deps, run tests.

    Returns a dict with keys: test_verified, tests_passed, tests_failed,
    failing_test_ids, deselect_args, install_log, test_log, python_version,
    verified_sha, error.
    """
    result: dict = {
        "test_verified": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "failing_test_ids": [],
        "deselect_args": [],
        "install_log": "",
        "test_log": "",
        "python_version": DEFAULT_PYTHON_VERSION,
        "verified_sha": "",
        "error": None,
    }

    try:
        client = docker.from_env()
    except DockerException as e:
        result["error"] = f"Docker not available: {e}"
        return result

    # ── Detection pass ─────────────────────────────────────────────────────────
    python_version = _detect_python_version(client, clone_url, default_branch)
    result["python_version"] = python_version
    image = f"python:{python_version}-slim"

    base = _install_script(clone_url, default_branch)

    # ── Test pass 1: full test run ─────────────────────────────────────────────
    output, _ = _run_container(client, image, base + "\n" + _run_script())

    if "CLONE_FAILED" in output:
        result["error"] = "git clone failed"
        result["test_log"] = output[-MAX_LOG_CHARS:]
        return result

    sha_match = re.search(r"VERIFIED_SHA=([0-9a-f]{40})", output)
    if sha_match:
        result["verified_sha"] = sha_match.group(1)

    result["install_log"], result["test_log"] = _split_logs(output)
    passed, failed, failing_ids = _parse_pytest_output(output)
    result["tests_passed"] = passed
    result["tests_failed"] = failed
    result["failing_test_ids"] = failing_ids

    if passed == 0 and failed == 0:
        result["error"] = "No tests collected"
        return result

    if failed == 0:
        result["test_verified"] = True
        return result

    # ── Test pass 2: re-run skipping known-failing tests ──────────────────────
    if not failing_ids:
        return result

    deselect = [f"--deselect {tid}" for tid in failing_ids]
    output2, _ = _run_container(client, image, base + "\n" + _run_script(deselect))
    result["test_log"] += "\n--- RERUN WITHOUT FAILING TESTS ---\n" + output2[-1000:]

    passed2, failed2, _ = _parse_pytest_output(output2)
    if failed2 == 0 and passed2 > 0:
        result["test_verified"] = True
        result["tests_passed"] = passed2
        result["tests_failed"] = 0
        result["deselect_args"] = deselect

    return result
