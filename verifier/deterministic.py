"""Deterministic repo verification: clone, detect Python version, run test suite.

This is the ground-truth check for whether a repo's tests pass on a clean checkout.
Used by the verifier pipeline before task generation.
"""

import re

import docker
from docker.errors import DockerException

from verifier.docker import (
    DEFAULT_PYTHON_VERSION,
    DETECTION_IMAGE,
    MAX_LOG_CHARS,
    _GIT_INSTALL,
    _detect_python_version,
    _install_script,
    _parse_pytest_output,
    _run_container,
    _run_script,
    _split_logs,
)

# Packages that require system GUI/media libraries unavailable in Docker slim images.
# Repos depending on these cannot be imported headlessly and are incompatible with
# our Docker-based check script verification.
_SYSTEM_LIB_PACKAGES = re.compile(
    r"(?i)\b(pyqt[456]?|pyside[26]?|wxpython|wx|python-vlc|vlc|kivy|"
    r"pygobject|pygtk|gtk|pygame|pyobjc|pyobjc-core|tkinter|"
    r"libX11|libgl1|libgthread|libvlc|sdl2)\b"
)

_DETECT_SYSTEM_LIBS_SCRIPT = """\
import re, sys
pattern = re.compile(
    r'(?i)\\b(pyqt[456]?|pyside[26]?|wxpython|wx|python-vlc|vlc|kivy|'
    r'pygobject|pygtk|gtk|pygame|pyobjc|pyobjc-core|'
    r'libX11|libgl1|libgthread|libvlc|sdl2)\\b'
)
files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'setup.cfg']
found = []
for f in files:
    try:
        content = open(f).read()
        matches = pattern.findall(content)
        if matches:
            found.extend(matches)
    except FileNotFoundError:
        pass
if found:
    print('SYSTEM_LIBS=' + ','.join(set(found)))
else:
    print('SYSTEM_LIBS=none')
"""


def _check_system_libs(client: docker.DockerClient, clone_url: str, branch: str) -> list[str]:
    """Clone repo and scan dependency files for known system-lib packages."""
    script = (
        _GIT_INSTALL
        + f"git clone --depth=1 --branch {branch} {clone_url} /repo 2>&1 "
        + f"|| {{ echo 'SYSTEM_LIBS=none'; exit 0; }}\n"
        + "cd /repo\n"
        + f"python3 -c '{_DETECT_SYSTEM_LIBS_SCRIPT}'\n"
    )
    output, _ = _run_container(client, DETECTION_IMAGE, script)
    m = re.search(r"SYSTEM_LIBS=(.+)", output)
    if m and m.group(1) != "none":
        return [lib.strip() for lib in m.group(1).split(",")]
    return []


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
        "system_libs": [],
        "error": None,
    }

    try:
        client = docker.from_env()
    except DockerException as e:
        result["error"] = f"Docker not available: {e}"
        return result

    # ── System libs check ──────────────────────────────────────────────────────
    system_libs = _check_system_libs(client, clone_url, default_branch)
    if system_libs:
        result["system_libs"] = system_libs
        result["error"] = f"system libs required: {', '.join(system_libs)}"
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
