"""Deterministic repo verification: clone, detect Python version, run test suite.

This is the ground-truth check for whether a repo's tests pass on a clean checkout.
Used by stage 2 (verify-repos), before task generation — see `main.py` for the CLI.
"""

import re
import time

import docker
from docker.errors import DockerException

from agents.deceptron.verifier.docker import (
    DEFAULT_PYTHON_VERSION,
    GIT_INSTALL,
    MAX_LOG_CHARS,
    SUPPORTED_PYTHON_VERSIONS,
    build_install_script,
    build_pytest_script,
    image_for,
    parse_pytest_output,
    run_container,
    split_logs,
)

# Packages that require system GUI/media libraries unavailable in Docker slim images.
# Repos depending on these cannot be imported headlessly and are incompatible with
# our Docker-based check script verification.
_DETECT_REPO_INFO_SCRIPT = r"""
import re, sys

# ── system libs ───────────────────────────────────────────────────────────────
_SYS_PAT = re.compile(
    r'(?i)\b(pyqt[456]?|pyside[26]?|wxpython|wx|python-vlc|vlc|kivy|'
    r'pygobject|pygtk|gtk|pygame|pyobjc|pyobjc-core|'
    r'libX11|libgl1|libgthread|libvlc|sdl2)\b'
)
found = []
for f in ['requirements.txt', 'pyproject.toml', 'setup.py', 'setup.cfg']:
    try:
        found.extend(_SYS_PAT.findall(open(f).read()))
    except FileNotFoundError:
        pass
print('SYSTEM_LIBS=' + (','.join(set(found)) if found else 'none'))

# ── python version ────────────────────────────────────────────────────────────
def _find(text, patterns):
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return '.'.join(m.group(1).split('.')[:2])
    return None

for fname, patterns in [
    ('.python-version',  [r'(\d+\.\d+)']),
    ('pyproject.toml',   [r'requires-python\s*=\s*["\\'>=<^~]*(\d+\.\d+)',
                          r'python\s*=\s*["\\'\\^~>=<]*(\d+\.\d+)']),
    ('setup.cfg',        [r'python_requires\s*=\s*[>=<^~]*(\d+\.\d+)']),
    ('setup.py',         [r'python_requires\s*=\s*["\\'>=<^~]*(\d+\.\d+)']),
]:
    try:
        ver = _find(open(fname).read(), patterns)
        if ver:
            print('PYTHON_VERSION=' + ver)
            sys.exit(0)
    except FileNotFoundError:
        pass
print('PYTHON_VERSION=3.11')
"""


def _detect_repo_info(
    client: docker.DockerClient, clone_url: str, branch: str
) -> tuple[list[str], str]:
    """Single container: clone repo, detect system libs AND python version.

    Returns (system_libs, python_version). Replaces two separate container
    calls (_check_system_libs + detect_python_version) to save ~60s per repo.
    """
    script = (
        GIT_INSTALL
        + f"git clone --depth=1 --branch {branch} {clone_url} /repo 2>&1 "
        + f"|| {{ echo 'SYSTEM_LIBS=none'; echo 'PYTHON_VERSION={DEFAULT_PYTHON_VERSION}'; exit 0; }}\n"
        + "cd /repo\n"
        + f"python3 -c '{_DETECT_REPO_INFO_SCRIPT}'\n"
    )
    output, _ = run_container(client, image_for(), script)

    sys_m = re.search(r"SYSTEM_LIBS=(.+)", output)
    system_libs: list[str] = []
    if sys_m and sys_m.group(1).strip() != "none":
        system_libs = [lib.strip() for lib in sys_m.group(1).split(",")]

    ver_m = re.search(r"PYTHON_VERSION=(\d+\.\d+)", output)
    python_version = DEFAULT_PYTHON_VERSION
    if ver_m and ver_m.group(1) in SUPPORTED_PYTHON_VERSIONS:
        python_version = ver_m.group(1)

    return system_libs, python_version


def verify_repo(clone_url: str, default_branch: str) -> dict:
    """Clone repo, detect Python version, install deps, run tests.

    Returns a dict with keys: test_verified, tests_passed, tests_failed,
    failing_test_ids, deselect_args, install_log, test_log, python_version,
    verified_sha, error.
    """
    t0 = time.monotonic()
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
        "duration_s": None,
    }

    def _done() -> dict:
        result["duration_s"] = round(time.monotonic() - t0, 1)
        return result

    try:
        client = docker.from_env()
    except DockerException as e:
        result["error"] = f"Docker not available: {e}"
        return _done()

    # ── Detection pass (single container: system libs + python version) ───────
    system_libs, python_version = _detect_repo_info(client, clone_url, default_branch)
    if system_libs:
        result["system_libs"] = system_libs
        result["error"] = f"system libs required: {', '.join(system_libs)}"
        return _done()
    result["python_version"] = python_version
    image = image_for(python_version)

    base = build_install_script(clone_url, default_branch)

    # ── Test pass 1: full test run ─────────────────────────────────────────────
    output, _ = run_container(client, image, base + "\n" + build_pytest_script())

    if "CLONE_FAILED" in output:
        result["error"] = "git clone failed"
        result["test_log"] = output[-MAX_LOG_CHARS:]
        return _done()

    sha_match = re.search(r"VERIFIED_SHA=([0-9a-f]{40})", output)
    if sha_match:
        result["verified_sha"] = sha_match.group(1)

    result["install_log"], result["test_log"] = split_logs(output)
    passed, failed, failing_ids = parse_pytest_output(output)
    result["tests_passed"] = passed
    result["tests_failed"] = failed
    result["failing_test_ids"] = failing_ids

    if passed == 0 and failed == 0:
        result["error"] = "No tests collected"
        return _done()

    if failed == 0:
        result["test_verified"] = True
        return _done()

    # ── Test pass 2: re-run skipping known-failing tests ──────────────────────
    if not failing_ids:
        return result

    deselect = [f"--deselect {tid}" for tid in failing_ids]
    output2, _ = run_container(client, image, base + "\n" + build_pytest_script(deselect))
    result["test_log"] += "\n--- RERUN WITHOUT FAILING TESTS ---\n" + output2[-1000:]

    passed2, failed2, _ = parse_pytest_output(output2)
    if failed2 == 0 and passed2 > 0:
        result["test_verified"] = True
        result["tests_passed"] = passed2
        result["tests_failed"] = 0
        result["deselect_args"] = deselect

    return _done()
