"""Docker-based test verification for cloned repositories.

Detection pass  — clone repo, detect required Python version.
Test pass 1     — install deps + run full test suite with correct Python image.
Test pass 2     — if some tests fail, re-run with --deselect for each failing ID.
                  If remaining tests pass, repo is usable with stored deselect_args.
"""

import re

import docker
from docker.errors import DockerException

DETECTION_IMAGE = "python:3.11-slim"
SUPPORTED_PYTHON_VERSIONS = {"3.9", "3.10", "3.11", "3.12", "3.13"}
DEFAULT_PYTHON_VERSION = "3.11"
TEST_TIMEOUT = 600       # seconds pytest is allowed to run (bash timeout command)
SOCKET_TIMEOUT = 660     # Docker API socket timeout — must exceed TEST_TIMEOUT
MAX_RETRIES = 2          # retries for transient connection errors
MAX_LOG_CHARS = 4000

# Bash snippet shared by all container runs: install git + clone
_GIT_INSTALL = """\
apt-get update -qq > /dev/null 2>&1
apt-get install -y git -qq > /dev/null 2>&1
"""

# Python snippet run inside container to detect required version
_DETECT_PY = """\
import re, sys

def _find(text, patterns):
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            parts = m.group(1).split('.')
            return '.'.join(parts[:2])
    return None

for fname, patterns in [
    ('.python-version',  [r'(\d+\.\d+)']),
    ('pyproject.toml',   [r'requires-python\s*=\s*["\\\\'\\''>=<^~]*(\d+\.\d+)',
                          r'python\s*=\s*["\\\\'\\''\\^~>=<]*(\d+\.\d+)']),
    ('setup.cfg',        [r'python_requires\s*=\s*[>=<^~]*(\d+\.\d+)']),
    ('setup.py',         [r'python_requires\s*=\s*["\\\\'\\''>=<^~]*(\d+\.\d+)']),
]:
    try:
        content = open(fname).read()
        ver = _find(content, patterns)
        if ver:
            print('PYTHON_VERSION=' + ver)
            sys.exit(0)
    except FileNotFoundError:
        pass

print('PYTHON_VERSION=3.11')
"""


def _run_container(
    client: docker.DockerClient, image: str, script: str
) -> tuple[str, str | None]:
    """Run a bash script in a fresh container. Always returns logs regardless of exit code.

    Retries up to MAX_RETRIES times on transient connection errors.
    Timeouts are handled by the bash `timeout` command inside the script.
    """
    import time

    last_err: str = ""
    for attempt in range(1, MAX_RETRIES + 1):
        container = None
        try:
            container = client.containers.run(
                image,
                command=["bash", "-c", script],
                detach=True,
                stdout=True,
                stderr=True,
                mem_limit="1g",
            )
            result = container.wait(timeout=SOCKET_TIMEOUT)
            output = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            exit_code = result.get("StatusCode", 0)

            if "PYTEST_TIMEOUT" in output:
                return output, "timeout"

            error = f"exit code {exit_code}" if exit_code != 0 else None
            return output, error

        except (DockerException, ConnectionError, OSError) as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(5)
        finally:
            if container:
                try:
                    container.remove(force=True)
                except DockerException:
                    pass

    return "", last_err


def _detect_python_version(client: docker.DockerClient, clone_url: str, branch: str) -> str:
    """Clone repo and detect required Python version. Returns 'major.minor' string."""
    script = (
        _GIT_INSTALL
        + f"git clone --depth=1 --branch {branch} {clone_url} /repo 2>&1 "
        + f"|| {{ echo 'PYTHON_VERSION={DEFAULT_PYTHON_VERSION}'; exit 0; }}\n"
        + "cd /repo\n"
        + f"python3 -c '{_DETECT_PY}'\n"
    )
    output, _ = _run_container(client, DETECTION_IMAGE, script)
    m = re.search(r"PYTHON_VERSION=(\d+\.\d+)", output)
    if m:
        version = m.group(1)
        if version in SUPPORTED_PYTHON_VERSIONS:
            return version
    return DEFAULT_PYTHON_VERSION


def _install_script(clone_url: str, branch: str) -> str:
    """Bash fragment: clone repo and install dependencies into /repo."""
    return (
        _GIT_INSTALL
        + f"git clone --depth=1 --branch {branch} {clone_url} /repo 2>&1 "
        + "|| { echo 'CLONE_FAILED'; exit 1; }\n"
        + "cd /repo\n"
        + "echo \"VERIFIED_SHA=$(git rev-parse HEAD)\"\n"
        + "echo '=== INSTALL START ==='\n"
        + "pip install pytest -q 2>&1\n"
        + "if [ -f requirements.txt ]; then\n"
        + "    pip install -r requirements.txt -q 2>&1 || echo 'WARN: requirements.txt install had errors'\n"
        + "fi\n"
        + "for req in requirements-dev.txt requirements-test.txt requirements_dev.txt requirements_test.txt; do\n"
        + "    [ -f $req ] && pip install -r $req -q 2>&1 || true\n"
        + "done\n"
        + "if [ -f setup.py ]; then\n"
        + "    pip install -e . -q 2>&1 || echo 'WARN: setup.py install had errors'\n"
        + "    pip install -e '.[dev]' -q 2>&1 || pip install -e '.[test]' -q 2>&1 || true\n"
        + "elif [ -f pyproject.toml ]; then\n"
        + "    pip install -e . -q 2>&1 || echo 'WARN: pyproject.toml install had errors'\n"
        + "    pip install -e '.[dev]' -q 2>&1 || pip install -e '.[test]' -q 2>&1 || true\n"
        + "fi\n"
        + "echo '=== INSTALL END ==='\n"
    )


def _run_script(deselect_args: list[str] = ()) -> str:
    """Bash fragment: run pytest from /repo with a hard time limit."""
    deselect = " ".join(deselect_args)
    # `timeout` exits with code 124 on expiry; pytest exits with 1 on failures — both are ok
    return (
        f"cd /repo && PYTHONPATH=/repo "
        f"timeout {TEST_TIMEOUT} pytest --tb=line -q --no-header {deselect} 2>&1 ; "
        f"STATUS=$? ; [ $STATUS -eq 124 ] && echo 'PYTEST_TIMEOUT' ; exit $STATUS"
    )


def _parse_pytest_output(output: str) -> tuple[int, int, list[str]]:
    """Return (passed, failed, failing_test_ids) from pytest stdout."""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", output)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", output)) else 0
    failing_ids = re.findall(r"FAILED\s+([\w/.\-]+::[\w\[\]\-]+)", output)
    return passed, failed, failing_ids


def _split_logs(output: str) -> tuple[str, str]:
    """Split combined container output into (install_log, test_log)."""
    if "=== INSTALL END ===" in output:
        install_part, test_part = output.split("=== INSTALL END ===", 1)
        return install_part[-MAX_LOG_CHARS:], test_part[-MAX_LOG_CHARS:]
    return "", output[-MAX_LOG_CHARS:]


def verify_repo(clone_url: str, default_branch: str) -> dict:
    """Clone repo, detect Python version, install deps, run tests."""
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
    output, err = _run_container(client, image, base + "\n" + _run_script())

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

    # ── Test pass 2: re-run without failing tests ──────────────────────────────
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
