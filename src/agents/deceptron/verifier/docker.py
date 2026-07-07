"""Docker-based test verification: detect Python version, install + run test
suite, retry with `--deselect` on a failing-but-usable suite."""

import os
import re
import time
from pathlib import Path

import docker
from docker.errors import DockerException

SUPPORTED_PYTHON_VERSIONS = {"3.9", "3.10", "3.11", "3.12", "3.13"}
DEFAULT_PYTHON_VERSION = "3.11"
TEST_TIMEOUT = 600       # seconds pytest is allowed to run (bash timeout command)
SOCKET_TIMEOUT = 660     # Docker API socket timeout — must exceed TEST_TIMEOUT
MAX_RETRIES = 2          # retries for transient connection errors
MAX_LOG_CHARS = 4000

# Set DECEPTRON_BASE_IMAGE_PREFIX=<repo>/<name> to use pre-built images that
# already have git, build-essential, python3-dev, and uv — skips ~13s of
# apt-get + pip install uv per container. Images are expected at
# <prefix>:<python-version> (e.g. danorel/deceptron-base:3.11).
# See docker/base/Dockerfile and the build instructions in .env.example.
_BASE_PREFIX = os.environ.get("DECEPTRON_BASE_IMAGE_PREFIX", "")


def image_for(python_version: str = DEFAULT_PYTHON_VERSION) -> str:
    """Return the Docker image tag for a given Python version."""
    if _BASE_PREFIX:
        return f"{_BASE_PREFIX}:{python_version}"
    return f"python:{python_version}-slim"


# These are empty strings when pre-built images are used (tools already present).
GIT_INSTALL = "" if _BASE_PREFIX else """\
apt-get update -qq > /dev/null 2>&1
apt-get install -y git build-essential python3-dev -qq > /dev/null 2>&1
"""
UV_INSTALL = "" if _BASE_PREFIX else "pip install -q uv 2>&1\n"
UV_PIP     = "uv pip install --system -q"

# Host-side uv cache dir. Mounted read-write into every container so uv
# reuses downloaded wheels/metadata across repo verification runs.
_UV_CACHE_HOST = str(Path.home() / ".cache" / "uv")
_UV_CACHE_GUEST = "/root/.cache/uv"


def run_container(
    client: docker.DockerClient, image: str, script: str
) -> tuple[str, str | None]:
    """Run a bash script in a fresh container, retrying on transient connection
    errors. `timeout` inside the script handles run-length limits."""
    Path(_UV_CACHE_HOST).mkdir(parents=True, exist_ok=True)
    volumes = {_UV_CACHE_HOST: {"bind": _UV_CACHE_GUEST, "mode": "rw"}}

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
                mem_limit="2g",
                volumes=volumes,
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


def build_install_script(clone_url: str, branch: str) -> str:
    """Bash fragment: clone repo and install dependencies into /repo."""
    return (
        GIT_INSTALL
        + UV_INSTALL
        + f"git clone --depth=1 --branch {branch} {clone_url} /repo 2>&1 "
        + "|| { echo 'CLONE_FAILED'; exit 1; }\n"
        + "cd /repo\n"
        + "echo \"VERIFIED_SHA=$(git rev-parse HEAD)\"\n"
        + "echo '=== INSTALL START ==='\n"
        + f"{UV_PIP} pytest pytest-xdist 2>&1\n"
        + "if [ -f requirements.txt ]; then\n"
        + f"    {UV_PIP} -r requirements.txt 2>&1 || echo 'WARN: requirements.txt install had errors'\n"
        + "fi\n"
        + "for req in requirements-dev.txt requirements-test.txt requirements_dev.txt requirements_test.txt; do\n"
        + f"    [ -f $req ] && {UV_PIP} -r $req 2>&1 || true\n"
        + "done\n"
        + "if [ -f setup.py ]; then\n"
        + f"    {UV_PIP} -e . 2>&1 || echo 'WARN: setup.py install had errors'\n"
        + f"    {UV_PIP} -e '.[dev]' 2>&1 || {UV_PIP} -e '.[test]' 2>&1 || true\n"
        + "elif [ -f pyproject.toml ]; then\n"
        + f"    {UV_PIP} -e . 2>&1 || echo 'WARN: pyproject.toml install had errors'\n"
        + f"    {UV_PIP} -e '.[dev]' 2>&1 || {UV_PIP} -e '.[test]' 2>&1 || true\n"
        + "fi\n"
        + "echo '=== INSTALL END ==='\n"
    )


def build_pytest_script(deselect_args: list[str] = ()) -> str:
    """Bash fragment: run pytest from /repo with a hard time limit.

    `-n auto` (pytest-xdist) parallelizes across cores — faster, but can
    flake on suites with shared state; if false failures show up in
    test_log, drop `-n auto` for that repo's deselect_args retry pass."""
    deselect = " ".join(deselect_args)
    # `timeout` exits with code 124 on expiry; pytest exits with 1 on failures — both are ok
    return (
        f"cd /repo && PYTHONPATH=/repo "
        f"timeout {TEST_TIMEOUT} pytest -n auto --tb=line -q --no-header {deselect} 2>&1 ; "
        f"STATUS=$? ; [ $STATUS -eq 124 ] && echo 'PYTEST_TIMEOUT' ; exit $STATUS"
    )


def parse_pytest_output(output: str) -> tuple[int, int, list[str]]:
    """Return (passed, failed, failing_test_ids) from pytest stdout."""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", output)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", output)) else 0
    failing_ids = re.findall(r"FAILED\s+([\w/.\-]+::[\w\[\]\-]+)", output)
    return passed, failed, failing_ids


def split_logs(output: str) -> tuple[str, str]:
    """Split combined container output into (install_log, test_log)."""
    if "=== INSTALL END ===" in output:
        install_part, test_part = output.split("=== INSTALL END ===", 1)
        return install_part[-MAX_LOG_CHARS:], test_part[-MAX_LOG_CHARS:]
    return "", output[-MAX_LOG_CHARS:]
