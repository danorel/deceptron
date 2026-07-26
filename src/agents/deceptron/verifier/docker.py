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


def check_docker_available() -> str | None:
    """Return an error message if the Docker daemon is unreachable, else None.
    Used to fail fast before a run that needs in-loop test containers."""
    try:
        docker.from_env(timeout=10).ping()
        return None
    except (DockerException, ConnectionError, OSError) as e:
        return str(e)


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


def build_deps_script() -> str:
    """Bash fragment: install pytest + the repo's own deps, assuming /repo exists.
    Shared by full verification (after clone) and the in-loop test session
    (repo mounted from the host)."""
    return (
        f"{UV_PIP} pytest pytest-xdist 2>&1\n"
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
    )


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
        + build_deps_script()
        + "echo '=== INSTALL END ==='\n"
    )


def build_pytest_script(deselect_args: list[str] = (), timeout_s: int = TEST_TIMEOUT) -> str:
    """Bash fragment: run pytest from /repo with a hard time limit.

    `-n auto` (pytest-xdist) parallelizes across cores — faster, but can
    flake on suites with shared state; if false failures show up in
    test_log, drop `-n auto` for that repo's deselect_args retry pass."""
    deselect = " ".join(deselect_args)
    # `timeout` exits with code 124 on expiry; pytest exits with 1 on failures — both are ok
    return (
        f"cd /repo && PYTHONPATH=/repo "
        f"timeout {timeout_s} pytest -n auto --tb=line -q --no-header {deselect} 2>&1 ; "
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


_SESSION_TEST_TIMEOUT = 180   # shorter than TEST_TIMEOUT — in-loop checks stay snappy
_MAX_TOOL_LOG_CHARS = 1500    # feedback returned to the agent; keep the tail short


class ContainerTestSession:
    """Long-lived container for in-loop `run_tests`: mounts the agent's live
    repo at /repo, installs deps once, then `docker exec`s pytest on demand.

    Faithful to the Docker verifier (same deps + pytest + deselect), but reused
    across turns so only the first call pays install cost. Call `close()` when
    the trajectory ends.

    >>> # s = ContainerTestSession(repo_dir, deselect_args)
    >>> # print(s.run_tests()); s.close()
    """

    def __init__(self, repo_dir: str, deselect_args: list[str], python_version: str = DEFAULT_PYTHON_VERSION) -> None:
        self._repo_dir = repo_dir
        self._deselect_args = list(deselect_args)
        self._image = image_for(python_version if python_version in SUPPORTED_PYTHON_VERSIONS else DEFAULT_PYTHON_VERSION)
        self._client: "docker.DockerClient | None" = None
        self._container = None
        self._installed = False

    def _ensure_started(self) -> None:
        if self._container is not None:
            return
        Path(_UV_CACHE_HOST).mkdir(parents=True, exist_ok=True)
        self._client = docker.from_env(timeout=SOCKET_TIMEOUT)
        self._container = self._client.containers.run(
            self._image,
            command=["sleep", "infinity"],
            detach=True,
            mem_limit="2g",
            volumes={
                self._repo_dir: {"bind": "/repo", "mode": "rw"},
                _UV_CACHE_HOST: {"bind": _UV_CACHE_GUEST, "mode": "rw"},
            },
        )

    def _exec(self, script: str) -> str:
        res = self._container.exec_run(["bash", "-c", script], demux=False)
        return res.output.decode("utf-8", errors="replace")

    def run_tests(self) -> str:
        """Install deps (first call only), run pytest, return a short summary."""
        try:
            self._ensure_started()
            if not self._installed:
                self._exec(GIT_INSTALL + UV_INSTALL + "cd /repo\n" + build_deps_script())
                self._installed = True
            out = self._exec(build_pytest_script(self._deselect_args, timeout_s=_SESSION_TEST_TIMEOUT))
        except (DockerException, ConnectionError, OSError) as e:
            return f"ERROR: could not run tests ({e}). Fix the code by reasoning about it instead."

        if "PYTEST_TIMEOUT" in out:
            return "TESTS TIMED OUT — suite too slow to run here; rely on careful reasoning instead."
        passed, failed, failing_ids = parse_pytest_output(out)
        head = f"TESTS: {passed} passed, {failed} failed."
        if failed == 0 and passed > 0:
            return head + " Suite is green."
        if passed == 0 and failed == 0:
            return ("TESTS: could not collect any tests (import error or bad path). Tail:\n"
                    + out[-_MAX_TOOL_LOG_CHARS:])
        tail = out[-_MAX_TOOL_LOG_CHARS:]
        ids = ("\nFailing: " + ", ".join(failing_ids[:10])) if failing_ids else ""
        return f"{head} Fix the failures below, then re-run.{ids}\n{tail}"

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.remove(force=True)
            except DockerException:
                pass
            self._container = None
