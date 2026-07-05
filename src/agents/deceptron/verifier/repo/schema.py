from agents.deceptron.miner.schema import Repo


class VerifiedRepo(Repo):
    test_verified: bool
    tests_passed: int
    tests_failed: int
    failing_test_ids: list[str]
    deselect_args: list[str]   # --deselect args to skip failing tests in future runs
    install_log: str = ""
    test_log: str = ""
    python_version: str
    verified_sha: str      # actual git SHA cloned during verification
    system_libs: list[str] = []   # GUI/native libs detected — repo is headless-incompatible
    error: str | None = None
    duration_s: float | None = None  # wall-clock seconds for Docker verification
