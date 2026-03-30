import re
import time
from datetime import UTC, datetime, timedelta

import requests

SEARCH_SLEEP = 2.0  # seconds between search API calls (30 req/min limit)

_TEST_INDICATORS: dict[str, set[str]] = {
    "python": {"tests", "test", "pytest.ini", "setup.cfg", "pyproject.toml"},
}


class GitHubAPIError(Exception):
    pass


class SearchExhausted(Exception):
    pass


class GitHubClient:
    _BASE = "https://api.github.com"

    def __init__(self, token: str) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def _request(self, method: str, url: str, **kwargs) -> dict | list:
        resp = self._session.request(method, url, **kwargs)

        if resp.status_code in (403, 429):
            remaining = resp.headers.get("X-RateLimit-Remaining", "1")
            if remaining == "0":
                reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                sleep_for = max(reset_ts - time.time() + 1, 1)
                time.sleep(sleep_for)
                resp = self._session.request(method, url, **kwargs)

        if resp.status_code == 422:
            raise SearchExhausted(url)

        if not resp.ok:
            try:
                msg = resp.json().get("message", resp.text)
            except Exception:
                msg = resp.text
            raise GitHubAPIError(f"HTTP {resp.status_code}: {msg}")

        return resp.json()

    def search_repos(self, language: str, page: int, per_page: int = 100) -> list[dict]:
        if page > 10:
            raise SearchExhausted("GitHub Search caps at 1000 results (10 pages)")

        pushed_after = (datetime.now(UTC) - timedelta(days=90)).date().isoformat()
        query = f"language:{language} stars:100..2000 pushed:>={pushed_after} fork:false"

        data = self._request(
            "GET",
            f"{self._BASE}/search/repositories",
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            },
        )
        time.sleep(SEARCH_SLEEP)

        total = data.get("total_count", 0)
        if (page - 1) * per_page >= total:
            return []

        return data.get("items", [])

    def get_contributors_count(self, full_name: str) -> int:
        """Returns contributor count by reading the Link header from a 1-item page."""
        resp = self._session.get(
            f"{self._BASE}/repos/{full_name}/contributors",
            params={"per_page": 1, "anon": "false"},
        )
        if resp.status_code == 204:
            return 0
        if not resp.ok:
            raise GitHubAPIError(f"HTTP {resp.status_code} for {full_name}/contributors")

        link = resp.headers.get("Link", "")
        match = re.search(r'page=(\d+)>;\s*rel="last"', link)
        if match:
            return int(match.group(1))

        # No Link header → all contributors fit in one page
        return len(resp.json())

    def get_default_branch_commit_date(self, full_name: str, default_branch: str) -> datetime | None:
        """Return the date of the latest commit on the default branch."""
        try:
            data = self._request(
                "GET",
                f"{self._BASE}/repos/{full_name}/commits",
                params={"sha": default_branch, "per_page": 1},
            )
        except GitHubAPIError:
            return None

        if not isinstance(data, list) or not data:
            return None

        raw_date = data[0]["commit"]["committer"]["date"]
        return datetime.fromisoformat(raw_date.replace("Z", "+00:00"))

    def check_ci_passing(self, full_name: str, default_branch: str) -> bool:
        """Return True if the latest GitHub Actions run on default branch succeeded.

        Returns True when there are no workflow runs (no CI is neutral).
        """
        try:
            data = self._request(
                "GET",
                f"{self._BASE}/repos/{full_name}/actions/runs",
                params={"branch": default_branch, "per_page": 1},
            )
        except GitHubAPIError:
            return True  # can't determine → don't penalise

        runs = data.get("workflow_runs", [])
        if not runs:
            return True  # no CI configured → neutral

        latest = runs[0]
        return latest.get("conclusion") == "success"

    def check_has_tests(self, full_name: str, language: str) -> bool:
        """Check root directory for test indicators specific to the language."""
        try:
            entries = self._request("GET", f"{self._BASE}/repos/{full_name}/contents/")
        except GitHubAPIError:
            return False

        if not isinstance(entries, list):
            return False

        names = {item["name"].lower() for item in entries}
        indicators = _TEST_INDICATORS.get(language.lower(), {"tests", "test"})
        return bool(names & indicators)
