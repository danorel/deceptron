from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Repo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    full_name: str
    url: str
    clone_url: str
    language: str
    stars: int
    contributors_count: int
    license: str
    pushed_at: datetime
    default_branch: str
    has_tests: bool

    @classmethod
    def from_github_response(
        cls, raw: dict, contributors_count: int, has_tests: bool
    ) -> "Repo":
        license_info = raw.get("license") or {}
        return cls(
            id=raw["id"],
            full_name=raw["full_name"],
            url=raw["html_url"],
            clone_url=raw["clone_url"],
            language=(raw.get("language") or "").lower(),
            stars=raw["stargazers_count"],
            contributors_count=contributors_count,
            license=license_info.get("spdx_id", "NOASSERTION"),
            pushed_at=raw["pushed_at"],
            default_branch=raw["default_branch"],
            has_tests=has_tests,
        )
