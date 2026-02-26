"""GitHub API client for the AI Code Reviewer.

Provides typed methods for PR metadata, diffs, and comment management.
All methods use the GitHub REST API v3 via ``requests``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

from tools.common import get_logger, redact_secrets

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PRMetadata:
    """Essential metadata for a pull request."""

    repo: str
    pr_number: int
    title: str = ""
    author: str = ""
    base_sha: str = ""
    head_sha: str = ""
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0


@dataclass
class ChangedFile:
    """A single file changed in a PR."""

    filename: str
    status: str = "modified"
    additions: int = 0
    deletions: int = 0
    patch: str = ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GitHubClient:
    """Thin wrapper around the GitHub REST API for PR review operations."""

    API_BASE = "https://api.github.com"

    def __init__(self, token: str | None = None, api_base: str | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.api_base = (api_base or self.API_BASE).rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        if self.token:
            self._session.headers["Authorization"] = f"Bearer {self.token}"

    # -- helpers -------------------------------------------------------------

    def _get(self, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.api_base}{path}"
        logger.debug("GET %s", url)
        resp = self._session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp

    def _post(self, path: str, json_body: dict[str, Any]) -> requests.Response:
        url = f"{self.api_base}{path}"
        logger.debug("POST %s", url)
        resp = self._session.post(url, json=json_body, timeout=30)
        resp.raise_for_status()
        return resp

    def _patch(self, path: str, json_body: dict[str, Any]) -> requests.Response:
        url = f"{self.api_base}{path}"
        logger.debug("PATCH %s", url)
        resp = self._session.patch(url, json=json_body, timeout=30)
        resp.raise_for_status()
        return resp

    # -- PR metadata ---------------------------------------------------------

    def get_pr_metadata(self, repo: str, pr: int) -> PRMetadata:
        """Fetch essential PR metadata."""
        data = self._get(f"/repos/{repo}/pulls/{pr}").json()
        return PRMetadata(
            repo=repo,
            pr_number=pr,
            title=data.get("title", ""),
            author=data.get("user", {}).get("login", ""),
            base_sha=data.get("base", {}).get("sha", ""),
            head_sha=data.get("head", {}).get("sha", ""),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            changed_files=data.get("changed_files", 0),
        )

    # -- Changed files -------------------------------------------------------

    def get_changed_files(self, repo: str, pr: int) -> list[ChangedFile]:
        """Return the list of files changed in the PR (paginated)."""
        files: list[ChangedFile] = []
        page = 1
        while True:
            resp = self._get(
                f"/repos/{repo}/pulls/{pr}/files",
                params={"per_page": 100, "page": page},
            )
            batch = resp.json()
            if not batch:
                break
            for f in batch:
                files.append(
                    ChangedFile(
                        filename=f["filename"],
                        status=f.get("status", "modified"),
                        additions=f.get("additions", 0),
                        deletions=f.get("deletions", 0),
                        patch=f.get("patch", ""),
                    )
                )
            page += 1
        return files

    # -- Unified diff --------------------------------------------------------

    def get_unified_diff(self, repo: str, pr: int) -> str:
        """Fetch the full unified diff for a PR."""
        resp = self._session.get(
            f"{self.api_base}/repos/{repo}/pulls/{pr}",
            headers={"Accept": "application/vnd.github.v3.diff"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.text

    # -- Bot comment management ----------------------------------------------

    def find_bot_comment(
        self, repo: str, pr: int, marker: str
    ) -> str | None:
        """Find an existing bot comment containing *marker*. Return comment ID or None."""
        page = 1
        while True:
            resp = self._get(
                f"/repos/{repo}/issues/{pr}/comments",
                params={"per_page": 100, "page": page},
            )
            comments = resp.json()
            if not comments:
                break
            for c in comments:
                if marker in c.get("body", ""):
                    return str(c["id"])
            page += 1
        return None

    def post_issue_comment(self, repo: str, pr: int, body: str) -> str:
        """Create a new issue comment and return its ID."""
        resp = self._post(
            f"/repos/{repo}/issues/{pr}/comments", {"body": body}
        )
        return str(resp.json()["id"])

    def update_issue_comment(
        self, repo: str, comment_id: str, body: str
    ) -> str:
        """Update an existing issue comment."""
        resp = self._patch(
            f"/repos/{repo}/issues/comments/{comment_id}", {"body": body}
        )
        return str(resp.json()["id"])

    # -- Inline comments (basic file:line) -----------------------------------

    def post_inline_comment_basic(
        self,
        repo: str,
        pr: int,
        commit_sha: str,
        path: str,
        line: int,
        body: str,
        side: str = "RIGHT",
    ) -> str:
        """Post a single-line review comment on a PR.

        Uses the pull request review comments API (not diff position).
        """
        payload: dict[str, Any] = {
            "body": body,
            "commit_id": commit_sha,
            "path": path,
            "line": line,
            "side": side,
        }
        resp = self._post(
            f"/repos/{repo}/pulls/{pr}/comments", payload
        )
        return str(resp.json()["id"])

    # -- Batch review (preferred for inline comments) ------------------------

    def post_review(
        self,
        repo: str,
        pr: int,
        commit_sha: str,
        body: str,
        comments: list[dict[str, Any]],
        event: str = "COMMENT",
    ) -> str:
        """Submit a pull request review with multiple inline comments at once."""
        payload: dict[str, Any] = {
            "commit_id": commit_sha,
            "body": body,
            "event": event,
            "comments": comments,
        }
        resp = self._post(f"/repos/{repo}/pulls/{pr}/reviews", payload)
        return str(resp.json()["id"])
