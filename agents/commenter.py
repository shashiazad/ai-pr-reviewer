"""CommentAgent — posts idempotent summary and inline comments via GitHub API."""

from __future__ import annotations

import json
from typing import Any

from tools.common import get_logger, redact_secrets
from tools.github_client import GitHubClient

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# CommentAgent
# ---------------------------------------------------------------------------

DEFAULT_SUMMARY_MARKER = "<!-- AI-REVIEW-SUMMARY -->"
BOT_NAME = "🤖 AI Code Reviewer"


class CommentAgent:
    """Posts review comments to GitHub.  Supports dry-run mode."""

    def __init__(
        self,
        gh: GitHubClient,
        policy: dict[str, Any] | None = None,
        dry_run: bool = False,
    ):
        self.gh = gh
        self.policy = policy or {}
        self.dry_run = dry_run
        self.summary_marker: str = self.policy.get(
            "summary_marker", DEFAULT_SUMMARY_MARKER
        )
        self.max_inline: int = self.policy.get("max_inline_comments", 20)

    # -- public API ----------------------------------------------------------

    def post_results(
        self,
        repo: str,
        pr_number: int,
        head_sha: str,
        issues: list[dict[str, Any]],
        summary_text: str,
    ) -> dict[str, Any]:
        """Post summary + inline comments.  Returns receipt dict."""
        receipt: dict[str, Any] = {
            "summary_id": None,
            "inline_ids": [],
            "dry_run": self.dry_run,
            "total_issues": len(issues),
        }

        # --- Summary comment (idempotent) ---
        summary_body = f"{self.summary_marker}\n\n**{BOT_NAME}**\n\n{summary_text}"
        receipt["summary_id"] = self._upsert_summary(repo, pr_number, summary_body)

        # --- Inline comments ---
        inline_issues = issues[: self.max_inline]
        if self.dry_run:
            receipt["inline_ids"] = [
                {"file": i.get("file"), "line": i.get("line"), "dry_run": True}
                for i in inline_issues
            ]
            logger.info("Dry-run: would post %d inline comments", len(inline_issues))
            return receipt

        # Prefer batch review API for efficiency
        review_comments = self._build_review_comments(inline_issues)
        has_errors = any(
            i.get("severity") in ("error", "warn") for i in issues
        )
        review_event = "REQUEST_CHANGES" if has_errors else "COMMENT"
        if review_comments:
            try:
                review_id = self.gh.post_review(
                    repo=repo,
                    pr=pr_number,
                    commit_sha=head_sha,
                    body=f"**{BOT_NAME}** found **{len(issues)}** issue(s) requiring attention.",
                    comments=review_comments,
                    event=review_event,
                )
                receipt["inline_ids"].append({"review_id": review_id})
                logger.info(
                    "Posted batch review with %d comments (review_id=%s)",
                    len(review_comments),
                    review_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Batch review failed (%s); falling back to individual comments",
                    str(exc)[:200],
                )
                self._post_individual_comments(
                    repo, pr_number, head_sha, inline_issues, receipt
                )

        return receipt

    # -- summary helpers -----------------------------------------------------

    def _upsert_summary(self, repo: str, pr_number: int, body: str) -> str | None:
        """Create or update the summary comment."""
        if self.dry_run:
            logger.info("Dry-run: would upsert summary comment")
            return "dry-run"

        existing_id = self.gh.find_bot_comment(repo, pr_number, self.summary_marker)
        if existing_id:
            cid = self.gh.update_issue_comment(repo, existing_id, body)
            logger.info("Updated summary comment %s", cid)
            return cid
        else:
            cid = self.gh.post_issue_comment(repo, pr_number, body)
            logger.info("Created summary comment %s", cid)
            return cid

    # -- inline helpers ------------------------------------------------------

    @staticmethod
    def _build_review_comments(
        issues: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert issue dicts to GitHub review comment payloads."""
        comments: list[dict[str, Any]] = []
        for iss in issues:
            path = iss.get("file", "")
            line = iss.get("line")
            if not path or not isinstance(line, int):
                continue

            severity = iss.get("severity", "info").upper()
            severity_emoji = {"ERROR": "🔴", "WARN": "🟡", "INFO": "🔵"}.get(severity, "🔵")
            category = iss.get("category", "")
            message = iss.get("message", "")
            suggestion = iss.get("suggestion")
            dedupe_hash = iss.get("_hash", "")

            body_parts = [
                f"{severity_emoji} **{severity}** | `{category}`",
                "",
                f"**What's wrong:** {message}",
            ]
            if suggestion:
                body_parts.extend([
                    "",
                    f"**How to fix:** {suggestion}",
                ])
            body_parts.append(f"\n---\n*— {BOT_NAME}*")
            if dedupe_hash:
                body_parts.append(f"<!-- ai-review-hash:{dedupe_hash} -->")

            comments.append({
                "path": path,
                "line": line,
                "side": "RIGHT",
                "body": "\n".join(body_parts),
            })
        return comments

    def _post_individual_comments(
        self,
        repo: str,
        pr_number: int,
        head_sha: str,
        issues: list[dict[str, Any]],
        receipt: dict[str, Any],
    ) -> None:
        """Fallback: post inline comments one by one."""
        for iss in issues:
            path = iss.get("file", "")
            line = iss.get("line")
            if not path or not isinstance(line, int):
                continue
            severity = iss.get("severity", "info").upper()
            severity_emoji = {"ERROR": "🔴", "WARN": "🟡", "INFO": "🔵"}.get(severity, "🔵")
            category = iss.get("category", "")
            message = iss.get("message", "")
            suggestion = iss.get("suggestion")
            body_parts = [
                f"{severity_emoji} **{severity}** | `{category}`",
                "",
                f"**What's wrong:** {message}",
            ]
            if suggestion:
                body_parts.extend(["", f"**How to fix:** {suggestion}"])
            body_parts.append(f"\n---\n*— {BOT_NAME}*")
            body = "\n".join(body_parts)
            try:
                cid = self.gh.post_inline_comment_basic(
                    repo=repo,
                    pr=pr_number,
                    commit_sha=head_sha,
                    path=path,
                    line=line,
                    body=body,
                )
                receipt["inline_ids"].append(cid)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to post inline comment on %s:%d — %s",
                    path,
                    line,
                    str(exc)[:200],
                )

    # -- dry-run output ------------------------------------------------------

    def dry_run_output(
        self,
        issues: list[dict[str, Any]],
        summary_text: str,
    ) -> str:
        """Return a JSON string of what would be posted (for --dry-run mode)."""
        return json.dumps(
            {
                "summary": summary_text,
                "comments": [
                    {
                        "file": i.get("file"),
                        "line": i.get("line"),
                        "severity": i.get("severity"),
                        "message": i.get("message"),
                        "suggestion": i.get("suggestion"),
                    }
                    for i in issues
                ],
            },
            indent=2,
        )
