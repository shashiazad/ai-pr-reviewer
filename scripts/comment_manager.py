"""GitHub comment posting for AI Code Reviewer.

Handles inline review comments, summary reviews, deduplication,
budget enforcement, nit consolidation, and idempotent summary updates.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import requests

from .config import ReviewerConfig

logger = logging.getLogger("ai-reviewer.comments")

SUMMARY_MARKER = "<!-- ai-reviewer-summary -->"
SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def issue_hash(issue: Dict[str, Any]) -> str:
    """Generate a stable hash for deduplication of an issue."""
    raw = f"{issue.get('file', '')}|{issue.get('line', '')}|{issue.get('category', '')}|{issue.get('message', '')[:100]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Budget & Prioritization
# ---------------------------------------------------------------------------

def prioritize_and_budget(
    issues: List[Dict[str, Any]],
    max_comments: int = 20,
    severity_threshold: str = "info",
) -> List[Dict[str, Any]]:
    """Sort issues by severity, apply threshold filter, and enforce budget."""
    threshold_rank = SEVERITY_ORDER.get(severity_threshold, 2)

    # Filter by threshold
    filtered = [
        i for i in issues
        if SEVERITY_ORDER.get(i.get("severity", "info"), 2) <= threshold_rank
    ]

    # Sort: error first, then warn, then info
    filtered.sort(key=lambda i: SEVERITY_ORDER.get(i.get("severity", "info"), 2))

    # Budget
    return filtered[:max_comments]


# ---------------------------------------------------------------------------
# Nit Consolidation
# ---------------------------------------------------------------------------

def consolidate_nits(issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group info-severity issues by file for consolidated commenting.

    Returns a dict: filename -> list of info issues.
    Non-info issues are not included (they get individual comments).
    """
    nit_groups: Dict[str, List[Dict[str, Any]]] = {}
    for issue in issues:
        if issue.get("severity") == "info":
            fname = issue.get("file", "unknown")
            nit_groups.setdefault(fname, []).append(issue)
    return nit_groups


# ---------------------------------------------------------------------------
# GitHub API Helpers
# ---------------------------------------------------------------------------

class CommentManager:
    """Manages posting review comments to GitHub PRs."""

    def __init__(self, cfg: ReviewerConfig) -> None:
        self.cfg = cfg
        self.api_base = f"https://api.github.com/repos/{cfg.github_repository}"
        self.headers = {
            "Authorization": f"Bearer {cfg.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._existing_hashes: Optional[set] = None

    def post_review(
        self,
        issues: List[Dict[str, Any]],
        summary_markdown: str,
        position_maps: Dict[str, Dict[int, int]],
    ) -> None:
        """Post inline review comments and a summary to the PR.

        Args:
            issues: Validated, prioritized, budgeted list of issues.
            summary_markdown: The PR summary in Markdown.
            position_maps: Dict of filename -> {line_number: diff_position}.
        """
        if self.cfg.dry_run:
            self._dry_run_output(issues, summary_markdown)
            return

        # Load existing comment hashes for dedup
        existing = self._get_existing_comment_hashes()

        # Separate nits from high-signal issues
        high_signal = [i for i in issues if i.get("severity") != "info"]
        nit_groups = consolidate_nits(issues)

        # Build review comments
        comments: List[Dict[str, Any]] = []

        for issue in high_signal:
            h = issue_hash(issue)
            if h in existing:
                logger.info("Skipping duplicate: %s", h)
                continue

            pos = self._resolve_position(issue, position_maps)
            if pos is None:
                continue

            body = self._format_inline_comment(issue)
            comments.append({
                "path": issue["file"],
                "position": pos,
                "body": body,
            })

        # Add consolidated nit comments
        if self.cfg.nit_consolidation:
            for fname, nits in nit_groups.items():
                # Dedupe nits
                new_nits = [n for n in nits if issue_hash(n) not in existing]
                if not new_nits:
                    continue

                body = self._format_consolidated_nits(new_nits)
                # Place at the first nit's position
                pos = self._resolve_position(new_nits[0], position_maps)
                if pos is None:
                    continue

                comments.append({
                    "path": fname,
                    "position": pos,
                    "body": body,
                })

        # Post as a single review with all comments
        self._submit_review(comments, summary_markdown)

    def _resolve_position(
        self,
        issue: Dict[str, Any],
        position_maps: Dict[str, Dict[int, int]],
    ) -> Optional[int]:
        """Resolve a line number to a diff position for inline commenting."""
        fname = issue.get("file", "")
        line = issue.get("line", 0)
        fmap = position_maps.get(fname, {})

        # Exact match
        if line in fmap:
            return fmap[line]

        # Try nearby lines (±3) for close matches
        for offset in range(1, 4):
            if (line + offset) in fmap:
                return fmap[line + offset]
            if (line - offset) in fmap:
                return fmap[line - offset]

        logger.warning(
            "Could not resolve position for %s:%d, skipping inline comment.",
            fname, line,
        )
        return None

    def _submit_review(
        self,
        comments: List[Dict[str, Any]],
        summary_markdown: str,
    ) -> None:
        """Submit a PR review with inline comments and summary body."""
        # Update or create the summary
        body = f"{SUMMARY_MARKER}\n{summary_markdown}"

        payload: Dict[str, Any] = {
            "body": body,
            "event": "COMMENT",
            "comments": comments,
        }

        if self.cfg.head_sha:
            payload["commit_id"] = self.cfg.head_sha

        url = f"{self.api_base}/pulls/{self.cfg.pr_number}/reviews"

        try:
            # Check for existing summary review to update
            existing_review_id = self._find_existing_summary_review()
            if existing_review_id:
                # Update existing review body (comments are additive)
                update_url = f"{url}/{existing_review_id}"
                update_payload = {"body": body}
                resp = requests.put(
                    update_url, headers=self.headers, json=update_payload, timeout=30
                )
                resp.raise_for_status()
                logger.info("Updated existing summary review #%s", existing_review_id)

                # Post new inline comments via a new review if any
                if comments:
                    new_review = {
                        "event": "COMMENT",
                        "comments": comments,
                    }
                    if self.cfg.head_sha:
                        new_review["commit_id"] = self.cfg.head_sha
                    resp = requests.post(
                        url, headers=self.headers, json=new_review, timeout=30
                    )
                    resp.raise_for_status()
            else:
                resp = requests.post(
                    url, headers=self.headers, json=payload, timeout=30
                )
                resp.raise_for_status()
                logger.info("Posted new review with %d inline comments.", len(comments))

        except requests.RequestException as exc:
            logger.error("Failed to post review: %s", str(exc)[:300])
            raise

    def _find_existing_summary_review(self) -> Optional[int]:
        """Find an existing AI reviewer summary review to update."""
        url = f"{self.api_base}/pulls/{self.cfg.pr_number}/reviews"
        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            for review in resp.json():
                body = review.get("body", "") or ""
                if SUMMARY_MARKER in body:
                    return review["id"]
        except requests.RequestException:
            pass
        return None

    def _get_existing_comment_hashes(self) -> set:
        """Fetch existing review comments and compute hashes for dedup."""
        if self._existing_hashes is not None:
            return self._existing_hashes

        self._existing_hashes = set()
        url = f"{self.api_base}/pulls/{self.cfg.pr_number}/comments"
        try:
            resp = requests.get(
                url, headers=self.headers, params={"per_page": 100}, timeout=30
            )
            resp.raise_for_status()
            for comment in resp.json():
                body = comment.get("body", "")
                # Extract hash from comment footer if present
                if "<!-- hash:" in body:
                    start = body.index("<!-- hash:") + len("<!-- hash:")
                    end = body.index("-->", start)
                    self._existing_hashes.add(body[start:end].strip())
        except requests.RequestException:
            pass

        return self._existing_hashes

    def _format_inline_comment(self, issue: Dict[str, Any]) -> str:
        """Format a single issue into a GitHub review comment body."""
        severity_emoji = {"error": "🔴", "warn": "🟡", "info": "🔵"}.get(
            issue.get("severity", "info"), "🔵"
        )
        parts = [
            f"{severity_emoji} **{issue.get('severity', 'info').upper()}** | {issue.get('category', 'general')}",
            "",
            issue.get("message", ""),
        ]
        if issue.get("suggestion"):
            parts.extend(["", f"**Suggestion:** {issue['suggestion']}"])

        h = issue_hash(issue)
        parts.extend(["", f"<!-- hash:{h} -->"])

        return "\n".join(parts)

    def _format_consolidated_nits(self, nits: List[Dict[str, Any]]) -> str:
        """Format multiple info-severity issues into a consolidated comment."""
        parts = [
            "🔵 **Minor Suggestions**",
            "",
        ]
        for nit in nits:
            line = nit.get("line", "?")
            msg = nit.get("message", "")
            parts.append(f"- **Line {line}**: {msg}")

        # Hash the consolidated set
        combined = "|".join(issue_hash(n) for n in nits)
        h = hashlib.sha256(combined.encode()).hexdigest()[:16]
        parts.extend(["", f"<!-- hash:{h} -->"])

        return "\n".join(parts)

    def _dry_run_output(
        self,
        issues: List[Dict[str, Any]],
        summary_markdown: str,
    ) -> None:
        """Print review output to stdout instead of posting to GitHub."""
        print("\n" + "=" * 70)
        print("AI CODE REVIEW — DRY RUN")
        print("=" * 70)

        if issues:
            print(f"\n--- Inline Comments ({len(issues)}) ---\n")
            for issue in issues:
                sev = issue.get("severity", "info").upper()
                print(f"  [{sev}] {issue.get('file', '?')}:{issue.get('line', '?')}")
                print(f"         {issue.get('category', '')}: {issue.get('message', '')}")
                if issue.get("suggestion"):
                    print(f"         Suggestion: {issue['suggestion']}")
                print()
        else:
            print("\nNo inline comments to post.")

        print("--- Summary ---\n")
        print(summary_markdown)
        print("\n" + "=" * 70)
