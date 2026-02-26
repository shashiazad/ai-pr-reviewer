#!/usr/bin/env python3
"""AI Code Reviewer — Main Entry Point.

Orchestrates the full review pipeline:
  1. Load configuration
  2. Fetch PR diff and metadata
  3. Parse and filter diff into per-file structures
  4. Run regex-based security pre-scan
  5. Send diff chunks to LLM for review
  6. Merge findings, deduplicate, apply budget
  7. Generate PR summary
  8. Post inline comments and summary review

Usage (from GitHub Actions):
    python scripts/review_pr.py

Environment:
    GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_EVENT_PATH — set by Actions.
    See config.py and DESIGN.md for all configuration options.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on path for model/ imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.config import ReviewerConfig, load_config
from scripts.diff_parser import (
    FileDiff,
    chunk_file_diff,
    fetch_pr_diff,
    fetch_pr_metadata,
    filter_files,
    get_diff_position_map,
    parse_diff,
)
from scripts.llm_client import LLMClient
from scripts.comment_manager import (
    CommentManager,
    prioritize_and_budget,
    issue_hash,
)
from scripts.prompts import detect_language
from scripts.security_scanner import (
    findings_to_issues,
    redact_secrets,
    scan_file_diff,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[ai-reviewer] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ai-reviewer")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def write_audit_log(cfg: ReviewerConfig, audit_data: Dict[str, Any]) -> None:
    """Write audit log to runner temp directory if enabled."""
    if not cfg.audit_log:
        return
    runner_temp = os.getenv("RUNNER_TEMP", os.path.join(_project_root, ".tmp"))
    os.makedirs(runner_temp, exist_ok=True)
    path = os.path.join(runner_temp, "ai-reviewer-audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, default=str)
    logger.info("Audit log written to %s", path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the AI code review pipeline. Returns 0 on success, 1 on error."""
    start_time = time.monotonic()

    # 1. Load config
    logger.info("Loading configuration...")
    cfg = load_config()

    if not cfg.enabled:
        logger.info("AI reviewer is disabled via config. Exiting.")
        return 0

    if not cfg.github_token:
        logger.error("GITHUB_TOKEN is not set. Cannot proceed.")
        return 1

    if not cfg.pr_number:
        logger.error("Could not determine PR number from GITHUB_EVENT_PATH.")
        return 1

    logger.info(
        "Reviewing PR #%d in %s (dry_run=%s)",
        cfg.pr_number, cfg.github_repository, cfg.dry_run,
    )

    # 2. Fetch PR diff and metadata
    logger.info("Fetching PR diff...")
    try:
        raw_diff = fetch_pr_diff(cfg)
        pr_meta = fetch_pr_metadata(cfg)
    except Exception as exc:
        logger.error("Failed to fetch PR data: %s", str(exc)[:300])
        return 1

    logger.info(
        "PR '%s' by %s — %d files, +%d/-%d lines",
        pr_meta["title"],
        pr_meta["author"],
        pr_meta["changed_files"],
        pr_meta["additions"],
        pr_meta["deletions"],
    )

    # Update head_sha from metadata if not set
    if not cfg.head_sha:
        cfg.head_sha = pr_meta.get("head_sha", "")

    # 3. Parse and filter diff
    logger.info("Parsing diff...")
    file_diffs = parse_diff(raw_diff)
    file_diffs = filter_files(file_diffs, cfg)
    logger.info("Files to review after filtering: %d", len(file_diffs))

    if not file_diffs:
        logger.info("No reviewable files in this PR. Exiting.")
        return 0

    # 4. Security pre-scan + 5. LLM review
    all_issues: List[Dict[str, Any]] = []
    position_maps: Dict[str, Dict[int, int]] = {}
    files_reviewed = 0

    llm_client = LLMClient(cfg)

    for fd in file_diffs:
        language = detect_language(fd.filename)
        if language and language not in cfg.languages:
            logger.info("Skipping %s (language %s not in config)", fd.filename, language)
            continue

        logger.info("Reviewing: %s (%s)", fd.filename, language or "unknown")
        files_reviewed += 1

        # Build position map for inline commenting
        position_maps[fd.filename] = get_diff_position_map(fd)

        # 4a. Security pre-scan (fast, regex-based)
        sec_findings = scan_file_diff(fd.filename, fd.hunks, language)
        if sec_findings:
            logger.info("  Security scanner: %d findings", len(sec_findings))
            all_issues.extend(findings_to_issues(sec_findings))

        # 5a. LLM review (per chunk)
        chunks = chunk_file_diff(fd)
        for i, chunk in enumerate(chunks):
            try:
                logger.info("  LLM review chunk %d/%d (%d lines)", i + 1, len(chunks), chunk.count("\n"))
                issues = llm_client.review_chunk(
                    filename=fd.filename,
                    language=language or "general",
                    diff_content=chunk,
                )
                if issues:
                    logger.info("  LLM found %d issues in chunk %d", len(issues), i + 1)
                    all_issues.extend(issues)
            except TimeoutError:
                logger.warning("Time budget exceeded, stopping LLM reviews.")
                break
            except Exception as exc:
                logger.error("LLM review failed for %s chunk %d: %s", fd.filename, i + 1, str(exc)[:200])
                continue

    # 6. Deduplicate and budget
    logger.info("Total raw issues: %d", len(all_issues))

    # Deduplicate
    seen_hashes: set = set()
    unique_issues: List[Dict[str, Any]] = []
    for issue in all_issues:
        h = issue_hash(issue)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_issues.append(issue)

    logger.info("Unique issues after dedup: %d", len(unique_issues))

    # Apply severity budget
    budgeted = prioritize_and_budget(
        unique_issues,
        max_comments=cfg.max_comments,
        severity_threshold=cfg.severity_threshold,
    )
    logger.info("Issues after budget: %d", len(budgeted))

    # 7. Generate summary
    summary_md = ""
    if cfg.summary.enabled:
        logger.info("Generating PR summary...")
        try:
            summary_md = llm_client.generate_summary(
                aggregated_results=unique_issues,
                pr_title=pr_meta["title"],
                pr_author=pr_meta["author"],
                files_changed=pr_meta["changed_files"],
                lines_added=pr_meta["additions"],
                lines_removed=pr_meta["deletions"],
            )
        except Exception as exc:
            logger.error("Summary generation failed: %s", str(exc)[:200])
            summary_md = _fallback_summary(unique_issues, pr_meta, files_reviewed)
    else:
        summary_md = _fallback_summary(unique_issues, pr_meta, files_reviewed)

    # 8. Post comments
    logger.info("Posting review...")
    comment_mgr = CommentManager(cfg)
    try:
        comment_mgr.post_review(
            issues=budgeted,
            summary_markdown=summary_md,
            position_maps=position_maps,
        )
    except Exception as exc:
        logger.error("Failed to post review: %s", str(exc)[:300])
        return 1

    # Stats
    elapsed = time.monotonic() - start_time
    llm_stats = llm_client.stats
    logger.info(
        "Done in %.1fs — %d files, %d issues posted, %d LLM calls",
        elapsed,
        files_reviewed,
        len(budgeted),
        llm_stats["total_llm_calls"],
    )

    # Audit log
    write_audit_log(cfg, {
        "pr_number": cfg.pr_number,
        "repository": cfg.github_repository,
        "files_reviewed": files_reviewed,
        "issues_found": len(unique_issues),
        "issues_posted": len(budgeted),
        "llm_calls": llm_stats["total_llm_calls"],
        "approx_tokens": llm_stats["approx_tokens_used"],
        "elapsed_seconds": round(elapsed, 1),
        "dry_run": cfg.dry_run,
    })

    return 0


def _fallback_summary(
    issues: List[Dict[str, Any]],
    pr_meta: Dict[str, Any],
    files_reviewed: int,
) -> str:
    """Generate a simple fallback summary without LLM."""
    error_count = sum(1 for i in issues if i.get("severity") == "error")
    warn_count = sum(1 for i in issues if i.get("severity") == "warn")
    info_count = sum(1 for i in issues if i.get("severity") == "info")

    if not issues:
        return "## AI Code Review Summary\n\n✓ No significant issues found.\n"

    lines = [
        "## AI Code Review Summary\n",
        f"**PR**: {pr_meta.get('title', 'N/A')}  ",
        f"**Author**: {pr_meta.get('author', 'N/A')}\n",
        "### Statistics",
        f"- Files reviewed: {files_reviewed}",
        f"- Issues found: {len(issues)} ({error_count} errors, {warn_count} warnings, {info_count} info)\n",
        "### Checklist",
        "- [ ] All flagged security issues addressed",
        "- [ ] Format/lint issues resolved",
        "- [ ] Error handling reviewed",
        "- [ ] Tests cover new/changed logic",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
