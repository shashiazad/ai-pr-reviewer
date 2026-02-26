#!/usr/bin/env python3
"""CLI entry point for the AI Code Reviewer.

Usage:
    python runner.py --repo owner/repo --pr 42
    python runner.py --repo owner/repo --pr 42 --dry-run
    python runner.py --repo owner/repo --pr 42 --audit-log audit.jsonl
    python runner.py --repo owner/repo --pr 42 --config .ai-reviewer.json

Environment overrides:
    GITHUB_TOKEN           — GitHub API token (required)
    AI_MODEL_BASE_URL      — override Dell Gateway base URL or localhost
    AI_MODEL_TOKEN         — static bearer token (testing only)
    AI_REVIEW_MAX_COMMENTS — max inline comments (default: 20)
    AI_REVIEW_SEVERITY     — minimum severity: info, warn, error
    AI_REVIEW_TEMPERATURE  — LLM temperature (default: 0.1)
    AI_REVIEW_TIMEOUT      — max runtime seconds (default: 300)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from agents.orchestrator import AgentOrchestrator
from tools.common import get_logger

logger = get_logger("runner")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="ai-code-reviewer",
        description="Fully agentic AI Code Reviewer for GitHub Pull Requests.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository in owner/repo format.",
    )
    parser.add_argument(
        "--pr",
        type=int,
        required=True,
        help="Pull request number.",
    )
    parser.add_argument(
        "--config",
        default=".ai-reviewer.json",
        help="Path to .ai-reviewer.json config file (default: .ai-reviewer.json).",
    )
    parser.add_argument(
        "--policy",
        default="configs/agent_policy.json",
        help="Path to agent policy JSON (default: configs/agent_policy.json).",
    )
    parser.add_argument(
        "--templates",
        default="policies/prompt_templates.json",
        help="Path to prompt templates JSON (default: policies/prompt_templates.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Output review as JSON without posting to GitHub.",
    )
    parser.add_argument(
        "--audit-log",
        default=None,
        help="Path to JSONL audit log file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the agentic AI code review pipeline."""
    args = parse_args(argv)

    # Validate required env
    if not os.environ.get("GITHUB_TOKEN"):
        logger.error("GITHUB_TOKEN environment variable is required")
        print("Error: GITHUB_TOKEN environment variable is not set.", file=sys.stderr)
        return 1

    # Apply env overrides to dry-run
    dry_run = args.dry_run or os.environ.get("AI_REVIEW_DRY_RUN", "").lower() in (
        "1", "true", "yes",
    )

    logger.info(
        "Starting review: repo=%s pr=%d dry_run=%s",
        args.repo,
        args.pr,
        dry_run,
    )

    orchestrator = AgentOrchestrator(
        repo=args.repo,
        pr_number=args.pr,
        config_path=args.config,
        policy_path=args.policy,
        templates_path=args.templates,
        dry_run=dry_run,
        audit_log=args.audit_log,
    )

    result = orchestrator.run()

    # Print result summary
    if dry_run:
        print(json.dumps(result, indent=2, default=str))
    else:
        final_state = result.get("final_state", "UNKNOWN")
        n_issues = result.get("total_validated_issues", 0)
        elapsed = result.get("elapsed_seconds", 0)
        logger.info(
            "Review finished: state=%s issues=%d elapsed=%.1fs",
            final_state,
            n_issues,
            elapsed,
        )

    # Exit code: 0 = success, 1 = failure
    if result.get("final_state") in ("DONE", "SUMMARIZE"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
