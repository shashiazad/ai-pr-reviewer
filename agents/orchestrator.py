"""AgentOrchestrator — ReAct state machine that drives the full review pipeline.

State machine:
  PLAN → GATHER_DIFF → RUN_LINTERS → REVIEW_CHUNKS → CRITIQUE_VALIDATE
  → POST_COMMENTS → SUMMARIZE → DONE

On failure at any state: SELF_HEAL_RETRY (bounded) → FAIL_GRACEFULLY
"""

from __future__ import annotations

import json
import os
import traceback
from enum import Enum
from typing import Any

from tools.common import (
    TimeBudget,
    get_logger,
    load_json_file,
    write_audit_log,
    env_int,
    env_or,
)
from tools.github_client import GitHubClient, PRMetadata
from tools.llm_client import LLMClient
from tools.linters import LintResult, run_all_linters
from tools.diff_utils import parse_unified_diff, build_line_map

from agents.planner import PlannerAgent, ReviewPlan
from agents.reviewer import ReviewerAgent
from agents.critic import CriticAgent
from agents.commenter import CommentAgent

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


class State(str, Enum):
    PLAN = "PLAN"
    GATHER_DIFF = "GATHER_DIFF"
    RUN_LINTERS = "RUN_LINTERS"
    REVIEW_CHUNKS = "REVIEW_CHUNKS"
    CRITIQUE_VALIDATE = "CRITIQUE_VALIDATE"
    POST_COMMENTS = "POST_COMMENTS"
    SUMMARIZE = "SUMMARIZE"
    DONE = "DONE"
    SELF_HEAL_RETRY = "SELF_HEAL_RETRY"
    FAIL_GRACEFULLY = "FAIL_GRACEFULLY"


# Ordered pipeline
_PIPELINE = [
    State.PLAN,
    State.GATHER_DIFF,
    State.RUN_LINTERS,
    State.REVIEW_CHUNKS,
    State.CRITIQUE_VALIDATE,
    State.POST_COMMENTS,
    State.SUMMARIZE,
    State.DONE,
]

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """Drives the end-to-end agentic review loop."""

    def __init__(
        self,
        repo: str,
        pr_number: int,
        config_path: str = ".ai-reviewer.json",
        policy_path: str = "configs/agent_policy.json",
        templates_path: str = "policies/prompt_templates.json",
        dry_run: bool = False,
        audit_log: str | None = None,
    ):
        self.repo = repo
        self.pr_number = pr_number
        self.dry_run = dry_run
        self.audit_log = audit_log

        # Load configuration layers
        self.config = load_json_file(config_path)
        self.policy = load_json_file(policy_path)
        self.templates = load_json_file(templates_path)

        # Budget
        orch_policy = self.policy.get("orchestrator", {})
        max_seconds = env_int("AI_REVIEW_TIMEOUT", orch_policy.get("max_runtime_seconds", 300))
        self.budget = TimeBudget(max_seconds)
        self.max_retries: int = orch_policy.get("max_retries_per_state", 2)
        self.max_comments: int = env_int(
            "AI_REVIEW_MAX_COMMENTS",
            orch_policy.get("max_comments", 20),
        )

        # Instantiate tools
        self.gh = GitHubClient()
        self.llm = LLMClient(
            max_retries=self.policy.get("reviewer", {}).get("schema_failure_retries", 2),
        )

        # Instantiate agents
        self.planner = PlannerAgent(self.policy.get("planner", {}))
        self.reviewer = ReviewerAgent(
            llm=self.llm,
            prompt_templates=self.templates,
            policy=self.policy.get("reviewer", {}),
        )
        self.critic = CriticAgent({
            **self.policy.get("critic", {}),
            "max_inline_comments": self.max_comments,
            "nit_consolidation": self.policy.get("commenter", {}).get("nit_consolidation", True),
            "nit_threshold": self.policy.get("commenter", {}).get("nit_threshold", 3),
        })
        self.commenter = CommentAgent(
            gh=self.gh,
            policy=self.policy.get("commenter", {}),
            dry_run=self.dry_run,
        )

        # Run state
        self._state = State.PLAN
        self._pr: PRMetadata | None = None
        self._raw_diff: str = ""
        self._plan: ReviewPlan | None = None
        self._linter_results: list[LintResult] = []
        self._raw_issues: list[dict[str, Any]] = []
        self._validated_issues: list[dict[str, Any]] = []
        self._summary: str = ""
        self._receipt: dict[str, Any] = {}

    # -- main entry point ----------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full pipeline.  Returns a receipt dict."""
        logger.info(
            "Orchestrator starting: repo=%s pr=%d dry_run=%s",
            self.repo,
            self.pr_number,
            self.dry_run,
        )
        retry_count: dict[str, int] = {}

        idx = 0
        while idx < len(_PIPELINE):
            state = _PIPELINE[idx]
            self._state = state

            if self.budget.expired and state not in (State.POST_COMMENTS, State.SUMMARIZE, State.DONE):
                logger.warning("Budget expired at state %s; jumping to POST_COMMENTS", state)
                idx = _PIPELINE.index(State.POST_COMMENTS)
                continue

            try:
                self._log_transition(state)
                self._execute_state(state)
                idx += 1
            except Exception as exc:  # noqa: BLE001
                state_key = state.value
                retry_count.setdefault(state_key, 0)
                retry_count[state_key] += 1

                if retry_count[state_key] <= self.max_retries:
                    logger.warning(
                        "State %s failed (attempt %d/%d): %s — retrying",
                        state_key,
                        retry_count[state_key],
                        self.max_retries,
                        str(exc)[:300],
                    )
                    # Stay on same index to retry
                else:
                    logger.error(
                        "State %s failed after %d retries: %s",
                        state_key,
                        self.max_retries,
                        str(exc)[:300],
                    )
                    self._fail_gracefully(exc)
                    break

        result = self._build_result()
        if self.audit_log:
            write_audit_log(self.audit_log, result)
        return result

    # -- state dispatch ------------------------------------------------------

    def _execute_state(self, state: State) -> None:
        dispatch = {
            State.PLAN: self._do_plan,
            State.GATHER_DIFF: self._do_gather_diff,
            State.RUN_LINTERS: self._do_run_linters,
            State.REVIEW_CHUNKS: self._do_review_chunks,
            State.CRITIQUE_VALIDATE: self._do_critique,
            State.POST_COMMENTS: self._do_post_comments,
            State.SUMMARIZE: self._do_summarize,
            State.DONE: self._do_done,
        }
        handler = dispatch.get(state)
        if handler:
            handler()
        else:
            raise ValueError(f"Unknown state: {state}")

    # -- individual state handlers -------------------------------------------

    def _do_plan(self) -> None:
        # Fetch PR metadata first (needed for planning)
        self._pr = self.gh.get_pr_metadata(self.repo, self.pr_number)
        logger.info("PR: %s by %s", self._pr.title, self._pr.author)

    def _do_gather_diff(self) -> None:
        assert self._pr is not None
        self._raw_diff = self.gh.get_unified_diff(self.repo, self.pr_number)
        self._plan = self.planner.plan(self._pr, self._raw_diff, self.budget)

        if self._plan.skip_review:
            logger.info("Plan says skip: %s", self._plan.skip_reason)
            # Jump directly to DONE by setting state appropriately
            self._summary = (
                f"## AI Code Review\n\nNo reviewable files found. "
                f"Reason: {self._plan.skip_reason}"
            )

    def _do_run_linters(self) -> None:
        if self._plan and self._plan.skip_review:
            return
        assert self._plan is not None
        enabled = {tool: True for tool in self._plan.linters_to_run}
        changed_files = [t.filename for t in self._plan.file_tasks]
        self._linter_results = run_all_linters(
            changed_files=changed_files, enabled=enabled
        )
        logger.info(
            "Linters completed: %d tools run",
            len(self._linter_results),
        )

    def _do_review_chunks(self) -> None:
        if self._plan and self._plan.skip_review:
            return
        assert self._plan is not None
        self._raw_issues, self._summary = self.reviewer.review_plan(
            self._plan,
            linter_results=self._linter_results,
            budget=self.budget,
        )
        logger.info("Reviewer produced %d raw issues", len(self._raw_issues))

    def _do_critique(self) -> None:
        if self._plan and self._plan.skip_review:
            return
        self._validated_issues = self.critic.critique(self._raw_issues)

        # Advisory contradiction check
        contradictions = self.critic.detect_contradictions(self._validated_issues)
        if contradictions:
            logger.warning(
                "Critic detected %d potential contradictions", len(contradictions)
            )

    def _do_post_comments(self) -> None:
        assert self._pr is not None
        self._receipt = self.commenter.post_results(
            repo=self.repo,
            pr_number=self.pr_number,
            head_sha=self._pr.head_sha,
            issues=self._validated_issues,
            summary_text=self._summary,
        )

    def _do_summarize(self) -> None:
        logger.info(
            "Review complete: %d validated issues, summary posted=%s",
            len(self._validated_issues),
            bool(self._receipt.get("summary_id")),
        )

    def _do_done(self) -> None:
        logger.info("Orchestrator finished in %.1fs", self.budget.elapsed)

    # -- failure handling ----------------------------------------------------

    def _fail_gracefully(self, exc: Exception) -> None:
        """Post a minimal failure summary to avoid silent failures."""
        self._state = State.FAIL_GRACEFULLY
        logger.error("Failing gracefully: %s", str(exc)[:500])

        failure_summary = (
            f"## AI Code Review — Failure\n\n"
            f"The automated review encountered an error and could not complete.\n\n"
            f"```\n{str(exc)[:300]}\n```\n\n"
            f"*Please review this PR manually.*"
        )

        try:
            if self._pr:
                self.commenter.post_results(
                    repo=self.repo,
                    pr_number=self.pr_number,
                    head_sha=self._pr.head_sha if self._pr else "",
                    issues=[],
                    summary_text=failure_summary,
                )
        except Exception as post_exc:  # noqa: BLE001
            logger.error("Failed to post failure summary: %s", str(post_exc)[:200])

    # -- helpers -------------------------------------------------------------

    def _log_transition(self, state: State) -> None:
        logger.info(
            "State: %s (elapsed=%.1fs, remaining=%.1fs)",
            state.value,
            self.budget.elapsed,
            self.budget.remaining,
        )

    def _build_result(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "pr_number": self.pr_number,
            "final_state": self._state.value,
            "elapsed_seconds": round(self.budget.elapsed, 2),
            "total_raw_issues": len(self._raw_issues),
            "total_validated_issues": len(self._validated_issues),
            "receipt": self._receipt,
            "dry_run": self.dry_run,
            "plan": self._plan.to_dict() if self._plan else None,
        }
