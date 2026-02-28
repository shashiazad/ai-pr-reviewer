"""AgentOrchestrator — LangGraph-powered pipeline that drives the full review.

The graph topology is defined in ``agents.graph`` and compiled into a
runnable that replaces the previous hand-rolled state-machine loop.

Pipeline:
  fetch_pr → plan_and_diff →[skip?]→ run_linters → review_chunks
                                 │                       │
                                 ▼              [budget expired?]
                          post_comments ◄── critique ◄───┘
                                 │
                                 ▼
                            summarize → END

On unrecoverable node failure → fail_gracefully → END
"""

from __future__ import annotations

import json
import os
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
from tools.linters import LintResult

from agents.planner import PlannerAgent, ReviewPlan
from agents.reviewer import ReviewerAgent
from agents.critic import CriticAgent
from agents.commenter import CommentAgent
from agents.graph import build_review_graph, ReviewState

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """Drives the end-to-end agentic review loop via a compiled LangGraph."""

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

        # Build compiled LangGraph
        self._graph = build_review_graph(
            gh=self.gh,
            llm=self.llm,
            planner=self.planner,
            reviewer=self.reviewer,
            critic=self.critic,
            commenter=self.commenter,
            budget=self.budget,
            max_retries=self.max_retries,
            max_comments=self.max_comments,
        )

    # -- main entry point ----------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the LangGraph review pipeline.  Returns a receipt dict."""
        logger.info(
            "Orchestrator starting (LangGraph): repo=%s pr=%d dry_run=%s",
            self.repo,
            self.pr_number,
            self.dry_run,
        )

        # Seed the initial graph state
        initial_state: ReviewState = {
            "repo": self.repo,
            "pr_number": self.pr_number,
            "dry_run": self.dry_run,
            "raw_diff": "",
            "linter_results": [],
            "raw_issues": [],
            "validated_issues": [],
            "summary": "",
            "receipt": {},
            "skip_review": False,
            "error": "",
            "current_node": "",
        }

        # Invoke the compiled graph
        final_state = self._graph.invoke(initial_state)

        result = self._build_result(final_state)
        if self.audit_log:
            write_audit_log(self.audit_log, result)
        return result

    # -- helpers -------------------------------------------------------------

    def _build_result(self, state: dict[str, Any]) -> dict[str, Any]:
        """Convert final graph state into the legacy receipt dict."""
        plan = state.get("plan")
        final_node = state.get("current_node", "unknown")
        has_error = bool(state.get("error"))

        if has_error:
            final_state_label = "FAIL_GRACEFULLY"
        elif final_node == "summarize":
            final_state_label = "DONE"
        else:
            final_state_label = final_node.upper()

        return {
            "repo": self.repo,
            "pr_number": self.pr_number,
            "final_state": final_state_label,
            "elapsed_seconds": round(self.budget.elapsed, 2),
            "total_raw_issues": len(state.get("raw_issues", [])),
            "total_validated_issues": len(state.get("validated_issues", [])),
            "receipt": state.get("receipt", {}),
            "dry_run": self.dry_run,
            "plan": plan.to_dict() if plan and hasattr(plan, "to_dict") else None,
        }
