"""LangGraph-based review pipeline graph.

Replaces the manual state-machine loop in orchestrator.py with a
compiled LangGraph StateGraph.  Each pipeline stage is a graph node;
conditional edges handle skip-review, budget expiry, and error routing.

Graph topology:
  fetch_pr ──► plan_and_diff ──►[skip?]──► run_linters ──► review_chunks
                                   │                             │
                                   ▼                     [budget expired?]
                            post_comments ◄── critique ◄────────┘
                                   │
                                   ▼
                              summarize ──► END

  On unrecoverable error from any node → fail_gracefully → END
"""

from __future__ import annotations

import traceback
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from tools.common import get_logger, TimeBudget
from tools.github_client import GitHubClient, PRMetadata
from tools.llm_client import LLMClient
from tools.linters import LintResult, run_all_linters

from agents.planner import PlannerAgent, ReviewPlan
from agents.reviewer import ReviewerAgent
from agents.critic import CriticAgent
from agents.commenter import CommentAgent

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class ReviewState(TypedDict, total=False):
    """Typed state flowing through the LangGraph review pipeline."""

    # --- Immutable inputs (set once by the orchestrator) ---
    repo: str
    pr_number: int
    dry_run: bool

    # --- Pipeline data (populated progressively by nodes) ---
    pr_metadata: Any          # PRMetadata
    raw_diff: str
    plan: Any                 # ReviewPlan
    linter_results: list      # list[LintResult]
    raw_issues: list          # list[dict]
    validated_issues: list    # list[dict]
    summary: str
    receipt: dict

    # --- Control flow ---
    skip_review: bool
    error: str
    current_node: str


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def build_review_graph(
    *,
    gh: GitHubClient,
    llm: LLMClient,
    planner: PlannerAgent,
    reviewer: ReviewerAgent,
    critic: CriticAgent,
    commenter: CommentAgent,
    budget: TimeBudget,
    max_retries: int = 2,
    max_comments: int = 20,
) -> Any:
    """Build and compile the LangGraph review pipeline.

    Agent instances are captured via closure so the state dict stays
    lightweight and serialisable.

    Returns a compiled ``StateGraph`` ready for ``.invoke(initial_state)``.
    """

    # -- helpers ---------------------------------------------------------

    def _retry(fn, state: dict, node_name: str) -> dict:
        """Execute *fn(state)* with bounded retries.  On exhaustion set error."""
        for attempt in range(max_retries + 1):
            try:
                return fn(state)
            except Exception as exc:  # noqa: BLE001
                if attempt < max_retries:
                    logger.warning(
                        "Node %s failed (attempt %d/%d): %s — retrying",
                        node_name,
                        attempt + 1,
                        max_retries,
                        str(exc)[:300],
                    )
                else:
                    logger.error(
                        "Node %s failed after %d retries: %s",
                        node_name,
                        max_retries + 1,
                        str(exc)[:300],
                    )
                    return {
                        "error": f"[{node_name}] {str(exc)[:500]}",
                        "current_node": node_name,
                    }
        return {"error": f"[{node_name}] unexpected retry exit", "current_node": node_name}

    # -- node implementations --------------------------------------------

    def fetch_pr(state: ReviewState) -> dict:
        """Fetch PR metadata from GitHub."""
        def _inner(s):
            pr = gh.get_pr_metadata(s["repo"], s["pr_number"])
            logger.info("PR: %s by %s", pr.title, pr.author)
            return {"pr_metadata": pr, "current_node": "fetch_pr"}
        return _retry(_inner, state, "fetch_pr")

    def plan_and_diff(state: ReviewState) -> dict:
        """Fetch unified diff and produce a bounded review plan."""
        def _inner(s):
            pr = s["pr_metadata"]
            raw_diff = gh.get_unified_diff(s["repo"], s["pr_number"])
            plan = planner.plan(pr, raw_diff, budget)
            skip = plan.skip_review
            result: dict = {
                "raw_diff": raw_diff,
                "plan": plan,
                "skip_review": skip,
                "current_node": "plan_and_diff",
            }
            if skip:
                logger.info("Plan says skip: %s", plan.skip_reason)
                result["summary"] = (
                    f"## AI Code Review\n\nNo reviewable files found. "
                    f"Reason: {plan.skip_reason}"
                )
            return result
        return _retry(_inner, state, "plan_and_diff")

    def run_linters_node(state: ReviewState) -> dict:
        """Run linters selected by the planner."""
        def _inner(s):
            plan: ReviewPlan = s["plan"]
            enabled = {tool: True for tool in plan.linters_to_run}
            changed_files = [t.filename for t in plan.file_tasks]
            results = run_all_linters(changed_files=changed_files, enabled=enabled)
            logger.info("Linters completed: %d tools run", len(results))
            return {"linter_results": results, "current_node": "run_linters"}
        return _retry(_inner, state, "run_linters")

    def review_chunks_node(state: ReviewState) -> dict:
        """Run LLM + deterministic review on each diff chunk."""
        def _inner(s):
            plan: ReviewPlan = s["plan"]
            linter_results = s.get("linter_results", [])
            raw_issues, summary = reviewer.review_plan(
                plan, linter_results=linter_results, budget=budget,
            )
            logger.info("Reviewer produced %d raw issues", len(raw_issues))
            return {
                "raw_issues": raw_issues,
                "summary": summary,
                "current_node": "review_chunks",
            }
        return _retry(_inner, state, "review_chunks")

    def critique_node(state: ReviewState) -> dict:
        """Validate, deduplicate, and budget-trim issues."""
        def _inner(s):
            validated = critic.critique(s.get("raw_issues", []))
            contradictions = critic.detect_contradictions(validated)
            if contradictions:
                logger.warning(
                    "Critic detected %d potential contradictions",
                    len(contradictions),
                )
            return {"validated_issues": validated, "current_node": "critique"}
        return _retry(_inner, state, "critique")

    def post_comments_node(state: ReviewState) -> dict:
        """Post summary + inline comments to GitHub."""
        def _inner(s):
            pr: PRMetadata = s["pr_metadata"]
            receipt = commenter.post_results(
                repo=s["repo"],
                pr_number=s["pr_number"],
                head_sha=pr.head_sha,
                issues=s.get("validated_issues", []),
                summary_text=s.get("summary", ""),
            )
            return {"receipt": receipt, "current_node": "post_comments"}
        return _retry(_inner, state, "post_comments")

    def summarize_node(state: ReviewState) -> dict:
        """Log final stats."""
        logger.info(
            "Review complete: %d validated issues, summary posted=%s, elapsed=%.1fs",
            len(state.get("validated_issues", [])),
            bool(state.get("receipt", {}).get("summary_id")),
            budget.elapsed,
        )
        return {"current_node": "summarize"}

    def fail_gracefully_node(state: ReviewState) -> dict:
        """Post a minimal failure summary so the PR is never silently ignored."""
        error_msg = state.get("error", "Unknown error")
        logger.error("Failing gracefully: %s", error_msg[:500])
        failure_summary = (
            f"## AI Code Review — Failure\n\n"
            f"The automated review encountered an error and could not complete.\n\n"
            f"```\n{error_msg[:300]}\n```\n\n"
            f"*Please review this PR manually.*"
        )
        try:
            pr = state.get("pr_metadata")
            if pr:
                commenter.post_results(
                    repo=state["repo"],
                    pr_number=state["pr_number"],
                    head_sha=pr.head_sha if hasattr(pr, "head_sha") else "",
                    issues=[],
                    summary_text=failure_summary,
                )
        except Exception as post_exc:  # noqa: BLE001
            logger.error("Failed to post failure summary: %s", str(post_exc)[:200])
        return {"current_node": "fail_gracefully"}

    # -- routing functions -----------------------------------------------

    def route_after_plan(state: ReviewState) -> str:
        """After planning, skip straight to posting if nothing to review."""
        if state.get("error"):
            return "fail_gracefully"
        if state.get("skip_review"):
            return "post_comments"
        return "run_linters"

    def route_after_linters(state: ReviewState) -> str:
        """Check for errors or budget expiry after linters."""
        if state.get("error"):
            return "fail_gracefully"
        if budget.expired:
            logger.warning("Budget expired after linters; jumping to post_comments")
            return "post_comments"
        return "review_chunks"

    def route_after_review(state: ReviewState) -> str:
        """Check for errors after review chunks."""
        if state.get("error"):
            return "fail_gracefully"
        return "critique"

    def route_after_critique(state: ReviewState) -> str:
        """Check for errors after critique."""
        if state.get("error"):
            return "fail_gracefully"
        return "post_comments"

    def route_after_post(state: ReviewState) -> str:
        """Check for errors after posting."""
        if state.get("error"):
            return "fail_gracefully"
        return "summarize"

    def route_after_fetch(state: ReviewState) -> str:
        if state.get("error"):
            return "fail_gracefully"
        return "plan_and_diff"

    # -- assemble graph --------------------------------------------------

    graph = StateGraph(ReviewState)

    # Add nodes
    graph.add_node("fetch_pr", fetch_pr)
    graph.add_node("plan_and_diff", plan_and_diff)
    graph.add_node("run_linters", run_linters_node)
    graph.add_node("review_chunks", review_chunks_node)
    graph.add_node("critique", critique_node)
    graph.add_node("post_comments", post_comments_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("fail_gracefully", fail_gracefully_node)

    # Set entry point
    graph.set_entry_point("fetch_pr")

    # Add edges with conditional routing for error / skip / budget
    graph.add_conditional_edges("fetch_pr", route_after_fetch)
    graph.add_conditional_edges("plan_and_diff", route_after_plan)
    graph.add_conditional_edges("run_linters", route_after_linters)
    graph.add_conditional_edges("review_chunks", route_after_review)
    graph.add_conditional_edges("critique", route_after_critique)
    graph.add_conditional_edges("post_comments", route_after_post)
    graph.add_edge("summarize", END)
    graph.add_edge("fail_gracefully", END)

    return graph.compile()
