"""PlannerAgent — decomposes a PR review into a bounded execution plan."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tools.common import get_logger, load_json_file, TimeBudget
from tools.diff_utils import (
    DiffChunk,
    FileDiff,
    filter_files,
    parse_unified_diff,
    chunk_file_diff,
)
from tools.github_client import PRMetadata

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------


@dataclass
class FileTask:
    """A single file to be reviewed, with its diff chunks."""

    filename: str
    language: str
    chunks: list[DiffChunk] = field(default_factory=list)
    linter_findings: list[str] = field(default_factory=list)


@dataclass
class ReviewPlan:
    """The bounded plan produced by PlannerAgent."""

    pr: PRMetadata
    file_tasks: list[FileTask] = field(default_factory=list)
    total_chunks: int = 0
    estimated_llm_calls: int = 0
    linters_to_run: list[str] = field(default_factory=list)
    skip_review: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr_number": self.pr.pr_number,
            "repo": self.pr.repo,
            "files": len(self.file_tasks),
            "total_chunks": self.total_chunks,
            "estimated_llm_calls": self.estimated_llm_calls,
            "linters": self.linters_to_run,
            "skip_review": self.skip_review,
            "skip_reason": self.skip_reason,
        }


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_DEFAULT_LANG_MAP: dict[str, str] = {
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".hcl": "terraform",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".py": "python",
    ".pyi": "python",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ksh": "bash",
}

_ANSIBLE_MARKERS = ("roles/", "playbooks/", "tasks/", "handlers/", "group_vars/", "host_vars/")


def detect_language(filename: str, lang_map: dict[str, str] | None = None) -> str:
    """Detect language from file extension and path markers."""
    lm = lang_map or _DEFAULT_LANG_MAP
    # Ansible override
    if any(marker in filename for marker in _ANSIBLE_MARKERS):
        ext = _ext(filename)
        if ext in (".yml", ".yaml"):
            return "ansible"
    return lm.get(_ext(filename), "generic")


def _ext(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx >= 0 else ""


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


class PlannerAgent:
    """Produces a ReviewPlan from PR metadata and raw diff."""

    def __init__(self, policy: dict[str, Any] | None = None):
        self.policy = policy or {}
        self.max_files = self.policy.get("max_files_per_run", 50)
        self.max_chunks_per_file = self.policy.get("max_chunks_per_file", 10)
        self.chunk_max_lines = self.policy.get("chunk_max_lines", 300)
        self.skip_patterns: list[str] = self.policy.get("skip_patterns", [
            "*.lock", "*.min.js", "*.min.css", "vendor/**",
            "node_modules/**", "dist/**", "*.pb.go", "*.generated.*",
        ])
        self.lang_map = _DEFAULT_LANG_MAP

    def plan(
        self,
        pr: PRMetadata,
        raw_diff: str,
        budget: TimeBudget | None = None,
    ) -> ReviewPlan:
        """Create a bounded ReviewPlan for the current PR."""
        result = ReviewPlan(pr=pr)

        # Parse and filter diff
        all_files = parse_unified_diff(raw_diff)
        filtered = filter_files(all_files, self.skip_patterns)

        if not filtered:
            result.skip_review = True
            result.skip_reason = "No reviewable files after filtering."
            logger.info("Plan: skip — %s", result.skip_reason)
            return result

        # Enforce file cap
        if len(filtered) > self.max_files:
            logger.warning(
                "PR has %d files; capping to %d", len(filtered), self.max_files
            )
            filtered = filtered[: self.max_files]

        # Build per-file tasks
        linter_set: set[str] = set()
        for fd in filtered:
            lang = detect_language(fd.filename, self.lang_map)
            chunks = chunk_file_diff(fd, self.chunk_max_lines)
            if len(chunks) > self.max_chunks_per_file:
                chunks = chunks[: self.max_chunks_per_file]

            task = FileTask(filename=fd.filename, language=lang, chunks=chunks)
            result.file_tasks.append(task)
            result.total_chunks += len(chunks)

            # Determine linters
            if lang == "terraform":
                linter_set.add("terraform")
            elif lang == "ansible":
                linter_set.add("ansible-lint")
            elif lang == "yaml":
                linter_set.add("yamllint")
            elif lang == "python":
                linter_set.add("black")
            elif lang == "bash":
                linter_set.add("shellcheck")

        result.linters_to_run = sorted(linter_set)
        # +1 for the summary call
        result.estimated_llm_calls = result.total_chunks + 1

        # Budget check
        if budget and budget.remaining < 30:
            logger.warning("Less than 30s remaining before planning completes")

        logger.info("Plan: %s", json.dumps(result.to_dict()))
        return result
