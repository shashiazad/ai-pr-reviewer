"""CriticAgent — validates, deduplicates, and budgets reviewer output."""

from __future__ import annotations

from typing import Any

from tools.common import get_logger, stable_hash

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------

SEVERITY_RANK: dict[str, int] = {"error": 0, "warn": 1, "info": 2}


def severity_key(issue: dict[str, Any]) -> int:
    """Return sort key: lower = higher priority."""
    return SEVERITY_RANK.get(issue.get("severity", "info"), 99)


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------


class CriticAgent:
    """Validates reviewer output: schema, dedup, contradictions, budget."""

    def __init__(self, policy: dict[str, Any] | None = None):
        self.policy = policy or {}
        self.hash_length: int = self.policy.get("dedupe_hash_length", 12)
        self.min_severity: str = self.policy.get("min_severity", "info")
        self.max_comments: int = self.policy.get("max_inline_comments", 20)
        self.nit_consolidation: bool = self.policy.get("nit_consolidation", True)
        self.nit_threshold: int = self.policy.get("nit_threshold", 3)

    # -- public API ----------------------------------------------------------

    def critique(
        self, issues: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Full validation pipeline: filter → dedupe → sort → budget → nit-merge.

        Returns the final validated list of issues ready for posting.
        """
        logger.info("Critic: received %d raw issues", len(issues))

        filtered = self._filter_severity(issues)
        deduped = self._deduplicate(filtered)
        sorted_issues = self._sort_by_severity(deduped)
        budgeted = self._apply_budget(sorted_issues)
        final = self._consolidate_nits(budgeted)

        logger.info(
            "Critic: %d → filtered %d → deduped %d → budgeted %d → final %d",
            len(issues),
            len(filtered),
            len(deduped),
            len(budgeted),
            len(final),
        )
        return final

    # -- pipeline stages -----------------------------------------------------

    def _filter_severity(
        self, issues: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove issues below the configured minimum severity."""
        min_rank = SEVERITY_RANK.get(self.min_severity, 2)
        return [i for i in issues if severity_key(i) <= min_rank]

    def _deduplicate(
        self, issues: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate issues using stable SHA-256 hashes."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for iss in issues:
            h = stable_hash(
                iss.get("file", ""),
                iss.get("line", 0),
                iss.get("message", ""),
                self.hash_length,
            )
            if h not in seen:
                seen.add(h)
                iss["_hash"] = h
                unique.append(iss)
            else:
                logger.debug("Dedupe: dropped duplicate %s", h)
        return unique

    def _sort_by_severity(
        self, issues: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Sort issues: error first, then warn, then info."""
        return sorted(issues, key=severity_key)

    def _apply_budget(
        self, issues: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Trim to max_comments, keeping highest-severity first."""
        if len(issues) <= self.max_comments:
            return issues
        logger.warning(
            "Budget: trimming %d issues to %d", len(issues), self.max_comments
        )
        return issues[: self.max_comments]

    def _consolidate_nits(
        self, issues: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge low-severity info issues in the same file into a single comment."""
        if not self.nit_consolidation:
            return issues

        # Separate nits from non-nits
        nits_by_file: dict[str, list[dict[str, Any]]] = {}
        non_nits: list[dict[str, Any]] = []

        for iss in issues:
            if iss.get("severity") == "info":
                fname = iss.get("file", "unknown")
                nits_by_file.setdefault(fname, []).append(iss)
            else:
                non_nits.append(iss)

        # Consolidate files with >= nit_threshold info issues
        consolidated: list[dict[str, Any]] = list(non_nits)
        for fname, nits in nits_by_file.items():
            if len(nits) >= self.nit_threshold:
                messages = [
                    f"- L{n.get('line', '?')}: {n.get('message', '')}"
                    for n in nits
                ]
                consolidated.append({
                    "file": fname,
                    "line": nits[0].get("line", 1),
                    "severity": "info",
                    "category": "nits",
                    "message": f"**{len(nits)} minor suggestions:**\n" + "\n".join(messages),
                    "suggestion": None,
                    "_hash": stable_hash(fname, 0, f"nit-group-{len(nits)}", self.hash_length),
                    "_consolidated": True,
                })
            else:
                consolidated.extend(nits)

        return consolidated

    # -- contradiction detection (advisory) ----------------------------------

    @staticmethod
    def detect_contradictions(
        issues: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Find pairs of issues on the same file+line with opposing suggestions.

        Returns list of (issue_a, issue_b) pairs that may contradict.
        This is advisory — the orchestrator decides whether to re-prompt.
        """
        by_loc: dict[str, list[dict[str, Any]]] = {}
        for iss in issues:
            key = f"{iss.get('file', '')}:{iss.get('line', 0)}"
            by_loc.setdefault(key, []).append(iss)

        contradictions: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for loc, group in by_loc.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a_sug = (group[i].get("suggestion") or "").strip().lower()
                    b_sug = (group[j].get("suggestion") or "").strip().lower()
                    if a_sug and b_sug and a_sug != b_sug:
                        contradictions.append((group[i], group[j]))
        return contradictions
