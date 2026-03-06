"""ReviewerAgent — runs per-chunk LLM reviews with deterministic prompts."""

from __future__ import annotations

import re
from typing import Any

from tools.common import get_logger, load_json_file, TimeBudget
from tools.llm_client import LLMClient
from tools.linters import LintResult
from agents.planner import FileTask, ReviewPlan

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Security pre-scan patterns (fast regex, no LLM needed)
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"""(?:password|passwd|pwd|secret|api.?key|access.?key|auth.?token|private.?key)\s*[:=]\s*['"][^'"]{8,}['"]""",
            re.IGNORECASE,
        ),
        "Potential hardcoded secret",
        "Move secrets to environment variables or a secret manager.",
    ),
    (
        re.compile(
            r"""(?:password|passwd|pwd|secret|api.?key|access.?key|auth.?token|private.?key)\s*=\s*[A-Za-z0-9_\-+/=]{8,}""",
            re.IGNORECASE,
        ),
        "Potential hardcoded secret",
        "Move secrets to environment variables or a secret manager.",
    ),
    (
        re.compile(r"""(?:AKIA|ASIA)[A-Z0-9]{16}"""),
        "Potential AWS access key ID",
        "Remove and rotate this key immediately. Use IAM roles or environment variables.",
    ),
    (
        re.compile(r"""-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"""),
        "Private key detected in code",
        "Never commit private keys. Use a secret manager.",
    ),
    (
        re.compile(r"""ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{40,}"""),
        "GitHub personal access token detected",
        "Rotate this token immediately and store in GitHub Actions secrets.",
    ),
    (
        re.compile(r"""Bearer\s+[A-Za-z0-9\-._~+/]{20,}"""),
        "Potential hardcoded Bearer token",
        "Use environment variables for tokens.",
    ),
]

_UNSAFE_PATTERNS: dict[str, list[tuple[re.Pattern[str], str, str, str]]] = {
    "python": [
        (re.compile(r"""\beval\s*\("""), "error", "Use of eval()", "Avoid eval() on untrusted input; use ast.literal_eval() if needed."),
        (re.compile(r"""subprocess\.\w+\(.*shell\s*=\s*True""", re.DOTALL), "warn", "subprocess with shell=True", "Prefer shell=False with a list of args."),
        (re.compile(r"""\bexcept\s*:"""), "warn", "Bare except clause", "Catch specific exceptions instead."),
        (re.compile(r"""\bos\.system\s*\("""), "warn", "os.system() usage", "Use subprocess.run() with a list of arguments instead of os.system()."),
        (re.compile(r"""\bexec\s*\("""), "warn", "exec() usage", "exec() can execute arbitrary code. Verify input is trusted."),
        (re.compile(r"""\bpickle\.loads?\s*\("""), "warn", "pickle.load() on untrusted data", "pickle can execute arbitrary code during deserialization. Verify data source."),
        (re.compile(r"""def\s+\w+\s*\([^)]*=\s*\[\s*\]"""), "warn", "Mutable default argument (list)", "Use None as default and create list inside function body: def f(x=None): x = x or []"),
        (re.compile(r"""def\s+\w+\s*\([^)]*=\s*\{\s*\}"""), "warn", "Mutable default argument (dict)", "Use None as default and create dict inside function body: def f(x=None): x = x or {}"),
    ],
    "go": [
        (re.compile(r"""\b_\s*=\s*\w+\."""), "warn", "Ignored error return value", "In Go, always check returned errors. Use `if err != nil` pattern instead of discarding with `_`."),
        (re.compile(r"""\bpanic\s*\("""), "warn", "Use of panic()", "Don't use panic() for normal error handling in Go. Return an error instead."),
        (re.compile(r"""\bgo\s+func\s*\("""), "info", "Anonymous goroutine detected", "Ensure goroutine has proper synchronization (WaitGroup, channel, or context) and error handling."),
        (re.compile(r"""\bfmt\.Errorf\([^)]*%v.*err"""), "info", "Error wrapping with %v instead of %w", "Use %w instead of %v in fmt.Errorf to preserve the error chain for errors.Is/As."),
        (re.compile(r"""\binit\s*\(\s*\)"""), "info", "init() function detected", "Avoid init() when possible — prefer explicit initialization for testability and clarity."),
        (re.compile(r"""\.(Close|Flush)\s*\("""), "info", "Resource cleanup without defer", "Consider using defer for cleanup calls to ensure they run even on early returns or panics."),
    ],
    "terraform": [
        (re.compile(r"""cidr_blocks\s*=\s*\[?"0\.0\.0\.0/0"?\]"""), "error", "Open CIDR 0.0.0.0/0 in security group", "Restrict to known IP ranges."),
    ],
    "bash": [
        (re.compile(r"""\beval\b"""), "warn", "Use of eval in shell", "Avoid eval with untrusted input."),
        (re.compile(r"""\brm\s+-rf\s+/"""), "error", "Potentially destructive rm -rf usage", "Avoid broad recursive deletes; scope paths safely and validate inputs."),
        (re.compile(r"""\[\s*\$[A-Za-z_][A-Za-z0-9_]*\s*=="""), "warn", "Unquoted shell variable in test expression", "Quote variables in test expressions to avoid word-splitting/globbing bugs."),
    ],
    "ansible": [
        (re.compile(r"""\bshell\s*:"""), "warn", "Ansible shell module usage", "Prefer built-in modules for idempotency."),
        (re.compile(r"""\bcommand\s*:"""), "info", "Ansible command module usage", "Consider if a built-in module can replace this command for idempotency."),
    ],
}


def _scan_for_secrets(lines: list[str], filename: str) -> list[dict[str, Any]]:
    """Fast regex scan for secrets on added lines only."""
    findings: list[dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        if not line.startswith("+"):
            continue
        content = line[1:]
        for pat, msg, suggestion in _SECRET_PATTERNS:
            if pat.search(content):
                findings.append({
                    "file": filename,
                    "line": i,
                    "severity": "error",
                    "category": "security",
                    "message": msg,
                    "suggestion": suggestion,
                })
    return findings


def _scan_unsafe_patterns(
    lines: list[str], filename: str, language: str
) -> list[dict[str, Any]]:
    """Fast regex scan for language-specific unsafe patterns."""
    patterns = _UNSAFE_PATTERNS.get(language, [])
    findings: list[dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        if not line.startswith("+"):
            continue
        content = line[1:]
        for pat, severity, msg, suggestion in patterns:
            if pat.search(content):
                findings.append({
                    "file": filename,
                    "line": i,
                    "severity": severity,
                    "category": "security",
                    "message": msg,
                    "suggestion": suggestion,
                })
    return findings


# ---------------------------------------------------------------------------
# ReviewerAgent
# ---------------------------------------------------------------------------


class ReviewerAgent:
    """Runs per-chunk LLM reviews and merges with linter + security findings."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_templates: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
    ):
        self.llm = llm
        self.templates = prompt_templates or {}
        self.policy = policy or {}
        self.system_prompt: str = self.templates.get(
            "system",
            "You are an objective, conservative code reviewer. Provide concise, "
            "high-signal comments with code-specific suggestions.",
        )

    # -- prompt building -----------------------------------------------------

    def _build_per_file_prompt(
        self,
        filename: str,
        language: str,
        diff_text: str,
        linter_context: str = "",
    ) -> str:
        tpl = self.templates.get("per_file", {}).get(
            "template",
            "Review this diff for {filename} ({language}).\n\nDiff:\n```\n{diff}\n```\n\n"
            "Return a JSON array of issues with fields: file, line, severity, category, message, suggestion.",
        )
        rubrics = self.templates.get("per_file", {}).get("rubrics", {})
        rubric = rubrics.get(language, "")

        context_section = ""
        if linter_context:
            context_section = f"Linter findings for context:\n{linter_context}\n"

        return tpl.format(
            filename=filename,
            language=language,
            diff=diff_text,
            rubric=rubric,
            context_section=context_section,
        )

    def _build_summary_prompt(
        self,
        pr_title: str,
        pr_number: int,
        pr_author: str,
        files_changed: int,
        additions: int,
        deletions: int,
        findings_summary: str,
    ) -> str:
        tpl = self.templates.get("summary", {}).get(
            "template",
            "Summarize this PR review.\nPR: {pr_title} (#{pr_number})\n"
            "Author: {pr_author}\nFiles changed: {files_changed}\n"
            "Additions: +{additions}, Deletions: -{deletions}\n\n"
            "Findings:\n{findings_summary}",
        )
        return tpl.format(
            pr_title=pr_title,
            pr_number=pr_number,
            pr_author=pr_author,
            files_changed=files_changed,
            additions=additions,
            deletions=deletions,
            findings_summary=findings_summary,
        )

    # -- review execution ----------------------------------------------------

    def review_plan(
        self,
        plan: ReviewPlan,
        linter_results: list[LintResult] | None = None,
        budget: TimeBudget | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Execute the review plan.

        Returns:
            (all_issues, summary_text)
        """
        all_issues: list[dict[str, Any]] = []
        linter_map = self._build_linter_map(linter_results or [])
        llm_attempts = 0
        llm_successes = 0

        # Phase 1: Security pre-scan (no LLM, instant)
        for task in plan.file_tasks:
            for chunk in task.chunks:
                lines = chunk.context.splitlines()
                all_issues.extend(_scan_for_secrets(lines, task.filename))
                all_issues.extend(
                    _scan_unsafe_patterns(lines, task.filename, task.language)
                )

        # Phase 2: LLM review per chunk
        for task in plan.file_tasks:
            if budget and budget.expired:
                logger.warning("Budget expired; switching to summarize mode")
                break

            linter_ctx = linter_map.get(task.filename, "")
            for chunk in task.chunks:
                if budget and budget.expired:
                    break

                prompt = self._build_per_file_prompt(
                    filename=task.filename,
                    language=task.language,
                    diff_text=chunk.context,
                    linter_context=linter_ctx,
                )
                llm_attempts += 1
                try:
                    issues = self.llm.review_chunk(self.system_prompt, prompt)
                    llm_successes += 1
                    all_issues.extend(issues)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "LLM review failed for %s: %s", task.filename, str(exc)[:200]
                    )

        if llm_attempts > 0 and llm_successes == 0:
            logger.error(
                "All LLM review chunk calls failed; continuing with deterministic findings only"
            )
            if not all_issues:
                fallback_file = plan.file_tasks[0].filename if plan.file_tasks else "unknown"
                all_issues.append(
                    {
                        "file": fallback_file,
                        "line": 1,
                        "severity": "warn",
                        "category": "operational",
                        "message": "Review could not fully complete due to a service issue (quota/auth/model). Some checks may be missing.",
                        "suggestion": "Check GEMINI_API_KEY validity, quota limits, and model access in Gemini API settings.",
                    }
                )

        # Phase 3: Generate summary
        findings_text = self._summarize_findings(all_issues)
        summary_prompt = self._build_summary_prompt(
            pr_title=plan.pr.title,
            pr_number=plan.pr.pr_number,
            pr_author=plan.pr.author,
            files_changed=plan.pr.changed_files,
            additions=plan.pr.additions,
            deletions=plan.pr.deletions,
            findings_summary=findings_text,
        )
        try:
            summary = self.llm.summarize_pr(self.system_prompt, summary_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("Summary generation failed: %s", str(exc)[:200])
            summary = self._fallback_summary(all_issues, plan.pr)

        return all_issues, summary

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _build_linter_map(results: list[LintResult]) -> dict[str, str]:
        """Map filenames to combined linter findings text."""
        mapping: dict[str, list[str]] = {}
        for lr in results:
            for finding in lr.findings:
                # parsable format: file:line:col: message
                parts = finding.split(":", 1)
                fname = parts[0].strip() if parts else ""
                if fname:
                    mapping.setdefault(fname, []).append(finding)
        return {k: "\n".join(v) for k, v in mapping.items()}

    @staticmethod
    def _summarize_findings(issues: list[dict[str, Any]]) -> str:
        """Create a compact text summary of all issues for the LLM summary prompt."""
        if not issues:
            return "No issues found."
        lines: list[str] = []
        for iss in issues[:30]:  # cap to avoid prompt overflow
            sev = iss.get("severity", "info").upper()
            f = iss.get("file", "?")
            ln = iss.get("line", "?")
            msg = iss.get("message", "")
            lines.append(f"[{sev}] {f}:{ln} — {msg}")
        if len(issues) > 30:
            lines.append(f"... and {len(issues) - 30} more issues")
        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(issues: list[dict[str, Any]], pr: Any) -> str:
        """Produce a minimal summary when LLM summarization fails."""
        errs = sum(1 for i in issues if i.get("severity") == "error")
        warns = sum(1 for i in issues if i.get("severity") == "warn")
        infos = sum(1 for i in issues if i.get("severity") == "info")
        return (
            f"## Code Review Summary\n\n"
            f"**PR:** {pr.title} (#{pr.pr_number})\n\n"
            f"Found **{len(issues)}** issues: "
            f"{errs} errors, {warns} warnings, {infos} info.\n\n"
            f"Please review the inline comments and address the flagged items."
        )
