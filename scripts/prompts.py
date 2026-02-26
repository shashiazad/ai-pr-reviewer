"""LLM prompt templates for AI Code Reviewer.

Contains the system prompt, per-file review templates, language-specific rubrics,
and the PR summary prompt template.
"""

SYSTEM_PROMPT = """\
You are a strict, concise code reviewer. You ONLY review the unified diff provided.

Critical rules:
1. ONLY flag lines starting with '+' (added/modified). Never comment on removed or context lines.
2. Every issue MUST cite an exact line number visible in the diff.
3. NEVER invent code, APIs, functions, or project details not in the diff.
4. If no real issues exist, return []. Do NOT fabricate issues.
5. Be specific: name the exact variable, function, or pattern that is wrong.
6. Suggest a concrete fix, not vague advice.
7. Severity: error = security flaw / data loss / crash; warn = bug risk / bad practice; info = style / readability.

Output: a raw JSON array only. No markdown fences, no text outside the array.
Schema per element:
{"file":"<str>","line":<int>,"severity":"error|warn|info","category":"<str>","message":"<concise>","suggestion":"<fix or null>"}

Return [] if nothing to flag.
"""

# ---------------------------------------------------------------------------
# Language-specific rubrics
# ---------------------------------------------------------------------------

RUBRIC_TERRAFORM = """\
Check: version pinning, 0.0.0.0/0 in security groups, overly broad IAM wildcards (*), \
missing backend config, hardcoded resource IDs, missing variable validation.
"""

RUBRIC_ANSIBLE = """\
Check: shell/command where a module exists, missing no_log on secrets, \
ignore_errors without justification, unpinned collections, non-idempotent tasks.
"""

RUBRIC_YAML = """\
Check: consistent 2-space indent, duplicate keys, unquoted booleans/numbers, broken anchors.
"""

RUBRIC_PYTHON = """\
Check: bare except, eval/exec on untrusted input, subprocess shell=True, os.system(), \
mutable default args, unclosed file handles, hardcoded secrets, missing type hints on public API.
"""

RUBRIC_BASH = """\
Check: missing set -euo pipefail, unquoted variables in tests/loops, eval usage, \
[ ] vs [[ ]], rm -rf with broad paths, hardcoded secrets, parsing ls output.
"""

RUBRICS = {
    "terraform": RUBRIC_TERRAFORM,
    "ansible": RUBRIC_ANSIBLE,
    "yaml": RUBRIC_YAML,
    "python": RUBRIC_PYTHON,
    "bash": RUBRIC_BASH,
}

# ---------------------------------------------------------------------------
# File extension to language mapping
# ---------------------------------------------------------------------------

EXTENSION_MAP = {
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".py": "python",
    ".sh": "bash",
    ".bash": "bash",
}

# Ansible detection: YAML files in certain paths
ANSIBLE_PATH_PATTERNS = [
    "playbooks/",
    "roles/",
    "tasks/",
    "handlers/",
    "defaults/",
    "vars/",
    "group_vars/",
    "host_vars/",
    "ansible/",
]


def detect_language(filename: str) -> str:
    """Detect the review language/rubric for a file based on extension and path."""
    import os

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Check for Ansible YAML first
    if ext in (".yml", ".yaml"):
        lower_path = filename.lower().replace("\\", "/")
        for pattern in ANSIBLE_PATH_PATTERNS:
            if pattern in lower_path:
                return "ansible"
        return "yaml"

    return EXTENSION_MAP.get(ext, "")


def build_per_file_prompt(filename: str, language: str, diff_content: str,
                          context: str = "") -> str:
    """Build the per-file review prompt with the appropriate rubric."""
    rubric = RUBRICS.get(language, "")
    if not rubric:
        rubric = "Check: readability, error handling, security, maintainability."

    parts = [
        f"File: `{filename}` | Language: {language}",
        "",
        rubric,
        "",
    ]

    if context:
        parts.extend([context, ""])

    parts.extend([
        "Diff:",
        "```",
        diff_content,
        "```",
        "",
        "Respond with a JSON array. Each element:",
        '{"file":"' + filename + '","line":<int>,"severity":"error|warn|info","category":"<str>","message":"<concise>","suggestion":"<fix or null>"}',
        "",
        "Example:",
        '[{"file":"app.py","line":12,"severity":"warn","category":"resource-leak","message":"File opened but never closed","suggestion":"Use `with open(path) as f:` instead"}]',
        "",
        "Return [] if no issues. Raw JSON only.",
    ])

    return "\n".join(parts)


SUMMARY_PROMPT_TEMPLATE = """\
Summarize this code review. Be factual — ONLY reference findings listed below.

PR: {pr_title} by {pr_author}
Changed: {files_changed} files, +{lines_added}/-{lines_removed} lines

Findings:
{aggregated_results_json}

Output this exact Markdown structure:

## AI Code Review Summary

**PR:** {pr_title}

### Top Issues
| # | Severity | File | Line | Issue |
|---|----------|------|------|-------|
(max 5 rows, severity-ordered. Skip table if 0 issues.)

### Checklist for Author
- [ ] (one actionable item per top issue)

### Stats
- Files reviewed: {files_changed}
- Issues: N errors, N warnings, N info

Do NOT invent issues not in the findings list. Raw Markdown, no code fences.
"""


def build_summary_prompt(
    aggregated_results: str,
    pr_title: str,
    pr_author: str,
    files_changed: int,
    lines_added: int,
    lines_removed: int,
    max_lines: int = 200,
) -> str:
    """Build the PR summary prompt."""
    return SUMMARY_PROMPT_TEMPLATE.format(
        aggregated_results_json=aggregated_results,
        pr_title=pr_title,
        pr_author=pr_author,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
        max_lines=max_lines,
    )
