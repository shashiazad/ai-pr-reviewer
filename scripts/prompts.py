"""LLM prompt templates for AI Code Reviewer.

Contains the system prompt, per-file review templates, language-specific rubrics,
and the PR summary prompt template.
"""

SYSTEM_PROMPT = """\
You are an objective, conservative code reviewer for enterprise infrastructure \
and automation code. Provide concise, high-signal comments with code-specific \
suggestions and short examples. Prefer deterministic, reproducible guidance.

Rules:
- Do NOT speculate beyond the diff context provided.
- Do NOT hallucinate APIs, functions, or project constraints not visible in the diff.
- Do NOT comment on things that are correct and need no changes.
- Focus on: correctness, security hygiene, readability, error handling, maintainability.
- Skip performance unless obviously relevant (e.g., N+1 queries, unbounded loops).
- Be constructive. Suggest fixes, not just problems.

Output contract: Respond ONLY with a valid JSON array. Each element must match:
{
  "file": "<filename>",
  "line": <int>,
  "severity": "error" | "warn" | "info",
  "category": "<string>",
  "message": "<concise issue description>",
  "suggestion": "<optional fix or code snippet>"
}

If no issues found, return: []
Do NOT wrap the JSON in markdown code fences. Return raw JSON only.\
"""

# ---------------------------------------------------------------------------
# Language-specific rubrics
# ---------------------------------------------------------------------------

RUBRIC_TERRAFORM = """\
- Provider and version pinning: all providers must have version constraints.
- State handling: verify backend configuration is present and appropriate.
- IAM: check for least-privilege; flag overly broad wildcards (*) in policies.
- Security groups: flag 0.0.0.0/0 ingress unless explicitly justified.
- Variable validation: ensure variables have type, description, and validation blocks where appropriate.
- Plan drift: note resources that may cause drift if applied partially.\
"""

RUBRIC_ANSIBLE = """\
- Idempotency: flag tasks that are not idempotent (e.g., raw shell without creates/removes).
- Handlers: verify notify/handler pairs are correct.
- Variable scoping: flag undefined or shadowed variables.
- Shell/command: flag shell/command usage when an Ansible module exists for the task.
- YAML syntax: check indentation consistency, anchor/alias correctness.\
"""

RUBRIC_YAML = """\
- Structure: verify proper indentation (consistent 2-space).
- Keys: flag duplicate keys.
- Anchors/aliases: verify correctness and no circular references.
- Schema: if the YAML represents a known schema (e.g., docker-compose, k8s manifest), check for common misconfigurations.\
"""

RUBRIC_PYTHON = """\
- PEP 8 / black / isort compatibility.
- Exceptions: flag bare except, broad Exception catches without re-raise, missing logging.
- Type hints: flag missing type annotations on public functions.
- Input validation: flag unvalidated external inputs.
- Subprocess: flag shell=True, unsanitized inputs to subprocess/os.system/os.popen.
- Path handling: flag string concatenation for paths; prefer pathlib.\
"""

RUBRIC_BASH = """\
- set -euo pipefail: must be present near top of script.
- Quoting: flag unquoted variables, especially in conditionals and loops.
- Globbing: flag unsafe glob patterns.
- ls parsing: flag parsing of ls output; suggest find/xargs.
- Dependencies: note external command dependencies and portability concerns.
- Shellcheck: note any patterns that shellcheck would flag.\
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
        rubric = "Apply general code review best practices: readability, error handling, security, maintainability."

    parts = [
        f"Review the following unified diff for `{filename}` ({language} file).",
        "Apply the review rubric below.",
        "",
        "--- RUBRIC ---",
        rubric,
        "",
        "--- DIFF ---",
        diff_content,
    ]

    if context:
        parts.extend([
            "",
            "--- SURROUNDING CONTEXT ---",
            context,
        ])

    parts.extend([
        "",
        "Return a JSON array of issues found. Follow the output contract strictly.",
    ])

    return "\n".join(parts)


SUMMARY_PROMPT_TEMPLATE = """\
You are summarizing an AI code review for a pull request.

Given the per-file review results below, produce a PR-level summary.

--- PER-FILE RESULTS ---
{aggregated_results_json}

--- PR METADATA ---
Title: {pr_title}
Author: {pr_author}
Files changed: {files_changed}
Lines added: {lines_added}
Lines removed: {lines_removed}

Output format (Markdown, max {max_lines} lines):

## AI Code Review Summary

### Top Issues (up to 5, severity-ordered)
| # | Severity | File | Line | Issue |
|---|----------|------|------|-------|

### Strengths
- (list positive aspects observed in the code)

### Checklist
- [ ] All flagged security issues addressed
- [ ] Format/lint issues resolved
- [ ] Error handling reviewed
- [ ] Tests cover new/changed logic

### Statistics
- Files reviewed: N
- Issues found: N (X errors, Y warnings, Z info)

Do NOT wrap the output in code fences. Return raw Markdown only.\
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
