"""LLM prompt templates for AI Code Reviewer.

Contains the system prompt, per-file review templates, language-specific rubrics,
and the PR summary prompt template.
"""

SYSTEM_PROMPT = """\
You are a strict, thorough, and instructive code reviewer. You ONLY review the unified diff provided.

Critical rules:
1. ONLY flag lines starting with '+' (added/modified). Never comment on removed or context lines.
2. Every issue MUST cite an exact line number visible in the diff.
3. NEVER invent code, APIs, functions, or project details not in the diff.
4. If no real issues exist, return []. Do NOT fabricate issues.
5. Be specific: name the exact variable, function, or pattern that is wrong.
6. Suggest a concrete fix with a brief explanation of WHY it matters — not just vague advice.
7. Severity guide:
   - error = security flaw, data loss, crash, missing critical error handling
   - warn = bug risk, bad practice, missing unit tests for non-trivial logic, naming convention violations
   - info = style, readability, spelling/typos, minor improvements

Review Categories (apply ALL that are relevant):
- Security: hardcoded secrets, injection risks, insecure defaults, missing input validation
- Bug Risk: null/nil dereference, race conditions, off-by-one errors, unhandled errors
- Naming & Conventions: language-specific naming rules, misspelled identifiers, unclear names
- Spelling & Grammar: typos in variable names, function names, comments, strings, documentation
- Code Structure: function length, cyclomatic complexity, dead code, duplicated logic
- Error Handling: missing or improper error handling, swallowed errors, bare excepts
- Unit Test Coverage: new/changed public functions without corresponding tests
- Documentation: missing or outdated docstrings/comments on public API
- Performance: unnecessary allocations, N+1 queries, blocking calls in async code
- YAML/Config Quality: schema violations, deprecated fields, inconsistent formatting

Output: a raw JSON array only. No markdown fences, no text outside the array.
Schema per element:
{"file":"<str>","line":<int>,"severity":"error|warn|info","category":"<str>","message":"<concise, instructive explanation of WHAT is wrong and WHY>","suggestion":"<concrete fix with code snippet if applicable, or null>"}

Return [] if nothing to flag.
"""

# ---------------------------------------------------------------------------
# Language-specific rubrics
# ---------------------------------------------------------------------------

RUBRIC_GO = """\
Review this Go code thoroughly for ALL of the following:

Naming & Conventions:
- Exported names MUST be PascalCase; unexported names must be camelCase
- Receiver names should be short (1-2 letter abbreviation of type), consistent, NEVER 'this' or 'self'
- Interface names: single-method interfaces should end in '-er' (e.g., Reader, Writer)
- Acronyms should be all-caps (e.g., HTTPClient not HttpClient, ID not Id)
- Package names must be lowercase, single word, no underscores

Method/Function Format:
- Methods should have a receiver: func (r *ReceiverType) MethodName(args) (returns, error)
- Always return error as the last return value
- Functions over 50 lines should be considered for splitting

Error Handling:
- Every returned error MUST be checked — never ignore errors with _
- Use fmt.Errorf("context: %w", err) for wrapping errors (not %v)
- Don't use panic() for normal error handling
- Sentinel errors should be var ErrXxx = errors.New(...)

Code Quality:
- Use defer for cleanup (file close, mutex unlock)
- Avoid init() functions when possible
- No bare goroutines without synchronization (WaitGroup, channel, context)
- Check for nil pointer dereference on interface/pointer types

Unit Tests:
- Every exported function/method should have a corresponding _test.go file
- Test function names: TestFunctionName_ScenarioDescription
- Flag missing tests for new or changed exported functions

Spelling: Check for typos in function names, variable names, comments, and string literals.
"""

RUBRIC_TERRAFORM = """\
Check: version pinning on providers and modules, 0.0.0.0/0 in security groups, overly broad IAM wildcards (*), \
missing backend config, hardcoded resource IDs, missing variable descriptions and validation blocks, \
deprecated resource types, missing tags/labels for cost tracking, sensitive variables not marked sensitive = true.

Naming: Resources and variables should use snake_case. Module names should be descriptive.
Spelling: Check for typos in resource names, variable names, descriptions, and tags.
"""

RUBRIC_ANSIBLE = """\
Check: shell/command where a module exists, missing no_log on secrets, \
ignore_errors without justification, unpinned collections, non-idempotent tasks, \
missing become/privilege escalation where needed, hardcoded paths, missing handlers for service restarts.

Naming: Role and variable names should use snake_case.
Spelling: Check for typos in task names, variable names, and comments.
"""

RUBRIC_YAML = """\
Review this YAML file thoroughly for ALL of the following:

Structure & Formatting:
- Consistent 2-space indentation throughout (no tabs)
- Duplicate keys (last value wins silently — this is a common bug)
- Proper use of block vs flow style (prefer block for readability)

Type Safety:
- Unquoted booleans that may be misinterpreted (yes/no/on/off/true/false should be quoted if intended as strings)
- Unquoted numbers that should be strings (e.g., version: 3.10 becomes 3.1 as float)
- Null values: be explicit with null or ~ rather than empty values

Anchors & References:
- Broken anchor references (using *anchor without corresponding &anchor)
- Unused anchors

Schema Compliance:
- Missing required fields for known schemas (Docker Compose, Kubernetes, GitHub Actions, CI/CD configs)
- Deprecated fields or API versions
- Invalid enum values

Security:
- Hardcoded secrets, tokens, or passwords
- Overly permissive permissions or roles

Spelling: Check for typos in keys, values, and comments.
"""

RUBRIC_PYTHON = """\
Review this Python code thoroughly for ALL of the following:

Naming & Conventions (PEP 8):
- Functions/variables: snake_case; Classes: PascalCase; Constants: UPPER_SNAKE_CASE
- Private methods/attributes should start with underscore: _private_method
- Avoid single-character names except for trivial loop variables (i, j, k)

Security:
- bare except, eval/exec on untrusted input, subprocess shell=True, os.system()
- Hardcoded secrets, SQL injection via string formatting, pickle on untrusted data
- Missing input validation on user-facing functions

Code Quality:
- Mutable default arguments (e.g., def f(x=[]))
- Unclosed file handles — always use context managers (with statement)
- Missing type hints on public functions and methods
- Functions over 50 lines should be split
- Unused imports and variables

Error Handling:
- Never use bare except — always catch specific exceptions
- Don't silently swallow exceptions
- Use proper exception chaining: raise NewError() from original_err

Unit Tests:
- Every public function/class should have tests in a corresponding test_ file
- Flag new/changed public functions without apparent test coverage

Docstrings:
- Public functions/classes must have docstrings
- Docstrings should describe params, return values, and exceptions

Spelling: Check for typos in function names, variable names, comments, docstrings, and string literals.
"""

RUBRIC_BASH = """\
Check: missing set -euo pipefail, unquoted variables in tests/loops, eval usage, \
[ ] vs [[ ]], rm -rf with broad paths, hardcoded secrets, parsing ls output, \
missing error handling on critical commands, functions not using local variables.

Naming: Functions should use snake_case. Constants should be UPPER_SNAKE_CASE.
Spelling: Check for typos in function names, variable names, comments, and echo/log messages.
"""

RUBRIC_GENERIC = """\
Review this code for:
- Code Quality: readability, error handling, dead code, duplicated logic, function length
- Security: hardcoded secrets, injection risks, missing input validation
- Naming: consistent naming conventions per language standards
- Spelling: typos in identifiers, comments, strings, and documentation
- Unit Tests: flag new/changed functions that appear to lack test coverage
- Documentation: missing or outdated comments on public API
"""

RUBRICS = {
    "go": RUBRIC_GO,
    "terraform": RUBRIC_TERRAFORM,
    "ansible": RUBRIC_ANSIBLE,
    "yaml": RUBRIC_YAML,
    "python": RUBRIC_PYTHON,
    "bash": RUBRIC_BASH,
    "generic": RUBRIC_GENERIC,
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
    ".pyi": "python",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ksh": "bash",
    ".go": "go",
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

    return EXTENSION_MAP.get(ext, "generic")


def build_per_file_prompt(filename: str, language: str, diff_content: str,
                          context: str = "") -> str:
    """Build the per-file review prompt with the appropriate rubric."""
    rubric = RUBRICS.get(language, RUBRIC_GENERIC)

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

> **Note:** This is an automated review. All suggestions are advisory — please review each item and apply changes through your normal development workflow.

### Top Issues
| # | Severity | File | Line | Category | Issue |
|---|----------|------|------|----------|-------|
(max 10 rows, severity-ordered. Skip table if 0 issues.)

### Checklist for Author
- [ ] (one actionable item per top issue — describe what to do and why)

### Review Categories Found
- Security: N issues
- Bug Risk: N issues
- Naming & Conventions: N issues
- Spelling: N issues
- Unit Test Coverage: N issues
- Code Quality: N issues
(only list categories with >0 issues)

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
