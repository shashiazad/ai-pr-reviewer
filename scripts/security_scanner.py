"""Regex-based security pre-scanner for AI Code Reviewer.

Performs fast, deterministic checks for common security issues before
sending diffs to the LLM. Results are merged with LLM findings.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SecurityFinding:
    """A security issue found by the regex scanner."""
    file: str
    line: int
    severity: str           # error, warn, info
    category: str
    message: str
    suggestion: str = ""


# ---------------------------------------------------------------------------
# Secret patterns (high confidence only — minimize false positives)
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (
        re.compile(
            r"""(?:password|passwd|pwd|secret|api.?key|access.?key|auth.?token|private.?key)\s*[:=]\s*['"][^'"]{8,}['"]""",
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
        re.compile(
            r"""-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----""",
        ),
        "Private key detected in code",
        "Never commit private keys. Use a secret manager.",
    ),
    (
        re.compile(
            r"""ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{40,}""",
        ),
        "GitHub personal access token detected",
        "Rotate this token immediately and store in GitHub Actions secrets.",
    ),
    (
        re.compile(r"""Bearer\s+[A-Za-z0-9\-._~+/]{20,}"""),
        "Potential hardcoded Bearer token",
        "Use environment variables for tokens.",
    ),
]

# ---------------------------------------------------------------------------
# Unsafe code patterns (language-specific)
# ---------------------------------------------------------------------------

_UNSAFE_PATTERNS: Dict[str, List[Tuple[re.Pattern, str, str, str]]] = {
    "python": [
        (
            re.compile(r"""\bsubprocess\.\w+\(.*shell\s*=\s*True""", re.DOTALL),
            "warn",
            "subprocess with shell=True",
            "Avoid shell=True to prevent shell injection. Use a list of arguments instead.",
        ),
        (
            re.compile(r"""\bos\.system\s*\("""),
            "warn",
            "os.system() usage",
            "Use subprocess.run() with a list of arguments instead of os.system().",
        ),
        (
            re.compile(r"""\beval\s*\("""),
            "error",
            "eval() usage",
            "eval() is a security risk. Use ast.literal_eval() or a safer alternative.",
        ),
        (
            re.compile(r"""\bexec\s*\("""),
            "warn",
            "exec() usage",
            "exec() can execute arbitrary code. Verify input is trusted.",
        ),
        (
            re.compile(r"""except\s*:"""),
            "warn",
            "Bare except clause",
            "Catch specific exceptions instead of bare except.",
        ),
        (
            re.compile(r"""\bpickle\.loads?\s*\("""),
            "warn",
            "pickle.load() on untrusted data",
            "pickle can execute arbitrary code during deserialization. Verify data source.",
        ),
    ],
    "bash": [
        (
            re.compile(r"""^\s*(?!#).*\$\{?\w+\}?(?!\")""", re.MULTILINE),
            "info",
            "Potentially unquoted variable",
            'Quote variables: "$VAR" instead of $VAR.',
        ),
        (
            re.compile(r"""\bls\b.*\|\s*(?:while|for|xargs)"""),
            "warn",
            "Parsing ls output",
            "Do not parse ls output. Use find with -print0 and xargs -0.",
        ),
    ],
    "terraform": [
        (
            re.compile(r"""cidr_blocks\s*=\s*\[\s*"0\.0\.0\.0/0"\s*\]"""),
            "warn",
            "Security group open to 0.0.0.0/0",
            "Restrict ingress CIDR blocks to specific IP ranges.",
        ),
        (
            re.compile(r"""\bresource\b.*"aws_iam_policy".*"Action"\s*:\s*"\*\"""", re.DOTALL),
            "error",
            "IAM policy with wildcard Action",
            "Follow least-privilege: specify only required actions.",
        ),
    ],
    "ansible": [
        (
            re.compile(r"""\bshell\s*:"""),
            "info",
            "Ansible shell module usage",
            "Prefer built-in Ansible modules over shell when possible for idempotency.",
        ),
        (
            re.compile(r"""\bcommand\s*:"""),
            "info",
            "Ansible command module usage",
            "Consider if a built-in module can replace this command for idempotency.",
        ),
    ],
}

# Log redaction pattern (used to mask secrets in log output)
SECRET_REDACTION_PATTERN = re.compile(
    r"""(?:password|secret|token|key|api.?key|client.?secret|private.?key|access.?key|bearer)\s*[:=\s]\s*\S+""",
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    """Redact potential secrets from a text string for safe logging."""
    return SECRET_REDACTION_PATTERN.sub("[REDACTED]", text)


def scan_diff_lines(
    filename: str,
    diff_lines: List[str],
    start_line: int,
    language: str = "",
) -> List[SecurityFinding]:
    """Scan diff lines for security issues using regex patterns.

    Only scans added lines (starting with '+') to avoid false positives
    on removed code.
    """
    findings: List[SecurityFinding] = []
    current_line = start_line

    for raw_line in diff_lines:
        # Only check added lines
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            line_content = raw_line[1:]  # strip the leading '+'
            line_num = current_line

            # Check secret patterns (all languages)
            for pattern, msg, suggestion in _SECRET_PATTERNS:
                if pattern.search(line_content):
                    findings.append(SecurityFinding(
                        file=filename,
                        line=line_num,
                        severity="error",
                        category="security/hardcoded-secret",
                        message=msg,
                        suggestion=suggestion,
                    ))

            # Check language-specific unsafe patterns
            lang_patterns = _UNSAFE_PATTERNS.get(language, [])
            for pattern, severity, msg, suggestion in lang_patterns:
                if pattern.search(line_content):
                    findings.append(SecurityFinding(
                        file=filename,
                        line=line_num,
                        severity=severity,
                        category=f"security/{language}",
                        message=msg,
                        suggestion=suggestion,
                    ))

        # Track line numbers for added and context lines
        if raw_line.startswith("+") or raw_line.startswith(" "):
            current_line += 1
        elif raw_line.startswith("-"):
            pass  # deleted line doesn't advance new-file line number

    return findings


def scan_file_diff(
    filename: str,
    hunks: list,
    language: str = "",
) -> List[SecurityFinding]:
    """Scan all hunks in a file diff for security issues."""
    all_findings: List[SecurityFinding] = []
    for hunk in hunks:
        findings = scan_diff_lines(
            filename=filename,
            diff_lines=hunk.lines,
            start_line=hunk.new_start,
            language=language,
        )
        all_findings.extend(findings)
    return all_findings


def findings_to_issues(findings: List[SecurityFinding]) -> List[Dict]:
    """Convert SecurityFinding objects to the standard issue dict format."""
    return [
        {
            "file": f.file,
            "line": f.line,
            "severity": f.severity,
            "category": f.category,
            "message": f.message,
            "suggestion": f.suggestion,
        }
        for f in findings
    ]
