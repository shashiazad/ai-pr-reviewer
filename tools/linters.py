"""Best-effort linter/formatter runners.

Each runner returns ``(ok: bool, findings: list[str])``.
If the tool is not installed the runner logs a warning and returns ``(True, [])``.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from tools.common import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class LintResult:
    """Outcome of a single linter invocation."""

    tool: str
    ok: bool
    findings: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    cmd: list[str],
    cwd: str | None = None,
    timeout: int = 60,
) -> tuple[int, str, str]:
    """Run *cmd* and return (returncode, stdout, stderr).  Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            shell=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return -1, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return -2, "", f"{cmd[0]}: timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return -3, "", str(exc)


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


# ---------------------------------------------------------------------------
# Individual runners
# ---------------------------------------------------------------------------


def run_terraform_fmt_check(cwd: str | None = None) -> LintResult:
    """Run ``terraform fmt -check -diff -recursive``."""
    tool = "terraform"
    if not _tool_available(tool):
        logger.warning("%s not installed; skipping", tool)
        return LintResult(tool=tool, ok=True, findings=[])

    rc, out, err = _run([tool, "fmt", "-check", "-diff", "-recursive"], cwd=cwd)
    findings = [ln for ln in (out + err).splitlines() if ln.strip()]
    return LintResult(tool=tool, ok=(rc == 0), findings=findings)


def run_yamllint(cwd: str | None = None) -> LintResult:
    """Run ``yamllint .`` with default config."""
    tool = "yamllint"
    if not _tool_available(tool):
        logger.warning("%s not installed; skipping", tool)
        return LintResult(tool=tool, ok=True, findings=[])

    rc, out, err = _run([tool, "-f", "parsable", "."], cwd=cwd)
    findings = [ln for ln in out.splitlines() if ln.strip()]
    return LintResult(tool=tool, ok=(rc == 0), findings=findings)


def run_ansible_lint(cwd: str | None = None) -> LintResult:
    """Run ``ansible-lint`` with parsable output."""
    tool = "ansible-lint"
    if not _tool_available(tool):
        logger.warning("%s not installed; skipping", tool)
        return LintResult(tool=tool, ok=True, findings=[])

    rc, out, err = _run([tool, "-p"], cwd=cwd)
    findings = [ln for ln in out.splitlines() if ln.strip()]
    return LintResult(tool=tool, ok=(rc == 0), findings=findings)


def run_black_check(cwd: str | None = None) -> LintResult:
    """Run ``black --check --diff .``."""
    tool = "black"
    if not _tool_available(tool):
        logger.warning("%s not installed; skipping", tool)
        return LintResult(tool=tool, ok=True, findings=[])

    rc, out, err = _run([tool, "--check", "--diff", "."], cwd=cwd)
    findings = [ln for ln in (out + err).splitlines() if ln.strip()]
    return LintResult(tool=tool, ok=(rc == 0), findings=findings)


def run_shellcheck(files: list[str] | None = None) -> LintResult:
    """Run ``shellcheck`` on the given files (or skip if none provided)."""
    tool = "shellcheck"
    if not _tool_available(tool):
        logger.warning("%s not installed; skipping", tool)
        return LintResult(tool=tool, ok=True, findings=[])
    if not files:
        return LintResult(tool=tool, ok=True, findings=[])

    rc, out, err = _run([tool, "-f", "gcc"] + files)
    findings = [ln for ln in (out + err).splitlines() if ln.strip()]
    return LintResult(tool=tool, ok=(rc == 0), findings=findings)


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------


def run_all_linters(
    changed_files: list[str] | None = None,
    cwd: str | None = None,
    enabled: dict[str, bool] | None = None,
) -> list[LintResult]:
    """Run all enabled linters and return combined results.

    *enabled* maps tool names to booleans; defaults to all enabled.
    """
    en = enabled or {}
    results: list[LintResult] = []

    if en.get("terraform", True):
        results.append(run_terraform_fmt_check(cwd=cwd))
    if en.get("yamllint", True):
        results.append(run_yamllint(cwd=cwd))
    if en.get("ansible-lint", True):
        results.append(run_ansible_lint(cwd=cwd))
    if en.get("black", True):
        results.append(run_black_check(cwd=cwd))
    if en.get("shellcheck", True):
        shell_files = [
            f for f in (changed_files or [])
            if f.endswith((".sh", ".bash", ".zsh", ".ksh"))
        ]
        results.append(run_shellcheck(shell_files))

    return results
