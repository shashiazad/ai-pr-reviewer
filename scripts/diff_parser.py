"""Diff fetching, parsing, and chunking for AI Code Reviewer.

Fetches the PR diff via GitHub API, parses it into per-file and per-hunk
structures, and applies skip patterns and chunking logic.
"""

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

from .config import ReviewerConfig


@dataclass
class DiffHunk:
    """A single hunk within a file diff."""
    header: str                     # e.g., @@ -10,6 +10,8 @@
    old_start: int = 0
    old_count: int = 0
    new_start: int = 0
    new_count: int = 0
    lines: List[str] = field(default_factory=list)
    content: str = ""


@dataclass
class FileDiff:
    """Parsed diff for a single file."""
    filename: str
    old_filename: Optional[str] = None
    status: str = "modified"         # added, modified, removed, renamed
    hunks: List[DiffHunk] = field(default_factory=list)
    raw_diff: str = ""
    additions: int = 0
    deletions: int = 0
    is_binary: bool = False


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_pr_diff(cfg: ReviewerConfig) -> str:
    """Fetch the raw unified diff for a PR from the GitHub API."""
    url = f"https://api.github.com/repos/{cfg.github_repository}/pulls/{cfg.pr_number}"
    headers = {
        "Authorization": f"Bearer {cfg.github_token}",
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_pr_metadata(cfg: ReviewerConfig) -> Dict:
    """Fetch PR metadata (title, author, stats) from the GitHub API."""
    url = f"https://api.github.com/repos/{cfg.github_repository}/pulls/{cfg.pr_number}"
    headers = {
        "Authorization": f"Bearer {cfg.github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        "title": data.get("title", ""),
        "author": data.get("user", {}).get("login", ""),
        "additions": data.get("additions", 0),
        "deletions": data.get("deletions", 0),
        "changed_files": data.get("changed_files", 0),
        "base_sha": data.get("base", {}).get("sha", ""),
        "head_sha": data.get("head", {}).get("sha", ""),
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_DIFF_FILE_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)


def parse_diff(raw_diff: str) -> List[FileDiff]:
    """Parse a unified diff string into a list of FileDiff objects."""
    files: List[FileDiff] = []
    current_file: Optional[FileDiff] = None
    current_hunk: Optional[DiffHunk] = None

    for line in raw_diff.splitlines(keepends=True):
        stripped = line.rstrip("\n\r")

        # New file header
        file_match = _DIFF_FILE_HEADER.match(stripped)
        if file_match:
            if current_hunk and current_file:
                current_hunk.content = "".join(current_hunk.lines)
                current_file.hunks.append(current_hunk)
            if current_file:
                current_file.raw_diff = _build_raw_diff(current_file)
                files.append(current_file)

            old_name, new_name = file_match.group(1), file_match.group(2)
            current_file = FileDiff(
                filename=new_name,
                old_filename=old_name if old_name != new_name else None,
            )
            current_hunk = None
            continue

        if current_file is None:
            continue

        # Detect binary
        if stripped.startswith("Binary files"):
            current_file.is_binary = True
            continue

        # Detect status from index/new/deleted lines
        if stripped.startswith("new file"):
            current_file.status = "added"
            continue
        if stripped.startswith("deleted file"):
            current_file.status = "removed"
            continue
        if stripped.startswith("rename from") or stripped.startswith("rename to"):
            current_file.status = "renamed"
            continue

        # Hunk header
        hunk_match = _HUNK_HEADER.match(stripped)
        if hunk_match:
            if current_hunk:
                current_hunk.content = "".join(current_hunk.lines)
                current_file.hunks.append(current_hunk)

            current_hunk = DiffHunk(
                header=stripped,
                old_start=int(hunk_match.group(1)),
                old_count=int(hunk_match.group(2) or 1),
                new_start=int(hunk_match.group(3)),
                new_count=int(hunk_match.group(4) or 1),
            )
            continue

        # Diff content lines
        if current_hunk is not None:
            current_hunk.lines.append(line)
            if stripped.startswith("+") and not stripped.startswith("+++"):
                current_file.additions += 1
            elif stripped.startswith("-") and not stripped.startswith("---"):
                current_file.deletions += 1

    # Flush last hunk/file
    if current_hunk and current_file:
        current_hunk.content = "".join(current_hunk.lines)
        current_file.hunks.append(current_hunk)
    if current_file:
        current_file.raw_diff = _build_raw_diff(current_file)
        files.append(current_file)

    return files


def _build_raw_diff(fd: FileDiff) -> str:
    """Reconstruct raw diff text from hunks."""
    parts = []
    for hunk in fd.hunks:
        parts.append(hunk.header + "\n")
        parts.extend(hunk.lines)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_files(
    file_diffs: List[FileDiff], cfg: ReviewerConfig
) -> List[FileDiff]:
    """Remove files matching skip patterns, binary files, and deleted files."""
    result = []
    for fd in file_diffs:
        if fd.is_binary:
            continue
        if fd.status == "removed":
            continue
        if _matches_skip_pattern(fd.filename, cfg.skip_patterns):
            continue
        result.append(fd)
    return result


def _matches_skip_pattern(filename: str, patterns: List[str]) -> bool:
    """Check if a filename matches any of the skip glob patterns."""
    for pattern in patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
        if fnmatch.fnmatch(filename.replace("\\", "/"), pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

MAX_CHUNK_LINES = 150


def chunk_file_diff(fd: FileDiff, max_lines: int = MAX_CHUNK_LINES) -> List[str]:
    """Split a file diff into chunks suitable for LLM input.

    Each chunk is a string containing one or more hunks, up to max_lines.
    If a single hunk exceeds max_lines, it is sent alone.
    """
    chunks: List[str] = []
    current_lines: List[str] = []
    current_count = 0

    for hunk in fd.hunks:
        hunk_line_count = len(hunk.lines) + 1  # +1 for header
        if current_count + hunk_line_count > max_lines and current_lines:
            chunks.append("".join(current_lines))
            current_lines = []
            current_count = 0

        current_lines.append(hunk.header + "\n")
        current_lines.extend(hunk.lines)
        current_count += hunk_line_count

    if current_lines:
        chunks.append("".join(current_lines))

    return chunks


def get_diff_position_map(fd: FileDiff) -> Dict[int, int]:
    """Build a mapping from new-file line number to diff position (1-indexed).

    The diff position is the line's position in the diff output, used by the
    GitHub API for posting inline review comments.
    """
    position_map: Dict[int, int] = {}
    position = 0

    for hunk in fd.hunks:
        position += 1  # hunk header
        current_line = hunk.new_start

        for line_text in hunk.lines:
            position += 1
            if line_text.startswith("+") or line_text.startswith(" "):
                position_map[current_line] = position
                current_line += 1
            elif line_text.startswith("-"):
                pass  # deleted line, no new-file line number
            else:
                current_line += 1

    return position_map
