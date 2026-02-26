"""Diff parsing, splitting, chunking, and line-mapping utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from tools.common import get_logger, load_json_file

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)"
)


@dataclass
class DiffHunk:
    """A single hunk inside a file diff."""

    header: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FileDiff:
    """Parsed diff for one file."""

    filename: str
    status: str = "modified"
    hunks: list[DiffHunk] = field(default_factory=list)
    is_binary: bool = False
    is_new: bool = False
    is_deleted: bool = False


@dataclass
class DiffChunk:
    """A chunk of diff content sized for LLM input."""

    file: str
    hunk_header: str
    context: str
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_unified_diff(raw_diff: str) -> list[FileDiff]:
    """Parse a full unified diff string into a list of ``FileDiff`` objects."""
    files: list[FileDiff] = []
    current_file: FileDiff | None = None
    current_hunk: DiffHunk | None = None

    for line in raw_diff.splitlines():
        # New file header
        if line.startswith("diff --git"):
            parts = line.split(" b/", 1)
            fname = parts[1] if len(parts) == 2 else ""
            current_file = FileDiff(filename=fname)
            files.append(current_file)
            current_hunk = None
            continue

        if current_file is None:
            continue

        # Binary marker
        if line.startswith("Binary files"):
            current_file.is_binary = True
            continue

        # New / deleted markers
        if line.startswith("new file mode"):
            current_file.is_new = True
            continue
        if line.startswith("deleted file mode"):
            current_file.is_deleted = True
            continue

        # Hunk header
        m = HUNK_HEADER_RE.match(line)
        if m:
            current_hunk = DiffHunk(
                header=line,
                old_start=int(m.group(1)),
                old_count=int(m.group(2) or 1),
                new_start=int(m.group(3)),
                new_count=int(m.group(4) or 1),
            )
            current_file.hunks.append(current_hunk)
            continue

        # Diff content lines (skip --- / +++ headers)
        if line.startswith("---") or line.startswith("+++"):
            continue

        if current_hunk is not None:
            current_hunk.lines.append(line)

    return files


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def should_skip_file(filename: str, skip_patterns: list[str]) -> bool:
    """Return True if *filename* matches any skip glob pattern."""
    from fnmatch import fnmatch

    for pat in skip_patterns:
        if fnmatch(filename, pat):
            return True
    return False


def filter_files(
    files: list[FileDiff],
    skip_patterns: list[str] | None = None,
) -> list[FileDiff]:
    """Remove binary, deleted, and pattern-matched files."""
    skip = skip_patterns or []
    kept: list[FileDiff] = []
    for fd in files:
        if fd.is_binary:
            logger.debug("skip binary: %s", fd.filename)
            continue
        if fd.is_deleted:
            logger.debug("skip deleted: %s", fd.filename)
            continue
        if should_skip_file(fd.filename, skip):
            logger.debug("skip pattern: %s", fd.filename)
            continue
        kept.append(fd)
    return kept


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_file_diff(fd: FileDiff, max_lines: int = 300) -> list[DiffChunk]:
    """Split a ``FileDiff`` into LLM-sized ``DiffChunk`` objects."""
    chunks: list[DiffChunk] = []
    for hunk in fd.hunks:
        if len(hunk.lines) <= max_lines:
            chunks.append(
                DiffChunk(
                    file=fd.filename,
                    hunk_header=hunk.header,
                    context="\n".join(hunk.lines),
                    start_line=hunk.new_start,
                    end_line=hunk.new_start + hunk.new_count - 1,
                )
            )
        else:
            # Split oversized hunk into sub-chunks
            for i in range(0, len(hunk.lines), max_lines):
                segment = hunk.lines[i : i + max_lines]
                offset = _count_new_lines(hunk.lines[:i])
                start = hunk.new_start + offset
                end = start + _count_new_lines(segment) - 1
                chunks.append(
                    DiffChunk(
                        file=fd.filename,
                        hunk_header=hunk.header,
                        context="\n".join(segment),
                        start_line=start,
                        end_line=max(start, end),
                    )
                )
    return chunks


def _count_new_lines(lines: list[str]) -> int:
    """Count lines that are additions or context (not removals)."""
    return sum(1 for ln in lines if not ln.startswith("-"))


# ---------------------------------------------------------------------------
# Line mapping
# ---------------------------------------------------------------------------


def build_line_map(fd: FileDiff) -> dict[int, int]:
    """Map new-file line numbers to 1-based diff positions for inline comments.

    GitHub's ``position`` parameter for review comments is the 1-based index
    into the diff hunk (including context and removal lines).
    """
    pos_map: dict[int, int] = {}
    position = 0
    for hunk in fd.hunks:
        position += 1  # hunk header counts as position 1
        new_line = hunk.new_start
        for line in hunk.lines:
            position += 1
            if line.startswith("+"):
                pos_map[new_line] = position
                new_line += 1
            elif line.startswith("-"):
                pass  # removed line — no new-file line number
            else:
                pos_map[new_line] = position
                new_line += 1
    return pos_map


# ---------------------------------------------------------------------------
# High-level split helper
# ---------------------------------------------------------------------------


def split_by_file_and_hunk(
    raw_diff: str,
    skip_patterns: list[str] | None = None,
    max_chunk_lines: int = 300,
) -> list[DiffChunk]:
    """Parse raw diff → filter → chunk.  Main entry point for the diff tool."""
    files = parse_unified_diff(raw_diff)
    files = filter_files(files, skip_patterns)
    chunks: list[DiffChunk] = []
    for fd in files:
        chunks.extend(chunk_file_diff(fd, max_chunk_lines))
    logger.info("Diff split into %d chunks across %d files", len(chunks), len(files))
    return chunks
