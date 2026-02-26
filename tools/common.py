"""Common utilities: JSONL logging, secret redaction, hashing, timers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

SECRET_PATTERN = re.compile(
    r"(?:password|passwd|pwd|secret|api.?key|access.?key|auth.?token"
    r"|private.?key|client.?secret|bearer|token)\s*[:=\s]\s*\S+",
    re.IGNORECASE,
)

GITHUB_TOKEN_PATTERN = re.compile(
    r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{40,}"
)

AWS_KEY_PATTERN = re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")

PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"
)

_ALL_SECRET_PATTERNS = [
    SECRET_PATTERN,
    GITHUB_TOKEN_PATTERN,
    AWS_KEY_PATTERN,
    PRIVATE_KEY_PATTERN,
]


def redact_secrets(text: str) -> str:
    """Replace potential secrets in *text* with [REDACTED]."""
    result = text
    for pat in _ALL_SECRET_PATTERNS:
        result = pat.sub("[REDACTED]", result)
    return result


# ---------------------------------------------------------------------------
# Stable hashing
# ---------------------------------------------------------------------------


def stable_hash(file: str, line: int, message: str, length: int = 12) -> str:
    """Return a truncated SHA-256 hex digest for deduplication."""
    payload = f"{file}:{line}:{message}".encode()
    return hashlib.sha256(payload).hexdigest()[:length]


# ---------------------------------------------------------------------------
# JSONL structured logger
# ---------------------------------------------------------------------------


class JSONLFormatter(logging.Formatter):
    """Emit each log record as a single JSON line with redaction."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact_secrets(record.getMessage()),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def get_logger(name: str, jsonl: bool = True) -> logging.Logger:
    """Return a logger configured for JSONL or plain output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    if jsonl:
        handler.setFormatter(JSONLFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Timer / budget helpers
# ---------------------------------------------------------------------------


class TimeBudget:
    """Track wall-clock time against a configurable budget."""

    def __init__(self, max_seconds: int = 300):
        self.max_seconds = max_seconds
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.elapsed >= self.max_seconds

    def check(self, label: str = "") -> None:
        """Raise RuntimeError if budget is exhausted."""
        if self.expired:
            raise RuntimeError(
                f"Time budget exhausted ({self.max_seconds}s) at step: {label}"
            )


@contextmanager
def timed_section(logger: logging.Logger, label: str) -> Generator[None, None, None]:
    """Context manager that logs duration of a section."""
    start = time.monotonic()
    logger.info("START %s", label)
    try:
        yield
    finally:
        dur = time.monotonic() - start
        logger.info("END %s (%.2fs)", label, dur)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return a dict; return {} on missing file."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def env_or(key: str, default: Any = None) -> Any:
    """Read an environment variable or return *default*."""
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    """Read an environment variable as int."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(key: str, default: float) -> float:
    """Read an environment variable as float."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def write_audit_log(path: str | Path, data: dict[str, Any]) -> None:
    """Append a redacted JSON record to the audit log file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    safe = json.loads(redact_secrets(json.dumps(data, default=str)))
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, default=str) + "\n")
