"""LLM interaction layer for AI Code Reviewer.

Wraps model.SharedGPTConnector with chunking, JSON schema validation,
retries with exponential backoff, and support for both hosted and local
model endpoints.
"""

import json
import logging
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure the project root is on sys.path so `model` package is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model import SharedGPTConnector

from .config import ReviewerConfig
from .prompts import SYSTEM_PROMPT, build_per_file_prompt, build_summary_prompt

logger = logging.getLogger("ai-reviewer.llm")

# ---------------------------------------------------------------------------
# JSON schema for per-file review issues
# ---------------------------------------------------------------------------

VALID_SEVERITIES = {"error", "warn", "info"}
REQUIRED_FIELDS = {"file", "line", "severity", "category", "message"}


def validate_issue(item: Dict[str, Any]) -> bool:
    """Validate a single review issue against the expected schema."""
    if not isinstance(item, dict):
        return False
    for field in REQUIRED_FIELDS:
        if field not in item:
            return False
    if item.get("severity") not in VALID_SEVERITIES:
        return False
    if not isinstance(item.get("line"), int):
        return False
    if not isinstance(item.get("message"), str) or not item["message"].strip():
        return False
    return True


def validate_issues(items: Any) -> List[Dict[str, Any]]:
    """Validate and filter a list of review issues."""
    if not isinstance(items, list):
        return []
    return [item for item in items if validate_issue(item)]


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Extract the first JSON array from LLM output text."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find array within text (LLM may wrap in markdown fences)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Wrapper around SharedGPTConnector with retry, validation, and chunking."""

    def __init__(self, cfg: ReviewerConfig) -> None:
        self.cfg = cfg
        self._total_calls = 0
        self._total_tokens_approx = 0
        self._start_time = time.monotonic()

        # Configure connector based on local vs hosted endpoint
        env_overrides: Dict[str, str] = {}
        if cfg.local_model_url:
            env_overrides["DELL_GATEWAY_BASE_URL"] = cfg.local_model_url
            env_overrides["USE_SSO"] = "false"

        # Apply env overrides temporarily for connector init
        original_env: Dict[str, Optional[str]] = {}
        for key, val in env_overrides.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = val

        try:
            self.connector = SharedGPTConnector(
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens_per_request,
            )
        finally:
            for key, val in original_env.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_llm_calls": self._total_calls,
            "approx_tokens_used": self._total_tokens_approx,
            "elapsed_seconds": round(time.monotonic() - self._start_time, 1),
        }

    def review_chunk(
        self,
        filename: str,
        language: str,
        diff_content: str,
        context: str = "",
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """Send a diff chunk to the LLM for review and return validated issues."""
        self._check_time_budget()

        user_prompt = build_per_file_prompt(filename, language, diff_content, context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_response = self._call_with_retry(messages, max_retries=max_retries)
        if not raw_response:
            return []

        items = extract_json_array(raw_response)
        valid = validate_issues(items)

        # If validation failed and we got some response, try reprompting once
        if not valid and raw_response.strip():
            logger.info("Schema validation failed, reprompting once for %s", filename)
            repair_msg = (
                "Your previous response was not valid JSON matching the schema. "
                "The error: expected a JSON array of objects with fields "
                "(file, line, severity, category, message, suggestion). "
                "Please try again. Return [] if no issues."
            )
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": repair_msg})
            raw_response = self._call_with_retry(messages, max_retries=1)
            if raw_response:
                items = extract_json_array(raw_response)
                valid = validate_issues(items)

        # Ensure all issues reference the correct file
        for issue in valid:
            issue["file"] = filename

        return valid

    def generate_summary(
        self,
        aggregated_results: List[Dict[str, Any]],
        pr_title: str,
        pr_author: str,
        files_changed: int,
        lines_added: int,
        lines_removed: int,
    ) -> str:
        """Generate a PR-level summary from aggregated review results."""
        self._check_time_budget()

        user_prompt = build_summary_prompt(
            aggregated_results=json.dumps(aggregated_results, indent=2),
            pr_title=pr_title,
            pr_author=pr_author,
            files_changed=files_changed,
            lines_added=lines_added,
            lines_removed=lines_removed,
            max_lines=self.cfg.summary.max_lines,
        )

        messages = [
            {"role": "system", "content": "You are a code review summarizer. Produce clear, concise Markdown summaries."},
            {"role": "user", "content": user_prompt},
        ]

        return self._call_with_retry(messages, max_retries=2) or ""

    def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
    ) -> str:
        """Call the LLM with exponential backoff retry."""
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                self._total_calls += 1
                response = self.connector.chat_completion(
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens_per_request,
                    stream=False,
                )
                # Rough token approximation for stats
                self._total_tokens_approx += len(response) // 4
                return response

            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    backoff = min(30, (2 ** attempt) + random.uniform(0, 1))
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        str(exc)[:200],
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "LLM call failed after %d attempts: %s",
                        max_retries + 1,
                        str(exc)[:200],
                    )

        return ""

    def _check_time_budget(self) -> None:
        """Raise if we've exceeded the 5-minute soft time budget."""
        elapsed = time.monotonic() - self._start_time
        if elapsed > 300:
            raise TimeoutError(
                f"AI reviewer time budget exceeded ({elapsed:.0f}s > 300s). "
                "Stopping to avoid workflow timeout."
            )
