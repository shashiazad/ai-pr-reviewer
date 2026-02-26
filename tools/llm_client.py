"""LLM client wrapping model.SharedGPTConnector with retry, JSON validation, and budgeting."""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any

from tools.common import get_logger, redact_secrets, env_or, env_float, env_int

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

REQUIRED_ISSUE_FIELDS = {"file", "line", "severity", "message"}
VALID_SEVERITIES = {"info", "warn", "error"}


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract the first JSON array from *text*, tolerating markdown fences."""
    cleaned = text
    if "```" in cleaned:
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = cleaned.replace("```", "")
    m = _JSON_ARRAY_RE.search(cleaned)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def validate_issue(item: Any) -> bool:
    """Return True if *item* is a well-formed review issue dict."""
    if not isinstance(item, dict):
        return False
    for key in REQUIRED_ISSUE_FIELDS:
        if key not in item:
            return False
    if item.get("severity") not in VALID_SEVERITIES:
        return False
    if not isinstance(item.get("line"), int):
        return False
    if not isinstance(item.get("message"), str) or not item["message"].strip():
        return False
    return True


def validate_issues(raw: list[Any]) -> list[dict[str, Any]]:
    """Filter *raw* to only valid issue dicts."""
    if not isinstance(raw, list):
        return []
    return [i for i in raw if validate_issue(i)]


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Wraps model.SharedGPTConnector with retry, schema validation, and budgeting.

    Environment overrides:
      AI_MODEL_BASE_URL  — override Dell Gateway base URL or point to localhost
      AI_MODEL_TOKEN     — static bearer token (testing only)
      AI_REVIEW_TEMPERATURE — prompt temperature
      AI_REVIEW_TIMEOUT  — per-call timeout in seconds
    """

    def __init__(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        timeout: int | None = None,
    ):
        self.temperature = temperature or env_float("AI_REVIEW_TEMPERATURE", 0.1)
        self.max_tokens = max_tokens or env_int("AI_REVIEW_MAX_TOKENS", 800)
        self.max_retries = max_retries
        self.timeout = timeout or env_int("AI_REVIEW_TIMEOUT", 120)

        # Lazy connector — created on first call
        self._connector: Any = None

    def _get_connector(self) -> Any:
        """Lazily import and instantiate SharedGPTConnector."""
        if self._connector is not None:
            return self._connector

        local_url = env_or("AI_MODEL_BASE_URL")
        if local_url:
            logger.info("Using override base URL: %s", redact_secrets(local_url))
            os.environ["DELL_GATEWAY_BASE_URL"] = local_url

        from model import SharedGPTConnector  # type: ignore[import-untyped]

        self._connector = SharedGPTConnector(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout,
        )
        return self._connector

    # -- core call with retry ------------------------------------------------

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request with exponential backoff + jitter."""
        connector = self._get_connector()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = connector.chat_completion(
                    messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                return raw
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        str(exc)[:200],
                        wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts: {last_err}")

    # -- review a diff chunk -------------------------------------------------

    def review_chunk(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Review a diff chunk; return validated issue list.

        On schema failure, re-prompts with the validator error (up to
        ``max_retries`` times), then returns whatever valid subset was found.
        """
        raw = self._call_llm(system_prompt, user_prompt, temperature, max_tokens)
        issues = extract_json_array(raw)
        valid = validate_issues(issues)

        if issues and not valid:
            # Re-prompt once with schema error hint
            logger.warning("Schema validation failed; attempting re-prompt")
            repair_prompt = (
                "Your previous response contained issues that did not match the "
                "required schema. Each item MUST have: file (str), line (int), "
                "severity (info|warn|error), message (str). "
                "Return a corrected JSON array."
            )
            raw2 = self._call_llm(system_prompt, repair_prompt, temperature, max_tokens)
            issues2 = extract_json_array(raw2)
            valid = validate_issues(issues2)

        return valid

    # -- summarize PR --------------------------------------------------------

    def summarize_pr(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a PR-level summary.  Returns raw markdown string."""
        return self._call_llm(
            system_prompt,
            user_prompt,
            temperature=temperature or 0.2,
            max_tokens=max_tokens or 1500,
        )
