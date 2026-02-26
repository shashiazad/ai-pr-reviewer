"""Configuration loader for AI Code Reviewer.

Loads settings from .ai-reviewer.json (repo-level) and environment variables.
Environment variables override JSON config values.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_CONFIG_PATH = ".ai-reviewer.json"
DEFAULT_SKIP_PATTERNS = [
    "*.lock",
    "*.min.js",
    "*.min.css",
    "*.pb.go",
    "*.generated.*",
    "vendor/**",
    "node_modules/**",
    "__pycache__/**",
    ".git/**",
]
SUPPORTED_LANGUAGES = ["terraform", "ansible", "yaml", "python", "bash"]


@dataclass
class FormatChecks:
    terraform: bool = True
    python_black: bool = True
    yamllint: bool = True
    shellcheck: bool = True
    ansible_lint: bool = True


@dataclass
class SummaryConfig:
    enabled: bool = True
    max_lines: int = 200


@dataclass
class ReviewerConfig:
    enabled: bool = True
    languages: List[str] = field(default_factory=lambda: list(SUPPORTED_LANGUAGES))
    max_comments: int = 20
    severity_threshold: str = "info"
    temperature: float = 0.1
    max_tokens_per_request: int = 1500
    skip_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_SKIP_PATTERNS))
    format_checks: FormatChecks = field(default_factory=FormatChecks)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    nit_consolidation: bool = True
    dry_run: bool = False
    audit_log: bool = False

    # Resolved at runtime
    github_token: str = ""
    github_repository: str = ""
    pr_number: int = 0
    base_ref: str = ""
    head_sha: str = ""
    dell_gateway_base_url: str = "https://aia.gateway.dell.com/genai/dev/v1"
    use_sso: bool = True
    local_model_url: Optional[str] = None


def load_config(config_path: Optional[str] = None) -> ReviewerConfig:
    """Load configuration from JSON file and environment variables."""
    cfg = ReviewerConfig()

    # --- Load JSON config ---
    path = Path(config_path or os.getenv("AI_REVIEWER_CONFIG", DEFAULT_CONFIG_PATH))
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _apply_json(cfg, data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ai-reviewer] Warning: Failed to load config from {path}: {exc}")

    # --- Override from environment ---
    cfg.github_token = os.getenv("GITHUB_TOKEN", "")
    cfg.github_repository = os.getenv("GITHUB_REPOSITORY", "")
    cfg.dell_gateway_base_url = os.getenv(
        "DELL_GATEWAY_BASE_URL", cfg.dell_gateway_base_url
    )
    cfg.use_sso = os.getenv("USE_SSO", "true").lower() == "true"
    cfg.local_model_url = os.getenv("LOCAL_MODEL_URL") or None

    if os.getenv("AI_REVIEWER_DRY_RUN", "").lower() in ("true", "1", "yes"):
        cfg.dry_run = True
    if os.getenv("AI_REVIEWER_AUDIT_LOG", "").lower() in ("true", "1", "yes"):
        cfg.audit_log = True
    if os.getenv("MODEL_TEMPERATURE"):
        cfg.temperature = float(os.getenv("MODEL_TEMPERATURE"))
    if os.getenv("MODEL_MAX_TOKENS"):
        cfg.max_tokens_per_request = int(os.getenv("MODEL_MAX_TOKENS"))

    # --- Resolve PR metadata from event ---
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if event_path and Path(event_path).exists():
        try:
            with open(event_path, "r", encoding="utf-8") as f:
                event = json.load(f)
            pr = event.get("pull_request", {})
            cfg.pr_number = pr.get("number", 0)
            cfg.base_ref = pr.get("base", {}).get("ref", "")
            cfg.head_sha = pr.get("head", {}).get("sha", "")
        except (json.JSONDecodeError, OSError):
            pass

    return cfg


def _apply_json(cfg: ReviewerConfig, data: Dict) -> None:
    """Apply values from parsed JSON config dict to the config dataclass."""
    if "enabled" in data:
        cfg.enabled = bool(data["enabled"])
    if "languages" in data:
        cfg.languages = [
            lang for lang in data["languages"] if lang in SUPPORTED_LANGUAGES
        ]
    if "max_comments" in data:
        cfg.max_comments = int(data["max_comments"])
    if "severity_threshold" in data and data["severity_threshold"] in (
        "info",
        "warn",
        "error",
    ):
        cfg.severity_threshold = data["severity_threshold"]
    if "temperature" in data:
        cfg.temperature = float(data["temperature"])
    if "max_tokens_per_request" in data:
        cfg.max_tokens_per_request = int(data["max_tokens_per_request"])
    if "skip_patterns" in data:
        cfg.skip_patterns = list(data["skip_patterns"])
    if "nit_consolidation" in data:
        cfg.nit_consolidation = bool(data["nit_consolidation"])
    if "dry_run" in data:
        cfg.dry_run = bool(data["dry_run"])
    if "audit_log" in data:
        cfg.audit_log = bool(data["audit_log"])

    fc = data.get("format_checks", {})
    if fc:
        cfg.format_checks = FormatChecks(
            terraform=fc.get("terraform", True),
            python_black=fc.get("python_black", True),
            yamllint=fc.get("yamllint", True),
            shellcheck=fc.get("shellcheck", True),
            ansible_lint=fc.get("ansible_lint", True),
        )

    sm = data.get("summary", {})
    if sm:
        cfg.summary = SummaryConfig(
            enabled=sm.get("enabled", True),
            max_lines=sm.get("max_lines", 200),
        )
