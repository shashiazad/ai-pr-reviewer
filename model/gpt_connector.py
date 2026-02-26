import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from .authentication_provider import AuthenticationProvider


DEFAULT_BASE_URL = "https://aia.gateway.dell.com/genai/dev/v1"
DEFAULT_MODEL = "gpt-oss-120b"


class SharedGPTConnector:
    """Reusable connector for Dell Gateway-hosted GPT models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        timeout_seconds: int = 120,
        verify_ssl: bool = True,
        env_path: Optional[str] = None,
    ) -> None:
        self._load_environment(env_path)
        self.base_url = os.getenv("DELL_GATEWAY_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"

        self.model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", str(max_tokens)))
        self.timeout_seconds = int(os.getenv("MODEL_TIMEOUT_SECONDS", str(timeout_seconds)))
        self.verify_ssl = _to_bool(os.getenv("MODEL_VERIFY_SSL", str(verify_ssl)))

        self.auth_provider = AuthenticationProvider()

    def _load_environment(self, env_path: Optional[str]) -> None:
        if env_path:
            load_dotenv(env_path)
            return

        model_dir = Path(__file__).resolve().parent
        default_env = model_dir / ".env"
        legacy_env = model_dir.parent / "model_prob" / ".env"

        if default_env.exists():
            load_dotenv(default_env)
        elif legacy_env.exists():
            load_dotenv(legacy_env)

    def _headers(self) -> Dict[str, str]:
        token = self.auth_provider.generate_auth_token()
        return {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **extra_payload: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.model_name,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        payload.update(extra_payload)

        response = requests.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        body = response.json()

        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices or not isinstance(choices, list):
            return ""

        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
        content = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, str):
            return content.strip()

        if content is None:
            return ""

        # Some chat APIs return content as a list of typed blocks.
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts).strip()

        return str(content).strip()

    def ask_text(self, prompt: str, **kwargs: Any) -> str:
        return self.chat_completion(messages=[{"role": "user", "content": prompt}], **kwargs)

    def ask_json(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        content = self.ask_text(prompt, **kwargs)
        return extract_first_json_object(content)


def extract_first_json_object(content: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}

    clean_json = match.group(0).replace("\t", "\\t")
    try:
        return json.loads(clean_json)
    except json.JSONDecodeError:
        return {}


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
