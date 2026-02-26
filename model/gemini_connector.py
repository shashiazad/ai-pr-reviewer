import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiConnector:
    """Reusable connector for Google Gemini models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        timeout_seconds: int = 120,
        env_path: Optional[str] = None,
    ) -> None:
        self._load_environment(env_path)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)

        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", DEFAULT_MODEL)
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", str(max_tokens)))
        self.timeout_seconds = int(os.getenv("MODEL_TIMEOUT_SECONDS", str(timeout_seconds)))

        self._model = genai.GenerativeModel(self.model_name)

    def _load_environment(self, env_path: Optional[str]) -> None:
        if env_path:
            load_dotenv(env_path)
            return
        model_dir = Path(__file__).resolve().parent
        default_env = model_dir / ".env"
        if default_env.exists():
            load_dotenv(default_env)

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
        """Send a chat completion request to Google Gemini.

        Accepts OpenAI-style messages (system/user/assistant) and translates
        them into the Gemini SDK format for seamless drop-in replacement.
        """
        contents: List[Dict[str, Any]] = []
        system_instruction: Optional[str] = None

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_instruction = text
            elif role == "assistant":
                contents.append({"role": "model", "parts": [text]})
            else:
                contents.append({"role": "user", "parts": [text]})

        # Rebuild model instance when system instruction or model name differs
        active_model = self._model
        target_model = model or self.model_name
        if system_instruction or target_model != self.model_name:
            active_model = genai.GenerativeModel(
                target_model,
                system_instruction=system_instruction,
            )

        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )

        response = active_model.generate_content(
            contents,
            generation_config=generation_config,
            request_options={"timeout": self.timeout_seconds},
        )

        if response.text:
            return response.text.strip()
        return ""

    def ask_text(self, prompt: str, **kwargs: Any) -> str:
        """Convenience: single user prompt → text response."""
        return self.chat_completion(messages=[{"role": "user", "content": prompt}], **kwargs)

    def ask_json(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Convenience: single user prompt → parsed JSON dict."""
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
