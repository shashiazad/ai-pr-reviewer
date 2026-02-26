# Gemini Model Connector

This directory contains the **shared model connector** that any Python agent can import to call the Google Gemini API in a unified way.

## What is included

- `gemini_connector.py`  
  Reusable `GeminiConnector` class for chat/text/JSON model calls.
- `__init__.py`  
  Re-export for clean imports.
- `requirements.txt`  
  Dependencies needed for this shared connector.

---

## 1) Install dependencies

From your project environment:

```bash
pip install -r model/requirements.txt
```

---

## 2) Configure environment variables

Create a `.env` file inside this `model/` directory:

```bash
GEMINI_API_KEY=your_api_key_here
```

Get a free API key at: https://aistudio.google.com/app/apikey

Required keys:

- `GEMINI_API_KEY` — your Google Gemini API key (required)

Optional tuning:

- `GEMINI_MODEL_NAME` — model to use (default: `gemini-2.0-flash`)
- `MODEL_TEMPERATURE` — generation temperature (default: `0.1`)
- `MODEL_MAX_TOKENS` — max output tokens (default: `1500`)
- `MODEL_TIMEOUT_SECONDS` — request timeout (default: `120`)

---

## 3) Exact code snippet to use in any agent file

Use this in `scanner-agent.py`, `cookbook-agent.py`, or any other Python agent:

```python
from model import GeminiConnector

# Uses model/.env automatically
gemini = GeminiConnector()

# Plain text response
answer = gemini.ask_text(
    "Explain CIS hardening in 5 concise bullets.",
    temperature=0.1,
    max_tokens=600,
)
print(answer)

# Structured JSON response (extracts first JSON object from model output)
json_answer = gemini.ask_json(
    "Return JSON only: {\"status\": \"ok\", \"checks\": [\"a\", \"b\"]}",
    temperature=0.0,
    max_tokens=300,
)
print(json_answer)
```

---

## 4) Advanced usage (message array)

```python
from model import GeminiConnector

gemini = GeminiConnector()

content = gemini.chat_completion(
    messages=[
        {"role": "system", "content": "You are a security automation assistant."},
        {"role": "user", "content": "Generate a safe shell audit command for sshd_config."},
    ],
    temperature=0.1,
    max_tokens=400,
)

print(content)
```

The connector translates OpenAI-style messages (`system`, `user`, `assistant`) into the
Gemini SDK format automatically, so existing code works with minimal changes.

---

## 5) Security best practices

- Do **not** commit your `GEMINI_API_KEY` to source control.
- Keep credentials in `.env` or secure secret stores.
- Add `model/.env` to your `.gitignore`.
