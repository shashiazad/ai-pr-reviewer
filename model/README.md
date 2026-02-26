# Shared GPT Model Connector (`gpt-oss-120b`)

This directory contains the **shared model connector** that any Python agent can import to call your Dell Gateway-hosted GPT model (`gpt-oss-120b`) in a unified way.

## What is included

- `gpt_connector.py`  
  Reusable `SharedGPTConnector` class for chat/text/JSON model calls.
- `authentication_provider.py`  
  Auth helpers (SSO + client credentials) used by the connector.
- `__init__.py`  
  Re-export for clean imports.
- `.env.example`  
  Environment variable template.
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

Copy `.env.example` to `.env` inside this same `model/` directory and fill values:

```bash
copy model\.env.example model\.env
```

Required keys:

- `DELL_GATEWAY_BASE_URL`
- `USE_SSO`
- If `USE_SSO=false`: `CLIENT_ID`, `CLIENT_SECRET`
- Optional tuning: `MODEL_NAME`, `MODEL_TEMPERATURE`, `MODEL_MAX_TOKENS`, `MODEL_TIMEOUT_SECONDS`, `MODEL_VERIFY_SSL`

> Default model is already set to `gpt-oss-120b`.

---

## 3) Exact code snippet to use in any agent file

Use this in `scanner-agent.py`, `cookbook-agent.py`, or any other Python agent:

```python
from model import SharedGPTConnector

# Uses model/.env automatically (falls back to legacy model_prob/.env if still present)
gpt = SharedGPTConnector(model_name="gpt-oss-120b")

# Plain text response
answer = gpt.ask_text(
    "Explain CIS hardening in 5 concise bullets.",
    temperature=0.1,
    max_tokens=600,
    stream=False,
)
print(answer)

# Structured JSON response (extracts first JSON object from model output)
json_answer = gpt.ask_json(
    "Return JSON only: {\"status\": \"ok\", \"checks\": [\"a\", \"b\"]}",
    temperature=0.0,
    max_tokens=300,
    stream=False,
)
print(json_answer)
```

---

## 4) Advanced usage (message array)

```python
from model import SharedGPTConnector

gpt = SharedGPTConnector()

content = gpt.chat_completion(
    messages=[
        {"role": "system", "content": "You are a security automation assistant."},
        {"role": "user", "content": "Generate a safe shell audit command for sshd_config."},
    ],
    temperature=0.1,
    max_tokens=400,
    stream=False,
)

print(content)
```

---

## 5) Migration note from `model_prob`

The connector is designed to replace the old `model_prob` usage:

- `model/gpt_connector.py` now centralizes model calls.
- `model/authentication_provider.py` carries over auth behavior.
- `SharedGPTConnector` auto-loads:
  1. `model/.env` (preferred)
  2. `model_prob/.env` (legacy fallback while migrating)

So you can migrate in phases and later delete `model_prob` safely after moving your env/dependencies.

---

## 6) Security best practices

- Do **not** commit real `CLIENT_SECRET` to source control.
- Keep credentials in `.env` or secure secret stores.
- Use `USE_SSO=true` when available.
