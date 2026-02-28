# AI PR Reviewer (Agentic, Gemini-powered)

An automated Pull Request reviewer that combines:

- deterministic checks (regex security scanning + linters),
- LLM review (Google Gemini), and
- structured GitHub review submission.

It reviews changed files in a PR, posts inline comments, generates a PR summary, and can submit **REQUEST_CHANGES** when blocking findings are present.

---

## 1) What this project is

`ai-pr-reviewer` is a Python-based, agentic code-review system for GitHub PRs.

The project has two execution styles:

1. **Agentic pipeline** (primary):
   - entry: `runner.py`
   - orchestrator: `agents/orchestrator.py`
   - agents: planner, reviewer, critic, commenter

2. **Script pipeline** (legacy/alternate):
   - entry: `scripts/review_pr.py`
   - components in `scripts/*`

Both pipelines review PR diffs and post comments through GitHub APIs.

---

## 2) Tech stack and dependencies

- **Language:** Python 3.11+
- **LLM provider:** Google Gemini via `langchain-google-genai` (`ChatGoogleGenerativeAI`)
- **Model default:** `gemini-2.0-flash`
- **Orchestration:** LangGraph `StateGraph` (compiled, conditional-edge pipeline)
- **LLM framework:** LangChain Core (message types, model abstraction)
- **GitHub integration:** GitHub REST API (PR metadata, comments, reviews)
- **CI/CD:** GitHub Actions (reusable workflow + caller workflow)
- **Static checks/linters:** Terraform, Ansible, YAML, Python, Shell toolchain (depending on changed files)

Key files:

- `runner.py` – CLI entrypoint for agentic flow
- `agents/orchestrator.py` – LangGraph-powered pipeline controller
- `agents/graph.py` – LangGraph state schema + compiled graph builder
- `agents/planner.py` – diff planning/chunking/linter selection
- `agents/reviewer.py` – regex + LLM review execution
- `agents/critic.py` – schema/duplication/severity/budget validation
- `agents/commenter.py` – GitHub review/comment posting
- `policies/prompt_templates.json` – strict prompt contracts
- `configs/agent_policy.json` – runtime behavior controls
- `.github/workflows/reusable-review.yml` – reusable CI workflow

---

## 3) How it works (end-to-end)

### 3.1 Trigger and environment

Typical trigger: Pull Request events via GitHub Actions.

Required env vars:

- `GITHUB_TOKEN`
- `GEMINI_API_KEY`
- `GITHUB_REPOSITORY`
- `PR_NUMBER`

Optional controls:

- `AI_REVIEW_MAX_COMMENTS`
- `AI_REVIEW_SEVERITY`
- `AI_REVIEW_TIMEOUT`
- `GEMINI_MODEL_NAME`

### 3.2 LangGraph orchestrator

The primary workflow is a compiled **LangGraph `StateGraph`** with typed state and conditional edges:

```
fetch_pr → plan_and_diff →[skip?]→ run_linters → review_chunks
                               │                       │
                               ▼              [budget expired?]
                        post_comments ◄── critique ◄───┘
                               │
                               ▼
                          summarize → END
```

Key LangGraph features used:

- **Typed state** (`ReviewState` TypedDict) flows between nodes
- **Conditional edges** for skip-review, budget expiry, and error routing
- **Per-node retry** with bounded attempts and exponential backoff
- **Graceful failure node** posts a failure summary to avoid silent failures
- Agent instances captured via **closure** (not serialised into state)

### 3.3 PlannerAgent

- Parses unified diff
- Filters skipped files (lockfiles, minified files, vendor/dist, generated files)
- Detects language by extension/path markers (Terraform, Ansible, YAML, Python, Bash)
- Chunks large diffs for token-safe LLM calls
- Selects relevant linters per language

### 3.4 ReviewerAgent

Two-phase detection:

1. **Deterministic pre-scan**
   - secret patterns
   - dangerous shell/code patterns

2. **LLM chunk review**
   - strict system/per-file prompts
   - JSON output contract
   - line-specific findings

Then produces a PR summary prompt.

If LLM calls fail (quota/auth/model problems), review still continues with deterministic findings and operational warning behavior.

### 3.5 CriticAgent

- filters by severity threshold
- deduplicates using stable hash
- sorts by severity (error > warn > info)
- enforces comment budget
- can consolidate many low-priority nits
- performs advisory contradiction checks

### 3.6 CommentAgent / CommentManager

- Posts summary + inline comments to PR
- Idempotent summary marker support
- Uses batch review API when possible
- Falls back to individual inline comments when batch fails
- Uses descriptive format:
  - severity/category
  - what's wrong
  - how to fix
- Review event behavior:
  - `REQUEST_CHANGES` when blocking findings exist (error/warn)
  - comment/approve behavior for non-blocking/clean runs

---

## 4) Prompting strategy and anti-hallucination techniques

Prompts are centralized in `policies/prompt_templates.json` and enforce:

- review only diff-visible content
- no speculation about hidden code
- exact line-numbered issues
- strict JSON schema output
- return empty array when no real issue exists
- concise markdown summary format from known findings only

This reduces hallucinations and improves parser reliability.

---

## 5) Security and reliability techniques used

- fail-fast checks for required secrets (`GITHUB_TOKEN`, `GEMINI_API_KEY`)
- deterministic regex scanning as safety net
- LLM retry + graceful fallback strategy
- review deduplication with stable hashing
- bounded runtime with time budget and retry policy
- comment budget controls to avoid PR spam
- dry-run support for non-posting test runs

---

## 6) Running locally

```bash
python runner.py \
  --repo owner/repo \
  --pr 123 \
  --config .ai-reviewer.json \
  --policy configs/agent_policy.json \
  --templates policies/prompt_templates.json
```

Dry-run mode:

```bash
python runner.py --repo owner/repo --pr 123 --dry-run
```

---

## 7) Configuration model

- `.ai-reviewer.json` (project-level behavior)
- `configs/agent_policy.json` (agent runtime policies)
- `policies/prompt_templates.json` (system/per-file/summary prompts)

Tune these to control:

- comment counts
- severity threshold
- chunk sizes
- retry/budget behavior
- prompt strictness and output contracts

---

## 8) CI workflow integration

Reusable workflow: `.github/workflows/reusable-review.yml`

Caller repositories can invoke it with `workflow_call`, pass:

- GitHub token (`token`)
- Gemini API key (`gemini_api_key`)

Required permission for posting reviews/comments:

```yaml
permissions:
  contents: read
  pull-requests: write
```

---

## 9) High-level architecture summary

1. **fetch_pr** — Fetch PR metadata from GitHub API
2. **plan_and_diff** — Fetch unified diff, filter files, chunk for LLM, select linters
3. **run_linters** — Execute relevant linters (Terraform, YAML, Ansible, Black, ShellCheck)
4. **review_chunks** — Deterministic regex scans + LLM review per chunk (via `langchain-google-genai`)
5. **critique** — Deduplicate, validate severity, enforce comment budget, detect contradictions
6. **post_comments** — Post inline comments + summary as GitHub review (`REQUEST_CHANGES` for blocking issues)
7. **summarize** — Log final stats and return structured receipt

All nodes are wired into a compiled **LangGraph `StateGraph`** with conditional routing for skip, budget, and error paths.

---

## 10) Current scope

Best suited for infrastructure/automation and general script-heavy repositories.

Supported-by-default language families include:

- Terraform / HCL
- Ansible / YAML
- Python
- Bash/Shell

The design is extensible for additional languages, scanners, and policy rules.
