# Project Description for Resume: AI PR Reviewer

## One-line summary
Built an agentic, Gemini-powered GitHub Pull Request review system that performs automated static + AI-assisted code review and posts line-level, actionable feedback with merge-blocking change requests.

## Resume-ready project description (short)
Designed and implemented an AI-driven PR reviewer using Python, Google Gemini, and GitHub Actions. Built a state-machine-based multi-agent architecture (Planner, Reviewer, Critic, Commenter) to parse diffs, run deterministic security/lint checks, generate strict JSON findings via LLM prompts, deduplicate/prioritize issues, and post descriptive inline comments to GitHub. Added reliable fallback behavior for LLM failures and enabled REQUEST_CHANGES workflow to block unsafe merges when high-severity issues are detected.

## Resume-ready project description (detailed)
Developed a production-oriented automated code review platform that integrates directly with GitHub PR workflows. The system combines deterministic scanning (regex-based secret/unsafe-pattern detection + linters) with prompt-constrained Gemini review to reduce hallucinations and improve signal quality. Architected a bounded orchestrator state machine with retry/fail-gracefully paths, severity-based issue budgeting, contradiction advisory checks, and idempotent summary updates. Improved developer experience through structured comments that explain what is wrong and how to fix it, while enforcing merge quality gates by submitting REQUEST_CHANGES for blocking findings.

## Key contributions to highlight
- Implemented end-to-end agentic pipeline for PR review automation.
- Integrated Gemini API with strict schema-constrained prompting.
- Added anti-hallucination prompt contracts and few-shot style guidance.
- Built resilient fallback path when LLM calls fail (quota/auth/model issues).
- Implemented issue deduplication, prioritization, and comment budgeting.
- Enabled merge blocking through REQUEST_CHANGES review events.
- Improved review comment quality with actionable fix-oriented structure.
- Automated execution through reusable GitHub Actions workflows.

## Tech stack
- Python
- Google Gemini API
- GitHub REST API
- GitHub Actions
- JSON policy/prompt configuration
- Linters (Terraform, Ansible, YAML, Python, Shell)

## ATS keyword block
AI code review, pull request automation, Gemini API, GitHub Actions, LLM orchestration, prompt engineering, agentic workflow, static analysis, CI/CD automation, secure coding, merge gating, Python backend.
