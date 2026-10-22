# A2A Invoice Action Agent

FastAPI service for the invoice-agent assignment. It implements the custom A2A 1.0 HTTP+JSON surface, durable task records, principal-scoped access, message idempotency, package decision caching, continuation validation, receipts, and atomic cancellation/completion transitions.

**Important:** this code is designed to follow the assignment contract, but hidden grader behavior and real invoice decisions must still be verified with the provided grader. A valid AI API key is required for business decision scoring. The app deliberately does not silently substitute a blanket `hold_invoice` answer when the key is missing.

## Local setup

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Windows: copy .env.example .env
```

Set `AI_API_KEY` in `.env` for real decisions. Do not commit `.env`. Start the service:

```bash
uvicorn app.main:app --reload
```

- Agent Card: `http://127.0.0.1:8000/.well-known/agent-card.json`
- Health: `http://127.0.0.1:8000/health`
- FastAPI docs: `http://127.0.0.1:8000/docs`

## Render deployment

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- `PUBLIC_BASE_URL`: set to the public origin only, e.g. `https://your-service.onrender.com`. The Agent Card advertises `https://your-service.onrender.com/a2a/`.
- Set `AI_API_KEY`, `AI_BASE_URL`, `AI_MODEL`, and `A2A_BEARER_TOKEN` in Render's Environment page. Never commit keys to GitHub.
- Use persistent storage. SQLite on Render's ephemeral filesystem can lose task and idempotency data after restarts/deploys. For SQLite, attach a persistent disk and use a database path on that disk; otherwise use a persistent managed SQL database.

## Routes

| Method | Path | Access | Success response |
|---|---|---|---|
| GET | `/.well-known/agent-card.json` | Public | Agent Card JSON |
| GET | `/health` | Public | `{"status":"ok"}` |
| POST | `/a2a/message:send` | Bearer + A2A version + A2A media type | `{"task": Task}` |
| GET | `/a2a/tasks` | Bearer + A2A version | `{"tasks": [Task, ...]}` for caller only |
| GET | `/a2a/tasks/{task_id}` | Bearer + A2A version; owner only | Task |
| POST | `/a2a/tasks/{task_id}:cancel` | Bearer + A2A version + A2A media type; owner only | Task |

Protected routes require `Authorization: Bearer <token>` and `A2A-Version: 1.0`. POST routes require `Content-Type: application/a2a+json`. Each distinct bearer token maps to a separate principal for isolation checks.

## Message contract

An initial `message` must contain a `messageId`, `role: ROLE_USER`, and a Part with media type `application/vnd.ga5.invoice-claim-batch+json`. Its data contains `batchId`, `policyRevision`, and `packages`. The response includes a `TASK_STATE_INPUT_REQUIRED` Task and one proposal artifact using `application/vnd.ga5.invoice-action-proposals+json`.

A continuation uses a new `messageId`, exact `taskId` and `contextId`, `role: ROLE_USER`, and media type `application/vnd.ga5.invoice-action-results+json`. Its batch and each package/action identity must match the stored proposal. Only `ACCEPTED` results become receipt executions. The receipt artifact uses `application/vnd.ga5.invoice-action-receipts+json`; rejected results are retained in history but do not execute.

Equivalent requests deduplicate by principal + `messageId` and canonical hash of `message` only; `configuration` is ignored. Reusing a message ID with different message content returns HTTP 409 `IDEMPOTENCY_CONFLICT`. Task completion and cancellation use conditional database state transitions so only one can win.

## AI decision cache

Decision cache keys use canonical package content plus `policyRevision`, excluding namespace-only package IDs. A new batch/message/task can therefore reuse stable package decisions while fresh evidence creates a new cache key. The AI is prompted to choose one permitted action, copy three decisive evidence references, provide facts, and cite evidence in the rationale. Output is validated before proposals are persisted.

## Tests

```bash
pytest -q
```

Tests mock the AI provider and do not require a live API key. Before grading, configure a valid key, deploy, inspect the public Agent Card, and test a full proposal → results continuation → receipt lifecycle.
