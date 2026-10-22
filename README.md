# A2A Invoice Action Agent

A FastAPI implementation scaffold for the A2A Invoice Agent assignment. It includes:
- Public Agent Card at `/.well-known/agent-card.json`
- Authenticated A2A 1.0 HTTP+JSON routes
- Invoice batch analysis through an OpenAI-compatible AI API
- Structured proposal validation, task history, and result continuation
- SQLite persistence, bearer-principal isolation, idempotency, cancellation, and tests
- Render deployment guidance

> Important: Treat this as a starter implementation and test it against the assignment's exact protocol/grader. The assignment's custom A2A media types and envelope requirements are implemented here, but hidden grader behavior may require small adjustments.

## 1. Local setup

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env  # Windows; on macOS/Linux use: cp .env.example .env
```

Edit `.env`. Add your AI API key if you want live AI decisions. The app can run without a key for local protocol testing, using a clearly marked conservative demo decision path; do not rely on demo mode for submission.

Start:

```bash
uvicorn app.main:app --reload
```

Agent Card: <http://127.0.0.1:8000/.well-known/agent-card.json>

## 2. Environment variables

- `AI_API_KEY`: secret AI provider key; configure it in Render's Environment page, not GitHub.
- `AI_BASE_URL`: OpenAI-compatible API root, e.g. `https://api.openai.com/v1`.
- `AI_MODEL`: model name supported by your provider.
- `A2A_BEARER_TOKEN`: token accepted by this deployment. For multi-principal tests, different bearer tokens are treated as different principals.
- `DATABASE_URL`: SQLAlchemy database URL.
- `PUBLIC_BASE_URL`: public origin/base URL used in Agent Card; set to your HTTPS Render URL.
- `APP_VERSION`: version string.
- `AI_TIMEOUT_SECONDS`: AI call timeout.

If using a provider with a non-OpenAI-compatible API, adapt `app/services/ai_service.py` to its request format.

## 3. Render deployment

1. Push this project to a GitHub repository. Do not push `.env`.
2. In Render, create a Web Service and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Set `AI_API_KEY`, `AI_BASE_URL`, `AI_MODEL`, `A2A_BEARER_TOKEN`, `DATABASE_URL`, and `PUBLIC_BASE_URL` in Render Environment.
6. Use a persistent database. SQLite on an ephemeral filesystem can lose data on redeploy/restart. If using a Render persistent disk with SQLite, mount it and point `DATABASE_URL` to that disk path; otherwise use a managed SQL database supported by your assignment.
7. Confirm the public Agent Card and API behavior over HTTPS.

## 4. A2A endpoints

- `GET /.well-known/agent-card.json` — public
- `POST /a2a/message:send` — authenticated
- `GET /a2a/tasks` — authenticated, lists only the current principal's tasks
- `GET /a2a/tasks/{task_id}` — authenticated and owner-only
- `POST /a2a/tasks/{task_id}:cancel` — authenticated and owner-only

All authenticated routes require:
- `Authorization: Bearer <token>`
- `A2A-Version: 1.0`
- `Content-Type: application/a2a+json` for POSTs

## 5. Request/continuation outline

Initial request message data should use media type:
`application/vnd.ga5.invoice-claim-batch+json`

The initial message data is expected to contain `batchId`, `policyRevision`, and `packages`. A valid result continuation uses `batchId` and `results`, where each result includes `packageId`, `actionId`, `action`, `outcome`, and `receiptNonce`.

The implementation accepts both direct message objects and common A2A-like message wrappers. Always compare the payload with your assignment's exact example request when integrating with the evaluator.

## 6. Tests

```bash
pytest -q
```

Tests use isolated in-memory SQLite and do not need an AI API key.

## 7. Security notes

- Never commit credentials.
- Demo mode is for local smoke testing only.
- The app validates AI output before creating proposals.
- AI output is not permission to execute a financial action. Only accepted result continuations produce receipts.
