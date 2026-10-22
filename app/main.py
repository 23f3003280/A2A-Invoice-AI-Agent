import json
from urllib.parse import urlparse
from fastapi import FastAPI, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.config import settings
from app.database import Base, engine, get_db
from app.models import TaskRecord
from app.security import authenticated_principal, require_a2a_content_type
from app.services.task_service import create_or_replay_task, PROPOSAL_MEDIA, RECEIPT_MEDIA

Base.metadata.create_all(bind=engine)
app = FastAPI(title="A2A Invoice Action Agent", version=settings.app_version)

def _task_url(task_id: str) -> str:
    return f"{settings.public_base_url}/a2a/tasks/{task_id}"

def _validate_public_base_url():
    parsed = urlparse(settings.public_base_url)
    if not parsed.scheme or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError("PUBLIC_BASE_URL must be a valid origin/base URL without credentials, query, or fragment")

@app.get("/.well-known/agent-card.json")
async def agent_card():
    _validate_public_base_url()
    base = settings.public_base_url.rstrip("/") + "/a2a/"
    return JSONResponse({
        "name": "Invoice Action Agent",
        "description": "Analyzes invoice claim batches and proposes safe invoice actions; execution receipts require an accepted result continuation.",
        "version": settings.app_version,
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "supportedInterfaces": [{
            "url": base,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
            "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
            "defaultOutputModes": [
                "application/vnd.ga5.invoice-action-proposals+json",
                "application/vnd.ga5.invoice-action-receipts+json"
            ]
        }],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Analyzes invoice evidence and policy to propose settle, approval, hold, duplicate rejection, or exception actions.",
            "tags": ["invoice", "finance", "evidence", "approval"]
        }]
    })

@app.post("/a2a/message:send")
async def send_message(
    request: Request,
    principal: str = Depends(authenticated_principal),
    db: Session = Depends(get_db),
):
    require_a2a_content_type(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        result = await create_or_replay_task(db, principal, body)
        if result.get("conflict"):
            return JSONResponse(status_code=409, media_type="application/a2a+json", content={
                "error": {"code": "IDEMPOTENCY_CONFLICT", "message": "messageId was already used with different message content"}
            })
        return JSONResponse({"task": result["task"]}, media_type="application/a2a+json")
    except PermissionError:
        raise HTTPException(status_code=404, detail={"code": "TASK_NOT_FOUND", "message": "Task not found"})
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail={"code": "INVALID_TASK_STATE", "message": str(exc)})
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_REQUEST", "message": str(exc)})
    except Exception:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": "Request could not be processed"})

@app.get("/a2a/tasks")
async def list_tasks(
    principal: str = Depends(authenticated_principal),
    db: Session = Depends(get_db),
):
    rows = db.scalars(select(TaskRecord).where(TaskRecord.principal_id == principal).order_by(TaskRecord.created_at.desc())).all()
    return JSONResponse({"tasks": [json.loads(row.task_json) for row in rows]}, media_type="application/a2a+json")

@app.get("/a2a/tasks/{task_id}")
async def get_task(
    task_id: str,
    principal: str = Depends(authenticated_principal),
    db: Session = Depends(get_db),
):
    row = db.get(TaskRecord, task_id)
    if not row or row.principal_id != principal:
        raise HTTPException(status_code=404, detail={"code": "TASK_NOT_FOUND", "message": "Task not found"})
    return JSONResponse(json.loads(row.task_json), media_type="application/a2a+json")

@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(
    task_id: str,
    request: Request,
    principal: str = Depends(authenticated_principal),
    db: Session = Depends(get_db),
):
    require_a2a_content_type(request)
    row = db.get(TaskRecord, task_id)
    if not row or row.principal_id != principal:
        raise HTTPException(status_code=404, detail={"code": "TASK_NOT_FOUND", "message": "Task not found"})
    task = json.loads(row.task_json)
    state = task.get("status", {}).get("state")
    if state in {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED"}:
        raise HTTPException(status_code=409, detail={"code": "TERMINAL_TASK", "message": "Terminal task cannot be changed"})
    task["status"] = {"state": "TASK_STATE_CANCELED"}
    task["metadata"].pop("principal", None)
    row.state = "TASK_STATE_CANCELED"
    row.task_json = json.dumps(task, ensure_ascii=False)
    db.commit()
    return JSONResponse(task, media_type="application/a2a+json")

@app.get("/health")
async def health():
    return {"status": "ok"}
