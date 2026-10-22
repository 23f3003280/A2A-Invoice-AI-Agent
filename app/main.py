import json
from datetime import datetime, timezone
from urllib.parse import urlparse
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from app.config import settings
from app.database import Base, engine, get_db
from app.models import TaskRecord
from app.security import authenticated_principal, require_a2a_content_type
from app.services.task_service import create_or_replay_task

Base.metadata.create_all(bind=engine)
app = FastAPI(title="A2A Invoice Action Agent", version=settings.app_version)


def _validate_public_base_url():
    parsed = urlparse(settings.public_base_url)
    if (parsed.scheme != "https" and not (parsed.hostname in {"127.0.0.1", "localhost"} and parsed.scheme == "http")) or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise RuntimeError("PUBLIC_BASE_URL must be the public HTTPS origin without path, credentials, query, or fragment")


@app.get("/.well-known/agent-card.json")
async def agent_card():
    _validate_public_base_url()
    base = settings.public_base_url.rstrip("/") + "/a2a/"
    return JSONResponse({
        "name": "Invoice Action Agent",
        "description": "Analyzes invoice claim batches and proposes evidence-grounded invoice actions. Execution receipts require an accepted result continuation.",
        "version": settings.app_version,
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "supportedInterfaces": [{
            "url": base,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Analyzes invoice evidence and delegated authority to propose settle, approval, hold, duplicate rejection, or exception actions.",
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
                "error": {"code": "IDEMPOTENCY_CONFLICT", "message": "Message ID was already used with different content."}
            })
        if not result.get("task"):
            raise HTTPException(status_code=503, detail={"code": "TASK_PROCESSING", "message": "Task is still processing; retry the same message ID."})
        return JSONResponse({"task": result["task"]}, media_type="application/a2a+json")
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(status_code=404, detail={"code": "TASK_NOT_FOUND", "message": "Task not found"})
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail={"code": "INVALID_TASK_STATE", "message": str(exc)})
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_REQUEST", "message": str(exc)})
    except Exception:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": "Request could not be processed"})


@app.get("/a2a/tasks")
async def list_tasks(principal: str = Depends(authenticated_principal), db: Session = Depends(get_db)):
    rows = db.scalars(select(TaskRecord).where(TaskRecord.principal_id == principal).order_by(TaskRecord.created_at.desc())).all()
    return JSONResponse({"tasks": [json.loads(row.task_json) for row in rows]}, media_type="application/a2a+json")


@app.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, principal: str = Depends(authenticated_principal), db: Session = Depends(get_db)):
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
    task["artifacts"] = [a for a in task.get("artifacts", []) if not any(
        p.get("mediaType") == "application/vnd.ga5.invoice-action-receipts+json" for p in a.get("parts", [])
    )]
    task_json = json.dumps(task, ensure_ascii=False)
    changed = db.execute(update(TaskRecord).where(
        TaskRecord.id == task_id,
        TaskRecord.principal_id == principal,
        TaskRecord.state.in_(["TASK_STATE_WORKING", "TASK_STATE_INPUT_REQUIRED"]),
    ).values(state="TASK_STATE_CANCELED", task_json=task_json, updated_at=datetime.now(timezone.utc)))
    if changed.rowcount != 1:
        db.rollback()
        raise HTTPException(status_code=409, detail={"code": "TASK_STATE_CONFLICT", "message": "Task changed concurrently; retry by reading the task."})
    db.commit()
    return JSONResponse(task, media_type="application/a2a+json")


@app.get("/health")
async def health():
    return {"status": "ok"}
