import json
import secrets
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models import TaskRecord
from app.services.normalization import safe_id, sha256_json, get_message_id, extract_data, extract_media_type
from app.services.ai_service import decide_batch
from app.services.validation import validate_decisions

PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
INPUT_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def _task_json(task_id, state, history, artifacts=None, metadata=None):
    return {
        "id": task_id,
        "contextId": (metadata or {}).get("contextId", task_id),
        "status": {"state": state},
        "history": history,
        "artifacts": artifacts or [],
        "metadata": metadata or {},
    }

def _user_history(message: dict) -> dict:
    result = dict(message)
    result.setdefault("role", "ROLE_USER")
    return result

def _part(media_type: str, data: dict) -> dict:
    return {"mediaType": media_type, "data": data}

async def create_or_replay_task(db: Session, principal: str, request_body: dict):
    message = request_body.get("message") if isinstance(request_body.get("message"), dict) else request_body
    if not isinstance(message, dict):
        raise ValueError("message must be an object")
    message_id = get_message_id(message)
    digest = sha256_json(message)  # configuration intentionally excluded
    existing = db.scalar(select(TaskRecord).where(
        TaskRecord.principal_id == principal,
        TaskRecord.message_id == message_id
    ))
    if existing:
        if existing.semantic_hash != digest:
            return {"conflict": True, "task": None}
        return {"conflict": False, "task": json.loads(existing.task_json)}

    data = extract_data(message)
    media_type = extract_media_type(message)
    if not data and isinstance(message.get("data"), dict):
        data = message["data"]
    if "results" in data:
        return await continue_task(db, principal, data, message)
    if media_type and media_type != INPUT_MEDIA:
        raise ValueError(f"Expected input media type {INPUT_MEDIA}")
    batch_id = str(data.get("batchId", ""))
    policy_revision = str(data.get("policyRevision", ""))
    packages = data.get("packages")
    if not batch_id or not isinstance(packages, list) or not packages:
        raise ValueError("Initial message data must include batchId and a non-empty packages array")
    if len(packages) > 100:
        raise ValueError("Too many packages in one batch")

    task_id = safe_id()
    context_id = str(message.get("contextId") or task_id)
    task = _task_json(task_id, "TASK_STATE_WORKING", [_user_history(message)], metadata={
        "contextId": context_id, "batchId": batch_id, "principal": principal
    })
    record = TaskRecord(
        id=task_id, principal_id=principal, message_id=message_id,
        semantic_hash=digest, batch_id=batch_id, state="TASK_STATE_WORKING",
        task_json=json.dumps(task, ensure_ascii=False), created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db.add(record)
    try:
        db.commit()
    except Exception:
        db.rollback()
        existing = db.scalar(select(TaskRecord).where(
            TaskRecord.principal_id == principal, TaskRecord.message_id == message_id
        ))
        if existing:
            if existing.semantic_hash != digest:
                return {"conflict": True, "task": None}
            return {"conflict": False, "task": json.loads(existing.task_json)}
        raise

    try:
        decisions = await decide_batch(batch_id, policy_revision, packages)
        proposals = validate_decisions(batch_id, packages, decisions)
        task["status"] = {"state": "TASK_STATE_INPUT_REQUIRED"}
        task["artifacts"] = [{"parts": [_part(PROPOSAL_MEDIA, {"batchId": batch_id, "proposals": proposals})]}]
        task["metadata"].pop("principal", None)
        task["metadata"]["proposalCount"] = len(proposals)
        record.state = "TASK_STATE_INPUT_REQUIRED"
    except Exception as exc:
        task["status"] = {"state": "TASK_STATE_FAILED", "message": {"message": "Unable to process invoice batch"}}
        task["metadata"].pop("principal", None)
        task["metadata"]["internalError"] = type(exc).__name__
        record.state = "TASK_STATE_FAILED"
    record.task_json = json.dumps(task, ensure_ascii=False)
    record.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"conflict": False, "task": task}

async def continue_task(db: Session, principal: str, data: dict, message: dict):
    batch_id = str(data.get("batchId", ""))
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Continuation must contain results array")
    # Continuation identifies the task through contextId or taskId in message.
    task_id = str(message.get("taskId") or message.get("task_id") or message.get("contextId") or "")
    if not task_id:
        raise ValueError("Continuation message must include taskId or contextId")
    record = db.get(TaskRecord, task_id)
    if not record or record.principal_id != principal:
        raise PermissionError("Task not found")
    task = json.loads(record.task_json)
    state = task.get("status", {}).get("state")
    if state != "TASK_STATE_INPUT_REQUIRED":
        raise RuntimeError("Task is not awaiting input")
    if batch_id != record.batch_id or batch_id != task.get("metadata", {}).get("batchId"):
        raise ValueError("Continuation batchId does not match")
    proposal_artifact = next((a for a in task.get("artifacts", []) for p in a.get("parts", []) if p.get("mediaType") == PROPOSAL_MEDIA), None)
    if not proposal_artifact:
        raise ValueError("Stored proposal artifact missing")
    proposals = proposal_artifact["parts"][0]["data"]["proposals"]
    by_package = {p["packageId"]: p for p in proposals}
    seen = set()
    executions = []
    for result in results:
        pid = str(result.get("packageId", ""))
        if pid in seen:
            raise ValueError("Duplicate package result")
        seen.add(pid)
        proposal = by_package.get(pid)
        if not proposal:
            raise ValueError("Unknown packageId")
        if result.get("actionId") != proposal["actionId"] or result.get("action") != proposal["action"]:
            raise ValueError("Result action identity does not match stored proposal")
        if result.get("outcome") not in {"ACCEPTED", "REJECTED"}:
            raise ValueError("Invalid outcome")
        if not isinstance(result.get("receiptNonce"), str) or not result["receiptNonce"]:
            raise ValueError("receiptNonce is required")
        if result["outcome"] == "ACCEPTED":
            executions.append({
                "packageId": pid, "actionId": proposal["actionId"], "action": proposal["action"],
                "outcome": "ACCEPTED", "receiptNonce": result["receiptNonce"],
                "facts": proposal["facts"], "evidenceRefs": proposal["evidenceRefs"]
            })
    if len(seen) != len(proposals):
        raise ValueError("Continuation must include one result per proposal")
    task["history"].append(_user_history(message))
    task["status"] = {"state": "TASK_STATE_COMPLETED"}
    task["artifacts"].append({"parts": [_part(RECEIPT_MEDIA, {"batchId": batch_id, "executions": executions})]})
    record.state = "TASK_STATE_COMPLETED"
    record.task_json = json.dumps(task, ensure_ascii=False)
    record.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"conflict": False, "task": task}
