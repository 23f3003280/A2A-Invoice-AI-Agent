import asyncio
import json
import secrets
from datetime import datetime, timezone
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session
from app.models import TaskRecord, MessageReplay, SemanticDecision
from app.services.normalization import (
    safe_id, sha256_json, get_message_id, extract_data, extract_media_type,
    semantic_package,
)
from app.services.ai_service import decide_batch, canonical_package_hash
from app.services.validation import validate_decisions

PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
INPUT_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
TERMINAL_STATES = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED"}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _task_json(task_id, state, history, artifacts=None, context_id=None, metadata=None):
    return {
        "id": task_id,
        "contextId": context_id or task_id,
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


def _replay(db: Session, principal: str, message_id: str, digest: str):
    row = db.scalar(select(MessageReplay).where(
        MessageReplay.principal_id == principal,
        MessageReplay.message_id == message_id,
    ))
    if row:
        if row.semantic_hash != digest:
            return {"conflict": True, "task": None}
        return {"conflict": False, "task": json.loads(row.response_json), "replay": True}
    task_row = db.scalar(select(TaskRecord).where(
        TaskRecord.principal_id == principal,
        TaskRecord.message_id == message_id,
    ))
    if task_row:
        if task_row.semantic_hash != digest:
            return {"conflict": True, "task": None}
        return {"conflict": False, "task": json.loads(task_row.task_json), "replay": True, "working": task_row.state == "TASK_STATE_WORKING"}
    return None


async def _wait_for_initial_task(db: Session, principal: str, message_id: str, digest: str):
    # Equivalent concurrent requests should return the completed proposal task, not a transient WORKING snapshot.
    for _ in range(380):
        await asyncio.sleep(0.1)
        db.expire_all()
        result = _replay(db, principal, message_id, digest)
        if result and not result.get("working"):
            return result
    result = _replay(db, principal, message_id, digest)
    return result or {"conflict": False, "task": None}


def _decision_without_namespace(decision: dict) -> dict:
    return {k: v for k, v in decision.items() if k not in {"packageId", "package_id", "id", "actionId", "action_id"}}


async def _cached_decisions(db: Session, batch_id: str, policy_revision: str, packages: list[dict]) -> list[dict]:
    hashes = [canonical_package_hash(semantic_package(p), policy_revision) for p in packages]
    cached_by_hash = {}
    for digest in set(hashes):
        row = db.get(SemanticDecision, digest)
        if row:
            cached_by_hash[digest] = json.loads(row.decision_json)
    missing_packages, missing_hashes = [], []
    for package, digest in zip(packages, hashes):
        if digest not in cached_by_hash and digest not in missing_hashes:
            missing_packages.append(package)
            missing_hashes.append(digest)
    if missing_packages:
        fresh = await decide_batch(batch_id, policy_revision, missing_packages)
        by_id = {str(d.get("packageId")): d for d in fresh}
        for package, digest in zip(missing_packages, missing_hashes):
            pid = str(package.get("packageId", package.get("id", "")))
            if pid not in by_id:
                raise ValueError(f"AI response omitted package {pid}")
            clean = _decision_without_namespace(by_id[pid])
            cached_by_hash[digest] = clean
    output = []
    for package, digest in zip(packages, hashes):
        decision = dict(cached_by_hash[digest])
        decision["packageId"] = str(package.get("packageId", package.get("id", "")))
        output.append(decision)
    return output


def _cache_validated_decisions(db: Session, policy_revision: str, packages: list[dict], decisions: list[dict]):
    by_id = {str(d.get("packageId")): d for d in decisions}
    pending = set()
    for package in packages:
        pid = str(package.get("packageId", package.get("id", "")))
        digest = canonical_package_hash(semantic_package(package), policy_revision)
        if digest not in pending and db.get(SemanticDecision, digest) is None:
            decision = _decision_without_namespace(by_id[pid])
            db.add(SemanticDecision(content_hash=digest, decision_json=json.dumps(decision, ensure_ascii=False)))
            pending.add(digest)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()


async def create_or_replay_task(db: Session, principal: str, request_body: dict):
    message = request_body.get("message") if isinstance(request_body.get("message"), dict) else request_body
    if not isinstance(message, dict):
        raise ValueError("message must be an object")
    message_id = get_message_id(message)
    digest = sha256_json(message)  # Configuration is intentionally ignored.

    previous = _replay(db, principal, message_id, digest)
    if previous:
        if previous.get("conflict"):
            return previous
        if previous.get("working"):
            return await _wait_for_initial_task(db, principal, message_id, digest)
        return previous

    data = extract_data(message)
    media_type = extract_media_type(message)
    if "results" in data:
        if media_type != RESULTS_MEDIA:
            raise ValueError(f"Continuation media type must be {RESULTS_MEDIA}")
        return await continue_task(db, principal, data, message, message_id, digest)
    if media_type != INPUT_MEDIA:
        raise ValueError(f"Initial message media type must be {INPUT_MEDIA}")
    if message.get("role", "ROLE_USER") != "ROLE_USER":
        raise ValueError("Initial message role must be ROLE_USER")

    batch_id = str(data.get("batchId", ""))
    policy_revision = str(data.get("policyRevision", ""))
    packages = data.get("packages")
    if not batch_id or not isinstance(packages, list) or not packages:
        raise ValueError("Initial message data must include batchId and a non-empty packages array")
    if len(packages) > 100:
        raise ValueError("Too many packages in one batch")

    task_id = safe_id()
    context_id = str(message.get("contextId") or task_id)
    # Persist a durable in-progress Task before calling the AI provider.
    task = _task_json(task_id, "TASK_STATE_WORKING", [_user_history(message)], context_id=context_id,
                      metadata={"batchId": batch_id})
    record = TaskRecord(
        id=task_id, principal_id=principal, message_id=message_id,
        semantic_hash=digest, batch_id=batch_id, state="TASK_STATE_WORKING",
        task_json=json.dumps(task, ensure_ascii=False), created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        previous = _replay(db, principal, message_id, digest)
        if previous:
            if previous.get("conflict"):
                return previous
            if previous.get("working"):
                return await _wait_for_initial_task(db, principal, message_id, digest)
            return previous
        raise

    try:
        decisions = await _cached_decisions(db, batch_id, policy_revision, packages)
        proposals = validate_decisions(batch_id, packages, decisions)
        _cache_validated_decisions(db, policy_revision, packages, decisions)
        task["status"] = {"state": "TASK_STATE_INPUT_REQUIRED"}
        task["artifacts"] = [{"parts": [_part(PROPOSAL_MEDIA, {"batchId": batch_id, "proposals": proposals})]}]
        task["metadata"]["proposalCount"] = len(proposals)
        new_state = "TASK_STATE_INPUT_REQUIRED"
    except Exception as exc:
        task["status"] = {"state": "TASK_STATE_FAILED", "message": {"message": "Unable to process invoice batch"}}
        task["metadata"]["failureType"] = type(exc).__name__
        new_state = "TASK_STATE_FAILED"

    # Do not overwrite a cancellation that won while the AI call was running.
    changed = db.execute(update(TaskRecord).where(
        TaskRecord.id == task_id,
        TaskRecord.principal_id == principal,
        TaskRecord.state == "TASK_STATE_WORKING",
    ).values(state=new_state, task_json=json.dumps(task, ensure_ascii=False), updated_at=datetime.now(timezone.utc)))
    db.commit()
    db.expire_all()
    saved = db.get(TaskRecord, task_id)
    if saved:
        return {"conflict": False, "task": json.loads(saved.task_json)}
    return {"conflict": False, "task": task}


async def continue_task(db: Session, principal: str, data: dict, message: dict, message_id: str, digest: str):
    batch_id = str(data.get("batchId", ""))
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Continuation must contain results array")
    task_id = str(message.get("taskId") or message.get("task_id") or "")
    context_id = str(message.get("contextId") or "")
    if not task_id or not context_id:
        raise ValueError("Continuation must include exact taskId and contextId")
    if message.get("role") != "ROLE_USER":
        raise ValueError("Continuation role must be ROLE_USER")
    record = db.get(TaskRecord, task_id)
    if not record or record.principal_id != principal:
        raise PermissionError("Task not found")
    task = json.loads(record.task_json)
    if context_id != task.get("contextId"):
        raise PermissionError("Task not found")
    if task.get("status", {}).get("state") != "TASK_STATE_INPUT_REQUIRED":
        raise RuntimeError("Task is not awaiting input")
    if batch_id != record.batch_id or batch_id != task.get("metadata", {}).get("batchId"):
        raise ValueError("Continuation batchId does not match")

    proposal_artifact = next((a for a in task.get("artifacts", []) for p in a.get("parts", [])
                              if p.get("mediaType") == PROPOSAL_MEDIA), None)
    if not proposal_artifact:
        raise ValueError("Stored proposal artifact missing")
    proposals = proposal_artifact["parts"][0]["data"].get("proposals", [])
    by_package = {p["packageId"]: p for p in proposals}
    seen, executions = set(), []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Each result must be an object")
        pid = str(result.get("packageId", ""))
        if not pid or pid in seen:
            raise ValueError("Missing or duplicate package result")
        seen.add(pid)
        proposal = by_package.get(pid)
        if not proposal:
            raise ValueError("Unknown packageId")
        if result.get("actionId") != proposal["actionId"] or result.get("action") != proposal["action"]:
            raise ValueError("Result action identity does not match stored proposal")
        if result.get("outcome") not in {"ACCEPTED", "REJECTED"}:
            raise ValueError("Invalid outcome")
        nonce = result.get("receiptNonce")
        if not isinstance(nonce, str) or not nonce:
            raise ValueError("receiptNonce is required")
        if result["outcome"] == "ACCEPTED":
            executions.append({
                "packageId": pid,
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "receiptNonce": nonce,
                "facts": proposal["facts"],
                "evidenceRefs": proposal["evidenceRefs"],
            })
    if len(seen) != len(proposals) or set(seen) != set(by_package):
        raise ValueError("Continuation must include exactly one result per proposal")

    task["history"].append(_user_history(message))
    task["status"] = {"state": "TASK_STATE_COMPLETED"}
    task["artifacts"].append({"parts": [_part(RECEIPT_MEDIA, {"batchId": batch_id, "executions": executions})]})
    task_json = json.dumps(task, ensure_ascii=False)
    # The task state update is a compare-and-swap. Cancellation and completion cannot both win.
    try:
        db.add(MessageReplay(
            principal_id=principal, message_id=message_id, semantic_hash=digest,
            task_id=task_id, response_json=task_json,
        ))
        changed = db.execute(update(TaskRecord).where(
            TaskRecord.id == task_id,
            TaskRecord.principal_id == principal,
            TaskRecord.state == "TASK_STATE_INPUT_REQUIRED",
        ).values(state="TASK_STATE_COMPLETED", task_json=task_json, updated_at=datetime.now(timezone.utc)))
        if changed.rowcount != 1:
            db.rollback()
            prior = _replay(db, principal, message_id, digest)
            if prior and not prior.get("conflict"):
                return prior
            raise RuntimeError("Task is no longer awaiting input")
        db.commit()
    except IntegrityError:
        db.rollback()
        prior = _replay(db, principal, message_id, digest)
        if prior:
            if prior.get("conflict"):
                return prior
            return prior
        raise RuntimeError("Task is no longer awaiting input")
    return {"conflict": False, "task": task}
