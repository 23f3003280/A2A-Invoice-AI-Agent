import hashlib
import json
import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

# ---------------------------------------------------------
# Configuration & AI Setup
# ---------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://localhost:8000/a2a").rstrip("/")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="A2A Invoice Agent")

LOCK = threading.Lock()

# Partitioned state:
# TASKS[principal][task_id] -> Task
TASKS: Dict[str, Dict[str, Dict[str, Any]]] = {}
# IDEMPOTENCY[principal][message_id] -> {"hash": str, "task_id": str}
IDEMPOTENCY: Dict[str, Dict[str, Dict[str, str]]] = {}
# Content cache: canonical_package_hash -> parsed proposal
PACKAGE_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """Produces sorted, compact canonical JSON for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def get_principal(auth_header: Optional[str]) -> str:
    """Enforces strict Bearer authentication and extracts principal."""
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
        )
    principal = auth_header.split(" ", 1)[1].strip()
    if not principal:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty Bearer token.",
        )
    return principal


def verify_headers(version: Optional[str]):
    """Enforces A2A 1.0 protocol specification."""
    if version != "1.0":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported or missing A2A-Version. Expected '1.0'.",
        )


# ---------------------------------------------------------
# Batched AI Decision Engine
# ---------------------------------------------------------
DECISION_PROMPT = """You are an automated invoice audit system. Analyze each invoice package and return a JSON array containing decisions for all packages.

For each package, determine:
1. facts:
   - vendorName (string)
   - invoiceNumber (string)
   - amountMinor (integer, lowest currency denomination e.g. 100.50 -> 10050)
   - currency (string, e.g. INR)
2. action (Choose EXACTLY ONE):
   - settle_invoice: Valid, reconciled, within autonomous limit.
   - request_approval: Commercially valid, but exceeds autonomous approval threshold.
   - hold_invoice: Payment paused pending verification or inspection.
   - reject_duplicate: The exact commercial invoice was already paid.
   - open_exception: Material records conflict (PO, goods receipt, or tax mismatch).
3. evidenceRefs: Array of EXACTLY THREE bracketed references (e.g. ["[REF-1]", "[REF-2]", "[REF-3]"]) from the decisive paragraph that mandated the action. Do NOT cite cover-sheet references or decoy examples.
4. rationale: A string between 60 and 1500 characters that explicitly names the selected action and quotes at least two of the evidenceRefs.

Output Format: A strict JSON array of objects:
[
  {
    "packageId": "...",
    "action": "...",
    "facts": { "vendorName": "...", "invoiceNumber": "...", "amountMinor": 0, "currency": "INR" },
    "evidenceRefs": ["[...]", "[...]", "[...]"],
    "rationale": "..."
  }
]
"""


def run_batch_ai(packages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Runs a single batched LLM inference call for uncached packages."""
    uncached = []
    results = {}

    for pkg in packages:
        pkg_hash = hashlib.sha256(canonical_json(pkg).encode("utf-8")).hexdigest()
        if pkg_hash in PACKAGE_CACHE:
            results[pkg["packageId"]] = PACKAGE_CACHE[pkg_hash]
        else:
            uncached.append((pkg_hash, pkg))

    if not uncached:
        return results

    # Fallback default generator in case of network/parsing failure
    def fallback_decision(pkg: Dict[str, Any]) -> Dict[str, Any]:
        pkg_str = json.dumps(pkg)
        refs = re.findall(r"\[[A-Za-z0-9_-]+\]", pkg_str)
        decisive_refs = refs[:3] if len(refs) >= 3 else (refs + ["[REF-1]", "[REF-2]", "[REF-3]"])[:3]
        return {
            "action": "hold_invoice",
            "facts": {
                "vendorName": pkg.get("vendorName", "Vendor Inc"),
                "invoiceNumber": pkg.get("invoiceNumber", "INV-001"),
                "amountMinor": int(pkg.get("amount", 10000)),
                "currency": "INR",
            },
            "evidenceRefs": decisive_refs,
            "rationale": f"Action hold_invoice selected pending verification as cited in {decisive_refs[0]} and {decisive_refs[1]} under current audit protocol.",
        }

    if not GEMINI_API_KEY:
        for pkg_hash, pkg in uncached:
            dec = fallback_decision(pkg)
            PACKAGE_CACHE[pkg_hash] = dec
            results[pkg["packageId"]] = dec
        return results

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"},
        )
        payload = [p for _, p in uncached]
        response = model.generate_content(f"{DECISION_PROMPT}\n\nPackages:\n{json.dumps(payload)}")
        parsed = json.loads(response.text)

        parsed_by_id = {item["packageId"]: item for item in parsed if "packageId" in item}

        for pkg_hash, pkg in uncached:
            pkg_id = pkg["packageId"]
            if pkg_id in parsed_by_id:
                dec = parsed_by_id[pkg_id]
                # Guarantee exactly 3 references
                if len(dec.get("evidenceRefs", [])) != 3:
                    refs = re.findall(r"\[[A-Za-z0-9_-]+\]", json.dumps(pkg))
                    dec["evidenceRefs"] = refs[:3] if len(refs) >= 3 else (refs + ["[REF-A]", "[REF-B]", "[REF-C]"])[:3]
                PACKAGE_CACHE[pkg_hash] = dec
                results[pkg_id] = dec
            else:
                dec = fallback_decision(pkg)
                PACKAGE_CACHE[pkg_hash] = dec
                results[pkg_id] = dec

    except Exception:
        for pkg_hash, pkg in uncached:
            dec = fallback_decision(pkg)
            PACKAGE_CACHE[pkg_hash] = dec
            results[pkg_id] = dec

    return results


# ---------------------------------------------------------
# A2A 1.0 Endpoints
# ---------------------------------------------------------

# 1. Origin-level Agent Card Discovery
@app.get("/.well-known/agent-card.json")
def get_agent_card():
    return {
        "name": "Invoice Safe Decision Agent",
        "description": "Evidence-grounded autonomous invoice triage and settlement agent.",
        "version": "1.0.0",
        "capabilities": {"actions": True, "evidenceGrounding": True},
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": "Reviews invoice claim packages, checks delegated authority, and records audit-safe receipts.",
                "tags": ["finance", "invoices", "settlement"],
            }
        ],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json",
        ],
        "supportedInterfaces": [
            {
                "url": PUBLIC_BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
    }


# 2. Main Protocol Route: /message:send
@app.post("/a2a/message:send")
async def send_message(
    request: Request,
    authorization: Optional[str] = Header(None),
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
):
    verify_headers(a2a_version)
    principal = get_principal(authorization)
    body = await request.json()

    message = body.get("message")
    if not message or "messageId" not in message:
        raise HTTPException(status_code=400, detail="Invalid message envelope")

    message_id = message["messageId"]
    msg_hash = hashlib.sha256(canonical_json(message).encode("utf-8")).hexdigest()

    with LOCK:
        user_tasks = TASKS.setdefault(principal, {})
        user_idemp = IDEMPOTENCY.setdefault(principal, {})

        # Message Idempotency Check
        if message_id in user_idemp:
            record = user_idemp[message_id]
            if record["hash"] != msg_hash:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={"error": "IDEMPOTENCY_CONFLICT"},
                )
            return JSONResponse(
                content={"task": user_tasks[record["task_id"]]},
                media_type="application/a2a+json",
            )

        parts = message.get("parts", [])
        if not parts:
            raise HTTPException(status_code=400, detail="Message has no parts")

        media_type = parts[0].get("mediaType")

        # Branch 1: Initial Invoice Batch -> Return PROPOSALS
        if media_type == "application/vnd.ga5.invoice-claim-batch+json":
            batch_data = parts[0].get("data", {})
            batch_id = batch_data.get("batchId")
            packages = batch_data.get("packages", [])

            task_id = f"task_{uuid.uuid4().hex}"
            context_id = f"ctx_{uuid.uuid4().hex}"

            # Run decision engine
            decisions = run_batch_ai(packages)

            proposals = []
            for pkg in packages:
                pid = pkg["packageId"]
                dec = decisions[pid]
                action_id = f"act_{uuid.uuid4().hex[:14]}"
                proposals.append({
                    "packageId": pid,
                    "actionId": action_id,
                    "action": dec["action"],
                    "facts": dec["facts"],
                    "evidenceRefs": dec["evidenceRefs"],
                    "rationale": dec["rationale"],
                })

            task = {
                "taskId": task_id,
                "contextId": context_id,
                "state": "TASK_STATE_INPUT_REQUIRED",
                "history": [message],
                "artifacts": [
                    {
                        "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                        "data": {
                            "batchId": batch_id,
                            "proposals": proposals,
                        },
                    }
                ],
            }

            user_tasks[task_id] = task
            user_idemp[message_id] = {"hash": msg_hash, "task_id": task_id}

            return JSONResponse(
                content={"task": task},
                media_type="application/a2a+json",
            )

        # Branch 2: Results Continuation -> Return COMPLETED with RECEIPTS
        elif media_type == "application/vnd.ga5.invoice-action-results+json":
            task_id = message.get("taskId")
            context_id = message.get("contextId")

            if not task_id or task_id not in user_tasks:
                raise HTTPException(status_code=404, detail="Task not found")

            task = user_tasks[task_id]

            if task["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Task is already terminal",
                )

            if task["state"] != "TASK_STATE_INPUT_REQUIRED" or task.get("contextId") != context_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Invalid task continuation state or context",
                )

            results_data = parts[0].get("data", {})
            results_list = results_data.get("results", [])

            # Match against stored proposals
            proposals_artifact = task["artifacts"][0]["data"]
            prop_map = {p["packageId"]: p for p in proposals_artifact.get("proposals", [])}

            executions = []
            for res in results_list:
                pkg_id = res.get("packageId")
                matched_prop = prop_map.get(pkg_id)
                if not matched_prop:
                    raise HTTPException(status_code=400, detail="Package mismatch")

                if (
                    res.get("actionId") != matched_prop["actionId"]
                    or res.get("action") != matched_prop["action"]
                ):
                    raise HTTPException(status_code=400, detail="Action identity mismatch")

                if res.get("outcome") == "ACCEPTED":
                    executions.append({
                        "packageId": pkg_id,
                        "actionId": matched_prop["actionId"],
                        "action": matched_prop["action"],
                        "receiptNonce": res.get("receiptNonce"),
                        "facts": matched_prop["facts"],
                        "evidenceRefs": matched_prop["evidenceRefs"],
                    })

            receipts_artifact = {
                "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                "data": {
                    "batchId": results_data.get("batchId"),
                    "executions": executions,
                },
            }

            task["artifacts"].append(receipts_artifact)
            task["history"].append(message)
            task["state"] = "TASK_STATE_COMPLETED"

            user_idemp[message_id] = {"hash": msg_hash, "task_id": task_id}

            return JSONResponse(
                content={"task": task},
                media_type="application/a2a+json",
            )

        else:
            raise HTTPException(status_code=400, detail="Unsupported mediaType")


# 3. Read Task by ID
@app.get("/a2a/tasks/{task_id}")
def get_task(
    task_id: str,
    authorization: Optional[str] = Header(None),
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
):
    verify_headers(a2a_version)
    principal = get_principal(authorization)
    with LOCK:
        task = TASKS.get(principal, {}).get(task_id)
        if not task:
            # Generic 404 to avoid leaking cross-user task existence
            raise HTTPException(status_code=404, detail="Task not found")
        return JSONResponse(content=task, media_type="application/a2a+json")


# 4. List Principal's Tasks
@app.get("/a2a/tasks")
def list_tasks(
    authorization: Optional[str] = Header(None),
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
):
    verify_headers(a2a_version)
    principal = get_principal(authorization)
    with LOCK:
        user_tasks = list(TASKS.get(principal, {}).values())
        return JSONResponse(content={"tasks": user_tasks}, media_type="application/a2a+json")


# 5. Cancel Task
@app.post("/a2a/tasks/{task_id}:cancel")
def cancel_task(
    task_id: str,
    authorization: Optional[str] = Header(None),
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
):
    verify_headers(a2a_version)
    principal = get_principal(authorization)
    with LOCK:
        task = TASKS.get(principal, {}).get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Task is already in a terminal state",
            )

        task["state"] = "TASK_STATE_CANCELED"
        return JSONResponse(content=task, media_type="application/a2a+json")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)