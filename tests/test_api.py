import json

INPUT_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"


def headers(token="test-token"):
    return {
        "Authorization": f"Bearer {token}",
        "A2A-Version": "1.0",
        "Content-Type": "application/a2a+json",
    }


def _package(pid="pkg-1"):
    return {
        "packageId": pid,
        "vendorName": "Example Vendor",
        "invoiceNumber": "INV-1",
        "amountMinor": 10000,
        "currency": "INR",
        "documents": [
            "[invoice paragraph] Invoice INV-1 is for INR 100.00.",
            "[purchase order paragraph] PO-1 confirms the invoice line items.",
            "[payment ledger paragraph] No previous payment for INV-1 is recorded."
        ]
    }


def _fake_decision(package):
    refs = ["[invoice paragraph]", "[purchase order paragraph]", "[payment ledger paragraph]"]
    return {
        "packageId": package["packageId"],
        "action": "settle_invoice",
        "vendorName": package["vendorName"],
        "invoiceNumber": package["invoiceNumber"],
        "amountMinor": package["amountMinor"],
        "currency": package["currency"],
        "evidenceRefs": refs,
        "rationale": f"settle_invoice is supported because the invoice matches the purchase order and no prior payment is recorded. Evidence: {refs[0]} and {refs[1]}.",
    }


def _initial_payload(message_id="m-1", package=None):
    return {
        "message": {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [{"mediaType": INPUT_MEDIA, "data": {
                "batchId": "b-1", "policyRevision": "p-1", "packages": [package or _package()]
            }}]
        },
        "configuration": {"returnImmediately": False, "acceptedOutputModes": [PROPOSAL_MEDIA, RECEIPT_MEDIA]}
    }


def test_agent_card_public(client):
    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.json()
    assert card["name"]
    assert card["skills"][0]["id"] == "invoice_action_agent"
    assert card["supportedInterfaces"][0]["protocolBinding"] == "HTTP+JSON"
    assert card["supportedInterfaces"][0]["protocolVersion"] == "1.0"
    assert INPUT_MEDIA in card["defaultInputModes"]
    assert PROPOSAL_MEDIA in card["defaultOutputModes"]
    assert RECEIPT_MEDIA in card["defaultOutputModes"]
    assert "defaultInputModes" not in card["supportedInterfaces"][0]


def test_auth_required(client):
    response = client.get("/a2a/tasks")
    assert response.status_code == 401


def test_version_and_content_type(client):
    h = headers()
    h["A2A-Version"] = "2.0"
    assert client.get("/a2a/tasks", headers=h).status_code == 400
    h = headers()
    h["Content-Type"] = "application/json"
    assert client.post("/a2a/message:send", headers=h, json={}).status_code == 400


def test_initial_task_and_receipt_lifecycle(client, monkeypatch):
    import app.services.task_service as service
    async def fake_decide(batch_id, policy_revision, packages):
        return [_fake_decision(p) for p in packages]
    monkeypatch.setattr(service, "decide_batch", fake_decide)
    response = client.post("/a2a/message:send", headers=headers(), json=_initial_payload())
    assert response.status_code == 200, response.text
    task = response.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert task["history"][0]["messageId"] == "m-1"
    assert len(task["artifacts"]) == 1
    proposal_part = task["artifacts"][0]["parts"][0]
    assert proposal_part["mediaType"] == PROPOSAL_MEDIA
    proposal = proposal_part["data"]["proposals"][0]
    continuation = {
        "message": {
            "messageId": "m-2", "taskId": task["id"], "contextId": task["contextId"], "role": "ROLE_USER",
            "parts": [{"mediaType": RESULTS_MEDIA, "data": {"batchId": "b-1", "results": [{
                "packageId": proposal["packageId"], "actionId": proposal["actionId"],
                "action": proposal["action"], "outcome": "ACCEPTED", "receiptNonce": "nonce-unpredictable-123"
            }]}}]
        }
    }
    continued = client.post("/a2a/message:send", headers=headers(), json=continuation)
    assert continued.status_code == 200, continued.text
    final_task = continued.json()["task"]
    assert final_task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert len(final_task["artifacts"]) == 2
    receipt = final_task["artifacts"][1]["parts"][0]
    assert receipt["mediaType"] == RECEIPT_MEDIA
    execution = receipt["data"]["executions"][0]
    assert execution["receiptNonce"] == "nonce-unpredictable-123"
    assert "outcome" not in execution
    # Same continuation ID and same message must replay the complete task.
    replay = client.post("/a2a/message:send", headers=headers(), json=continuation)
    assert replay.status_code == 200
    assert replay.json()["task"] == final_task


def test_idempotency_conflict_and_configuration_ignored(client, monkeypatch):
    import app.services.task_service as service
    async def fake_decide(batch_id, policy_revision, packages):
        return [_fake_decision(p) for p in packages]
    monkeypatch.setattr(service, "decide_batch", fake_decide)
    payload = _initial_payload()
    first = client.post("/a2a/message:send", headers=headers(), json=payload)
    assert first.status_code == 200
    payload["configuration"]["returnImmediately"] = True
    replay = client.post("/a2a/message:send", headers=headers(), json=payload)
    assert replay.status_code == 200
    assert replay.json()["task"] == first.json()["task"]
    changed = _initial_payload(package=_package("pkg-different"))
    conflict = client.post("/a2a/message:send", headers=headers(), json=changed)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_task_isolation(client, monkeypatch):
    import app.services.task_service as service
    async def fake_decide(batch_id, policy_revision, packages):
        return [_fake_decision(p) for p in packages]
    monkeypatch.setattr(service, "decide_batch", fake_decide)
    response = client.post("/a2a/message:send", headers=headers("owner-token"), json=_initial_payload())
    task_id = response.json()["task"]["id"]
    outsider = client.get(f"/a2a/tasks/{task_id}", headers=headers("outsider-token"))
    assert outsider.status_code == 404
    outsider_list = client.get("/a2a/tasks", headers=headers("outsider-token"))
    assert outsider_list.status_code == 200
    assert outsider_list.json() == {"tasks": []}


def test_cancel_terminal_conflict(client, monkeypatch):
    import app.services.task_service as service
    async def fake_decide(batch_id, policy_revision, packages):
        return [_fake_decision(p) for p in packages]
    monkeypatch.setattr(service, "decide_batch", fake_decide)
    response = client.post("/a2a/message:send", headers=headers(), json=_initial_payload())
    task_id = response.json()["task"]["id"]
    cancelled = client.post(f"/a2a/tasks/{task_id}:cancel", headers=headers(), json={})
    assert cancelled.status_code == 200
    assert cancelled.json()["status"]["state"] == "TASK_STATE_CANCELED"
    again = client.post(f"/a2a/tasks/{task_id}:cancel", headers=headers(), json={})
    assert again.status_code == 409
