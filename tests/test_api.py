import json

def headers(token="test-token"):
    return {
        "Authorization": f"Bearer {token}",
        "A2A-Version": "1.0",
        "Content-Type": "application/a2a+json",
    }

def test_agent_card_public(client):
    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.json()
    assert card["name"]
    assert card["skills"][0]["id"] == "invoice_action_agent"

def test_auth_required(client):
    response = client.get("/a2a/tasks")
    assert response.status_code == 401

def test_initial_task_demo_mode(client, monkeypatch):
    import app.services.ai_service as ai
    async def fake_decide(batch_id, policy_revision, packages):
        out = []
        for p in packages:
            pid = p["packageId"]
            refs = ["[invoice paragraph]", "[purchase order paragraph]", "[payment ledger paragraph]"]
            out.append({
                "packageId": pid, "action": "hold_invoice",
                "vendorName": "Example Vendor", "invoiceNumber": "INV-1",
                "amountMinor": 10000, "currency": "INR", "evidenceRefs": refs,
                "rationale": f"hold_invoice pending verification because the source records need review. Evidence: {refs[0]} and {refs[1]}."
            })
        return out
    monkeypatch.setattr(ai, "decide_batch", fake_decide)
    payload = {
        "message": {
            "messageId": "m-1", "role": "ROLE_USER",
            "parts": [{"mediaType": "application/vnd.ga5.invoice-claim-batch+json",
                       "data": {"batchId": "b-1", "policyRevision": "p-1",
                                "packages": [{"packageId": "pkg-1", "vendorName": "Example Vendor",
                                              "invoiceNumber": "INV-1", "amountMinor": 10000, "currency": "INR"}]}}]
        },
        "configuration": {"acceptedOutputModes": ["application/vnd.ga5.invoice-action-proposals+json"]}
    }
    response = client.post("/a2a/message:send", headers=headers(), json=payload)
    assert response.status_code == 200, response.text
    task = response.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert len(task["artifacts"][0]["parts"][0]["data"]["proposals"]) == 1

def test_task_isolation(client):
    response = client.get("/a2a/tasks", headers={"Authorization": "Bearer one", "A2A-Version": "1.0"})
    assert response.status_code == 200
    assert response.json() == {"tasks": []}
