import json
from typing import Any
import httpx
from app.config import settings
from app.services.normalization import sha256_json

SYSTEM_PROMPT = """
You are an invoice-action decision assistant. You analyze ONLY the supplied synthetic invoice package evidence and policy.
Choose exactly one action per package:
- settle_invoice: valid, reconciled, and within autonomous authority.
- request_approval: commercially valid but outside delegated authority.
- hold_invoice: pause until a specific verification completes.
- reject_duplicate: the same commercial invoice was already paid; same vendor/amount alone is NOT a duplicate.
- open_exception: material records conflict and need an exception workflow.
Do not invent facts, policy rules, invoice numbers, amounts, or evidence references.
For every package return vendorName, invoiceNumber, amountMinor (integer minor currency units; INR rupees * 100), currency, exactly three decisive bracketed evidenceRefs copied from determining evidence paragraphs, and a rationale 60-1500 characters that names the action and cites at least two evidence references.
Return JSON only, with {"proposals":[{"packageId":"...","action":"...","vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR","evidenceRefs":["[ref]","[ref]","[ref]"],"rationale":"..."}]}.
""".strip()

def _package_prompt(batch_id: str, policy_revision: str, packages: list[dict]) -> str:
    return json.dumps({
        "batchId": batch_id,
        "policyRevision": policy_revision,
        "instructions": "Analyze each package independently. First identify the paragraph that controls the action; ignore cover sheets, archived examples, and training decoys. Copy exactly three decisive bracketed evidence references verbatim from that controlling evidence. Do not use references that are merely present but not decisive. Distinguish invoice identity from vendor/amount similarity, and apply the policy revision and delegated authority explicitly.",
        "packages": packages,
    }, ensure_ascii=False)

async def decide_batch(batch_id: str, policy_revision: str, packages: list[dict]) -> list[dict]:
    if not settings.ai_api_key:
        raise RuntimeError("AI_API_KEY is required for evidence-grounded invoice decisions")
    url = settings.ai_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.ai_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": settings.ai_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _package_prompt(batch_id, policy_revision, packages)},
        ],
    }
    async with httpx.AsyncClient(timeout=settings.ai_timeout_seconds) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    decisions = parsed.get("proposals")
    if not isinstance(decisions, list):
        raise ValueError("AI response must contain a proposals array")
    by_id = {str(x.get("packageId")): x for x in decisions if isinstance(x, dict) and x.get("packageId") is not None}
    result = []
    for package in packages:
        pid = str(package.get("packageId", package.get("id", "")))
        if pid not in by_id:
            raise ValueError(f"AI response omitted package {pid}")
        result.append(by_id[pid])
    return result

def canonical_package_hash(package: dict, policy_revision: str) -> str:
    return sha256_json({"package": package, "policyRevision": policy_revision})
