import re
import secrets
from app.schemas import ACTIONS
from app.services.normalization import evidence_ref_is_bracketed

def _field(package: dict, *keys, default=""):
    for key in keys:
        if package.get(key) is not None:
            return package.get(key)
    return default

def validate_decisions(batch_id: str, packages: list[dict], decisions: list[dict]) -> list[dict]:
    if len(decisions) != len(packages):
        raise ValueError("Exactly one decision per package is required")
    package_ids = [str(_field(p, "packageId", "id")) for p in packages]
    if len(set(package_ids)) != len(package_ids) or "" in package_ids:
        raise ValueError("Every package needs a unique packageId")
    decision_map = {str(d.get("packageId", "")): d for d in decisions}
    if len(decision_map) != len(decisions) or set(decision_map) != set(package_ids):
        raise ValueError("AI output package IDs do not match input package IDs")

    result = []
    for package in packages:
        pid = str(_field(package, "packageId", "id"))
        d = decision_map[pid]
        action = str(d.get("action", ""))
        if action not in ACTIONS:
            raise ValueError(f"Invalid action for package {pid}")
        vendor = str(_field(d, "vendorName", "vendor_name", default=_field(package, "vendorName", "vendor", default="UNKNOWN")))
        invoice = str(_field(d, "invoiceNumber", "invoice_number", default=_field(package, "invoiceNumber", "invoice_no", default="UNKNOWN")))
        currency = str(_field(d, "currency", default=_field(package, "currency", default="INR")))
        amount = d.get("amountMinor", d.get("amount_minor", _field(package, "amountMinor", "amount_minor", default=0)))
        try:
            amount = int(amount)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid amountMinor for package {pid}")
        refs = d.get("evidenceRefs", d.get("evidence_refs", []))
        if not isinstance(refs, list) or len(refs) != 3 or not all(isinstance(r, str) and evidence_ref_is_bracketed(r) for r in refs):
            raise ValueError(f"Exactly three bracketed evidence references required for package {pid}")
        if len(set(refs)) != 3:
            raise ValueError(f"Evidence references must be distinct for package {pid}")
        rationale = str(d.get("rationale", "")).strip()
        if not 60 <= len(rationale) <= 1500:
            raise ValueError(f"Rationale must be 60-1500 characters for package {pid}")
        if action not in rationale:
            raise ValueError(f"Rationale must name action {action}")
        cited = sum(1 for ref in refs if ref in rationale)
        if cited < 2:
            raise ValueError(f"Rationale must cite at least two evidence references for package {pid}")
        result.append({
            "packageId": pid,
            "actionId": secrets.token_urlsafe(18),
            "action": action,
            "facts": {"vendorName": vendor, "invoiceNumber": invoice, "amountMinor": amount, "currency": currency},
            "evidenceRefs": refs,
            "rationale": rationale,
        })
    return result
