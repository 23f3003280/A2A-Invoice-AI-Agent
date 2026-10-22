from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict

ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
OUTCOMES = {"ACCEPTED", "REJECTED"}

class A2AMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    messageId: str | None = None
    message_id: str | None = None
    role: str | None = None
    parts: list[dict[str, Any]] | None = None
    content: Any | None = None
    data: dict[str, Any] | None = None

class SendRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: dict[str, Any]
    configuration: dict[str, Any] | None = None

class ProposalFacts(BaseModel):
    vendorName: str
    invoiceNumber: str
    amountMinor: int
    currency: str

class Proposal(BaseModel):
    packageId: str
    actionId: str = Field(min_length=12)
    action: str
    facts: ProposalFacts
    evidenceRefs: list[str]
    rationale: str

class ResultItem(BaseModel):
    packageId: str
    actionId: str
    action: str
    outcome: str
    receiptNonce: str

class ContinuationData(BaseModel):
    batchId: str
    results: list[ResultItem]
