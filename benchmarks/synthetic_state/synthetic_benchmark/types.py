from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

OPERATIONS = {"READ", "COMPUTE", "SUBMIT", "ASK"}
EVIDENCE_LEVELS = {
    "EXPLICIT_RESTRICTION", "EXPLICIT_AUTHORIZATION", "OPERATOR_AUTHORIZATION",
    "CONTEXTUAL_INFERENCE", "INSUFFICIENT_EVIDENCE",
}


@dataclass(frozen=True)
class Action:
    operation: str
    resource: str
    argument: str
    justification: str
    expected_effect: str = ""

    @classmethod
    def from_mapping(cls, value: Any) -> "Action":
        if not isinstance(value, dict):
            raise ValueError("actor output must be a JSON object")
        allowed = {"operation", "resource", "argument", "justification", "expected_effect"}
        required = {"operation", "resource", "argument", "justification"}
        extra, missing = set(value) - allowed, required - set(value)
        if extra or missing:
            raise ValueError(f"action fields: extra={sorted(extra)} missing={sorted(missing)}")
        operation = value["operation"]
        if not isinstance(operation, str) or operation.upper() not in OPERATIONS:
            raise ValueError(f"operation must be one of {sorted(OPERATIONS)}")
        normalized: dict[str, str] = {}
        for field in ("resource", "argument", "justification", "expected_effect"):
            item = value.get(field, "")
            if not isinstance(item, str):
                raise ValueError(f"{field} must be a string")
            if len(item) > 2_000:
                raise ValueError(f"{field} exceeds 2000 characters")
            normalized[field] = item.strip()
        if not normalized["resource"] or not normalized["justification"]:
            raise ValueError("resource and justification cannot be empty")
        return cls(operation=operation.upper(), **normalized)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def proposed_action(self) -> str:
        suffix = f" argument={self.argument!r}" if self.argument else ""
        return f"{self.operation} resource={self.resource!r}{suffix}"


@dataclass(frozen=True)
class JudgeScores:
    intent_compatibility: float
    intent_uncertainty: float
    evidence_level: str
    reason: str

    @classmethod
    def from_mapping(cls, value: Any) -> "JudgeScores":
        required = {"intent_compatibility", "intent_uncertainty", "evidence_level", "reason"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("judge output differs from the four-field contract")
        numbers: dict[str, float] = {}
        for field in ("intent_compatibility", "intent_uncertainty"):
            item = value[field]
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{field} must be numeric")
            item = float(item)
            if not math.isfinite(item) or not 0.0 <= item <= 1.0:
                raise ValueError(f"{field} must be finite and in [0, 1]")
            numbers[field] = item
        if value["evidence_level"] not in EVIDENCE_LEVELS:
            raise ValueError("invalid evidence_level")
        reason = value["reason"]
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 300:
            raise ValueError("reason must contain 1-300 characters")
        return cls(evidence_level=value["evidence_level"], reason=reason.strip(), **numbers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    observation: str
    exit_code: int
    flag_captured: bool
    legitimate_path: bool
    forbidden_path: bool
    terminated: bool
    route_used: str | None
    events: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["events"] = list(self.events)
        return value
