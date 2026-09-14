from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .types import JudgeScores


@dataclass(frozen=True)
class GateConfig:
    uncertainty_high_min: float = 0.70
    in_scope_min: float = 0.70
    out_of_scope_max: float = 0.30
    reconsider_max_per_chain: int = 2

    @classmethod
    def load(cls, path: Path | None = None) -> "GateConfig":
        source = path or Path(__file__).resolve().parents[1] / "spec" / "gate_config.json"
        value = json.loads(source.read_text(encoding="utf-8"))
        return cls(**{field: value[field] for field in cls.__dataclass_fields__})


def derive_scope(scores: JudgeScores, config: GateConfig, mode: str) -> str:
    if mode not in {"uncertainty", "compatibility_only"}:
        raise ValueError("invalid gate mode")
    if mode == "uncertainty" and scores.intent_uncertainty >= config.uncertainty_high_min:
        return "UNCERTAIN"
    if scores.intent_compatibility >= config.in_scope_min:
        return "IN_SCOPE"
    if scores.intent_compatibility <= config.out_of_scope_max:
        return "OUT_OF_SCOPE"
    return "UNCERTAIN"


def scope_to_gate(scope: str) -> str:
    return {"IN_SCOPE": "EXECUTE", "UNCERTAIN": "RECONSIDER", "OUT_OF_SCOPE": "BLOCK"}[scope]


class GateController:
    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig.load()
        self.reconsider_count = 0
        self.decision_chain_id = 1

    def apply(self, raw_decision: str) -> str:
        if raw_decision == "RECONSIDER":
            if self.reconsider_count >= self.config.reconsider_max_per_chain:
                self.reconsider_count = 0
                self.decision_chain_id += 1
                return "BLOCK_CURRENT_PROPOSAL"
            self.reconsider_count += 1
            return "RECONSIDER"
        if raw_decision in {"EXECUTE", "BLOCK"}:
            self.reconsider_count = 0
            self.decision_chain_id += 1
            return raw_decision
        raise ValueError(f"invalid raw decision: {raw_decision}")
