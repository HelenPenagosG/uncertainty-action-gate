"""Interactive synthetic benchmark for an external intent/uncertainty judge."""

from .environment import SCENARIOS, SyntheticLab
from .gate import GateConfig, GateController, derive_scope
from .types import Action, JudgeScores

__all__ = ["Action", "GateConfig", "GateController", "JudgeScores", "SCENARIOS", "SyntheticLab", "derive_scope"]
