"""logger.py — Interfaz 3 (con Persona 4: Análisis / Telemetría).

Registro exhaustivo de eventos. En cada paso del loop se appendea:

- Una fila al CSV agregado `logs/experiment_runs.csv` (una fila por paso, de
  TODAS las corridas) con las columnas obligatorias acordadas con P4.
- Una línea JSON al archivo por corrida `logs/run_{run_id}.jsonl` con el detalle
  completo (incluye razonamiento crudo, veredicto del Juez, observación, etc.).

Columnas obligatorias del CSV:
    run_id, condition, step, proposed_action, judge_score, is_blocked,
    executed_action, captured_flag, touched_forbidden_vuln, timestamp

El escritor es seguro para corridas concurrentes: el CSV compartido se protege
con un `asyncio.Lock` de proceso. Los JSONL son por corrida, así que no compiten.
"""
from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CSV_FILENAME = "experiment_runs.csv"

CSV_COLUMNS = [
    "run_id",
    "condition",
    "step",
    "proposed_action",
    "judge_score",
    "is_blocked",
    "executed_action",
    "captured_flag",
    "touched_forbidden_vuln",
    "timestamp",
    "experimental_arm", "gate_mode", "intent_compatibility", "intent_uncertainty",
    "evidence_level", "derived_scope", "raw_gate_decision", "applied_gate_decision",
    "judge_source", "judge_failure", "fallback_decision", "executed", "reconsider_count",
    "prompt_version", "judge_model", "temperature", "latency", "input_tokens",
    "output_tokens", "judge_max_tokens", "judge_output_id", "decision_chain_id",
]


def _utc_now_iso() -> str:
    """Timestamp ISO-8601 en UTC (con sufijo Z)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class StepRecord:
    """Registro estructurado de un paso del loop ReAct."""

    run_id: str
    condition: str
    step: int
    proposed_action: str
    executed_action: str            # "" si fue bloqueado / no ejecutado
    is_blocked: bool
    captured_flag: bool
    touched_forbidden_vuln: bool
    judge_score: Optional[float] = None      # None en A/B. = intent_uncertainty (continuidad CSV)
    judge_reasoning: str = ""                 # = reason del Juez
    # Juez de Nivel 3 (dos ejes) + gate derivado. Van al JSONL para P4; el CSV
    # mantiene las columnas obligatorias acordadas (judge_score, is_blocked).
    intent_compatibility: Optional[float] = None
    intent_uncertainty: Optional[float] = None
    evidence_level: str = ""                 # EXPLICIT_RESTRICTION|..._AUTHORIZATION|CONTEXTUAL_INFERENCE|INSUFFICIENT_EVIDENCE
    scope_label: str = ""                    # IN_SCOPE|UNCERTAIN|OUT_OF_SCOPE (derived_scope)
    gate_mode: str = ""                      # uncertainty (C-UNC) | compatibility_only (C-COMP)
    gate_decision: str = ""                  # gate crudo del scope: EXECUTE|RECONSIDER|BLOCK
    gate_applied: str = ""                   # gate efectivo tras política de reconsideración
    judge_source: str = ""                   # prefilter|llm|mock|error
    judge_prompt_version: str = ""           # versión del prompt del Juez (reproducibilidad)
    thought: str = ""
    justification: str = ""                  # objetivo observable declarado por el actor
    expected_effect: str = ""
    observation: str = ""
    raw_agent_output: str = ""
    exit_code: Optional[int] = None
    judge_error: Optional[str] = None
    experimental_arm: str = ""
    derived_scope: Optional[str] = None
    raw_gate_decision: Optional[str] = None
    applied_gate_decision: str = ""
    judge_failure: bool = False
    fallback_decision: Optional[str] = None
    executed: bool = False
    reconsider_count: int = 0
    prompt_version: str = ""
    judge_model: str = ""
    temperature: Optional[float] = None
    latency: Optional[float] = None  # seconds, judge attempts only
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    judge_max_tokens: Optional[int] = None
    judge_output_id: Optional[str] = None
    decision_chain_id: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_csv_row(self) -> dict[str, Any]:
        """Proyección a las columnas obligatorias del CSV."""
        data = asdict(self)
        return {key: data[key] for key in CSV_COLUMNS}


# Lock de proceso para el CSV compartido (una sola instancia de proceso escribe).
_CSV_LOCK = asyncio.Lock()


class RunLogger:
    """Escritor de telemetría para una corrida concreta.

    Se instancia una por `run_id`. El JSONL es exclusivo de la corrida; el CSV es
    compartido y se serializa con un lock de módulo.
    """

    def __init__(self, run_id: str, condition: str, log_dir: Path) -> None:
        self.run_id = run_id
        self.condition = condition
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_dir / CSV_FILENAME
        self._jsonl_path = log_dir / f"run_{run_id}.jsonl"
        if self._csv_path.exists() and self._csv_path.stat().st_size:
            with self._csv_path.open(encoding="utf-8", newline="") as fh:
                if next(csv.reader(fh)) != CSV_COLUMNS:
                    raise ValueError("Legacy CSV schema: choose a new --log-dir; historical logs are preserved.")
        if self._jsonl_path.exists():
            raise ValueError(f"run_id already exists: {run_id}; choose another ID or --log-dir")
        self._output_path = log_dir / f"run_{run_id}.judge_outputs.jsonl"
        self._config: dict = {}

        # Reinicia el JSONL de esta corrida (idempotencia si se re-ejecuta el id).
        self._jsonl_path.write_text("", encoding="utf-8")
        self._records: list[StepRecord] = []

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    async def log_step(self, record: StepRecord) -> None:
        """Persiste un paso en JSONL (detalle) y CSV (agregado)."""
        self._records.append(record)
        self._append_jsonl(record)
        await self._append_csv(record)

    def _append_jsonl(self, record: StepRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    async def _append_csv(self, record: StepRecord) -> None:
        async with _CSV_LOCK:
            # Crea encabezado si el archivo es nuevo o está vacío.
            new_file = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
            with self._csv_path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(record.to_csv_row())

    def write_run_summary(self, summary: dict[str, Any]) -> None:
        """Escribe un resumen de la corrida en `logs/run_{run_id}.summary.json`."""
        path = self._log_dir / f"run_{self.run_id}.summary.json"
        from collections import Counter
        summary = {**summary, "effective_config": self._config,
                   "judge_failure_count": sum(r.judge_failure for r in self._records),
                   "executed_count": sum(r.executed for r in self._records),
                   "applied_gate_counts": dict(Counter(r.applied_gate_decision for r in self._records)),
                   "judge_output_ids": [r.judge_output_id for r in self._records if r.judge_output_id]}
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_config(self, config: dict) -> None:
        self._config = config
        path = self._log_dir / f"run_{self.run_id}.config.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def log_judge_output(self, output: dict, *, step: int) -> None:
        record = {**output, "run_id": self.run_id, "case_id": f"{self.run_id}:{step}",
                  "step": step, "rep": 1}
        with self._output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def history_for_judge(self) -> list[dict[str, Any]]:
        """Historial compacto para pasar al Juez (Interfaz 2)."""
        return [
            {
                "step": r.step,
                "proposed_action": r.proposed_action,
                "executed_action": r.executed_action,
                "observation": r.observation,
            }
            for r in self._records
        ]
