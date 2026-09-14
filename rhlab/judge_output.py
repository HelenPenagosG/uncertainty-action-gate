"""Una evaluación lógica del juez, con intentos crudos y derivaciones pareadas."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from rhlab.judge_spec import JudgeSpec, extract_json, validate_output
from rhlab.llm_client import LLMError

SCOPE_TO_GATE = {"IN_SCOPE": "EXECUTE", "UNCERTAIN": "RECONSIDER", "OUT_OF_SCOPE": "BLOCK"}
ARMS = {"C-UNC": "uncertainty", "C-COMP": "compatibility_only"}


@dataclass
class JudgeOutput:
    judge_output_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    scores: dict | None = None
    attempts: list[dict] = field(default_factory=list)
    judge_input: dict = field(default_factory=dict)
    judge_model: str = ""
    prompt_version: str = ""
    temperature: float = 0.0
    judge_max_tokens: int | None = None
    latency: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    json_valid_first: bool = False
    json_valid_final: bool = False
    error: str | None = None
    raw_available: bool = True
    source: str = "llm"

    @property
    def judge_failure(self) -> bool:
        return not self.json_valid_final

    def to_dict(self) -> dict:
        return {**asdict(self), "judge_failure": self.judge_failure,
                "fallback_decision": "RECONSIDER" if self.judge_failure else None}


async def collect_judge_output(client, spec: JudgeSpec, judge_input: dict, *,
                               max_tokens: int, temperature: float = 0.0,
                               json_mode: bool = False) -> JudgeOutput:
    """Una salida compartida por ambos gates. Solo se repite si el intento falla."""
    output = JudgeOutput(judge_input=judge_input, judge_model=client.model,
                         prompt_version=spec.version, temperature=temperature,
                         judge_max_tokens=max_tokens)
    messages = [{"role": "system", "content": spec.prompt},
                {"role": "user", "content": json.dumps(judge_input, ensure_ascii=False)}]
    start = time.perf_counter()
    for attempt_index in range(spec.retry_count + 1):
        attempt: dict[str, Any] = {"attempt": attempt_index + 1, "content": None,
                                   "raw_response": None, "valid": False, "error": None}
        attempt_start = time.perf_counter()
        try:
            resp = await client.chat(messages, json_mode=json_mode, temperature=temperature,
                                     max_tokens=max_tokens)
            attempt.update(content=resp.content, raw_response=resp.raw)
            if not isinstance(resp.content, str):
                raise ValueError("Judge content is not a string")
            obj = extract_json(resp.content)
            valid, error = validate_output(obj)
            attempt.update(valid=valid, error=error or None)
            if valid:
                output.scores = obj
                output.json_valid_final = True
        except (LLMError, ValueError, TypeError) as exc:
            attempt["error"] = str(exc)
            if getattr(exc, "raw", None) is not None:
                attempt["raw_response"] = exc.raw
        attempt["latency"] = time.perf_counter() - attempt_start
        raw = attempt["raw_response"] if isinstance(attempt["raw_response"], dict) else {}
        usage = raw.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        # Totales de TODOS los intentos; desconocido si falta uso en cualquiera.
        attempt["input_tokens"] = usage.get("prompt_tokens", raw.get("prompt_eval_count"))
        attempt["output_tokens"] = usage.get("completion_tokens", raw.get("eval_count"))
        output.attempts.append(attempt)
        if attempt_index == 0:
            output.json_valid_first = attempt["valid"]
        output.error = attempt["error"]
        if output.json_valid_final:
            break
    output.latency = time.perf_counter() - start
    output.raw_available = any(a["content"] is not None or a["raw_response"] is not None for a in output.attempts)
    for key in ("input_tokens", "output_tokens"):
        values = [a[key] for a in output.attempts]
        setattr(output, key, sum(values) if all(isinstance(v, int) for v in values) else None)
    return output


def paired_derivations(output: JudgeOutput, spec: JudgeSpec) -> list[dict]:
    """Función pura: ambos brazos referencian el MISMO objeto/scores e identificador."""
    rows = []
    for arm, mode in ARMS.items():
        scores = output.scores if output.json_valid_final else None
        scope = spec.derive(scores["intent_compatibility"], scores["intent_uncertainty"], mode) if scores else None
        raw_gate = SCOPE_TO_GATE[scope] if scope else None
        rows.append({
            "judge_output_id": output.judge_output_id, "experimental_arm": arm,
            "gate_mode": mode, "intent_compatibility": scores["intent_compatibility"] if scores else None,
            "intent_uncertainty": scores["intent_uncertainty"] if scores else None,
            "evidence_level": scores["evidence_level"] if scores else None,
            "reason": scores["reason"] if scores else None,
            "derived_scope": scope, "raw_gate_decision": raw_gate,
            "fallback_decision": "RECONSIDER" if output.judge_failure else None,
            "applied_gate_decision": raw_gate or "RECONSIDER",
            "decision_application": "derived_only", "executed": False,
            "reconsider_count": None,
            "judge_failure": output.judge_failure, "judge_source": output.source,
            "json_valid_first": output.json_valid_first, "json_valid_final": output.json_valid_final,
            "prompt_version": output.prompt_version, "judge_model": output.judge_model,
            "temperature": output.temperature, "judge_max_tokens": output.judge_max_tokens,
            "latency": output.latency,
            "input_tokens": output.input_tokens, "output_tokens": output.output_tokens,
        })
    return rows
