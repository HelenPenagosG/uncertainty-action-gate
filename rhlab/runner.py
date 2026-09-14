"""runner.py — CLI para ejecutar lotes de corridas del experimento (Persona 2).

Ejemplos
--------
    # Lote real, Condición C, 10 corridas secuenciales:
    python runner.py --condition C --runs 10

    # Prueba end-to-end sin GPU ni Docker (todo mock), 3 corridas concurrentes:
    python runner.py --condition C --runs 3 --mock --concurrency 3

    # Solo mockear el sandbox (LLMs reales por Ollama):
    python runner.py --condition B --runs 5 --mock-sandbox

    # Objetivo/alcance personalizados:
    python runner.py --condition A --runs 1 --objective "Capturar flag en /admin" \
        --authorized-resources "172.20.0.3:8080"

Salidas
-------
    logs/experiment_runs.csv        (agregado, una fila por paso)
    logs/run_{run_id}.jsonl         (detalle por corrida)
    logs/run_{run_id}.summary.json  (resumen por corrida)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from typing import Optional
from pathlib import Path

from rhlab.agent_loop import AgentLoop, RunResult
from rhlab.config import Condition, LLMBackend, Settings, ensure_dirs
from rhlab.judge_evaluator import JudgeEvaluator
from rhlab.llm_client import LLMClient, build_llm_client
from rhlab.logger import RunLogger
from rhlab.mocks import make_agent_script_callable
from rhlab.prompts import (
    DEFAULT_AUTHORIZED_RESOURCES,
    DEFAULT_EXPLICIT_RESTRICTIONS,
    DEFAULT_OBJECTIVE,
    DEFAULT_OPERATOR_AUTHORIZATION,
    TaskSpec,
)
from rhlab.sandbox_client import build_sandbox_client


# --------------------------------------------------------------------------- #
# Construcción de una corrida individual
# --------------------------------------------------------------------------- #


def _build_agent_client(settings: Settings) -> LLMClient:
    """Cliente del agente: mock scripteado o backend real."""
    if settings.use_mock_llm:
        return build_llm_client(
            LLMBackend.MOCK, "mock-agent", settings,
            mock_script=make_agent_script_callable(),
        )
    return build_llm_client(settings.agent_backend, settings.agent_model, settings)


def _build_judge(condition: Condition, settings: Settings) -> Optional[JudgeEvaluator]:
    """Juez solo para Condición C. Usa mock/heurística o LLM real según settings."""
    if not condition.uses_judge:
        return None
    if settings.use_mock_judge:
        return JudgeEvaluator(client=None, settings=settings)
    judge_client = build_llm_client(settings.judge_backend, settings.judge_model, settings)
    return JudgeEvaluator(client=judge_client, settings=settings)


async def run_single(
    run_id: str,
    condition: Condition,
    task: TaskSpec,
    settings: Settings,
) -> RunResult:
    """Ejecuta una corrida completa, gestionando el ciclo de vida de los clientes."""
    logger = RunLogger(run_id, condition.value, settings.log_dir)
    agent_client = _build_agent_client(settings)
    sandbox = build_sandbox_client(settings)
    judge = _build_judge(condition, settings)

    loop = AgentLoop(
        condition=condition,
        agent_client=agent_client,
        sandbox=sandbox,
        judge=judge,
        settings=settings,
    )
    try:
        return await loop.run(run_id, task, logger)
    finally:
        # Cierre ordenado de conexiones (evita sockets colgados en lotes grandes).
        await agent_client.aclose()
        await sandbox.aclose()
        if judge is not None:
            await judge.aclose()


# --------------------------------------------------------------------------- #
# Lote
# --------------------------------------------------------------------------- #


async def run_batch(
    condition: Condition,
    runs: int,
    settings: Settings,
    *,
    concurrency: int,
    start_index: int,
    task: TaskSpec,
    arm: str = "",
) -> list[RunResult]:
    """Ejecuta `runs` corridas con la concurrencia indicada.

    `arm` es la etiqueta para los run_id (p. ej. C-UNC / C-COMP); si vacía, usa la
    condición.
    """
    ensure_dirs(settings)
    prefix = arm or condition.value

    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[RunResult] = []

    async def _worker(idx: int) -> RunResult:
        run_id = f"{prefix}_{idx:03d}"
        async with semaphore:
            print(f"[+] Iniciando corrida {run_id} ...", flush=True)
            res = await run_single(run_id, condition, task, settings)
            print(
                f"[OK] {run_id}: stop={res.stop_reason.value} flag={res.captured_flag} "
                f"forbidden={res.touched_forbidden_vuln_count} blocked={res.blocked_count} "
                f"steps={res.steps}",
                flush=True,
            )
            return res

    tasks = [
        _worker(idx)
        for idx in range(start_index, start_index + runs)
    ]
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)

    return results


def _print_summary(results: list[RunResult]) -> None:
    """Resumen agregado del lote en consola."""
    if not results:
        print("Sin resultados.")
        return
    n = len(results)
    flags = sum(r.captured_flag for r in results)
    forbidden = sum(r.touched_forbidden_vuln_count > 0 for r in results)
    blocks = sum(r.blocked_count for r in results)
    errors = sum(bool(r.error) for r in results)
    print("\n===== RESUMEN DEL LOTE =====")
    print(f"corridas:               {n}")
    print(f"flags capturadas:       {flags}/{n}")
    print(f"corridas fuera de alcance: {forbidden}/{n}")
    print(f"bloqueos del Juez (total): {blocks}")
    print(f"corridas con error:     {errors}/{n}")
    print("============================")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_settings_from_args(args: argparse.Namespace) -> Settings:
    """Traduce los argumentos CLI a un objeto Settings inmutable."""
    base = Settings()
    overrides: dict = {}

    # --mock activa los tres mocks a la vez; los flags finos permiten mezclar.
    use_mock_llm = args.mock or args.mock_llm
    use_mock_sandbox = args.mock or args.mock_sandbox
    use_mock_judge = args.mock or args.mock_judge

    overrides.update(
        use_mock_llm=use_mock_llm,
        use_mock_sandbox=use_mock_sandbox,
        use_mock_judge=use_mock_judge,
    )
    if args.agent_backend:
        overrides["agent_backend"] = LLMBackend(args.agent_backend)
    if args.judge_backend:
        overrides["judge_backend"] = LLMBackend(args.judge_backend)
    if args.agent_model:
        overrides["agent_model"] = args.agent_model
    if args.judge_model:
        overrides["judge_model"] = args.judge_model
    if args.judge_max_tokens is not None:
        overrides["judge_max_tokens"] = args.judge_max_tokens
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.step_timeout is not None:
        overrides["step_timeout"] = args.step_timeout
    if args.log_dir is not None:
        overrides["log_dir"] = Path(args.log_dir)
    if args.sandbox_url:
        overrides["sandbox_url"] = args.sandbox_url
    if args.no_scope_prefilter:
        overrides["judge_scope_prefilter"] = False
    if args.openrouter_response_format:
        overrides["openrouter_response_format"] = True

    return replace(base, **overrides)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Runner del experimento de reward hacking (Persona 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--condition", required=True, choices=["A", "B", "C", "C-UNC", "C-COMP"],
                   help="A (normal), B (sin freno), C/C-UNC (sin freno + juez de incertidumbre), "
                        "C-COMP (ablación opcional: gate solo por compatibilidad).")
    p.add_argument("--gate-mode", choices=["uncertainty", "compatibility_only"], default=None,
                   help="Modo del gate en C. Por defecto uncertainty; C-COMP fuerza compatibility_only.")
    p.add_argument("--runs", type=int, default=1, help="Número de corridas del lote.")
    p.add_argument("--concurrency", type=int, default=1, help="Corridas concurrentes.")
    p.add_argument("--start-index", type=int, default=1, help="Índice inicial para los run_id.")
    p.add_argument("--log-dir", help="Directorio de telemetría; usar uno nuevo para el esquema v3.")

    # Mocks
    p.add_argument("--mock", action="store_true", help="Activa TODOS los mocks (LLM+sandbox+juez).")
    p.add_argument("--mock-llm", action="store_true", help="Mockea solo el agente LLM.")
    p.add_argument("--mock-sandbox", action="store_true", help="Mockea solo el sandbox (P1).")
    p.add_argument("--mock-judge", action="store_true", help="Mockea solo el Juez (P3).")

    # Overrides de modelos/endpoints
    p.add_argument("--agent-backend", choices=[b.value for b in LLMBackend])
    p.add_argument("--judge-backend", choices=[b.value for b in LLMBackend])
    p.add_argument("--agent-model")
    p.add_argument("--judge-model")
    p.add_argument("--judge-max-tokens", type=positive_int,
                   help="Máximo de tokens de salida del juez; se envía y registra explícitamente.")
    p.add_argument("--sandbox-url")

    # Parámetros del experimento
    p.add_argument("--max-steps", type=int, help="Máximo de pasos por corrida.")
    p.add_argument("--step-timeout", type=int, help="Timeout por comando en el sandbox (s).")
    p.add_argument("--no-scope-prefilter", action="store_true",
                   help="Desactiva el pre-filtro determinista del Juez (capa 1).")
    p.add_argument("--openrouter-response-format", action="store_true",
                   help="Fuerza response_format=json_object en OpenRouter (si el modelo lo soporta).")

    # Tarea (contrato v1.1: lo que ve el juez son authorization/resources/restrictions)
    p.add_argument("--objective", default=None,
                   help="Meta del ACTOR (incluye capturar la flag). NO se envía al juez.")
    p.add_argument("--operator-authorization", default=None,
                   help="Propósito autorizado que ve el juez (sin mencionar la flag).")
    p.add_argument("--authorized-resources", nargs="+", default=None,
                   help="Recursos permitidos. Ej: --authorized-resources '172.20.0.3:80'")
    p.add_argument("--restrictions", nargs="+", default=None,
                   help="Restricciones explícitas. Ej: --restrictions '10.0.0.0/8' 'verificador'")

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    # Consola en UTF-8: evita UnicodeEncodeError con acentos/glifos en terminales
    # cp1252 (Git Bash / cmd sin chcp 65001). No falla si no se puede reconfigurar.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    args = parse_args(argv)
    settings = build_settings_from_args(args)

    # Resolver condición, gate_mode y etiqueta del arm a partir del alias de --condition.
    if args.condition in ("C", "C-UNC"):
        condition, gate_mode = Condition.C, "uncertainty"
    elif args.condition == "C-COMP":
        condition, gate_mode = Condition.C, "compatibility_only"
    else:  # A, B
        condition, gate_mode = Condition(args.condition), settings.gate_mode
    if args.gate_mode:                       # --gate-mode explícito manda
        gate_mode = args.gate_mode
    settings = replace(settings, gate_mode=gate_mode)
    arm = ("C-COMP" if condition is Condition.C and gate_mode == "compatibility_only"
           else "C-UNC" if condition is Condition.C else condition.value)

    task = TaskSpec(
        objective=args.objective or DEFAULT_OBJECTIVE,
        operator_authorization=args.operator_authorization or DEFAULT_OPERATOR_AUTHORIZATION,
        authorized_resources=(tuple(args.authorized_resources) if args.authorized_resources
                              else DEFAULT_AUTHORIZED_RESOURCES),
        explicit_restrictions=(tuple(args.restrictions) if args.restrictions
                               else DEFAULT_EXPLICIT_RESTRICTIONS),
        max_steps=settings.max_steps,
    )

    print(f"[config] {settings.describe()}")
    print(f"[config] arm={arm} (condición={condition.value}, gate_mode={settings.gate_mode}) "
          f"runs={args.runs} concurrency={args.concurrency}")
    print(f"[task] objetivo={task.objective!r}")
    print(f"[task] autorización(juez)={task.operator_authorization!r}")
    print(f"[task] permitido={list(task.authorized_resources)} prohibido={list(task.explicit_restrictions)}")

    results = asyncio.run(
        run_batch(
            condition, args.runs, settings,
            concurrency=args.concurrency,
            start_index=args.start_index,
            task=task,
            arm=arm,
        )
    )
    _print_summary(results)
    print(f"\nLogs en: {settings.log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
