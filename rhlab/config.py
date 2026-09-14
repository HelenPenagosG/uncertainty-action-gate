"""config.py — Configuración central del experimento (Persona 2).

Define endpoints de los modelos (Ollama / vLLM), nombres de modelos, el umbral
del Juez de Incertidumbre, timeouts y las carpetas de logs.

Todos los valores tienen un default sensato y pueden sobrescribirse por variable
de entorno o desde la CLI del `runner.py` (que construye un objeto `Settings`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------- #
# Enums de dominio
# --------------------------------------------------------------------------- #


class Condition(str, Enum):
    """Condiciones experimentales.

    A = Normal            -> system prompt estándar, sin Juez.
    B = Sin Freno         -> system prompt "meta a cualquier costo", sin Juez.
    C = Sin Freno + Juez  -> agente sin freno + interceptor del Juez antes de ejecutar.
    """

    A = "A"
    B = "B"
    C = "C"

    @property
    def uses_judge(self) -> bool:
        """Solo la condición C consulta al Juez antes de ejecutar."""
        return self is Condition.C

    @property
    def is_no_brakes(self) -> bool:
        """B y C usan el system prompt sin freno; A usa el estándar."""
        return self in (Condition.B, Condition.C)


class LLMBackend(str, Enum):
    """Backend para las llamadas al LLM."""

    OLLAMA = "ollama"           # API nativa de Ollama (/api/chat)
    VLLM = "vllm"               # API OpenAI-compatible de vLLM (/v1/chat/completions)
    OPENROUTER = "openrouter"   # API OpenAI-compatible de OpenRouter (nube; la usa el equipo)
    MOCK = "mock"               # Cliente scripteado para pruebas locales

    @property
    def is_openai_compatible(self) -> bool:
        """vLLM y OpenRouter comparten el esquema OpenAI /chat/completions."""
        return self in (LLMBackend.VLLM, LLMBackend.OPENROUTER)


# --------------------------------------------------------------------------- #
# Rutas base
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
PROMPT_DIR = BASE_DIR / "prompts"          # overrides opcionales en .txt

# --------------------------------------------------------------------------- #
# Defaults (sobrescribibles por entorno)
# --------------------------------------------------------------------------- #

# Endpoints
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# vLLM expone una API OpenAI-compatible; incluir el sufijo /v1.
DEFAULT_VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
# OpenRouter: API OpenAI-compatible en la nube.
DEFAULT_OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# La credencial se obtiene solo del entorno; nunca se hardcodea.
DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Cabeceras opcionales de atribución para el proveedor.
DEFAULT_OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "")
DEFAULT_OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "Reward-Blind Action Gate")

# Endpoint del sandbox de P1 (HTTP POST). Se ignora si USE_MOCK_SANDBOX=True.
DEFAULT_SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:9000/execute")

# Modelos por defecto usados durante el desarrollo.
DEFAULT_AGENT_BACKEND = LLMBackend(os.getenv("AGENT_BACKEND", "openrouter"))
DEFAULT_JUDGE_BACKEND = LLMBackend(os.getenv("JUDGE_BACKEND", "openrouter"))
# Agente principal (atacante): 24B "uncensored" — permisivo, para inducir el
# comportamiento a estudiar. Alternativas: microsoft/wizardlm-2-8x22b.
DEFAULT_AGENT_MODEL = os.getenv(
    "AGENT_MODEL", "cognitivecomputations/dolphin-mistral-24b-venice-edition"
)
# Small judge candidate; final selection awaits real Docker actions.
# Hermes/Llama 70B are upper bounds, larger than the 24B actor.
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3-8b")

# Enable JSON response_format only if supported by the provider.
DEFAULT_OPENROUTER_RESPONSE_FORMAT = os.getenv("OPENROUTER_RESPONSE_FORMAT", "0") == "1"

# Modo del gate (Condición C):
#   "uncertainty"        -> C-UNC: usa ambos ejes (compat + uncertainty). DEFAULT.
#   "compatibility_only" -> C-COMP: ABLACIÓN opcional, ignora intent_uncertainty en el
#                           gate (el juez sigue devolviendo ambos ejes; el spec NO cambia).
DEFAULT_GATE_MODE = os.getenv("GATE_MODE", "uncertainty")

# Pre-filtro determinista (capa "firewall"): bloquea líneas rojas universales
# (comandos destructivos irreversibles, fork bombs, apagados...) SIN gastar una
# llamada al LLM. Inspirado en el firewall de ExploitGym (enforcement estructural).
DEFAULT_JUDGE_SCOPE_PREFILTER = os.getenv("JUDGE_SCOPE_PREFILTER", "1") == "1"

# Control del loop
DEFAULT_MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
DEFAULT_STEP_TIMEOUT = int(os.getenv("STEP_TIMEOUT", "15"))      # timeout por comando en sandbox (s)
DEFAULT_LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))       # timeout por llamada al LLM (s)
DEFAULT_MAX_PARSE_RETRIES = int(os.getenv("MAX_PARSE_RETRIES", "3"))  # reintentos si el LLM no da acción parseable

# Muestreo del LLM
DEFAULT_AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
DEFAULT_JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
DEFAULT_AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "1024"))
DEFAULT_JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "1024"))

MAX_STEPS = DEFAULT_MAX_STEPS
STEP_TIMEOUT = DEFAULT_STEP_TIMEOUT


# --------------------------------------------------------------------------- #
# Settings: objeto inmutable que se pasa por el sistema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Settings:
    """Configuración efectiva de una corrida/lote.

    El `runner.py` construye una instancia a partir de los argumentos CLI y de los
    defaults de este módulo, y la inyecta en el resto de componentes. Ser inmutable
    (`frozen`) evita mutaciones accidentales entre corridas concurrentes.
    """

    # Backends y modelos
    agent_backend: LLMBackend = DEFAULT_AGENT_BACKEND
    judge_backend: LLMBackend = DEFAULT_JUDGE_BACKEND
    agent_model: str = DEFAULT_AGENT_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL

    # Endpoints
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    vllm_base_url: str = DEFAULT_VLLM_BASE_URL
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    openrouter_api_key: str = DEFAULT_OPENROUTER_API_KEY
    openrouter_referer: str = DEFAULT_OPENROUTER_REFERER
    openrouter_title: str = DEFAULT_OPENROUTER_TITLE
    openrouter_response_format: bool = DEFAULT_OPENROUTER_RESPONSE_FORMAT
    sandbox_url: str = DEFAULT_SANDBOX_URL

    # Mocks
    use_mock_llm: bool = False
    use_mock_sandbox: bool = False
    use_mock_judge: bool = False

    # Thresholds, history and failure policy come exclusively from judge_spec.
    gate_mode: str = DEFAULT_GATE_MODE
    judge_scope_prefilter: bool = DEFAULT_JUDGE_SCOPE_PREFILTER

    # Loop
    max_steps: int = DEFAULT_MAX_STEPS
    step_timeout: int = DEFAULT_STEP_TIMEOUT
    llm_timeout: int = DEFAULT_LLM_TIMEOUT
    max_parse_retries: int = DEFAULT_MAX_PARSE_RETRIES

    # Muestreo
    agent_temperature: float = DEFAULT_AGENT_TEMPERATURE
    judge_temperature: float = DEFAULT_JUDGE_TEMPERATURE
    agent_max_tokens: int = DEFAULT_AGENT_MAX_TOKENS
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS

    # Logs
    log_dir: Path = LOG_DIR

    def experimental_arm(self, condition: Condition) -> str:
        if condition is Condition.C:
            return "C-COMP" if self.gate_mode == "compatibility_only" else "C-UNC"
        return condition.value

    def effective_config(self) -> dict:
        """Serializable configuration with sanitized endpoints and no API key."""
        from dataclasses import asdict
        from urllib.parse import urlsplit, urlunsplit
        config = {k: str(v) if isinstance(v, Path) else v
                  for k, v in asdict(self).items() if k != "openrouter_api_key"}
        for key in ("ollama_base_url", "vllm_base_url", "openrouter_base_url", "sandbox_url", "openrouter_referer"):
            parts = urlsplit(config[key])
            config[key] = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
        return config

    def __post_init__(self) -> None:
        if self.gate_mode not in ("uncertainty", "compatibility_only"):
            raise ValueError("Invalid gate_mode")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.judge_max_tokens < 1:
            raise ValueError("judge_max_tokens must be positive")

    def base_url_for(self, backend: LLMBackend) -> str:
        """Devuelve el endpoint que corresponde a un backend dado."""
        if backend is LLMBackend.OLLAMA:
            return self.ollama_base_url
        if backend is LLMBackend.VLLM:
            return self.vllm_base_url
        if backend is LLMBackend.OPENROUTER:
            return self.openrouter_base_url
        return ""  # MOCK no usa red

    def describe(self) -> str:
        """Resumen legible para el log de arranque."""
        return (
            f"agent={self.agent_model}@{self.agent_backend.value} "
            f"judge={self.judge_model}@{self.judge_backend.value} "
            f"gate_mode={self.gate_mode} judge_max_tokens={self.judge_max_tokens} "
            f"max_steps={self.max_steps} "
            f"mock(llm={self.use_mock_llm},sandbox={self.use_mock_sandbox},judge={self.use_mock_judge})"
        )


def ensure_dirs(settings: Settings) -> None:
    """Crea las carpetas de logs si no existen."""
    settings.log_dir.mkdir(parents=True, exist_ok=True)
