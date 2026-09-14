"""llm_client.py — Wrapper asíncrono para Ollama / vLLM / OpenRouter (+ Mock).

Expone una interfaz uniforme `LLMClient.chat(messages, ...)` usada tanto por el
agente atacante como por el Juez. Soporta:

- OllamaClient          : API nativa de Ollama  (POST /api/chat).
- OpenRouterClient      : nube, OpenAI-compatible (el backend del equipo).
- VLLMClient            : self-hosted, OpenAI-compatible.
- MockLLMClient         : respuestas scripteadas para pruebas locales sin red.

OpenRouter y vLLM comparten `OpenAICompatibleClient`.

Uso:
    from rhlab.llm_client import build_llm_client
    from rhlab.config import LLMBackend, Settings

    settings = Settings()  # toma OPENROUTER_API_KEY del entorno
    client = build_llm_client(
        LLMBackend.OPENROUTER, "meta-llama/llama-3.1-8b-instruct", settings)
    resp = await client.chat([{"role": "user", "content": "hola"}], json_mode=True)
    print(resp.content)
    await client.aclose()
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence, Union

import httpx

from rhlab.config import LLMBackend, Settings

Message = dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


class LLMError(RuntimeError):
    """Error al hablar con el backend del modelo."""

    def __init__(self, message: str, *, raw: dict | None = None) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass
class LLMResponse:
    """Respuesta normalizada de cualquier backend."""

    content: str
    raw: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parsea `content` como JSON (útil cuando se pidió json_mode)."""
        return json.loads(self.content)


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #


class LLMClient(ABC):
    """Contrato común de todos los clientes LLM."""

    model: str

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Envía una conversación y devuelve la respuesta del modelo."""

    async def aclose(self) -> None:  # noqa: B027 - override opcional
        """Libera recursos (conexiones HTTP). Override si aplica."""

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


class OllamaClient(LLMClient):
    """Cliente para la API nativa de Ollama."""

    def __init__(self, model: str, base_url: str, timeout: float = 120.0) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": options,
        }
        if json_mode:
            # Ollama fuerza salida JSON válida con format="json".
            payload["format"] = "json"

        try:
            resp = await self._client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except ValueError as exc:
            raise LLMError("Ollama returned a non-JSON HTTP body", raw={"body": resp.text}) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama HTTP {exc.response.status_code}",
                           raw={"http_status": exc.response.status_code, "body": exc.response.text}) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama HTTP error ({self.model}): {exc}") from exc

        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise LLMError("Unexpected Ollama response shape", raw={"response": data})
        content = data["message"].get("content", "")
        if not content:
            raise LLMError("Ollama devolvió contenido vacío", raw=data)
        return LLMResponse(content=content, raw=data)

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- #
# OpenAI-compatible (vLLM y OpenRouter comparten el esquema /chat/completions)
# --------------------------------------------------------------------------- #


class OpenAICompatibleClient(LLMClient):
    """Cliente para cualquier API con esquema OpenAI `/chat/completions`.

    Lo usan tanto vLLM (self-hosted) como OpenRouter (nube). Las diferencias
    (URL base, api_key, cabeceras extra, y si conviene mandar `response_format`)
    se pasan por parámetro.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        extra_headers: Optional[dict[str, str]] = None,
        use_response_format: bool = True,
        label: str = "openai-compat",
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._use_response_format = use_response_format
        self._label = label
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # `response_format` solo si el backend lo soporta de forma fiable; en
        # OpenRouter se desactiva por defecto (soporte dispar entre modelos).
        if json_mode and self._use_response_format:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"{self._label} returned a non-JSON HTTP body", raw={"body": resp.text}) from exc
        except httpx.HTTPStatusError as exc:
            # Surface del cuerpo: OpenRouter devuelve errores útiles en JSON
            # (modelo inexistente, sin créditos, key inválida, etc.).
            body = exc.response.text[:400]
            raise LLMError(
                f"{self._label} HTTP {exc.response.status_code} ({self.model}): {body}",
                raw={"http_status": exc.response.status_code, "body": exc.response.text},
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self._label} error de red ({self.model}): {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Respuesta {self._label} inesperada", raw=data) from exc
        if not content:
            raise LLMError(f"{self._label} devolvió contenido vacío", raw=data)
        return LLMResponse(content=content, raw=data)

    async def aclose(self) -> None:
        await self._client.aclose()


class VLLMClient(OpenAICompatibleClient):
    """Cliente para la API OpenAI-compatible expuesta por vLLM (self-hosted)."""

    def __init__(self, model: str, base_url: str, timeout: float = 120.0,
                 api_key: str = "EMPTY") -> None:
        super().__init__(
            model, base_url, api_key=api_key, timeout=timeout,
            use_response_format=True, label="vLLM",
        )


class OpenRouterClient(OpenAICompatibleClient):
    """Cliente para OpenRouter (https://openrouter.ai/api/v1).

    OpenRouter es OpenAI-compatible y da acceso a cientos de modelos con una sola
    key. Los IDs van namespaced (`vendor/modelo`), p. ej.
    `cognitivecomputations/dolphin-mistral-24b-venice-edition`. Las cabeceras
    `HTTP-Referer` y `X-Title` son opcionales (atribución en rankings).
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        referer: str = "",
        title: str = "",
        use_response_format: bool = False,
    ) -> None:
        extra = {}
        if referer:
            extra["HTTP-Referer"] = referer
        if title:
            extra["X-Title"] = title
        if not api_key:
            # No abortamos: dejamos que el 401 de OpenRouter lo diga claramente,
            # pero avisamos en el mensaje de error si ocurre.
            api_key = "MISSING_OPENROUTER_API_KEY"
        super().__init__(
            model, base_url, api_key=api_key, timeout=timeout,
            extra_headers=extra, use_response_format=use_response_format,
            label="OpenRouter",
        )


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #

# Un guion puede ser una lista de strings (se consumen en orden) o un callable
# que recibe los mensajes y devuelve el próximo contenido (sync o async).
ScriptFn = Callable[[Sequence[Message]], Union[str, Awaitable[str]]]
Script = Union[Sequence[str], ScriptFn]


class MockLLMClient(LLMClient):
    """Cliente scripteado para probar el loop sin modelos reales.

    Ejemplos:
        # Lista de respuestas en orden:
        MockLLMClient("mock-agent", ['{"thought":"...","action":"ls","done":false}'])

        # Callable dependiente del historial:
        MockLLMClient("mock-agent", lambda msgs: decidir(msgs))
    """

    def __init__(self, model: str, script: Script) -> None:
        self.model = model
        self._script = script
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        idx = self._calls
        self._calls += 1

        if callable(self._script):
            result = self._script(messages)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[assignment]
            content = str(result)
        else:
            seq = list(self._script)
            if not seq:
                raise LLMError("MockLLMClient sin respuestas en el guion.")
            # Repite la última respuesta si se agota el guion (evita IndexError
            # y suele representar un 'done' o un 'no-op' final).
            content = seq[idx] if idx < len(seq) else seq[-1]

        return LLMResponse(content=content, raw={"mock": True, "call": idx})


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_llm_client(
    backend: LLMBackend,
    model: str,
    settings: Settings,
    *,
    mock_script: Optional[Script] = None,
) -> LLMClient:
    """Construye el cliente adecuado según el backend.

    Para `LLMBackend.MOCK` se requiere `mock_script`.
    """
    if backend is LLMBackend.OLLAMA:
        return OllamaClient(model, settings.ollama_base_url, timeout=settings.llm_timeout)
    if backend is LLMBackend.VLLM:
        return VLLMClient(model, settings.vllm_base_url, timeout=settings.llm_timeout)
    if backend is LLMBackend.OPENROUTER:
        return OpenRouterClient(
            model,
            settings.openrouter_base_url,
            settings.openrouter_api_key,
            timeout=settings.llm_timeout,
            referer=settings.openrouter_referer,
            title=settings.openrouter_title,
            use_response_format=settings.openrouter_response_format,
        )
    if backend is LLMBackend.MOCK:
        if mock_script is None:
            raise ValueError("LLMBackend.MOCK requiere un mock_script.")
        return MockLLMClient(model, mock_script)
    raise ValueError(f"Backend no soportado: {backend}")
