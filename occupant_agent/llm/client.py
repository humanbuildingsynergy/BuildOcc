"""
Unified LLM client for Anthropic, OpenAI, Google Gemini, and Ollama.

Returns parsed JSON dicts. Callers (step, receive_signal, reflect) validate
structure with Pydantic or manual key checks after the call.

Supported providers:
    "anthropic" — Claude (Haiku for step, Sonnet for reflect). API key: ANTHROPIC_API_KEY
    "openai"    — GPT-4o-mini for step, GPT-4o for reflect. API key: OPENAI_API_KEY
    "google"    — Gemini Flash for step, Gemini Pro for reflect. API key: GOOGLE_API_KEY
    "ollama"    — Local open models (Llama, Mistral, Qwen…). No API key needed.
                  Host: OLLAMA_HOST (default: http://localhost:11434)
                  Set OLLAMA_MODEL to override the default model.

Usage:
    from occupant_agent.llm.client import call_llm
    result = call_llm(system="...", user="...", provider="anthropic")
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd or any parent directory

# ── Default models: fast/cheap tier for high-frequency step() calls (~96×/sim-day) ──

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai":    "gpt-4o-mini",
    "google":    "gemini-2.0-flash",
    "ollama":    "llama3.2",           # override with OLLAMA_MODEL env var
}

# Smarter models for low-frequency reflect() synthesis
REFLECT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-1.5-pro",
    "ollama":    "llama3.2",           # same model; Ollama hardware is the constraint
}

# Temperature=0 ensures deterministic, reproducible outputs for scientific validation.
DEFAULT_TEMPERATURE: float = 0.0

_KNOWN_PROVIDERS = frozenset(_DEFAULT_MODELS)


class LLMParseError(ValueError):
    """Raised when the LLM response is not valid JSON."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


def call_llm(
    system: str,
    user: str,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    """
    Call an LLM and return the parsed JSON response as a dict.

    Args:
        system:      System prompt (persona context, instructions).
        user:        User turn (current situation, action request).
        provider:    "anthropic" | "openai" | "google" | "ollama"
        model:       Model ID; None → provider default.
        max_tokens:  Maximum output tokens.
        temperature: Sampling temperature. Default 0.0 for scientific reproducibility.

    Returns:
        Parsed dict from the LLM's JSON response.

    Raises:
        LLMParseError: Response was not valid JSON.
        ValueError:    Unknown provider.
        RuntimeError:  API call failed (wraps underlying exception).
    """
    if provider not in _KNOWN_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Expected one of: {sorted(_KNOWN_PROVIDERS)}"
        )

    # Ollama: allow OLLAMA_MODEL env var to override the default
    if provider == "ollama" and model is None:
        model = os.getenv("OLLAMA_MODEL", _DEFAULT_MODELS["ollama"])
    else:
        model = model or _DEFAULT_MODELS[provider]

    raw: str
    try:
        if provider == "anthropic":
            raw = _call_anthropic(system, user, model, max_tokens, temperature)
        elif provider == "openai":
            raw = _call_openai(system, user, model, max_tokens, temperature)
        elif provider == "google":
            raw = _call_google(system, user, model, max_tokens, temperature)
        elif provider == "ollama":
            raw = _call_ollama(system, user, model, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")
    except (LLMParseError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(f"LLM API call failed ({provider}/{model}): {exc}") from exc

    # Strip markdown code fences if the model wrapped the JSON
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMParseError(
            f"LLM ({provider}/{model}) returned non-JSON: {exc}",
            raw=raw,
        ) from exc


# ── Provider implementations ──────────────────────────────────────────────────

def _call_anthropic(
    system: str, user: str, model: str, max_tokens: int, temperature: float
) -> str:
    import anthropic  # lazy import

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def _call_openai(
    system: str, user: str, model: str, max_tokens: int, temperature: float
) -> str:
    import openai  # lazy import

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_google(
    system: str, user: str, model: str, max_tokens: int, temperature: float
) -> str:
    """
    Google Gemini via the google-generativeai SDK.
    Install: pip install "buildocc[google]"
    API key: GOOGLE_API_KEY environment variable.
    """
    try:
        import google.generativeai as genai  # lazy import
    except ImportError:
        raise RuntimeError(
            "google-generativeai is not installed. "
            'Run: pip install "buildocc[google]"'
        )

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    generation_config = genai.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        response_mime_type="application/json",   # enforces JSON output
    )
    model_obj = genai.GenerativeModel(
        model_name=model,
        system_instruction=system,
        generation_config=generation_config,
    )
    response = model_obj.generate_content(user)
    return response.text or ""


def _call_ollama(
    system: str, user: str, model: str, max_tokens: int, temperature: float
) -> str:
    """
    Ollama local inference via its OpenAI-compatible REST API.
    Install Ollama: https://ollama.com — no extra Python package needed.
    Host: OLLAMA_HOST env var (default: http://localhost:11434).
    Model: OLLAMA_MODEL env var (default: llama3.2).
    """
    import openai  # lazy import — Ollama is OpenAI-API-compatible

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = openai.OpenAI(base_url=f"{host}/v1", api_key="ollama")

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        # JSON mode support varies by model; prompt-based JSON is the fallback.
        # The code-fence stripping in call_llm() handles wrapped responses.
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""
