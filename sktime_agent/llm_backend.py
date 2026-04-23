"""LLM backend abstraction — OpenAI, Anthropic, Gemini, LangChain."""
from __future__ import annotations

import os
from typing import Any


class LLMBackend:
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return resp.choices[0].message.content


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0,
        )
        return resp.content[0].text


class GeminiBackend(LLMBackend):
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        self._model_name = model
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    def complete(self, system: str, user: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")
        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config={"response_mime_type": "application/json"},
            system_instruction=system,
        )
        resp = model.generate_content(user)
        return resp.text


class LangChainBackend(LLMBackend):
    """Wrap any LangChain LLM or ChatModel."""

    def __init__(self, llm: Any):
        self._llm = llm

    def complete(self, system: str, user: str) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [SystemMessage(content=system), HumanMessage(content=user)]
            resp = self._llm.invoke(messages)
            return resp.content if hasattr(resp, "content") else str(resp)
        except ImportError:
            return self._llm.predict(f"{system}\n\n{user}")


def create_backend(llm_backend: str | Any) -> LLMBackend:
    if isinstance(llm_backend, LLMBackend):
        return llm_backend
    if isinstance(llm_backend, str):
        backends = {
            "openai": OpenAIBackend,
            "anthropic": AnthropicBackend,
            "gemini": GeminiBackend,
        }
        if llm_backend not in backends:
            raise ValueError(
                f"Unknown backend {llm_backend!r}. "
                f"Use one of {list(backends)} or pass a LangChain LLM object."
            )
        return backends[llm_backend]()
    return LangChainBackend(llm_backend)
