# src/models.py
"""
models.py

Loads all LLM endpoint configurations and credentials from config.yaml and
provides:

  • completion_with_backoff(...) — a thin HTTP wrapper with retries and backoff
  • AzureLLM.invoke(messages) — a unified interface for GPT, Claude, Gemini, Llama, etc.
  • get_tokenizer / count_tokens — utilities for token-counting and usage tracking
  • load_model(model_key, …) — factory that returns a ready-to-use AzureLLM

To use your own deployments or keys, simply update the endpoints and api_keys
in config.yaml; no code changes required.
"""

import json
import time
import logging
from typing import Any, List

import requests
from requests import Session

from rich.console import Console
from rich.logging import RichHandler

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

import tiktoken
from src.config import CONFIG

from src.utils.logging import console

# —————————————————————————————
# Global HTTP session & logger
# —————————————————————————————
session = Session()


# —————————————————————————————
# 1) completion_with_backoff
# —————————————————————————————
def completion_with_backoff(**kwargs) -> str:
    """
    Thin HTTP wrapper around the various LLM proxy endpoints with automatic
    retries, connection reuse (global `session`) and a 30-second timeout.

    Expected kwargs:
        platform, api_base, api_key, deployment_identifier, messages / prompt_text …
    """
    platform = kwargs.get("platform", "").lower()
    if platform not in {"gpt", "claude", "gemini", "llama", "o3-mini", "deepseek", "gemini-flash", "gpt-4.1", "gpt-4.1-mini"}:
        raise ValueError("Unsupported platform")

    api_base      = kwargs.get("api_base", "")
    api_key       = kwargs.get("api_key", "")
    deployment_id = kwargs.get("deployment_identifier", "")
    max_tokens    = kwargs.get("max_tokens", CONFIG.model_loading.default_max_tokens)
    temperature   = kwargs.get("temperature", CONFIG.model_loading.default_temperature)
    api_version   = kwargs.get("api_version", "")
    messages      = kwargs.get("messages", [])
    prompt_text   = kwargs.get("prompt_text", "")
    safety_settings   = kwargs.get("safety_settings", [])
    generation_config = kwargs.get("generation_config", {})

    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": api_key
    }

    # Exponential backoff parameters
    initial_delay = 1  # seconds - initial sleep duration
    max_delay     = 60 # seconds - maximum sleep duration
    backoff_factor = 2 # exponential factor
    
    delay = initial_delay  # start with initial delay
    resp = None

    for attempt in range(5):
        try:
            # -------- build request per provider --------------------------------
            if platform in {"gpt", "gpt-4.1", "gpt-4.1-mini"}:
                url  = f"{api_base}/deployments/{deployment_id}/chat/completions?api-version={api_version}"
                body = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

            elif platform == "claude":
                url  = api_base
                body = {
                    "model_id": deployment_id,
                    "prompt_text": prompt_text
                }

            elif platform == "gemini":
                url  = api_base
                body = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt_text}]
                    }],
                    "safety_settings": safety_settings,
                    "generation_config": generation_config
                }

            elif platform == "gemini-flash":
                url  = api_base
                body = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt_text}]
                    }],
                    "safety_settings": safety_settings,
                    "generation_config": generation_config
                }

            elif platform == "llama":
                url  = api_base
                # Format messages properly for Llama
                formatted = kwargs.get("messages", [])
                # If we were given prompt_text instead of messages, convert it
                if not formatted and "prompt_text" in kwargs:
                    formatted = [{"role": "user", "content": prompt_text}]
                body = {
                    "model": deployment_id,
                    "messages": formatted,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

            elif platform == "o3-mini":
                url  = api_base
                body = {
                    "messages": messages,
                    "max_completion_tokens": max_tokens
                }

            elif platform == "deepseek":
                url  = api_base
                body = {
                    "model": deployment_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 1,
                    "stream": False
                }

            # -------- send the request ------------------------------------------
            resp = session.post(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # -------- normalise output ------------------------------------------
            if platform in {"gpt", "gpt-4.1", "gpt-4.1-mini", "llama", "o3-mini", "deepseek"}:
                msg      = data["choices"][0]["message"]
                content  = msg.get("content")           # may be None
                if content in (None, ""):               # DeepSeek quirk
                    content = json.dumps(msg.get("tool_calls", {})) or ""
                return content

            if platform == "claude":
                cont = data.get("content")
                if isinstance(cont, list):
                    return "".join(p.get("text", "") for p in cont)
                return data.get("completion") or data.get("response", "")

            if platform in {"gemini", "gemini-flash"}:
                # handle both streaming‑array and non‑streaming dict
                if isinstance(data, list):
                    data = {"candidates": [c for chunk in data for c in chunk.get("candidates", [])]}
                return "".join(
                    part.get("text", "")
                    for cand in data.get("candidates", [])
                    for part in cand.get("content", {}).get("parts", [])
                )

        except requests.RequestException as e:
            console.log(f"[red]Request failed on attempt {attempt + 1}: {e}")
            status = getattr(resp, "status_code", None)
            if status == 429:
                console.log(f"[yellow]Rate limit hit. Backing off for {delay:.2f} seconds…")
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                time.sleep(initial_delay)
                
            # reset for next retry
            resp = None

    return f"[ERROR] exceeded retries for {platform}"


# —————————————————————————————
# 2) AzureLLM class 
# —————————————————————————————
class AzureLLM:
    """
    Wrapper for various endpoints: GPT, Claude, Gemini, or Llama, 
    exposing an .invoke(messages) method that returns the text.
    """
    def __init__(
        self,
        platform: str,
        api_base: str,
        api_key: str,
        deployment_identifier: str,
        api_version: str = "",
        temperature: float = 0.1,
        max_tokens: int = 800
    ):
        self.platform = platform.lower()
        self.api_base = api_base
        self.api_key = api_key
        self.deployment_identifier = deployment_identifier
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Token tracking
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def invoke(self, messages: List[Any]) -> str:
        # Count input tokens
        input_tokens = 0
        for m in messages:
            if isinstance(m, SystemMessage):
                input_tokens += count_tokens(m.content, self.platform)
            elif isinstance(m, HumanMessage):
                input_tokens += count_tokens(m.content, self.platform)
            elif isinstance(m, AIMessage):
                input_tokens += count_tokens(m.content, self.platform)
            elif isinstance(m, ToolMessage):
                input_tokens += count_tokens(str(m.content), self.platform)

        # Convert messages to the correct format for the chosen platform
        if self.platform in ["gpt", "gpt-4.1", "gpt-4.1-mini", "llama", "deepseek", "o3-mini"]:
            # For all these platforms, convert to dict messages
            mapped_msgs = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    role = "system"
                elif isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                elif isinstance(m, ToolMessage):
                    content = m.content
                    if not isinstance(content, str):
                        try:
                            content = json.dumps(content)
                        except:
                            content = str(content)
                    mapped_msgs.append({
                        "role": "assistant",
                        "content": f"[Tool Output]\n{content}"
                    })
                    # Skip adding this message again
                    continue
                else:
                    role = "user"
                mapped_msgs.append({"role": role, "content": m.content})

            # For o3-mini, ensure there's at least one message
            if self.platform == "o3-mini" and not mapped_msgs:
                mapped_msgs = [{"role": "user", "content": "Please analyze this case."}]

            # Adjust parameters based on platform
            params = {
                "platform":             self.platform,
                "api_base":             self.api_base,
                "api_key":              self.api_key,
                "deployment_identifier":self.deployment_identifier,
                "api_version":          self.api_version,
                "messages":             mapped_msgs
            }
            # For o3-mini, use max_completion_tokens instead of max_tokens
            if self.platform == "o3-mini":
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"]    = self.max_tokens
                params["temperature"]   = self.temperature

            result = completion_with_backoff(**params)

        elif self.platform in ["claude", "gemini", "gemini-flash"]:
            # FIXED: Use message-style formatting for Claude/Gemini instead of concatenating
            # This preserves conversation structure better
            prompt_text = ""
            for m in messages:
                if isinstance(m, SystemMessage):
                    prompt_text += f"<system>\n{m.content}\n</system>\n\n"
                elif isinstance(m, HumanMessage):
                    prompt_text += f"Human: {m.content}\n\n"
                elif isinstance(m, AIMessage):
                    prompt_text += f"Assistant: {m.content}\n\n"
                elif isinstance(m, ToolMessage):
                    prompt_text += f"Tool Result: {m.content}\n\n"
                else:
                    prompt_text += f"{m.content}\n\n"

            # End with an "Assistant: " prompt to indicate it's the model's turn
            if not prompt_text.endswith("Assistant: "):
                prompt_text += "Assistant: "

            if self.platform == "claude":
                result = completion_with_backoff(
                    platform="claude",
                    api_base=self.api_base,
                    api_key=self.api_key,  # Use the instance's API key
                    deployment_identifier=self.deployment_identifier,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    prompt_text=prompt_text
                )
            else:  # "gemini" or "gemini-flash"
                result = completion_with_backoff(
                    platform="gemini",
                    api_base=self.api_base,
                    api_key=self.api_key,  # Use the instance's API key
                    deployment_identifier=self.deployment_identifier,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    prompt_text=prompt_text,
                    safety_settings=[{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                                      "threshold": "BLOCK_LOW_AND_ABOVE"}],
                    generation_config={"temperature": self.temperature, "topP": 0.8, "topK": 40}
                )

        else:
            result = "Error: Unsupported platform"

        # Count output tokens
        output_tokens = count_tokens(result, self.platform)

        # Store token counts
        self.last_input_tokens = input_tokens
        self.last_output_tokens = output_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return result


# —————————————————————————————
# 3) Token counting
# —————————————————————————————
def get_tokenizer(model_name):
    """
    Get the appropriate tokenizer based on the model name.
    """
    model_name = model_name.lower()
    
    try:
        if "gpt-4" in model_name or "gpt4" in model_name or "gpt-4o" in model_name:
            return tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model_name or "gpt3" in model_name:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif "claude" in model_name or "llama" in model_name or "gemini" in model_name or "deepseek" in model_name:
            # Directly get the cl100k_base encoding instead of trying to map it to a model
            return tiktoken.get_encoding("cl100k_base")
        else:
            # Default to cl100k_base as a fallback, using get_encoding
            return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error getting tokenizer: {e}")
        

def count_tokens(text, model_name):
    """
    Count tokens for a given text using the appropriate tokenizer.
    """
    try:
        tokenizer = get_tokenizer(model_name)
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback to simple approximation if tiktoken fails
        print(f"Token counting error: {e}")
        return len(text) // 4  # Rough approximation


# —————————————————————————————
# 4) load_model
# —————————————————————————————
def load_model(
    model_id: str,
    matcher: bool = False,
    temperature: float = 0.1,
) -> AzureLLM:
    """
    Instantiate AzureLLM by mapping model_id to its API base, deployment_id,
    and version via CONFIG or edge-case defaults exactly as in the original.
    """
    platform = model_id.lower()

    # GPT-4.1 family on East US2
    if platform == "gpt-4.1":
        api_base              = CONFIG.endpoints.gpt41.api_base
        deployment_identifier = CONFIG.endpoints.gpt41.model_id
        api_version           = CONFIG.endpoints.gpt41.api_version

    elif platform == "gpt-4.1-mini":
        api_base              = CONFIG.endpoints.gpt41mini.api_base
        deployment_identifier = CONFIG.endpoints.gpt41mini.model_id
        api_version           = CONFIG.endpoints.gpt41mini.api_version

    # GPT-4 default
    elif platform == "gpt":
        api_base              = CONFIG.endpoints.gpt.api_base
        deployment_identifier = CONFIG.endpoints.gpt.model_id
        api_version           = CONFIG.endpoints.gpt.api_version

    elif platform == "claude":
        api_base              = CONFIG.endpoints.claude.api_base
        deployment_identifier = CONFIG.endpoints.claude.model_id
        api_version           = ""   # Claude doesn't use this

    elif platform == "gemini-flash":
        api_base              = CONFIG.endpoints.gemini_flash.api_base
        deployment_identifier = CONFIG.endpoints.gemini_flash.model_id
        api_version           = ""

    elif platform == "gemini":
        api_base              = CONFIG.endpoints.gemini.api_base
        deployment_identifier = CONFIG.endpoints.gemini.model_id
        api_version           = ""  # Gemini doesn't use this

    elif platform == "llama":
        api_base              = CONFIG.endpoints.llama.api_base
        deployment_identifier = CONFIG.endpoints.llama.model_id
        api_version           = ""   # Llama doesn't use this

    elif platform == "o3-mini":
        api_base              = CONFIG.endpoints.o3_mini.api_base
        deployment_identifier = CONFIG.endpoints.o3_mini.model_id
        api_version           = CONFIG.endpoints.o3_mini.api_version

    elif platform == "deepseek":
        api_base              = CONFIG.endpoints.deepseek_r1.api_base
        deployment_identifier = CONFIG.endpoints.deepseek_r1.model_id
        api_version           = ""    # DeepSeek doesn't use this

    else:
        # Default to OpenAI (but warn)
        console.log(f"[yellow]Unknown model '{model_id}', defaulting to gpt")
        api_base              = CONFIG.endpoints.gpt.api_base
        deployment_identifier = CONFIG.endpoints.gpt.model_id
        api_version           = CONFIG.endpoints.gpt.api_version
        platform              = "gpt"  # Force to GPT mode for unknown models

    
    if platform == "gpt":
        tok_cap = CONFIG.model_loading.matcher_max_tokens
    else:
        tok_cap = CONFIG.model_loading.matcher_max_tokens if matcher else CONFIG.model_loading.default_max_tokens


    return AzureLLM(
        platform=platform,
        api_base=api_base,
        api_key=CONFIG.api_keys.azure_key,
        deployment_identifier=deployment_identifier,
        api_version=api_version,
        temperature=temperature,
        max_tokens=tok_cap
    )

