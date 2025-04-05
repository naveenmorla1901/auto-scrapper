"""
Configuration settings for the application.
"""
import os
from typing import Any, Dict, List, Optional, Union

# Try to import secrets, but don't fail if file doesn't exist
try:
    from .secrets import get_config, get_api_key
except ImportError:
    # Fallback functions if secrets.py is not available
    def get_config(key: str, default: Any = None) -> Any:
        return os.getenv(key, default)

    def get_api_key(provider: str) -> Optional[str]:
        provider = provider.upper()
        return os.getenv(f"{provider}_API_KEY")

# Application settings
DEBUG = get_config("DEBUG", False)
SECRET_KEY = os.getenv("SECRET_KEY", "insecure-dev-key-change-this-in-production")
DEFAULT_TIMEOUT = int(get_config("DEFAULT_EXECUTION_TIMEOUT", 60))
MAX_ATTEMPTS = int(get_config("MAX_ATTEMPTS", 3))

# Helper LLM settings
DEFAULT_HELPER_MODEL = "gemini-2.0-flash-lite"
DEFAULT_HELPER_TEMP = 0.2

# Supported LLM providers and models
SUPPORTED_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-lite", "gemini-1.0-pro", "gemini-1.0-ultra"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3.5-sonnet"],
    "mistral": ["mistral-large", "mistral-medium", "mistral-small"],
    "meta": ["llama-3-70b", "llama-3-8b"],
    "deepseek": ["deepseek-coder", "deepseek-chat", "deepseek-llm-67b", "deepseek-llm-7b"],
    "cohere": ["command-r", "command-r-plus", "command"],
    "ai21": ["j2-ultra", "j2-mid"],
    "together": ["yi-34b", "qwen-72b", "falcon-180b"]
}

def get_provider_from_model(model: str) -> str:
    """
    Determine the provider based on the model name.

    Args:
        model: The model name

    Returns:
        The provider name (openai, google, anthropic, etc.)
    """
    model = model.lower()

    if "gpt" in model:
        return "openai"
    elif "gemini" in model:
        return "google"
    elif "claude" in model:
        return "anthropic"
    elif "mistral" in model:
        return "mistral"
    elif "llama" in model:
        return "meta"
    elif "deepseek" in model:
        return "deepseek"
    elif "command" in model:
        return "cohere"
    elif "j2-" in model:
        return "ai21"
    elif any(x in model for x in ["yi-", "qwen", "falcon"]):
        return "together"
    else:
        raise ValueError(f"Unknown model provider: {model}")