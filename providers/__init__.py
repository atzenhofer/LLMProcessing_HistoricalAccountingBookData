# Author: Max Vogeltanz, University of Graz, 2026

# Script to initialize different providers for LLM API Processing in project "Aldersbach Digital"


def make_client(provider: str, cfg: dict | None = None):
    p = provider.lower()
    if p == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient()
    if p == "openai":
        from .openai_client import OpenAIClient
        openai_cfg = (cfg or {}).get("providers", {}).get("openai", {})
        return OpenAIClient(
            api_key=openai_cfg.get("api_key"),
            base_url=openai_cfg.get("base_url"),
            extra_body=openai_cfg.get("extra_body"),
        )
    if p == "vllm":
        from .openai_client import OpenAIClient
        vllm_cfg = (cfg or {}).get("providers", {}).get("vllm", {})
        return OpenAIClient(
            api_key=vllm_cfg.get("api_key"),
            base_url=vllm_cfg.get("base_url"),
            extra_body=vllm_cfg.get("extra_body"),
        )
    if p == "mistral":
        from .mistral_client import MistralClient
        return MistralClient()
    if p == "gemini":
        from .gemini_client import GeminiClient
        return GeminiClient()
    raise ValueError(f"Unknown provider: {provider}")
