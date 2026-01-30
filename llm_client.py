"""
LLM Client for RQ1/RQ2 Experiments

This module provides a unified interface for calling different LLM providers
(OpenAI, Anthropic) for root cause diagnosis.

Includes cost tracking for experiment budget management.
"""
from __future__ import annotations

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from config import (
    LLM_CONFIG, 
    OPENAI_API_KEY, 
    ANTHROPIC_API_KEY,
    DEEPSEEK_API_KEY,
    TOGETHER_API_KEY,
    GROQ_API_KEY,
    MIN_REQUEST_INTERVAL,
    MAX_RETRIES,
    RATE_LIMIT_BASE_DELAY
)

logger = logging.getLogger(__name__)


# ============================================================================
# COST TRACKING
# ============================================================================

# Pricing per 1M tokens (as of 2024-2025, update as needed)
# Source: https://openai.com/pricing, https://www.anthropic.com/pricing
MODEL_PRICING = {
    # OpenAI models (per 1M tokens)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic models (per 1M tokens)
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # DeepSeek models (per 1M tokens)
    "deepseek-coder": {"input": 0.14, "output": 0.28},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    # Llama models via Together AI (per 1M tokens)
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {"input": 0.18, "output": 0.18},
    # Llama models via Groq (per 1M tokens)
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},  # deprecated
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}


@dataclass
class UsageStats:
    """Track token usage and costs for a single API call."""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class CostTracker:
    """
    Track cumulative LLM usage and costs across an experiment.
    """
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    
    # Per-model breakdown
    model_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Per-component breakdown (e.g., "tier3_classification", "cluster_verification")
    component_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def record_usage(
        self, 
        model: str, 
        input_tokens: int, 
        output_tokens: int,
        component: str = "default"
    ) -> UsageStats:
        """
        Record a single API call's usage.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            component: Component name for breakdown (e.g., "tier3", "verification")
            
        Returns:
            UsageStats for this call
        """
        # Calculate costs
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        # Update totals
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        
        # Update model stats
        if model not in self.model_stats:
            self.model_stats[model] = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0
            }
        self.model_stats[model]["calls"] += 1
        self.model_stats[model]["input_tokens"] += input_tokens
        self.model_stats[model]["output_tokens"] += output_tokens
        self.model_stats[model]["cost"] += total_cost
        
        # Update component stats
        if component not in self.component_stats:
            self.component_stats[component] = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0
            }
        self.component_stats[component]["calls"] += 1
        self.component_stats[component]["input_tokens"] += input_tokens
        self.component_stats[component]["output_tokens"] += output_tokens
        self.component_stats[component]["cost"] += total_cost
        
        return UsageStats(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all usage."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "model_breakdown": self.model_stats,
            "component_breakdown": self.component_stats
        }
    
    def format_summary(self) -> str:
        """Format a human-readable summary."""
        lines = [
            "=" * 60,
            "LLM COST SUMMARY",
            "=" * 60,
            f"Total API Calls: {self.total_calls}",
            f"Total Input Tokens: {self.total_input_tokens:,}",
            f"Total Output Tokens: {self.total_output_tokens:,}",
            f"Total Tokens: {self.total_input_tokens + self.total_output_tokens:,}",
            f"Total Cost: ${self.total_cost:.4f} USD",
            "",
            "By Model:",
        ]
        
        for model, stats in self.model_stats.items():
            lines.append(f"  {model}:")
            lines.append(f"    Calls: {stats['calls']}, Tokens: {stats['input_tokens'] + stats['output_tokens']:,}, Cost: ${stats['cost']:.4f}")
        
        if self.component_stats:
            lines.append("")
            lines.append("By Component:")
            for component, stats in self.component_stats.items():
                lines.append(f"  {component}:")
                lines.append(f"    Calls: {stats['calls']}, Tokens: {stats['input_tokens'] + stats['output_tokens']:,}, Cost: ${stats['cost']:.4f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# Global cost tracker instance
_global_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return _global_cost_tracker


def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _global_cost_tracker
    _global_cost_tracker = CostTracker()


# ============================================================================
# LLM RESULT CACHING
# ============================================================================

import hashlib
import json
from pathlib import Path

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent / ".llm_cache"


class LLMCache:
    """
    Simple file-based cache for LLM responses.
    
    Saves money by avoiding repeated API calls for the same prompts.
    Cache key is based on: model + prompt + system_prompt
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a unique cache key."""
        content = f"{model}||{system_prompt or ''}||{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, model: str, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Get cached response if available.
        
        Returns:
            Cached response string, or None if not cached
        """
        if not self.enabled:
            return None
        
        key = self._make_key(model, prompt, system_prompt)
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.hits += 1
                    logger.debug(f"Cache hit: {key}")
                    return data.get("response")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Cache read error: {e}")
                return None
        
        self.misses += 1
        return None
    
    def set(
        self, 
        model: str, 
        prompt: str, 
        response: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Cache a response.
        
        Args:
            model: Model name
            prompt: The prompt sent
            response: The response received
            system_prompt: Optional system prompt
            metadata: Optional metadata to store (e.g., token counts)
        """
        if not self.enabled:
            return
        
        key = self._make_key(model, prompt, system_prompt)
        cache_path = self._get_cache_path(key)
        
        data = {
            "model": model,
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response": response,
            "timestamp": time.time(),
        }
        if metadata:
            data["metadata"] = metadata
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Cached response: {key}")
        except IOError as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear(self):
        """Clear all cached responses."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir),
            "cached_responses": len(cache_files),
            "total_size_kb": round(total_size / 1024, 2),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


# Global cache instance
_global_cache = LLMCache(enabled=True)


def get_llm_cache() -> LLMCache:
    """Get the global LLM cache instance."""
    return _global_cache


def set_cache_enabled(enabled: bool):
    """Enable or disable the global cache."""
    _global_cache.enabled = enabled


def clear_llm_cache():
    """Clear the global LLM cache."""
    _global_cache.clear()


# ============================================================================
# LLM CLIENT BASE CLASS
# ============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM clients with cost tracking."""
    
    def __init__(self):
        self.cost_tracker = get_cost_tracker()
        self._current_component = "default"
    
    def set_component(self, component: str):
        """Set the current component for cost tracking."""
        self._current_component = component
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used."""
        pass
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return self.cost_tracker.get_summary()


class OpenAIClient(LLMClient):
    """OpenAI API client with rate limit handling and cost tracking."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        
        try:
            from openai import OpenAI, RateLimitError, APIError
            self.RateLimitError = RateLimitError
            self.APIError = APIError
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e
        
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.config = LLM_CONFIG["openai"]
        self.model = model or self.config["model"]
        
        # Rate limiting: minimum delay between requests (seconds)
        self.min_request_interval = MIN_REQUEST_INTERVAL
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """Ensure minimum interval between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using OpenAI API with robust retry logic and cost tracking."""
        # Check cache first
        cache = get_llm_cache()
        cached_response = cache.get(self.model, prompt, system_prompt)
        if cached_response is not None:
            logger.debug(f"Using cached response for {self.model}")
            return cached_response
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        max_retries = MAX_RETRIES
        base_delay = RATE_LIMIT_BASE_DELAY
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self._wait_for_rate_limit()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"]
                )
                
                self.last_request_time = time.time()
                
                # Track usage and cost
                metadata = None
                if response.usage:
                    usage = self.cost_tracker.record_usage(
                        model=self.model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        component=self._current_component
                    )
                    metadata = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "cost": usage.total_cost
                    }
                    logger.debug(
                        f"API call: {usage.total_tokens} tokens, ${usage.total_cost:.4f} "
                        f"(component: {self._current_component})"
                    )
                
                result = response.choices[0].message.content
                
                # Cache the response
                cache.set(self.model, prompt, result, system_prompt, metadata)
                
                return result
                
            except self.RateLimitError as e:
                # Rate limit error - use longer backoff
                wait_time = base_delay * (2 ** attempt)  # 5, 10, 20, 40, 80 seconds
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except self.APIError as e:
                # Other API errors
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
                    
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        return ""
    
    def get_model_name(self) -> str:
        return self.model


class AnthropicClient(LLMClient):
    """Anthropic API client with rate limit handling and cost tracking."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        
        try:
            from anthropic import Anthropic, RateLimitError, APIError
            self.RateLimitError = RateLimitError
            self.APIError = APIError
        except ImportError as e:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from e
        
        self.api_key = api_key or ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = Anthropic(api_key=self.api_key)
        self.config = LLM_CONFIG["anthropic"]
        self.model = model or self.config["model"]
        
        # Rate limiting
        self.min_request_interval = MIN_REQUEST_INTERVAL
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """Ensure minimum interval between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using Anthropic API with robust retry logic and cost tracking."""
        # Check cache first
        cache = get_llm_cache()
        cached_response = cache.get(self.model, prompt, system_prompt)
        if cached_response is not None:
            logger.debug(f"Using cached response for {self.model}")
            return cached_response
        
        max_retries = MAX_RETRIES
        base_delay = RATE_LIMIT_BASE_DELAY
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.config["max_tokens"],
                    system=system_prompt or "You are an expert software engineer specializing in test flakiness analysis.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                self.last_request_time = time.time()
                
                # Track usage and cost
                metadata = None
                if response.usage:
                    usage = self.cost_tracker.record_usage(
                        model=self.model,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        component=self._current_component
                    )
                    metadata = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "cost": usage.total_cost
                    }
                    logger.debug(
                        f"API call: {usage.total_tokens} tokens, ${usage.total_cost:.4f} "
                        f"(component: {self._current_component})"
                    )
                
                result = response.content[0].text
                
                # Cache the response
                cache.set(self.model, prompt, result, system_prompt, metadata)
                
                return result
                
            except self.RateLimitError as e:
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except self.APIError as e:
                logger.warning(f"Anthropic API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
                    
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        return ""
    
    def get_model_name(self) -> str:
        return self.model


class OpenAICompatibleClient(LLMClient):
    """
    Client for OpenAI-compatible APIs (DeepSeek, Together AI, Groq, etc.)
    
    These providers use the same API format as OpenAI, just with different
    base URLs and API keys.
    """
    
    def __init__(
        self, 
        provider: str,
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        super().__init__()
        
        try:
            from openai import OpenAI, RateLimitError, APIError
            self.RateLimitError = RateLimitError
            self.APIError = APIError
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e
        
        self.provider = provider
        self.config = LLM_CONFIG.get(provider, {})
        
        # Get API key based on provider
        if api_key:
            self.api_key = api_key
        elif provider == "deepseek":
            self.api_key = DEEPSEEK_API_KEY
        elif provider == "together":
            self.api_key = TOGETHER_API_KEY
        elif provider == "groq":
            self.api_key = GROQ_API_KEY
        else:
            self.api_key = None
        
        if not self.api_key:
            env_var = f"{provider.upper()}_API_KEY"
            raise ValueError(f"API key not provided. Set {env_var} environment variable.")
        
        # Get base URL
        self.base_url = base_url or self.config.get("base_url")
        if not self.base_url:
            raise ValueError(f"Base URL not configured for provider: {provider}")
        
        # Initialize client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.model = model or self.config.get("model")
        if not self.model:
            raise ValueError(f"Model not configured for provider: {provider}")
        
        # Rate limiting
        self.min_request_interval = MIN_REQUEST_INTERVAL
        self.last_request_time = 0
        
        logger.info(f"Initialized {provider} client with model: {self.model}")
    
    def _wait_for_rate_limit(self):
        """Ensure minimum interval between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using OpenAI-compatible API."""
        # Check cache first
        cache = get_llm_cache()
        cached_response = cache.get(self.model, prompt, system_prompt)
        if cached_response is not None:
            logger.debug(f"Using cached response for {self.model}")
            return cached_response
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        max_retries = MAX_RETRIES
        base_delay = RATE_LIMIT_BASE_DELAY
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.get("temperature", 0.1),
                    max_tokens=self.config.get("max_tokens", 2000)
                )
                
                self.last_request_time = time.time()
                
                # Track usage and cost
                metadata = None
                if response.usage:
                    usage = self.cost_tracker.record_usage(
                        model=self.model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        component=self._current_component
                    )
                    metadata = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "cost": usage.total_cost
                    }
                    logger.debug(
                        f"API call ({self.provider}): {usage.total_tokens} tokens, "
                        f"${usage.total_cost:.4f} (component: {self._current_component})"
                    )
                
                result = response.choices[0].message.content
                
                # Cache the response
                cache.set(self.model, prompt, result, system_prompt, metadata)
                
                return result
                
            except self.RateLimitError as e:
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except self.APIError as e:
                logger.warning(f"{self.provider} API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
                    
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        return ""
    
    def get_model_name(self) -> str:
        return self.model


def create_llm_client(provider: str = "openai", **kwargs) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: One of "openai", "anthropic", "deepseek", "together", "groq"
        **kwargs: Additional arguments passed to the client constructor
        
    Returns:
        LLMClient instance
        
    Examples:
        # OpenAI (GPT-4o)
        client = create_llm_client("openai")
        
        # Anthropic (Claude 3.5 Sonnet)
        client = create_llm_client("anthropic")
        
        # DeepSeek Coder
        client = create_llm_client("deepseek")
        
        # Llama 3.1 70B via Together AI
        client = create_llm_client("together")
        
        # Llama 3.1 70B via Groq (faster, has free tier)
        client = create_llm_client("groq")
    """
    # Direct client classes
    direct_providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient
    }
    
    # OpenAI-compatible providers
    compatible_providers = ["deepseek", "together", "groq"]
    
    if provider in direct_providers:
        return direct_providers[provider](**kwargs)
    elif provider in compatible_providers:
        return OpenAICompatibleClient(provider=provider, **kwargs)
    else:
        all_providers = list(direct_providers.keys()) + compatible_providers
        raise ValueError(f"Unknown provider: {provider}. Available: {all_providers}")


# ============================================================================
# PROMPTS FOR ROOT CAUSE DIAGNOSIS
# ============================================================================

SYSTEM_PROMPT = """You are an expert software engineer specializing in analyzing flaky tests. 
Your task is to identify the root causes of test flakiness based on test code and stack traces.

Focus on identifying:
1. The underlying technical issue causing the flakiness
2. Whether it's related to: networking, resources, configuration, external dependencies, concurrency, timing, or filesystem
3. Specific details that explain WHY the test fails intermittently

Be concise and specific in your diagnosis. Focus on the root cause, not just symptoms."""


INDIVIDUAL_ANALYSIS_PROMPT = """Analyze this flaky test and identify its root cause.

## Test Name
{test_name}

## Test Code
```java
{source_code}
```

## Stack Trace (sample failure)
```
{stack_trace}
```

## Question
What is the root cause of this flaky test failure? Provide a concise diagnosis (2-3 sentences) focusing on the underlying technical issue, not just the symptoms shown in the stack trace."""


AGGREGATION_PROMPT = """The following are root cause diagnoses for {num_tests} flaky tests that consistently fail TOGETHER in the same test runs. They are believed to share a common root cause.

## Individual Diagnoses
{individual_diagnoses}

## Question
Based on these individual diagnoses, what is the most likely SHARED root cause that explains why ALL these tests fail together at the same time?

Provide a concise diagnosis (2-3 sentences) that identifies the common underlying issue."""


COLLECTIVE_ANALYSIS_PROMPT = """The following {num_tests} flaky tests consistently fail TOGETHER in the same test runs. They are believed to share a common root cause.

{test_details}

## Question
These tests fail together in the same test runs. What is the SHARED root cause that explains why ALL of them fail at the same time?

Provide a concise diagnosis (2-3 sentences) focusing on the common underlying issue that causes all these tests to fail together."""


def format_test_for_collective_prompt(test, index: int) -> str:
    """Format a single test for the collective analysis prompt."""
    parts = [f"### Test {index + 1}: {test.name}"]
    
    if test.source_code:
        parts.append(f"\n**Code:**\n```java\n{test.source_code}\n```")
    
    if test.stack_traces:
        parts.append(f"\n**Stack Trace (sample):**\n```\n{test.stack_traces[0]}\n```")
    
    return "\n".join(parts)


def format_individual_diagnoses(diagnoses: list[tuple[str, str]]) -> str:
    """Format individual diagnoses for the aggregation prompt."""
    parts = []
    for i, (test_name, diagnosis) in enumerate(diagnoses):
        parts.append(f"### Test {i + 1}: {test_name}\n{diagnosis}")
    return "\n\n".join(parts)

