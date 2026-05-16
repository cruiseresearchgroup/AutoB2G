"""
Utilities for interacting with LLMs in the SOCIA system.
"""

import os
import json
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
from utils.api_usage import record_api_call

def load_api_key(key_name: str) -> Optional[str]:
    """
    Load API key from environment or keys.py file
    
    Args:
        key_name: Name of the API key, e.g., "OPENAI_API_KEY"
        
    Returns:
        Optional[str]: The API key value, or None if not found
    """
    env_value = os.environ.get(key_name)
    if env_value:
        return env_value

    try:
        # Import the keys module to access the hardcoded API key
        import keys
        # Return the hardcoded API key
        return getattr(keys, key_name, None)
    except ImportError:
        # Return None if keys.py doesn't exist
        return None

class LLMProvider:
    """
    Base class for LLM providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM provider.
        
        Args:
            config: Configuration dictionary for the LLM provider
        """
        self.config = config
        self.logger = logging.getLogger("SOCIA.LLMProvider")
        
        # Cache effective max tokens to avoid repeated computation
        self.effective_max_tokens = compute_effective_max_tokens(config)
    
    def call(self, prompt: str, reasoning: Optional[Dict[str, Any]] = None) -> str:
        """
        Call the LLM with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            reasoning: Optional reasoning parameters for advanced models
        
        Returns:
            The LLM's response
        """
        raise NotImplementedError("Subclasses must implement this method")


def compute_effective_max_tokens(provider_config: Dict[str, Any]) -> int:
    """
    Compute the effective max tokens based on provider configuration.
    - Responses API: prefer max_output_tokens, fallback to max_tokens, then 4000
    - Chat Completions: prefer max_tokens, fallback to max_output_tokens, then 4000
    """
    try:
        use_responses_api = provider_config.get("use_responses_api", False)
        cfg_max_tokens = provider_config.get("max_tokens")
        cfg_max_output_tokens = provider_config.get("max_output_tokens")
        if use_responses_api:
            effective_max = (
                cfg_max_output_tokens
                if cfg_max_output_tokens is not None
                else (cfg_max_tokens if cfg_max_tokens is not None else 4000)
            )
        else:
            effective_max = (
                cfg_max_tokens
                if cfg_max_tokens is not None
                else (cfg_max_output_tokens if cfg_max_output_tokens is not None else 4000)
            )
        return int(effective_max)
    except Exception:
        return 4000


class OpenAIProvider(LLMProvider):
    """
    LLM provider using OpenAI's API.
    """
    
    def call(self, prompt: str, reasoning: Optional[Dict[str, Any]] = None) -> str:
        """
        Call OpenAI's API with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            reasoning: Optional reasoning parameters for advanced models
        
        Returns:
            The LLM's response
        """
        try:
            import openai
            from openai import OpenAI
            
            # Get API key from keys.py file
            api_key = load_api_key("OPENAI_API_KEY")
            
            # Use API key from config only as fallback
            if not api_key:
                api_key = self.config.get("api_key")
                
            if not api_key:
                self.logger.error("OpenAI API key not found in keys.py")
                return "Error: OpenAI API key not found in keys.py"
            
            # Initialize client
            client = OpenAI(api_key=api_key)
            
            # Configure request parameters
            model = self.config.get("model", "gpt-4o")
            temperature = self.config.get("temperature", 0.7)
            # Use cached effective max tokens
            use_responses_api = self.config.get("use_responses_api", False)
            effective_max = self.effective_max_tokens

            def _extract_from_responses(resp_obj):
                # Prefer unified SDK helper when available
                if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
                    return resp_obj.output_text
                # Fallback: try to navigate common structures
                try:
                    # New SDK formats may expose output as a list of content parts
                    output = getattr(resp_obj, "output", None)
                    if output and isinstance(output, list):
                        content = output[0].get("content") if isinstance(output[0], dict) else None
                        if content and isinstance(content, list) and len(content) > 0:
                            text = content[0].get("text")
                            if isinstance(text, str):
                                return text
                except Exception:
                    pass
                # Last resort: string cast
                return str(resp_obj)

            # Some newer models (e.g., gpt-5 family) require Responses API with
            # `max_output_tokens` instead of Chat Completions `max_tokens`.
            # Strategy:
            # 1) If explicitly configured, call responses API directly.
            # 2) Otherwise, try chat.completions; on 400 unsupported_parameter for
            #    max_tokens, retry via responses API using max_output_tokens.
            try:
                if use_responses_api:
                    # Base parameters for responses API
                    responses_kwargs = {
                        "model": model,
                        "input": [
                            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                        ],
                        # Do not pass temperature for Responses API (some models like gpt-5 don't support it)
                        "max_output_tokens": effective_max,
                    }
                    
                    # If reasoning parameters are provided, use them and adjust max_output_tokens
                    if reasoning:
                        responses_kwargs.update({
                            "reasoning": reasoning,
                            "max_output_tokens": 100000  # Larger output for reasoning models
                        })
                    
                    try:
                        resp = client.responses.create(**responses_kwargs)
                        usage = record_api_call("openai", "responses", model, True)
                        self.logger.info(f"OpenAI API call recorded. Total API calls: {usage['total_calls']}")
                    except Exception as api_err:
                        record_api_call("openai", "responses", model, False, str(api_err))
                        raise
                    return _extract_from_responses(resp)
                else:
                    chat_kwargs = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": effective_max,
                    }
                    try:
                        resp = client.chat.completions.create(**chat_kwargs)
                        usage = record_api_call("openai", "chat.completions", model, True)
                        self.logger.info(f"OpenAI API call recorded. Total API calls: {usage['total_calls']}")
                    except Exception as api_err:
                        record_api_call("openai", "chat.completions", model, False, str(api_err))
                        raise
                    return resp.choices[0].message.content
            except Exception as call_err:
                err_str = str(call_err)
                # Detect the specific parameter error and retry with Responses API
                if "Unsupported parameter: 'max_tokens'" in err_str and "max_output_tokens" in err_str:
                    try:
                        responses_kwargs = {
                            "model": model,
                            "input": [
                                {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                            ],
                            # Do not pass temperature for Responses API retry
                            "max_output_tokens": effective_max,
                        }
                        
                        # Apply reasoning parameters if provided
                        if reasoning:
                            responses_kwargs.update({
                                "reasoning": reasoning,
                                "max_output_tokens": 100000  # Larger output for reasoning models
                            })
                        
                        try:
                            resp = client.responses.create(**responses_kwargs)
                            usage = record_api_call("openai", "responses", model, True)
                            self.logger.info(f"OpenAI API call recorded. Total API calls: {usage['total_calls']}")
                        except Exception as api_err:
                            record_api_call("openai", "responses", model, False, str(api_err))
                            raise
                        return _extract_from_responses(resp)
                    except Exception as resp_err:
                        self.logger.error(f"OpenAI Responses API retry failed: {resp_err}")
                        return f"Error: {str(resp_err)}"
                else:
                    # Other errors: log and return
                    raise
        
        except ImportError:
            self.logger.error("openai package not installed")
            return "Error: openai package not installed"
        
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return f"Error: {str(e)}"


class GeminiProvider(LLMProvider):
    """
    LLM provider using Google's Gemini API.
    """
    
    def call(self, prompt: str, reasoning: Optional[Dict[str, Any]] = None) -> str:
        """
        Call Google's Gemini API with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            reasoning: Optional reasoning parameters (not used by Gemini)
        
        Returns:
            The LLM's response
        """
        try:
            import google.generativeai as genai
            
            # Get API key from keys.py file
            api_key = load_api_key("GEMINI_API_KEY")
            
            # Use API key from config only as fallback
            if not api_key:
                api_key = self.config.get("api_key")
                
            if not api_key:
                self.logger.error("Gemini API key not found in keys.py")
                return "Error: Gemini API key not found in keys.py"
            
            # Configure the Gemini API
            genai.configure(api_key=api_key)
            
            # IMPORTANT: Use the exact model name from config
            if "model" not in self.config:
                self.logger.warning("Model not specified in config, using gemini-pro as fallback")
                model = "models/gemini-pro"
            else:
                # Get model name EXACTLY as specified in config, no default value
                model = self.config.get("model")
            
            temperature = self.config.get("temperature", 0.7)
            max_tokens = self.config.get("max_tokens", 8192)
            
            # Log the full model information
            self.logger.info(f"Using Gemini model: {model}")
            
            # Initialize model
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 0
            }
            
            # Get the generative model
            model_instance = genai.GenerativeModel(model_name=model,
                                            generation_config=generation_config)
            
            # Call the API
            try:
                response = model_instance.generate_content(prompt)
                usage = record_api_call("gemini", "generate_content", model, True)
                self.logger.info(f"Gemini API call recorded. Total API calls: {usage['total_calls']}")
            except Exception as api_err:
                record_api_call("gemini", "generate_content", model, False, str(api_err))
                raise
            
            # Extract response text
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                return ''.join([part.text for part in response.parts if hasattr(part, 'text')])
            else:
                self.logger.warning("Unexpected response format from Gemini")
                return str(response)
        
        except ImportError:
            self.logger.error("google-generativeai package not installed")
            return "Error: google-generativeai package not installed"
        
        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {e}")
            return f"Error: {str(e)}"


class AnthropicProvider(LLMProvider):
    """
    LLM provider using Anthropic's API.
    """
    
    def call(self, prompt: str, reasoning: Optional[Dict[str, Any]] = None) -> str:
        """
        Call Anthropic's API with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            reasoning: Optional reasoning parameters (not used by Anthropic)
        
        Returns:
            The LLM's response
        """
        try:
            from anthropic import Anthropic

            # Get API key from keys.py file
            api_key = load_api_key("ANTHROPIC_API_KEY")
            
            # Use API key from config only as fallback
            if not api_key:
                api_key = self.config.get("api_key")
                
            if not api_key:
                self.logger.error("Anthropic API key not found in keys.py")
                return "Error: Anthropic API key not found in keys.py"

            model = self.config.get("model", "claude-sonnet-4-20250514")
            temperature = self.config.get("temperature", 0.7)
            max_tokens = self.config.get("max_tokens", self.effective_max_tokens)
            timeout = self.config.get("timeout", 180)

            client = Anthropic(api_key=api_key, timeout=timeout)

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                usage = record_api_call("anthropic", "messages", model, True)
                self.logger.info(f"Anthropic API call recorded. Total API calls: {usage['total_calls']}")
            except Exception as api_err:
                record_api_call("anthropic", "messages", model, False, str(api_err))
                raise

            text_parts = []
            for block in getattr(response, "content", []):
                text = getattr(block, "text", None)
                if text:
                    text_parts.append(text)

            if text_parts:
                return "".join(text_parts)

            self.logger.warning("Unexpected response format from Anthropic")
            return str(response)
            
        except ImportError:
            self.logger.error("anthropic package not installed")
            return "Error: anthropic package not installed"
            
        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {e}")
            return f"Error: {str(e)}"


class NvidiaProvider(LLMProvider):
    """
    LLM provider using NVIDIA NIM chat completions.
    """

    def call(self, prompt: str, reasoning: Optional[Dict[str, Any]] = None) -> str:
        """
        Call NVIDIA's chat completions API with the provided prompt.

        Args:
            prompt: The prompt to send to the LLM
            reasoning: Optional reasoning parameters (not used by NVIDIA)

        Returns:
            The LLM's response
        """
        try:
            import requests

            api_key = load_api_key("NVIDIA_API_KEY")
            if not api_key:
                api_key = self.config.get("api_key")

            if not api_key:
                self.logger.error("NVIDIA API key not found in keys.py")
                return "Error: NVIDIA API key not found in keys.py"

            model = self.config.get("model", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
            url = self.config.get("url", "https://integrate.api.nvidia.com/v1/chat/completions")
            temperature = self.config.get("temperature", 0.6)
            max_tokens = self.config.get("max_tokens", self.effective_max_tokens)
            timeout = self.config.get("timeout", 180)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if not response.ok:
                    raise RuntimeError(
                        f"NVIDIA API HTTP {response.status_code}: {response.text[:1000]}"
                    )
                usage = record_api_call("nvidia", "chat.completions", model, True)
                self.logger.info(f"NVIDIA API call recorded. Total API calls: {usage['total_calls']}")
            except Exception as api_err:
                record_api_call("nvidia", "chat.completions", model, False, str(api_err))
                raise

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except ImportError:
            self.logger.error("requests package not installed")
            return "Error: requests package not installed"

        except Exception as e:
            self.logger.error(f"Error calling NVIDIA API: {e}")
            return f"Error: {str(e)}"


class LlamaProvider(LLMProvider):
    """
    LLM provider using local Llama models.
    """
    
    def call(self, prompt: str, reasoning: Optional[Dict[str, Any]] = None) -> str:
        """
        Call a local Llama model with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            reasoning: Optional reasoning parameters (not used by Llama)
        
        Returns:
            The LLM's response
        """
        try:
            # Check model path configuration
            model_path = self.config.get("model_path")
            if not model_path:
                self.logger.error("Llama model path not found in configuration")
                return "Error: Llama model path not configured"
            
            # Implementation for Llama model
            # Currently just returns a placeholder message
            self.logger.warning("Llama provider not yet fully implemented")
            return "Error: Llama provider not yet fully implemented"
            
        except ImportError:
            self.logger.error("llama package not installed")
            return "Error: llama package not installed"
            
        except Exception as e:
            self.logger.error(f"Error calling Llama model: {e}")
            return f"Error: {str(e)}"


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing and development.
    """
    
    def call(self, prompt: str, reasoning: Optional[Dict[str, Any]] = None) -> str:
        """
        Return a mock response based on the prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            reasoning: Optional reasoning parameters (not used by mock)
        
        Returns:
            A mock response
        """
        self.logger.info("Using mock LLM provider")
        
        # Return a simple mock response
        return "This is a mock response from the LLM provider."


def get_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """
    Get an LLM provider based on the provided configuration.
    
    Args:
        config: Configuration dictionary for the LLM provider
    
    Returns:
        An LLM provider instance
    """
    # Get the provider name from the config
    provider_name = os.environ.get("LLM_PROVIDER") or config.get("provider", "mock")
    provider_name = provider_name.lower()
    
    # Load the global config to get provider-specific settings
    try:
        import yaml
        with open("config.yaml", 'r') as f:
            global_config = yaml.safe_load(f)
        
        providers_config = global_config.get("llm_providers", {})
        
        # Get provider-specific config if available
        if provider_name in providers_config:
            provider_config = dict(providers_config.get(provider_name, {}))
            
            # Direct debugging of the actual model name coming from config
            if provider_name == "gemini":
                model_name = provider_config.get('model', 'NO_MODEL_FOUND_IN_CONFIG')
                logging.getLogger("SOCIA.LLMProvider").info(f"Loading Gemini model from config: {model_name}")
        else:
            provider_config = {}

        model_override = os.environ.get("LLM_MODEL")
        if model_override:
            provider_config["model"] = model_override
            logging.getLogger("SOCIA.LLMProvider").info(
                f"Overriding {provider_name} model from LLM_MODEL: {model_override}"
            )
            
        # Create and return the appropriate provider
        if provider_name == "openai":
            return OpenAIProvider(provider_config)
        elif provider_name == "gemini":
            return GeminiProvider(provider_config)
        elif provider_name == "anthropic":
            return AnthropicProvider(provider_config)
        elif provider_name == "nvidia":
            return NvidiaProvider(provider_config)
        elif provider_name == "llama":
            return LlamaProvider(provider_config)
        else:
            return MockProvider(provider_config)
    
    except Exception as e:
        logging.getLogger("SOCIA.LLMProvider").error(f"Error loading provider config: {e}")
        # Fallback to basic provider without specific config
        if provider_name == "openai":
            return OpenAIProvider({})
        elif provider_name == "gemini":
            return GeminiProvider({})
        elif provider_name == "anthropic":
            return AnthropicProvider({})
        elif provider_name == "nvidia":
            return NvidiaProvider({})
        elif provider_name == "llama":
            return LlamaProvider({})
        else:
            return MockProvider({}) 
