import ollama
from mistralai import Mistral
import cohere
from functools import lru_cache
from typing import List, Dict, Any, Optional
import os

from .core_v2 import MISTRAL_MODEL, mistral_api_key, MODEL_BACKEND, COHERE_MODEL, cohere_api_key

# ==================== CONFIGURATION ====================

CTX_SIZE = 10000

# ==================== CACHED CLIENT MANAGEMENT ====================

@lru_cache(maxsize=1)
def _get_mistral_client() -> Mistral:
    """Get cached Mistral client instance."""
    return Mistral(api_key=mistral_api_key)

@lru_cache(maxsize=1)
def _get_cohere_client() -> cohere.ClientV2:
    """Get cached Cohere client instance."""
    return cohere.ClientV2(cohere_api_key)

# ==================== HELPER FUNCTIONS ====================

def _format_messages_inline(messages: List[Dict[str, str]], system: Optional[str] = None) -> List[Dict[str, str]]:
    """Inline message formatting - replaces the standalone format_messages function."""
    formatted_messages = []
    if system:
        formatted_messages.append({"role": "system", "content": system})
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            formatted_messages.append(msg)
    return formatted_messages

# ==================== BACKEND-SPECIFIC FUNCTIONS ====================

def _get_ollama_response_internal(model: str, messages: List[Dict[str, str]], 
                                context_size: int = 10000, temperature: float = 0.7) -> str:
    """Internal Ollama response handler."""
    try:
        opt = ollama.Options(num_ctx=context_size, temperature=temperature)
        response = ollama.chat(
            model=model,
            messages=messages,
            options=opt
        )
        return response['message']['content']
    except Exception as e:
        raise Exception(f"Error generating Ollama response: {str(e)}")

def _get_mistral_response_internal(messages: List[Dict[str, str]], 
                                 system: Optional[str] = None, 
                                 temperature: float = 0.7) -> str:
    """Internal Mistral response handler."""
    try:
        client = _get_mistral_client()
        formatted_messages = _format_messages_inline(messages, system)
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=formatted_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error in Mistral response: {str(e)}")

def _get_cohere_response_internal(messages: List[Dict[str, str]], 
                                system: Optional[str] = None, 
                                temperature: float = 0.7) -> str:
    """Internal Cohere response handler."""
    try:
        client = _get_cohere_client()
        formatted_messages = _format_messages_inline(messages, system)
        response = client.chat(
            model=COHERE_MODEL,
            messages=formatted_messages
        )
        return response.message.content[0].text
    except Exception as e:
        raise Exception(f"Error in Cohere response: {str(e)}")

# ==================== PUBLIC API FUNCTIONS ====================
# These maintain exact backward compatibility with original llm_utils.py

def get_available_models() -> List[str]:
    """Get available Ollama models.
    
    Maintains exact backward compatibility with original function.
    """
    try:
        models = ollama.list()
        return [model['model'] for model in models['models']]
    except Exception as e:
        raise Exception(f"Could not connect to Ollama: {str(e)}")

def get_ollama_response(model: str, messages: List[Dict[str, str]], 
                       context_size: int = 10000, temperature: float = 0.7) -> str:
    """Get response from Ollama model.
    
    Maintains exact backward compatibility with original function.
    """
    return _get_ollama_response_internal(model, messages, context_size, temperature)

def get_ollama_response_backup(model: str, messages: List[Dict[str, str]], 
                              system: str = "", temperature: float = 0.7) -> str:
    """Backup Ollama response function with system prompt support.
    
    Maintains exact backward compatibility with original function.
    """
    try:
        formatted_messages = _format_messages_inline(messages, system if system else None)
        opt = ollama.Options(num_ctx=CTX_SIZE, temperature=temperature)
        response = ollama.chat(
            model=model,
            messages=formatted_messages,
            options=opt
        )
        return response['message']['content']
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

def get_ollama_response_mistral(messages: List[Dict[str, str]], 
                               system: str = "", temperature: float = 0.7) -> str:
    """Get response from Mistral API.
    
    Maintains exact backward compatibility with original function.
    """
    return _get_mistral_response_internal(messages, system if system else None, temperature)

def get_ollama_response_mistral_backup(model: str = "no model", messages: Any = "", 
                                     system: str = "", temperature: float = 0.7) -> str:
    """Backup Mistral response function.
    
    Maintains exact backward compatibility with original function.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return _get_mistral_response_internal(messages, system if system else None, temperature)

def get_ollama_response_cohere(messages: List[Dict[str, str]], 
                              system: str = "", temperature: float = 0.7) -> str:
    """Get response from Cohere API.
    
    Maintains exact backward compatibility with original function.
    """
    return _get_cohere_response_internal(messages, system if system else None, temperature)

def get_ollama_response_cohere_backup(model: str = "no model", messages: Any = "", 
                                    system: str = "", temperature: float = 0.7) -> str:
    """Backup Cohere response function.
    
    Maintains exact backward compatibility with original function.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return _get_cohere_response_internal(messages, system if system else None, temperature)

def get_llm_response_backup(model: str = "", messages: Any = "", 
                           system: str = "", temperature: float = 0.7) -> str:
    """Unified backup function that calls appropriate backend.
    
    Maintains exact backward compatibility with original function.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    if MODEL_BACKEND == "mistral":
        return get_ollama_response_mistral_backup(model, messages, system, temperature)
    elif MODEL_BACKEND == "cohere":
        return get_ollama_response_cohere_backup(model, messages, system, temperature)
    elif MODEL_BACKEND == "ollama":
        return get_ollama_response_backup(model, messages, system, temperature)
    else:
        raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}. Must be 'ollama', 'mistral', or 'cohere'")

def get_llm_response(model: str = "", messages: Any = "", 
                    system: str = "", temperature: float = 0.7) -> str:
    """Unified function that calls the appropriate backend based on MODEL_BACKEND.
    
    Maintains exact backward compatibility with original function.
    """
    # Handle string messages for backward compatibility
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    if MODEL_BACKEND == "mistral":
        return _get_mistral_response_internal(messages, system if system else None, temperature)
    elif MODEL_BACKEND == "cohere":
        return _get_cohere_response_internal(messages, system if system else None, temperature)
    elif MODEL_BACKEND == "ollama":
        return _get_ollama_response_internal(model, messages, CTX_SIZE, temperature)
    else:
        raise Exception(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")

# ==================== DEPRECATED FUNCTIONS ====================
# Kept for full backward compatibility but implemented using new internals

def format_messages(messages: List[Dict[str, str]], system: Optional[str] = None) -> List[Dict[str, str]]:
    """Deprecated: Use _format_messages_inline instead.
    
    Kept for backward compatibility only.
    """
    return _format_messages_inline(messages, system)