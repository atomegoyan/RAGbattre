import ollama
from mistralai import Mistral
import cohere

from .core import MISTRAL_MODEL, mistral_api_key, MODEL_BACKEND, COHERE_MODEL, cohere_api_key

CTX_SIZE = 10000

def format_messages(messages, system=None):
    """
    Formats messages for the LLM API.
    Args:
        messages: List of message dictionaries.
        system: Optional system prompt string.
    Returns:
        List of formatted messages.
    """
    formatted_messages = []
    if system:
        formatted_messages.append({"role": "system", "content": system})
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            formatted_messages.append(msg)
    return formatted_messages

def get_available_models():
    try:
        models = ollama.list()
        return [model['model'] for model in models['models']]
    except Exception as e:
        raise Exception(f"Could not connect to Ollama: {str(e)}")

def get_ollama_response_backup(model, messages, system="", temperature=0.7):
    try:
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                formatted_messages.append(msg)
        opt = ollama.Options(num_ctx=CTX_SIZE, temperature=temperature)
        response = ollama.chat(
            model=model,
            messages=formatted_messages,
            options=opt
        )
        return  response['message']['content']
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")
    
def get_ollama_response(model, messages,context_size = 10000, temperature=0.7):
    try:
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(msg)
        opt = ollama.Options(num_ctx=context_size, temperature=temperature)
        response = ollama.chat(
            model=model,
            messages=formatted_messages,
            options=opt
        )
        return  response['message']['content']
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")
    
def get_ollama_response_mistral_backup(model="no model",messages ="",system="",temperature=0.7):
    client = Mistral(api_key=mistral_api_key)
    formatted_messages = []
    if system:
        formatted_messages.append({"role": "system", "content": system})
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            formatted_messages.append(msg)
    response = client.chat.complete(
        model= MISTRAL_MODEL,
        messages = formatted_messages
    )
    return response.choices[0].message.content

def get_ollama_response_mistral(messages, system="", temperature=0.7):
    try:
        client = Mistral(api_key=mistral_api_key)
        formatted_messages = format_messages(messages, system)
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=formatted_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error in Mistral response: {str(e)}")

def get_ollama_response_cohere_backup(model="no model",messages ="",system="",temperature=0.7):
    client = cohere.ClientV2(cohere_api_key) 
    formatted_messages = []
    if system:
        formatted_messages.append({"role": "system", "content": system})
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            formatted_messages.append(msg)
    response = client.chat(
        model= COHERE_MODEL,
        messages = formatted_messages
    )
    return response.message.content[0].text 

def get_ollama_response_cohere(messages, system="", temperature=0.7):
    try:
        client = cohere.ClientV2(cohere_api_key)
        formatted_messages = format_messages(messages, system)
        response = client.chat(
            model=COHERE_MODEL,
            messages=formatted_messages
        )
        return response.message.content[0].text
    except Exception as e:
        raise Exception(f"Error in Cohere response: {str(e)}")


def get_llm_response_backup(model="", messages="", system="", temperature=0.7):
    """
    Unified function that calls either Ollama or Mistral based on MODEL_BACKEND configuration.
    
    Args:
        model: Model name (used for Ollama, ignored for Mistral)
        messages: List of message dictionaries
        system: System prompt string
        temperature: Temperature for generation
    
    Returns:
        str: Generated response
    """
    if MODEL_BACKEND == "mistral":
        return get_ollama_response_mistral(model=model, messages=messages, system=system, temperature=temperature)
    elif MODEL_BACKEND == "cohere":
        return get_ollama_response_cohere(model=model,messages=messages,system=system,temperature=temperature)
    elif MODEL_BACKEND == "ollama":
        return get_ollama_response(model=model, messages=messages, temperature=temperature)
    else:
        raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}. Must be 'ollama' or 'mistral'")

def get_llm_response(model = "", messages="", system="", temperature=0.7):
    """
    Unified function that calls the appropriate backend based on MODEL_BACKEND.
    """
    if MODEL_BACKEND == "mistral":
        return get_ollama_response_mistral(messages, system, temperature)
    elif MODEL_BACKEND == "cohere":
        return get_ollama_response_cohere(messages, system, temperature)
    elif MODEL_BACKEND == "ollama":
        return get_ollama_response(model=model, messages=messages, temperature=temperature)
    else:
        raise Exception(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")