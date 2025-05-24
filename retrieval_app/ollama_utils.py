import ollama
from mistralai import Mistral

from config import MISTRAL_MODEL , mistral_api_key

CTX_SIZE = 10000



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
    
def get_ollama_response_mistral(model="no model",messages ="",system="",temperature=0.7):
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

