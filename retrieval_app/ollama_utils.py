import ollama

CTX_SIZE = 20000
def get_available_models():
    try:
        models = ollama.list()
        return [model['model'] for model in models['models']]
    except Exception as e:
        raise Exception(f"Could not connect to Ollama: {str(e)}")

def get_ollama_response(model, messages, system="", temperature=0.7):
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

def simple(model="gemma3:27b",messages=[{"role":"system","content":"Tu es un assistant"}],temperature=0):
    opt = ollama.Options(num_ctx=CTX_SIZE, temperature=temperature)
    response = ollama.chat(
                model="gemma3:27b",
                messages=messages,
                options=opt
            )
    return response["message"]["content"]