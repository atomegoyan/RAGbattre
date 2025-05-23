import os
import ast

# Base directories
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
#DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data_extract"))
#CORPUS_DIR = os.path.join(BASE_DIR,"data","corpus")
CORPUS_DIR = os.path.join(DATA_DIR,"corpus")
MAX_CHAR_DISPLAY = 1000
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings_cs1')


# Device configuration
#EMBEDDING_DEVICE = "cuda"
#RERANKING_DEVICE = "cuda"
EMBEDDING_DEVICE = "cpu"
RERANKING_DEVICE = "cpu"

# Default parameters
DEFAULT_COLLECTION = "1881-01-20"
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
#DEFAULT_GENERATION_MODEL = "gemma3:27b"
DEFAULT_GENERATION_MODEL = "llama3.2:1b"
EXAMPLE_QUESTIONS_FILE = os.path.join(DATA_DIR, "questions_strat1.jsonl")




# Prompt configuration
# Default query
DEFAULT_QUERY = "Qui est le président de la séance ?"

SYSTEM_PROMPT_SOURCE = """Tu es un expert en extraction précise d'informations à partir de documents. Ta tâche principale est de localiser avec une précision absolue la source exacte d'une réponse dans un ensemble de documents.

Règles cruciales :
1. Tu dois TOUJOURS renvoyer un dictionnaire Python
2. Le dictionnaire DOIT contenir exactement deux clés :
   - `document_id`: L'identifiant unique du document source
   - `texte_source`: Le texte source EXACT sans aucune modification, correction ou reformulation
3. Si aucune réponse n'est trouvée, les valeurs seront `None`
4. Le texte source doit être copié mot pour mot depuis le document original
5. la source renvoyée doit contenir tout le contexte nécessaire pour répondre à la question"""



def generer_prompt_utilisateur_local(identifiants,documents, query):

    documents_numerotes = [f"Document {i}:\n{doc}" for i, doc in zip(identifiants,documents)]
    
    prompt_utilisateur = f"""Voici les documents à analyser :
{chr(10).join(documents_numerotes)}

Question à résoudre : {query}

Réponds UNIQUEMENT sous forme de dictionnaire Python en respectant strictement les règles suivantes :
- Identifie le document source de la réponse
- Copie le texte source mot pour mot
- Ne modifie JAMAIS le texte original
- La source renvoyée doit contenir suffisament de contexte pour pouvoir répondre à la question
- Retourne un dictionnaire avec `document_id` et `texte_source`

Exemple de format de réponse attendu :
{{
    "id du document": 2,
    "texte_source": "Texte exact copié du document source"
}}
"""
    
    return prompt_utilisateur

def extract_document_data(input_string):
    """
    Extract data from a string containing a Python dictionary-like representation.
    
    Args:
        input_string (str): Input string containing a Python dictionary.
    
    Returns:
        dict: A dictionary containing document_id and texte_source
    """
    try:
        # Remove code block markers and leading/trailing whitespace
        clean_string = input_string.strip('`').strip()
        
        # Remove 'python' identifier if present
        if clean_string.startswith('python\n'):
            clean_string = clean_string[7:]
        
        # Use ast.literal_eval to safely parse the dictionary
        parsed_dict = ast.literal_eval(clean_string.strip())
        
        # Ensure the result is a dictionary
        if not isinstance(parsed_dict, dict):
            raise ValueError("Parsed content is not a dictionary")
        
        # Extract document_id and texte_source
        document_id = parsed_dict.get('document_id')
        texte_source = parsed_dict.get('texte_source')
        
        # If both are None, return the full dictionary
        if document_id is None and texte_source is None:
            return parsed_dict
        
        # Return a dictionary with the extracted values
        return {
            'document_id': document_id,
            'texte_source': texte_source
        }
    
    except (SyntaxError, ValueError, TypeError) as e:
        # If parsing fails, return a dictionary with input as both document_id and texte_source
        return {
            'document_id': input_string,
            'texte_source': input_string
        }