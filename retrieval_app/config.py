import os

# Base directories
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
CORPUS_DIR = os.path.join(BASE_DIR,"data","corpus")
MAX_CHAR_DISPLAY = 1000
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings_cs1')


# Device configuration
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

def generer_prompt_utilisateur(identifiants,documents, query):

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
```python
{{
    "document_id": 2,
    "texte_source": "Texte exact copié du document source"
}}
```"""
    
    return prompt_utilisateur