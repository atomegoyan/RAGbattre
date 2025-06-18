import os
import json
import regex as re
import chromadb
import sys
import ast
from chromadb.utils import embedding_functions
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from functools import lru_cache
from typing import Optional, Tuple, List

# ==================== CONFIGURATION ====================

# Base directories
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings_cs1')

# Display and processing limits
MAX_CHAR_DISPLAY = 1000

# Device configuration
EMBEDDING_DEVICE = "cpu"
RERANKING_DEVICE = "cpu"

# Default parameters
DEFAULT_COLLECTION = "1881-01-20"
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
DEFAULT_GENERATION_MODEL = "llama3.2:1b"
TEMPERATURE = 0.0

# Model configurations
MISTRAL_MODEL = "mistral-large-latest"
COHERE_MODEL = "command-a-03-2025"

# Model backend configuration - "ollama", "mistral", or "cohere"
MODEL_BACKEND = "mistral"

# File paths
EXAMPLE_QUESTIONS_FILE = os.path.join(DATA_DIR, "questions_strat1.jsonl")

# Default query
DEFAULT_QUERY = "Qui est le président de la séance ?"

# API Keys (use environment variables with fallbacks)
mistral_api_key = os.getenv("MISTRAL_API_KEY", "spJ5ykWOWSiVCjbdf1PvfSX5XPVLse0x")
cohere_api_key = os.getenv("COHERE_API_KEY", "ckYq5LUA0oVWB1Pd9Clv4RW3xhJzb3PgQqHUvzRS")

# System prompts
SYSTEM_PROMPT_SOURCE = """Tu es un expert en extraction précise d'informations à partir de documents. Ta tâche principale est de localiser avec une précision absolue la source exacte d'une réponse dans un ensemble de documents.

Règles cruciales :
1. Tu dois TOUJOURS renvoyer un dictionnaire Python
2. Le dictionnaire DOIT contenir exactement deux clés :
   - `document_id`: L'identifiant unique du document source
   - `texte_source`: Le texte source EXACT sans aucune modification, correction ou reformulation
3. Si aucune réponse n'est trouvée, les valeurs seront `None`
4. Le texte source doit être copié mot pour mot depuis le document original
5. la source renvoyée doit contenir tout le contexte nécessaire pour répondre à la question"""

# ==================== CACHED MODEL MANAGEMENT ====================

@lru_cache(maxsize=1)
def _get_reranking_model():
    """Lazily load and cache the reranking model."""
    device = torch.device(RERANKING_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3").to(device)
    return tokenizer, model, device

# ==================== HELPER FUNCTIONS ====================

def _ensure_client(client: Optional[chromadb.PersistentClient] = None) -> chromadb.PersistentClient:
    """Ensure a ChromaDB client exists, create one if needed."""
    if client is None:
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=EMBEDDINGS_DIR)
    return client

def _create_embedding_function(embedding_model: str) -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Create and configure embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        trust_remote_code=True,
        device=EMBEDDING_DEVICE,
        normalize_embeddings=True
    )

def _execute_query(query: str, collection, n_results: int = 10, 
                  strategy: str = "standard", **kwargs) -> Tuple[List[str], List[str]]:
    """Internal unified query execution with different strategies."""
    try:
        if strategy == "standard":
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results["ids"][0], results["documents"][0]
            
        elif strategy == "filtered":
            word_filter = kwargs.get("word_to_filter", "")
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where_document={"$contains": word_filter}
            )
            return results["ids"][0], results["documents"][0]
            
        elif strategy == "regex_filtered":
            regex_pattern = kwargs.get("regex_pattern", "")
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            pattern = re.compile(regex_pattern)
            filtered_documents = []
            filtered_ids = []
            
            for id_, doc in zip(results["ids"][0], results['documents'][0]):
                if pattern.search(doc):
                    filtered_documents.append(doc)
                    filtered_ids.append(id_)
            
            return filtered_ids, filtered_documents
            
        elif strategy == "reranked":
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            ids_reranked, docs_reranked = _rerank_documents(
                query, results["documents"][0], n_results
            )
            return [results["ids"][0][i] for i in ids_reranked], docs_reranked
            
        else:
            raise ValueError(f"Unknown query strategy: {strategy}")
            
    except Exception as e:
        raise Exception(f"Error in {strategy} query: {str(e)}")

def _rerank_documents(question: str, docs: List[str], n_rank: int) -> Tuple[List[int], List[str]]:
    """Rerank documents using the cached reranking model."""
    tokenizer, ranking_model, device = _get_reranking_model()
    
    pairs = [[question, docs[i]] for i in range(len(docs))]
    
    with torch.no_grad():
        inputs = tokenizer(pairs, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
        scores = ranking_model(inputs, return_dict=True).logits.view(-1,).float()
    
    similarity_scores = scores.tolist()
    top_k_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:n_rank]
    top_k_documents = [docs[i] for i in top_k_indices]
    
    return top_k_indices, top_k_documents

# ==================== PUBLIC API FUNCTIONS ====================
# These maintain exact backward compatibility with original core.py

def initialize_chromadb(collection_name: str, embedding_model: str, 
                       client: Optional[chromadb.PersistentClient] = None) -> Tuple[chromadb.PersistentClient, 
                                                                                   chromadb.Collection, 
                                                                                   embedding_functions.SentenceTransformerEmbeddingFunction]:
    """Initialize ChromaDB with a given collection and embedding model.
    
    Maintains exact backward compatibility with original function.
    """
    client = _ensure_client(client)
    embedding_function = _create_embedding_function(embedding_model)
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    return client, collection, embedding_function

def query_documents(query: str, collection, n_results: int = 10) -> Tuple[List[str], List[str]]:
    """Query documents from ChromaDB.
    
    Maintains exact backward compatibility with original function.
    """
    return _execute_query(query, collection, n_results, strategy="standard")

def query_documents_reranking(query: str, collection, n_results: int = 10) -> Tuple[List[str], List[str]]:
    """Query documents from ChromaDB with reranking.
    
    Maintains exact backward compatibility with original function.
    """
    return _execute_query(query, collection, n_results, strategy="reranked")

def query_documents_filtered(query: str, collection, word_to_filter: str = "", n_results: int = 10) -> Tuple[List[str], List[str]]:
    """Query documents from ChromaDB with word filtering.
    
    Maintains exact backward compatibility with original function.
    """
    return _execute_query(query, collection, n_results, strategy="filtered", word_to_filter=word_to_filter)

def query_documents_regex_filtering(query: str, collection, regex_pattern: str = "", n_results: int = 10) -> Tuple[List[str], List[str]]:
    """Query documents from ChromaDB with regex filtering.
    
    Maintains exact backward compatibility with original function.
    """
    return _execute_query(query, collection, n_results, strategy="regex_filtered", regex_pattern=regex_pattern)

def query_seance(seance: str, corpus_path: str) -> str:
    """Query session text from corpus.
    
    Maintains exact backward compatibility with original function.
    """
    with open(os.path.join(corpus_path, seance + ".txt")) as f:
        text = f.read()
        return text[:MAX_CHAR_DISPLAY]

def get_available_collections(client: chromadb.PersistentClient) -> List:
    """Get list of collections from ChromaDB client.
    
    Maintains exact backward compatibility with original function.
    """
    try:
        return client.list_collections()
    except Exception as e:
        raise Exception(f"Error getting collections: {str(e)}")

def load_example_questions(jsonl_path: str) -> List[dict]:
    """Load example questions from a JSONL file.
    
    Maintains exact backward compatibility with original function.
    """
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading example questions: {e}")
        return []

def rerank_retrieved(question: str, docs: List[str], n_rank: int) -> Tuple[List[int], List[str]]:
    """Rerank retrieved documents.
    
    Maintains exact backward compatibility with original function.
    """
    return _rerank_documents(question, docs, n_rank)

def generer_prompt_utilisateur_local(identifiants: List[str], documents: List[str], query: str) -> str:
    """Generate local user prompt.
    
    Maintains exact backward compatibility with original function.
    """
    documents_numerotes = [f"Document {i}:\n{doc}" for i, doc in zip(identifiants, documents)]
    
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

def extract_document_data(input_string: str) -> dict:
    """Extract data from a string containing a Python dictionary-like representation.
    
    Maintains exact backward compatibility with original function.
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