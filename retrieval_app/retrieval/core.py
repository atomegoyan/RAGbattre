import os
import json
import streamlit as st
import regex as re
import chromadb
import sys
import ast
from chromadb.utils import embedding_functions
sys.path.append(os.getcwd())
from retrieval_app.config import EMBEDDINGS_DIR , MAX_CHAR_DISPLAY
from retrieval_app.config import EMBEDDING_DEVICE , RERANKING_DEVICE

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
reranking_device = torch.device(RERANKING_DEVICE)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
ranking_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3").to(reranking_device)

def initialize_chromadb(collection_name, embedding_model):
    """Initialize ChromaDB with a given collection and embedding model."""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    #st.session_state.client = chromadb.PersistentClient(path=EMBEDDINGS_DIR)

    try:
        collections = get_available_collections(st.session_state.client)
        #st.session_state.available_collections = [col for col in collections]
    except Exception as e:
        st.error(f"Error fetching collections: {str(e)}")
        #st.session_state.available_collections = [collection_name]

    st.session_state.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embedding_model,trust_remote_code=True,device="cpu",normalize_embeddings=True)

    st.session_state.collection = st.session_state.client.get_or_create_collection(
        name=collection_name,
        embedding_function=st.session_state.embedding_function
    )
   

def query_documents(query, n_results=10):
    """Query documents from ChromaDB."""
    try:
        results = st.session_state.collection.query(
            query_texts=query,
            n_results=n_results
        )
        return results["ids"][0], results["documents"][0]
    except Exception as e:
        raise Exception(f"Error querying documents: {str(e)}")
    
def query_documents_reranking(query, n_results=10):
    """Query documents from ChromaDB."""
    try:
        results = st.session_state.collection.query(
            query_texts=query,
            n_results=n_results
        )

        ids_reranked , docs_reranked = rerank_retrieved(query,results["documents"][0],n_rank=n_results)
        return [results["ids"][0][i] for i in ids_reranked] , docs_reranked
#        return results["ids"][0], results["documents"][0]
    except Exception as e:
        raise Exception(f"Error querying documents: {str(e)}")
    
def query_documents_filtered(query, word_to_filter = "",n_results=10):
    """Query documents from ChromaDB."""
    try:
        results = st.session_state.collection.query(
            query_texts=query,
            n_results=n_results,where_document={"$contains":word_to_filter}
        )
        return results["ids"][0], results["documents"][0]
    except Exception as e:
        raise Exception(f"Error querying documents: {str(e)}")
    
def query_documents_regex_filtering(query, regex_pattern = "",n_results=10):
    """Query documents from ChromaDB."""
    filtered_documents = []
    filtered_ids = []
    try:
        results = st.session_state.collection.query(
            query_texts=query,
            n_results=n_results
        )

        pattern = re.compile(regex_pattern)
        for id , doc in zip(results["ids"][0],results['documents'][0]):
            if pattern.search(doc):
                filtered_documents.append(doc)
                filtered_ids.append(id)

        return filtered_ids , filtered_documents
    except Exception as e:
        raise Exception(f"Error querying documents: {str(e)}")

def query_seance(seance,corpus_path) :
    with open(os.path.join(corpus_path,seance+".txt")) as f : 
        text = f.read()
        return text[:MAX_CHAR_DISPLAY]

def get_available_collections(client):
    """Get list of collections from ChromaDB client."""
    try:
        return client.list_collections()
    except Exception as e:
        raise Exception(f"Error getting collections: {str(e)}")

@st.cache_data(show_spinner=False)
def load_example_questions(jsonl_path):
    """Load example questions from a JSONL file."""
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        st.error(f"Error loading example questions: {e}")
        return []

def rerank_retrieved(question,docs,n_rank) : 
    pairs = [[question,docs[i]] for i in range(len(docs))]
    with torch.no_grad():
        inputs = tokenizer(pairs,return_tensors="pt",truncation=True,padding=True).input_ids.to(reranking_device)
        scores = ranking_model(inputs,return_dict=True).logits.view(-1,).float()
        #print(scores)
    similarity_scores = scores.tolist()
    top_k_indices = sorted(range(len(similarity_scores)),key=lambda i : similarity_scores[i],reverse=True)[:n_rank]
    top_k_documents = [docs[i] for i in top_k_indices]
    return top_k_indices , top_k_documents

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