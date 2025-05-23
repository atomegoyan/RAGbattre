import sys
import streamlit as st
import ollama
import pandas as pd
import time
import os
import json
import regex as re
import chromadb
from chromadb.utils import embedding_functions
sys.path.append(os.getcwd())
from retrieval_app.retrieval.core import (
    initialize_chromadb,
    query_documents,
    get_available_collections,
    load_example_questions,
    query_seance,
    query_documents_filtered,
    query_documents_regex_filtering,
    query_documents_reranking,
    extract_document_data
)
from retrieval_app.ollama_utils import (
    get_available_models,
    get_ollama_response,
    get_ollama_response_backup
)
from retrieval_app.config import BASE_DIR, DATA_DIR, DEFAULT_QUERY, DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_DIR, EXAMPLE_QUESTIONS_FILE, CORPUS_DIR, \
                                SYSTEM_PROMPT_SOURCE, DEFAULT_GENERATION_MODEL , generer_prompt_utilisateur_local


def main():
    st.set_page_config(
        page_title="Parliamentary Debate Analyzer", 
        page_icon="📝",
        layout="wide"
    )
    
    st.title("Parliamentary Debate Analysis Tool")
    
    # Define application modes
    modes = ["RAG Mode","Document Retrieval", "Chat with Ollama"]
    selected_mode = st.sidebar.radio("Select Mode", modes)
    
    if selected_mode == "Chat with Ollama":
        chat_mode()
    elif selected_mode == "RAG Mode":
        rag_mode()
    else:
        retrieval_mode()

def chat_mode():
    st.subheader("Chat with Ollama LLM")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Chat Configuration")
        
        # Get available models
        try:
            models = get_available_models()
            default_index = 0
        except Exception as e:
            st.error(f"Error connecting to Ollama: {str(e)}")
            models = ["llama3", "mistral", "phi3", "gemma", "mixtral", "llama2"]
            default_index = 0
        
        model = st.selectbox(
            "Choose your Ollama model",
            models,
            index=default_index
        )
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful assistant that analyzes parliamentary debates. Provide clear, concise analysis.",
            height=100
        )
        
        # Test connection button
        if st.button("Test Ollama Connection"):
            try:
                models = get_available_models()
                st.success(f"Connected to Ollama! Available models: {', '.join(models)}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about parliamentary debates..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Call Ollama API
            try:
                with st.spinner("Thinking..."):
                    response = get_ollama_response_backup(
                        model=model,
                        messages=st.session_state.messages,
                        system=system_prompt,
                        temperature=temperature
                    )
                    
                    message_placeholder.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}\n\nMake sure Ollama is running with the selected model.")

def retrieval_mode():
    st.subheader("Document Retrieval Mode")
    
    # Initialize session state
    _initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Retrieval Configuration")
        
        collection_name = _setup_collection_selector()
        n_results, use_regex_filter, use_reranking, regex_pattern = _setup_retrieval_options()
        
        _display_example_questions(collection_name)
        _setup_chromadb_controls(collection_name)
    
    # Display session info in main area
    _display_session_info_main(collection_name)
    
    # Query input and search
    query = _handle_query_input()
    _handle_document_search(query, collection_name, n_results, use_regex_filter, use_reranking, regex_pattern)

def rag_mode():
    st.subheader("RAG Mode: Retrieval Augmented Generation")
    
    # Initialize session state
    _initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("RAG Configuration")
        
        collection_name = _setup_collection_selector()
        model, temperature, n_results, system_prompt, use_reranking = _setup_rag_options()
        
        _display_example_questions(collection_name)
        _setup_chromadb_controls(collection_name)

    # Display session info in main area
    _display_session_info_main(collection_name)
    
    # Query input and RAG generation
    query = _handle_rag_query_input()
    _handle_rag_generation(query, collection_name, model, temperature, n_results, system_prompt, use_reranking)


# Helper functions for cleaner code organization
def _initialize_session_state():
    """Initialize session state variables."""
    if 'client' not in st.session_state:
        st.session_state.client = chromadb.PersistentClient(path=EMBEDDINGS_DIR)
    
    if 'available_collections' not in st.session_state:
        st.session_state.available_collections = [DEFAULT_COLLECTION]
        try:
            collections = get_available_collections(st.session_state.client)
            st.session_state.available_collections = [col.name for col in collections]
        except Exception:
            pass

def _setup_collection_selector():
    """Setup collection selector dropdown."""
    return st.selectbox(
        "Select Collection",
        st.session_state.available_collections,
        index=0
    )

def _setup_retrieval_options():
    """Setup retrieval configuration options."""
    n_results = st.slider("Number of results", min_value=1, max_value=20, value=3)
    use_regex_filter = st.checkbox("Enable regex filtering")
    
    # Show regex pattern input only when filtering is enabled
    regex_pattern = ""
    if use_regex_filter:
        regex_pattern = st.text_input("Enter regex pattern", "")
    
    use_reranking = st.checkbox("Enable document reranking")
    return n_results, use_regex_filter, use_reranking, regex_pattern

def _setup_chromadb_controls(collection_name):
    """Setup ChromaDB initialization controls."""
    if st.button("Initialize ChromaDB"):
        try:
            client, collection, embedding_function = initialize_chromadb(
                collection_name, DEFAULT_EMBEDDING_MODEL, st.session_state.client
            )
            st.session_state.client = client
            st.session_state.collection = collection
            st.session_state.embedding_function = embedding_function
            st.session_state.chroma_initialized = True
            st.success(f"ChromaDB initialized with collection: {collection_name}")
            
            collections = get_available_collections(st.session_state.client)
            st.session_state.available_collections = [col.name for col in collections]
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            st.session_state.chroma_initialized = False

def _display_session_info_main(collection_name):
    """Display session information in main area."""
    with st.expander(f"📖 Débat analysé - Séance du {collection_name}", expanded=False):
        try:
            text = query_seance(collection_name, CORPUS_DIR)
            # Create a scrollable container with fixed height
            st.markdown(
                f"""
                <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                    {text.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error loading session info: {str(e)}")

def _display_example_questions(collection_name):
    """Display example questions for the selected collection."""
    example_questions = load_example_questions(EXAMPLE_QUESTIONS_FILE)
    filtered_questions = [q for q in example_questions if q["file_name"] == collection_name]
    
    if filtered_questions:
        with st.expander("💡 Show Example Questions"):
            for i, q in enumerate(filtered_questions):
                st.markdown(f"**Q{i+1}:** {q['question']}")
                st.markdown(f"**Answer Excerpt:** {q['source']}")
                if st.button(f"Use this question", key=f"use_q_{i}"):
                    st.session_state.query = q["question"]
                    st.session_state.expected_source = q["source"]
                st.markdown("---")
    else:
        st.info("No example questions available for this collection.")

def _handle_query_input():
    """Handle query input and state management."""
    default_query = st.session_state.get("query", "Qui est le président de la séance ?")
    query = st.text_input("Enter your query", value=default_query)
    
    if query != st.session_state.get("query", ""):
        st.session_state.query = query
        st.session_state.expected_source = None
    
    return query

def _ensure_chromadb_initialized(collection_name):
    """Ensure ChromaDB is initialized."""
    if not hasattr(st.session_state, 'chroma_initialized') or not st.session_state.chroma_initialized:
        try:
            client, collection, embedding_function = initialize_chromadb(
                collection_name, DEFAULT_EMBEDDING_MODEL, st.session_state.client
            )
            st.session_state.client = client
            st.session_state.collection = collection
            st.session_state.embedding_function = embedding_function
            st.session_state.chroma_initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            st.session_state.chroma_initialized = False
            return False
    return True

def _handle_document_search(query, collection_name, n_results, use_regex_filter, use_reranking, regex_pattern):
    """Handle document search with different options."""
    if st.button("Search Documents"):
        if not _ensure_chromadb_initialized(collection_name):
            return
        
        try:
            with st.spinner("Searching for relevant documents..."):
                if use_regex_filter and regex_pattern:
                    identifiants, docs = query_documents_regex_filtering(
                        query, st.session_state.collection, regex_pattern, n_results
                    )
                    display_docs_case_sensitive(identifiants, docs, header_message="regex retrieval")
                elif use_reranking:
                    identifiants, docs = query_documents(query, st.session_state.collection, n_results)
                    display_docs_case_sensitive(identifiants, docs, header_message="Naive retrieval")
                    identifiants_ranked, docs_ranked = query_documents_reranking(
                        query, st.session_state.collection, n_results
                    )
                    display_docs_case_sensitive(identifiants_ranked, docs_ranked, header_message="Reranked retrieval")
                else:
                    identifiants, docs = query_documents(query, st.session_state.collection, n_results)
                    display_docs_case_sensitive(identifiants, docs, header_message="Naive retrieval")
                    display_docs_sidebar(identifiants, docs)
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")

def _setup_rag_options():
    """Setup RAG-specific configuration options."""
    model = DEFAULT_GENERATION_MODEL
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    n_results = st.slider("Number of context documents", min_value=1, max_value=20, value=3)
    
    system_prompt = st.text_area(
        "RAG System Prompt",
        value=""""Tu es un assistant utile qui répond aux questions en te basant sur le contexte fourni.
        Si la réponse ne se trouve pas dans le contexte, réponds : Je n'ai pas assez d'informations pour répondre à cette question.""",
        height=150
    )
    
    use_reranking = st.checkbox("Enable document reranking")
    
    return model, temperature, n_results, system_prompt, use_reranking

def _handle_rag_query_input():
    """Handle RAG query input."""
    return st.text_input("Enter your query about the documents", 
                        value=st.session_state.get("query", "Qui est le président de la séance ?"))

def _handle_rag_generation(query, collection_name, model, temperature, n_results, system_prompt, use_reranking):
    """Handle RAG generation process."""
    if st.button("Generate RAG Response"):
        if not _ensure_chromadb_initialized(collection_name):
            return
        
        try:
            with st.spinner("Performing Retrieval Augmented Generation..."):
                # Retrieve documents
                if use_reranking:
                    identifiants, docs = query_documents_reranking(query, st.session_state.collection, n_results)
                else:
                    identifiants, docs = query_documents(query, st.session_state.collection, n_results)
                
                # Display retrieved documents
                _display_retrieved_documents(identifiants, docs)
                
                # Generate and display RAG response
                context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)])
                response = _generate_rag_response(query, context, model, system_prompt, temperature)
                
                st.subheader("RAG Response")
                st.markdown(response)
                
                # Generate and display source information
                _generate_and_display_source(identifiants, docs, query, model, temperature)
        
        except Exception as e:
            st.error(f"Error in RAG generation: {str(e)}")

def _display_retrieved_documents(identifiants, docs):
    """Display the retrieved documents."""
    st.subheader(f"Retrieved Documents (Top {len(docs)})")
    for i, doc in enumerate(docs):
        with st.expander(f"Document {identifiants[i]} || Document {i+1}"):
            st.markdown(doc)
            st.markdown("---")

def _generate_rag_response(query, context, model, system_prompt, temperature):
    """Generate RAG response using the LLM."""
    rag_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}\n\n"}
    ]
    
    return get_ollama_response_backup(
        model=model,
        messages=rag_messages,
        system=system_prompt,
        temperature=temperature
    )

def _generate_and_display_source(identifiants, docs, query, model, temperature):
    """Generate and display source information."""
    st.subheader("RAG Source")
    source_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SOURCE},
        {"role": "user", "content": generer_prompt_utilisateur_local(identifiants, docs, query)}
    ]
    
    source = get_ollama_response_backup(
        model=model,
        messages=source_messages,
        system=SYSTEM_PROMPT_SOURCE,
        temperature=temperature
    )
    
    parsed_source = extract_document_data(source)
    st.markdown(parsed_source)
    
    id_doc, doc = query_documents(parsed_source["texte_source"], st.session_state.collection, 1)
    st.markdown("### doc id")
    st.markdown(id_doc[0])
    st.markdown("### texte")
    st.markdown(doc[0][:100])



def display_docs_old(identifiants,docs,header_message=f"Naive retrieval"):
    st.subheader(f"{header_message} - Top {len(docs)} Results:")
    expected_source = st.session_state.get("expected_source", None)
    found_in_any_doc = False
    for i, doc in enumerate(docs):
        contains_source = False

        if expected_source and expected_source in doc:
            contains_source = True
            found_in_any_doc = True
            highlighted_doc = re.sub(re.escape(expected_source),f"<mark>{expected_source}</mark>",doc,flags=re.IGNORECASE)
        else:
            highlighted_doc = doc

        symbol = "✅" if contains_source else "❌"
        with st.expander(f"{symbol} Document {identifiants[i]} || Document {i+1}"):
            st.markdown(highlighted_doc, unsafe_allow_html=True)
            st.markdown("---")

def display_docs_case_sensitive(identifiants, docs, header_message="Naive retrieval"):
    st.subheader(f"{header_message} - Top {len(docs)} Results:")
    expected_source = st.session_state.get("expected_source", None)

    for i, doc in enumerate(docs):
        if expected_source:
            contains_source = expected_source in doc
            if contains_source:
                # Highlight exact matches only (case-sensitive)
                highlighted_doc = re.sub(
                    re.escape(expected_source), 
                    f"<mark>{expected_source}</mark>", 
                    doc
                )
            else:
                highlighted_doc = doc
            symbol = "✅" if contains_source else "❌"
            label = f"{symbol} Document {identifiants[i]} || Document {i+1}"
        else:
            highlighted_doc = doc
            label = f"Document {identifiants[i]} || Document {i+1}"

        with st.expander(label):
            st.markdown(highlighted_doc, unsafe_allow_html=True)
            st.markdown("---") 
def display_docs_sidebar(identifiants, docs, header_message="Naive retrieval"):
    with st.sidebar:
        st.subheader(f"{header_message} - Top {len(docs)} Results:")
        expected_source = st.session_state.get("expected_source", None)

        for i, doc in enumerate(docs):
            if expected_source:
                contains_source = expected_source in doc
                if contains_source:
                    # Highlight exact matches only (case-sensitive)
                    highlighted_doc = re.sub(
                        re.escape(expected_source), 
                        f"<mark>{expected_source}</mark>", 
                        doc
                    )
                else:
                    highlighted_doc = doc
                symbol = "✅" if contains_source else "❌"
                label = f"{symbol} Document {identifiants[i]} || Doc {i+1}"
            else:
                highlighted_doc = doc
                label = f"Document {identifiants[i]} || Doc {i+1}"

            with st.expander(label):
                # More compact display - truncate long documents
                truncated_doc = highlighted_doc[:500] + "..." if len(highlighted_doc) > 500 else highlighted_doc
                st.markdown(truncated_doc, unsafe_allow_html=True)


if __name__ == "__main__":
    main()