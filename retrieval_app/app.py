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
    get_ollama_response_backup,
    get_ollama_response_mistral,
    get_llm_response
)
from retrieval_app.config import BASE_DIR, DATA_DIR, DEFAULT_QUERY, DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_DIR, EXAMPLE_QUESTIONS_FILE, CORPUS_DIR, \
                                SYSTEM_PROMPT_SOURCE, DEFAULT_GENERATION_MODEL, generer_prompt_utilisateur_local, MODEL_BACKEND


def main():
    st.set_page_config(
        page_title="Parliamentary Debate Analyzer", 
        page_icon="📝",
        layout="wide"
    )
    
    st.title("Parliamentary Debate Analysis Tool")
    
    # Define application modes
    modes = ["Document Retrieval","RAG Mode", "Chat with Ollama"]
    selected_mode = st.sidebar.radio("Select Mode", modes)
    
    if selected_mode == "Chat with Ollama":
        chat_mode()
    elif selected_mode == "RAG Mode":
        rag_mode()
    else:
        retrieval_mode()

def chat_mode():
    st.subheader("Chat with Ollama LLM")
    
    # Initialize chat session state
    _initialize_chat_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Chat Configuration")
        
        model, temperature, system_prompt = _setup_chat_options()
        _setup_ollama_connection_test()
    
    # Display chat history and handle input
    _display_chat_history()
    _handle_chat_input(model, temperature, system_prompt)

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
    st.markdown("XXX")
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
        _initialize_chromadb_if_needed(collection_name, show_success=True)

def _display_session_info_main(collection_name):
    """Display session information in main area."""
    with st.expander(f"📖 Débat analysé - Séance du {collection_name}", expanded=False):
        try:
            text = query_seance(collection_name, CORPUS_DIR)
            # Create a scrollable container with fixed height
            st.markdown(text)
        except Exception as e:
            st.error(f"Error loading session info: {str(e)}")


def _display_example_questions(collection_name):
    """Display example questions for the selected collection."""
    example_questions = load_example_questions(EXAMPLE_QUESTIONS_FILE)
    filtered_questions = [q for q in example_questions if q["file_name"] == collection_name]
    
    if filtered_questions:
        with st.expander("💡 Show Example Questions"):
            # Create a scrollable container using st.container with height
            container = st.container(height=400)
            
            with container:
                for i, q in enumerate(filtered_questions):
                    # Use columns for a more compact layout with toggles
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        # Show truncated question
                        question_preview = q['question'][:80] + "..." if len(q['question']) > 80 else q['question']
                        st.markdown(f"**Q{i+1}:** {question_preview}")
                    
                    with col2:
                        # Toggle button to show/hide details
                        toggle_key = f"toggle_q_{i}"
                        if st.button("📖", key=toggle_key, help="Show details"):
                            st.session_state[f"show_details_{i}"] = not st.session_state.get(f"show_details_{i}", False)
                    
                    # Show details if toggled
                    if st.session_state.get(f"show_details_{i}", False):
                        st.markdown(f"**Answer Excerpt:** {q['source']}")
                        if st.button(f"Use this question", key=f"use_q_{i}"):
                            st.session_state.query = q["question"]
                            st.session_state.expected_source = q["source"]
                    
                    st.markdown("---")
    else:
        st.info("No example questions available for this collection.")

def _handle_query_input(prompt_text="Enter your query", default_text="Qui est le président de la séance ?"):
    """Handle query input and state management."""
    default_query = st.session_state.get("query", default_text)
    query = st.text_input(prompt_text, value=default_query)
    
    if query != st.session_state.get("query", ""):
        st.session_state.query = query
        st.session_state.expected_source = None
    
    return query

def _initialize_chromadb_if_needed(collection_name, show_success=False):
    """Initialize ChromaDB if not already done."""
    if not hasattr(st.session_state, 'chroma_initialized') or not st.session_state.chroma_initialized:
        try:
            client, collection, embedding_function = initialize_chromadb(
                collection_name, DEFAULT_EMBEDDING_MODEL, st.session_state.client
            )
            st.session_state.client = client
            st.session_state.collection = collection
            st.session_state.embedding_function = embedding_function
            st.session_state.chroma_initialized = True
            
            if show_success:
                st.success(f"ChromaDB initialized with collection: {collection_name}")
                collections = get_available_collections(st.session_state.client)
                st.session_state.available_collections = [col.name for col in collections]
            
            return True
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            st.session_state.chroma_initialized = False
            return False
    return True

def _handle_document_search(query, collection_name, n_results, use_regex_filter, use_reranking, regex_pattern):
    """Handle document search with different options."""
    if st.button("Search Documents"):
        if not _initialize_chromadb_if_needed(collection_name):
            return
        
        try:
            with st.spinner("Searching for relevant documents..."):
                if use_regex_filter and regex_pattern:
                    identifiants, docs = query_documents_regex_filtering(
                        query, st.session_state.collection, regex_pattern, n_results
                    )
                    display_documents(identifiants, docs, "regex retrieval")
                elif use_reranking:
                    identifiants, docs = query_documents(query, st.session_state.collection, n_results)
                    display_documents(identifiants, docs, "Naive retrieval")
                    identifiants_ranked, docs_ranked = query_documents_reranking(
                        query, st.session_state.collection, n_results
                    )
                    display_documents(identifiants_ranked, docs_ranked, "Reranked retrieval")
                else:
                    identifiants, docs = query_documents(query, st.session_state.collection, n_results)
                    display_documents(identifiants, docs, "Naive retrieval")
                    display_documents(identifiants, docs, "Naive retrieval", location="sidebar", truncate=True)
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
    return _handle_query_input("Enter your query about the documents")

def _handle_rag_generation(query, collection_name, model, temperature, n_results, system_prompt, use_reranking):
    """Handle RAG generation process."""
    if st.button("Generate RAG Response"):
        if not _initialize_chromadb_if_needed(collection_name):
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
    display_documents(identifiants, docs, "Retrieved Documents")

def _generate_rag_response(query, context, model, system_prompt, temperature):
    """Generate RAG response using the LLM."""
    rag_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}\n\n"}
    ]
    return get_llm_response(
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
    
    source = get_llm_response(
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


# Chat mode helper functions
def _initialize_chat_session_state():
    """Initialize chat session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def _setup_chat_options():
    """Setup chat configuration options."""
    # Get available models only if using Ollama backend
    if MODEL_BACKEND == "ollama":
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
    else:
        # For Mistral backend, model is predefined
        model = "mistral-large-latest"
        st.info(f"Using Mistral backend with model: {model}")
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant that analyzes parliamentary debates. Provide clear, concise analysis.",
        height=100
    )
    
    return model, temperature, system_prompt

def _setup_ollama_connection_test():
    """Setup connection test button based on backend."""
    if MODEL_BACKEND == "ollama":
        if st.button("Test Ollama Connection"):
            try:
                models = get_available_models()
                st.success(f"Connected to Ollama! Available models: {', '.join(models)}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
    else:
        if st.button("Test Mistral Connection"):
            try:
                # Simple test by making a minimal request
                test_response = get_llm_response(
                    model="",
                    messages=[{"role": "user", "content": "Hello"}],
                    system="",
                    temperature=0.0
                )
                st.success("Connected to Mistral API successfully!")
            except Exception as e:
                st.error(f"Mistral connection failed: {str(e)}")

def _display_chat_history():
    """Display chat messages from history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def _handle_chat_input(model, temperature, system_prompt):
    """Handle chat input and response generation."""
    if prompt := st.chat_input("Ask about parliamentary debates..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            _generate_and_display_chat_response(model, temperature, system_prompt)

def _generate_and_display_chat_response(model, temperature, system_prompt):
    """Generate and display chat response."""
    message_placeholder = st.empty()
    
    try:
        with st.spinner("Thinking..."):
            response = get_llm_response(
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



def display_documents(identifiants, docs, header_message="Results", location="main", truncate=False):
    """Unified document display function for all modes."""
    expected_source = st.session_state.get("expected_source", None)
    
    if location == "sidebar":
        with st.sidebar:
            st.subheader(f"{header_message} - Top {len(docs)} Results:")
            
            for i, doc in enumerate(docs):
                # Process document highlighting and symbols
                if expected_source:
                    contains_source = expected_source in doc
                    if contains_source:
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
                
                # Apply truncation if requested
                if truncate and len(highlighted_doc) > 500:
                    highlighted_doc = highlighted_doc[:500] + "..."
                
                with st.expander(label):
                    st.markdown(highlighted_doc, unsafe_allow_html=True)
    else:
        # Main area display
        st.subheader(f"{header_message} - Top {len(docs)} Results:")
        
        for i, doc in enumerate(docs):
            # Process document highlighting and symbols
            if expected_source:
                contains_source = expected_source in doc
                if contains_source:
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
            
            # Apply truncation if requested
            if truncate and len(highlighted_doc) > 500:
                highlighted_doc = highlighted_doc[:500] + "..."
            
            with st.expander(label):
                st.markdown(highlighted_doc, unsafe_allow_html=True)
                st.markdown("---")


if __name__ == "__main__":
    main()