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
from retrieval_app.core import (
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
from retrieval_app.llm_utils import (
    get_available_models,
    get_ollama_response,
    get_ollama_response_backup,
    get_ollama_response_mistral,
    get_llm_response
)
from retrieval_app.core import BASE_DIR, DATA_DIR, DEFAULT_QUERY, DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_DIR, EXAMPLE_QUESTIONS_FILE, CORPUS_DIR, \
                                SYSTEM_PROMPT_SOURCE, DEFAULT_GENERATION_MODEL, generer_prompt_utilisateur_local, MODEL_BACKEND, TEMPERATURE


def main():
    st.set_page_config(
        page_title="Parliamentary Debate Analyzer", 
        page_icon="📝",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    .main > div {
        padding-top: 1.5rem;
    }
    h2 {
        margin-top: 0.5rem !important;
        padding-top: 0.5rem !important;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2a5298;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        border: none;
        border-radius: 25px;
        font-weight: 600;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## Rabattre")
    
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
    # Initialize chat session state
    _initialize_chat_session_state()
    
    # Sidebar configuration with organized sections
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Configuration Section
        with st.expander("📋 Model Settings", expanded=True):
            model, system_prompt = _setup_chat_options_enhanced()
        
        # Connection Testing Section
        with st.expander("🔧 Connection", expanded=False):
            _setup_ollama_connection_test()
    
    # Main chat area
    _display_chat_history()
    _handle_chat_input(model, system_prompt)

def retrieval_mode():
    #st.markdown("### Document Retrieval Mode")
    
    # Initialize session state
    _initialize_session_state()
    
    # Sidebar configuration with organized sections
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Basic Configuration Section
        with st.expander("📋 Basic Settings", expanded=True):
            collection_name = _setup_collection_selector()
            n_results = st.slider("Number of results", min_value=1, max_value=20, value=3)
        
        # Advanced Options Section
        with st.expander("🔧 Advanced Options", expanded=False):
            use_regex_filter, regex_pattern = _setup_regex_options()
            use_reranking = st.checkbox("🎯 Enable document reranking")
        
        # Auto-initialize ChromaDB
        _auto_initialize_chromadb(collection_name)
        
        # Example Questions Section
        _display_example_questions(collection_name)
    
    # Main area with better organization
    # 1. Query Input (prominent position)
    #st.markdown("#### Search Query")
    query = _handle_query_input(prompt_text="", default_text="Qui est le président de la séance ?")
    
    # 2. Search Results
    _handle_document_search_enhanced(query, collection_name, n_results, use_regex_filter, use_reranking, regex_pattern)
    
    # 3. Session Info (moved to bottom)
    st.markdown("---")
    _display_session_info_main(collection_name)

def rag_mode():
    # Initialize session state
    _initialize_session_state()
    
    # Sidebar configuration with organized sections
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Basic Configuration Section
        with st.expander("📋 Basic Settings", expanded=True):
            collection_name = _setup_collection_selector()
            model = _setup_model_selector()
            n_results = st.slider("Number of context documents", min_value=1, max_value=20, value=3)
        
        # Advanced Options Section
        with st.expander("🔧 Advanced Options", expanded=False):
            system_prompt = _setup_system_prompt()
            use_reranking = st.checkbox("🎯 Enable document reranking")
        
        # Auto-initialize ChromaDB
        _auto_initialize_chromadb(collection_name)
        
        # Example Questions Section
        _display_example_questions(collection_name)

    # Main area with better organization
    # 1. Query Input (prominent position)
    query = _handle_query_input(prompt_text="", default_text="Posez votre question sur les documents")
    
    # 2. RAG Generation
    _handle_rag_generation_enhanced(query, collection_name, model, n_results, system_prompt, use_reranking)
    
    # 3. Session Info (moved to bottom)
    st.markdown("---")
    _display_session_info_main(collection_name)


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

def _setup_regex_options():
    """Setup regex filtering options with smart suggestions."""
    use_regex_filter = st.checkbox("🔎 Enable regex filtering")
    
    regex_pattern = ""
    if use_regex_filter:
        # Smart regex suggestions
        regex_suggestions = {
            "Speaker mentions": r"M\. [A-Z][a-z]+",
            "Questions": r"\?",
            "Votes": r"vote|voter|votation",
            "Custom": ""
        }
        
        suggestion = st.selectbox("Choose pattern or custom:", list(regex_suggestions.keys()))
        
        if suggestion == "Custom":
            regex_pattern = st.text_input("📝 Enter custom regex pattern:", "")
        else:
            regex_pattern = regex_suggestions[suggestion]
            st.code(f"Pattern: {regex_pattern}")
    
    return use_regex_filter, regex_pattern

def _auto_initialize_chromadb(collection_name):
    """Auto-initialize ChromaDB with status indicator."""
    if not hasattr(st.session_state, 'chroma_initialized') or not st.session_state.chroma_initialized:
        with st.spinner("⚙️ Initializing ChromaDB..."):
            success = _initialize_chromadb_if_needed(collection_name, show_success=False)
            if success:
                st.success("✓ ChromaDB ready")
            else:
                st.error("❌ ChromaDB initialization failed")
                if st.button("🔄 Retry Initialization"):
                    st.rerun()
    else:
        st.success("✓ ChromaDB connected")

def _display_session_info_main(collection_name):
    """Display session information in main area with better styling."""
    st.markdown("#### Session Information")
    with st.expander(f"📖 Débat analysé - Séance du {collection_name}", expanded=False):
        try:
            text = query_seance(collection_name, CORPUS_DIR)
            
            # Info header
            st.info(f"📅 Session: {collection_name}")
            
            # Scrollable content
            container = st.container(height=400)
            with container:
                st.markdown(text)
                
        except Exception as e:
            st.error(f"❌ Error loading session info: {str(e)}")


def _display_example_questions(collection_name):
    """Display example questions for the selected collection with enhanced styling."""
    example_questions = load_example_questions(EXAMPLE_QUESTIONS_FILE)
    filtered_questions = [q for q in example_questions if q["file_name"] == collection_name]
    
    if filtered_questions:
        with st.expander(f"💡 Example Questions ({len(filtered_questions)} available)", expanded=False):
            # Create a scrollable container using st.container with height
            container = st.container(height=350)
            
            with container:
                for i, q in enumerate(filtered_questions):
                    # Use columns for a more compact layout with toggles
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        # Show truncated question with better formatting
                        question_preview = q['question'][:70] + "..." if len(q['question']) > 70 else q['question']
                        st.markdown(f"**Q{i+1}:** {question_preview}")
                    
                    with col2:
                        # Toggle button to show/hide details
                        toggle_key = f"toggle_q_{i}"
                        if st.button("👁️", key=toggle_key, help="Show details", use_container_width=True):
                            st.session_state[f"show_details_{i}"] = not st.session_state.get(f"show_details_{i}", False)
                    
                    # Show details if toggled
                    if st.session_state.get(f"show_details_{i}", False):
                        st.markdown(f"**📝 Answer Excerpt:** {q['source'][:200]}...")
                        if st.button(f"✅ Use this question", key=f"use_q_{i}", type="secondary"):
                            st.session_state.query = q["question"]
                            st.session_state.expected_source = q["source"]
                            st.success("✓ Question loaded!")
                    
                    st.divider()
    else:
        st.info("ℹ️ No example questions available for this collection.")

def _handle_query_input(prompt_text="Enter your query", default_text="Qui est le président de la séance ?"):
    """Handle query input and state management."""
    default_query = st.session_state.get("query", default_text)
    
    # If prompt_text is empty, use placeholder instead of label
    if prompt_text.strip() == "":
        query = st.text_input("query", value=default_query, placeholder=default_text, label_visibility="collapsed")
    else:
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

def _handle_document_search_enhanced(query, collection_name, n_results, use_regex_filter, use_reranking, regex_pattern):
    """Enhanced document search with better UI and no duplications."""
    
    # Search button with better styling
    search_col, status_col = st.columns([2, 3])
    
    with search_col:
        search_clicked = st.button("🔍 Search Documents", type="primary", use_container_width=True)
    
    if search_clicked and query.strip():
        if not _initialize_chromadb_if_needed(collection_name):
            return
        
        try:
            with st.spinner("🔍 Searching for relevant documents..."):
                # Determine search strategy
                if use_regex_filter and regex_pattern:
                    strategy = "🔎 Regex Filtered Search"
                    identifiants, docs = query_documents_regex_filtering(
                        query, st.session_state.collection, regex_pattern, n_results
                    )
                    _display_search_results(identifiants, docs, strategy, use_reranking)
                    
                elif use_reranking:
                    # Show both naive and reranked results in tabs
                    strategy = "🎯 Reranked Search"
                    
                    # Get initial results
                    identifiants_naive, docs_naive = query_documents(query, st.session_state.collection, n_results)
                    
                    # Get reranked results
                    identifiants_ranked, docs_ranked = query_documents_reranking(
                        query, st.session_state.collection, n_results
                    )
                    
                    # Display in tabs
                    tab1, tab2 = st.tabs(["📄 Standard Results", "🎯 Reranked Results"])
                    
                    with tab1:
                        display_documents(identifiants_naive, docs_naive, "Standard Search Results")
                    
                    with tab2:
                        display_documents(identifiants_ranked, docs_ranked, "Reranked Search Results")
                        
                else:
                    strategy = "📄 Standard Search"
                    identifiants, docs = query_documents(query, st.session_state.collection, n_results)
                    _display_search_results(identifiants, docs, strategy, False)
                    
                # Show search summary
                #with status_col:
                #    st.success(f"✓ Found {len(docs) if 'docs' in locals() else 0} documents")
                    
        except Exception as e:
            st.error(f"❌ Error querying documents: {str(e)}")
    
    elif search_clicked and not query.strip():
        st.warning("⚠️ Please enter a search query")

def _display_search_results(identifiants, docs, strategy, show_sidebar_preview=False):
    """Display search results with optional sidebar preview."""
    display_documents(identifiants, docs, strategy)
    
    # Optional sidebar preview for standard search
    if show_sidebar_preview and len(docs) > 0:
        with st.sidebar:
            st.markdown("### 👁️ Quick Preview")
            with st.container(height=300):
                for i, doc in enumerate(docs[:3]):  # Show top 3 in sidebar
                    st.markdown(f"**Doc {i+1}:** {doc[:100]}...")
                    st.markdown("---")

def _setup_model_selector():
    """Setup model selection for RAG mode."""
    return DEFAULT_GENERATION_MODEL

def _setup_system_prompt():
    """Setup system prompt configuration."""
    return st.text_area(
        "System Prompt",
        value=""""Tu es un assistant utile qui répond aux questions en te basant sur le contexte fourni.
        Si la réponse ne se trouve pas dans le contexte, réponds : Je n'ai pas assez d'informations pour répondre à cette question.""",
        height=120
    )

def _handle_rag_generation_enhanced(query, collection_name, model, n_results, system_prompt, use_reranking):
    """Enhanced RAG generation with better UI."""
    
    # Generate button with better styling
    generate_col, status_col = st.columns([2, 3])
    
    with generate_col:
        generate_clicked = st.button("🤖 Generate Response", type="primary", use_container_width=True)
    
    if generate_clicked and query.strip():
        if not _initialize_chromadb_if_needed(collection_name):
            return
        
        try:
            with st.spinner("🤖 Performing Retrieval Augmented Generation..."):
                # Retrieve documents
                if use_reranking:
                    # Get both standard and reranked results for comparison
                    identifiants_standard, docs_standard = query_documents(query, st.session_state.collection, n_results)
                    identifiants_reranked, docs_reranked = query_documents_reranking(query, st.session_state.collection, n_results)
                    
                    # Display in tabs for comparison
                    tab1, tab2 = st.tabs(["📄 Standard Results", "🎯 Reranked Results"])
                    
                    with tab1:
                        display_documents(identifiants_standard, docs_standard, "Standard Retrieval - Context Documents")
                        
                        # Generate RAG response with standard results
                        context_standard = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs_standard)])
                        response_standard = _generate_rag_response(query, context_standard, model, system_prompt)
                        
                        st.markdown("### 🤖 Generated Response (Standard)")
                        st.markdown(response_standard)
                    
                    with tab2:
                        display_documents(identifiants_reranked, docs_reranked, "Reranked Retrieval - Context Documents")
                        
                        # Generate RAG response with reranked results
                        context_reranked = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs_reranked)])
                        response_reranked = _generate_rag_response(query, context_reranked, model, system_prompt)
                        
                        st.markdown("### 🤖 Generated Response (Reranked)")
                        st.markdown(response_reranked)
                    
                    # Use reranked results for source information
                    identifiants, docs = identifiants_reranked, docs_reranked
                    
                else:
                    identifiants, docs = query_documents(query, st.session_state.collection, n_results)
                    
                    # Display retrieved documents
                    display_documents(identifiants, docs, "📄 Standard Retrieval - Context Documents")
                    
                    # Generate and display RAG response
                    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)])
                    response = _generate_rag_response(query, context, model, system_prompt)
                    
                    st.markdown("### 🤖 Generated Response")
                    st.markdown(response)
                
                # Generate and display source information
                _generate_and_display_source(identifiants, docs, query, model)
                
                # Show generation summary
                #with status_col:
                #    st.success(f"✓ Generated from {len(docs)} documents")
        
        except Exception as e:
            st.error(f"❌ Error in RAG generation: {str(e)}")
    
    elif generate_clicked and not query.strip():
        st.warning("⚠️ Please enter a question")

def _display_retrieved_documents(identifiants, docs):
    """Display the retrieved documents."""
    display_documents(identifiants, docs, "Retrieved Documents")

def _generate_rag_response(query, context, model, system_prompt):
    """Generate RAG response using the LLM."""
    rag_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}\n\n"}
    ]
    return get_llm_response(
        model=model,
        messages=rag_messages,
        system=system_prompt,
        temperature=TEMPERATURE
    )

def _generate_and_display_source(identifiants, docs, query, model):
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
        temperature=TEMPERATURE
    )
    
    parsed_source = extract_document_data(source)
    st.markdown(parsed_source)
    
    #id_doc, doc = query_documents(parsed_source["texte_source"], st.session_state.collection, 1)
    #st.markdown("### doc id")
    #st.markdown(id_doc[0])
    #st.markdown("### texte")
    #st.markdown(doc[0][:100])


# Chat mode helper functions
def _initialize_chat_session_state():
    """Initialize chat session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def _setup_chat_options_enhanced():
    """Setup enhanced chat configuration options."""
    # Get available models only if using Ollama backend
    if MODEL_BACKEND == "ollama":
        try:
            models = get_available_models()
            default_index = 0
            st.success("✓ Ollama connected")
        except Exception as e:
            st.error(f"❌ Error connecting to Ollama: {str(e)}")
            models = ["llama3", "mistral", "phi3", "gemma", "mixtral", "llama2"]
            default_index = 0
        
        model = st.selectbox(
            "🤖 Choose your Ollama model",
            models,
            index=default_index
        )
    else:
        # For Mistral backend, model is predefined
        model = "mistral-large-latest"
        st.info(f"🔗 Using Mistral backend with model: {model}")
    
    system_prompt = st.text_area(
        "📝 System Prompt",
        value="You are a helpful assistant that analyzes parliamentary debates. Provide clear, concise analysis.",
        height=100
    )
    
    return model, system_prompt

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
                    temperature=TEMPERATURE
                )
                st.success("Connected to Mistral API successfully!")
            except Exception as e:
                st.error(f"Mistral connection failed: {str(e)}")

def _display_chat_history():
    """Display chat messages from history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def _handle_chat_input(model, system_prompt):
    """Handle chat input and response generation."""
    if prompt := st.chat_input("Ask about parliamentary debates..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            _generate_and_display_chat_response(model, system_prompt)

def _generate_and_display_chat_response(model, system_prompt):
    """Generate and display chat response."""
    message_placeholder = st.empty()
    
    try:
        with st.spinner("Thinking..."):
            response = get_llm_response(
                model=model,
                messages=st.session_state.messages,
                system=system_prompt,
                temperature=TEMPERATURE
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