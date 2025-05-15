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
    get_ollama_response
)
from retrieval_app.config import BASE_DIR, DATA_DIR, DEFAULT_QUERY, DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_DIR, EXAMPLE_QUESTIONS_FILE, CORPUS_DIR, \
                                SYSTEM_PROMPT_SOURCE, DEFAULT_GENERATION_MODEL , generer_prompt_utilisateur


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
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0, step=0.1)
        
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
                    response = get_ollama_response(
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
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Retrieval Configuration")
        
        # Input for data directory
        base_dir = BASE_DIR#st.text_input("Base Directory", value=BASE_DIR)
        data_dir = DATA_DIR#st.text_input("Data Directory", value=DATA_DIR)
        embedding_model = DEFAULT_EMBEDDING_MODEL
        embeddings_dir = EMBEDDINGS_DIR
        st.session_state.client = chromadb.PersistentClient(path=embeddings_dir)
        # Input for collection name
        #collection_name = st.text_input("Collection Name", value="1916-05-19")

        if not hasattr(st.session_state, 'available_collections'):
            st.session_state.available_collections = [DEFAULT_COLLECTION]  # Default value
            
            # Try to initialize and get collections if we have client info
            if hasattr(st.session_state, 'client'):
                try:
                    collections = get_available_collections(st.session_state.client)
                    st.session_state.available_collections = [col for col in collections]
                except Exception:
                    pass

        # Display the collection selection dropdown
        collection_name = st.selectbox(
            "Select Collection",
            st.session_state.available_collections,
            index=0
        )

# Load example questions file
        example_path = EXAMPLE_QUESTIONS_FILE
        example_questions = load_example_questions(example_path)

        # Filter based on collection name
        filtered_questions = [q for q in example_questions if q["file_name"] == collection_name]

        if filtered_questions:
            #st.markdown("### 💡 Example Questions for this Collection")
            # with st.expander(f"questions"):
            #     for i, q in enumerate(filtered_questions):
            #         #with st.expander(f"Example {i+1}: {q['question']}"):
            #         #    st.markdown(f"**Answer Excerpt**: {q['source']}")
            #         with st.expander(f"Example {i+1}: {q['question']}"):
            #             st.markdown(f"**Answer Excerpt**: {q['source']}")
            #             if st.button(f"Use this question", key=f"use_q_{i}"):
            #                 st.session_state.query = q["question"]
            #                 st.session_state.expected_source = q["source"]

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
        
        # Number of results to return
        n_results = st.slider("Number of results", min_value=1, max_value=20, value=3)

        use_regex_filter = st.checkbox("Enable regex filtering")
        use_reranking = st.checkbox("Enable document reranking")
        # Text input for regex pattern (if the checkbox is selected)
        regex_pattern = st.text_input("Enter regex pattern (if any)", "")
        
        # Model selection for embedding
        #embedding_model = st.selectbox(
        #    "Embedding Model",
        #    [DEFAULT_EMBEDDING_MODEL],
        #    index=0
        #)
        
        
        # Initialize ChromaDB button
        if st.button("Initialize ChromaDB"):
            try:
                initialize_chromadb( collection_name, embedding_model)
                st.session_state.chroma_initialized = True
                st.success(f"ChromaDB initialized with collection: {collection_name}")
                collections = get_available_collections(st.session_state.client)
                st.session_state.available_collections = [col for col in collections]
                #st.session_state.available_collections = ['aa']
                
            except Exception as e:
                st.error(f"Error initializing ChromaDB: {str(e)}")
                st.session_state.chroma_initialized = False


        st.markdown("### Débat analysé")
        text = query_seance(collection_name,CORPUS_DIR)
        with st.expander(f"Séance du {collection_name}"):
            st.markdown(f"{text}")
    # Query input
    #query = st.text_input("Enter your query", "Qui est le président de la séance ?")
    default_query = st.session_state.get("query", "Qui est le président de la séance ?")
    query = st.text_input("Enter your query", value=default_query)
    
    # If the user typed a new query manually, reset the expected source
    if query != st.session_state.get("query", ""):
        st.session_state.query = query
        st.session_state.expected_source = None

    if st.button("Search Documents"):
        if not hasattr(st.session_state, 'chroma_initialized') or not st.session_state.chroma_initialized:
            try:
                initialize_chromadb(collection_name, embedding_model)
                st.session_state.chroma_initialized = True
            except Exception as e:
                st.error(f"Error initializing ChromaDB: {str(e)}")
                st.session_state.chroma_initialized = False
                return
        
        try:
            with st.spinner("Searching for relevant documents..."):

                if use_regex_filter and regex_pattern:
                    identifiants , docs = query_documents_regex_filtering(query, n_results = n_results , regex_pattern = regex_pattern)
                    display_docs(identifiants,docs,header_message="regex retrieval")
                elif use_reranking:
                    identifiants , docs = query_documents(query,n_results)
                    display_docs(identifiants,docs,header_message="Naive retrieval")
                    identifiants_ranked , docs_ranked = query_documents_reranking(query,n_results)
                    display_docs(identifiants_ranked,docs_ranked,header_message="Reranked retrieval")
                else : 
                    identifiants , docs = query_documents(query,n_results)
                    display_docs(identifiants,docs,header_message="Naive retrieval")
                    display_docs_sidebar(identifiants,docs)

                #for i, doc in enumerate(docs):
                #    with st.expander(f"Document {identifiants[i]} ||Document {i+1}"):
                #        st.markdown(doc)
                #        st.markdown("---")


                


                
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")

def rag_mode():
    st.subheader("RAG Mode: Retrieval Augmented Generation")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("RAG Configuration")
        st.session_state.client = chromadb.PersistentClient(path=EMBEDDINGS_DIR)
        # ChromaDB Collection Selection
        if not hasattr(st.session_state, 'available_collections'):
            st.session_state.available_collections = [DEFAULT_COLLECTION]
            
            # Try to initialize and get collections if we have client info
            if hasattr(st.session_state, 'client'):
                try:
                    collections = get_available_collections(st.session_state.client)
                    st.session_state.available_collections = [col for col in collections]
                except Exception:
                    pass

        collection_name = st.selectbox(
            "Select Collection",
            st.session_state.available_collections,
            index=0
        )

        # Model and LLM Configuration
        # try:
        #     models = get_available_models()
        #     default_index = 0
        # except Exception as e:
        #     st.error(f"Error connecting to Ollama: {str(e)}")
        #     models = ["llama3", "mistral", "phi3", "gemma", "mixtral", "llama2"]
        #     default_index = 0
        
        # model= st.selectbox(
        #     "Choose Ollama model for RAG",
        #     models,
        #     index=default_index
        # )
        model = DEFAULT_GENERATION_MODEL

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        # Retrieval Configuration
        n_results = st.slider("Number of context documents", min_value=1, max_value=20, value=3)
        
        # RAG Prompt Configuration
        system_prompt = st.text_area(
            "RAG System Prompt",
            value=""""Tu es un assistant utile qui répond aux questions en te basant sur le contexte fourni.
            Si la réponse ne se trouve pas dans le contexte, réponds : Je n'ai pas assez d'informations pour répondre à cette question.""",
            height=150
        )

        # Optional Filtering
        use_reranking = st.checkbox("Enable document reranking")

    # Query Input
    query = st.text_input("Enter your query about the documents", 
                           value=st.session_state.get("query", "Qui est le président de la séance ?"))
    
    # Search and RAG Button
    if st.button("Generate RAG Response"):
        # Initialize ChromaDB if not already done
        if not hasattr(st.session_state, 'chroma_initialized') or not st.session_state.chroma_initialized:
            try:
                initialize_chromadb(collection_name, DEFAULT_EMBEDDING_MODEL)
                st.session_state.chroma_initialized = True
            except Exception as e:
                st.error(f"Error initializing ChromaDB: {str(e)}")
                st.session_state.chroma_initialized = False
                return
        
        try:
            with st.spinner("Performing Retrieval Augmented Generation..."):
                # Retrieve documents
                if use_reranking:
                    identifiants, docs = query_documents_reranking(query, n_results)
                else:
                    identifiants, docs = query_documents(query, n_results)
                
                # Prepare context for RAG
                context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)])
                
                # Prepare RAG prompt
                rag_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}\n\n"}
                ]
                
                # Generate response using Ollama
                response = get_ollama_response(
                    model=model,
                    messages=rag_messages,
                    system=system_prompt,
                    temperature=temperature
                )
                
                # Display RAG Response
                st.subheader("RAG Response")
                st.markdown(response)
                
                # Display Retrieved Documents
                st.subheader(f"Retrieved Documents (Top {len(docs)})")
                for i, doc in enumerate(docs):
                    with st.expander(f"Document {identifiants[i]} || Document {i+1}"):
                        st.markdown(doc)
                        st.markdown("---")
                
                st.subheader("RAG Source")
                source_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SOURCE},
                    {"role": "user", "content": generer_prompt_utilisateur(identifiants,docs,query)}
                ]
                source = get_ollama_response(model=model,messages=source_messages,
                                             system=SYSTEM_PROMPT_SOURCE,
                                             temperature=temperature)
                parsed_source = extract_document_data(source)
                st.markdown(parsed_source)
                id_doc , doc = query_documents(query = parsed_source["texte_source"],n_results=1)
                st.markdown("### doc id")
                st.markdown(id_doc[0])
                st.markdown("### texte")
                st.markdown(doc[0][:100])
        
        except Exception as e:
            st.error(f"Error in RAG generation: {str(e)}")



def display_docs(identifiants,docs,header_message=f"Naive retrieval"):
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
def display_docs_sidebar(identifiants,docs,header_message=f"Naive retrieval"):
    with st.sidebar:
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
if __name__ == "__main__":
    main()