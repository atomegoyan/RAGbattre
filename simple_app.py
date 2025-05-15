import streamlit as st
import ollama

# Set page configuration
st.set_page_config(page_title="Chat Ollama", layout="centered")

# App title
st.title("Chat Ollama")

# Sidebar for model selection
with st.sidebar:
    model = st.selectbox(
        "Modèle",
        ["gemma3:27b", "mistral", "gemma", "phi", "llama3", "mixtral"],
        index=0
    )
    temperature = st.slider("Température", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
prompt = st.chat_input("Pfffn...")

# When user submits a message
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            # Configure Ollama options
            opt = ollama.Options(num_ctx=context_size, temperature=temperature)
            
            # Send prompt to Ollama
            response = ollama.chat(
                model=model,
                messages=st.session_state.messages,
                options=opt
            )
            
            # Display response
            answer = response["message"]["content"]
            response_placeholder.write(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            response_placeholder.error(f"Erreur: {str(e)}")

# Clear chat button
if st.sidebar.button("Effacer la conversation"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Simple Chat avec Ollama")