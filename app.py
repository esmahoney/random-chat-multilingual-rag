"""
RAG Document Chat - Streamlit App
A simple chat interface for querying documents using RAG.
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Page configuration
st.set_page_config(
    page_title="Random Chat",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .source-tag {
        display: inline-block;
        background-color: #f0f2f6;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embeddings_and_db():
    """Load embeddings and vector store (cached - doesn't need API key)"""
    
    # Load multilingual embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load FAISS index
    index_path = "notebooks/faiss_index_multilingual"
    if not os.path.exists(index_path):
        st.error(f"FAISS index not found at `{index_path}`")
        st.info("Run the notebook first to create the index.")
        st.stop()
    
    db = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return db


def create_rag_chain(api_key):
    """Create RAG chain with the provided API key"""
    
    # Load cached embeddings and database
    db = load_embeddings_and_db()
    
    # Initialize LLM with user's API key
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    
    # Create RAG chain
    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer. If the answer is not in the context, say so.
If the context is in German, you may translate key points to English in your answer.

Context:
{context}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain


def main():
    # Header
    st.title("Random Chat")
    st.markdown("Ask questions about **Paracetamol** (German) or **ReAct** (English research paper)")
    
    # Sidebar with API key input and info
    with st.sidebar:
        st.header("API Key")
        
        # Check for environment variable as fallback
        env_key = os.getenv("GROQ_API_KEY")
        
        api_key = st.text_input(
            "Groq API Key", 
            type="password",
            placeholder="gsk_...",
            help="Get a free API key at [console.groq.com](https://console.groq.com)"
        )
        
        # Use environment key as fallback if no user key provided
        if not api_key and env_key:
            api_key = env_key
            st.caption("✓ Using default API key")
        
        if not api_key:
            st.warning("Please enter your Groq API key to start chatting")
            st.markdown("""
            **Get a free key:**
            1. Go to [console.groq.com](https://console.groq.com)
            2. Sign up (free)
            3. Create an API key
            4. Paste it above
            """)
        
        st.divider()
        
        st.header("About")
        st.markdown("""
        This RAG (Retrieval-Augmented Generation) system uses:
        
        - **Embeddings**: Multilingual MPNet
        - **Vector Store**: FAISS
        - **LLM**: Llama 3.1 8B (via Groq)
        
        ---
        
        **Documents indexed:**
        - Paracetamol leaflet (German)
        - ReAct research paper (English)
        
        ---
        
        **Example questions:**
        - What is paracetamol used for?
        - What are the side effects?
        - What is ReAct?
        - How does ReAct improve reasoning?
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Stop here if no API key
    if not api_key:
        st.info("Enter your Groq API key in the sidebar to start chatting")
        return
    
    # Load RAG chain with API key
    try:
        with st.spinner("Loading AI model..."):
            rag_chain = create_rag_chain(api_key)
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.info("Please check your API key is valid")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(f"📄 Sources: {message['sources']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    # Get unique sources
                    sources = set(
                        doc.metadata.get('source', 'Unknown').split('/')[-1]
                        for doc in response.get('context', [])
                    )
                    sources_str = ", ".join(sources) if sources else "No sources"
                    
                    # Display response
                    st.markdown(answer)
                    st.caption(f"Sources: {sources_str}")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_str
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
