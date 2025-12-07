"""
RAG Document Chat - Streamlit App
A simple chat interface for querying documents using RAG.
"""

import streamlit as st
from dotenv import load_dotenv
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
def load_rag_chain():
    """Load embeddings, vector store, and create RAG chain (cached)"""
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in environment variables!")
        st.info("Create a `.env` file with your GROQ_API_KEY")
        st.stop()
    
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
    
    # Initialize LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
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
    
    # Load RAG chain
    with st.spinner("Loading AI model..."):
        rag_chain = load_rag_chain()
    
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
    
    # Sidebar with info
    with st.sidebar:
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


if __name__ == "__main__":
    main()

