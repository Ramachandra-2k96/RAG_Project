import streamlit as st

# Page config MUST be first
st.set_page_config(page_title="Chat with Ramachandra", page_icon="💬", layout="wide")

from dotenv import load_dotenv
import os
from langchain_cerebras import ChatCerebras
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Custom CSS for better UI
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .tool-message {
        background: #262630;
        color: white;
        padding: 12px 16px;
        border-radius: 12px;
        width: 50%;
        overflow: hidden;
        margin: 8px 0;
        font-size: 14px;
        border-left: 4px solid #4CAF50;
    }
    
    .tool-header {
        font-weight: bold;
        font-size: 13px;
        color: #E8F5E8;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 1rem 0;
    }
    
    .user-message {
        background: #303030;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px auto;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .assistant-message {
        background: #212121;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px auto 8px 0;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .thinking-indicator {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 10px 15px;
        border-radius: 20px;
        margin: 8px auto 8px 0;
        max-width: 80%;
        color: #666;
        font-style: italic;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .title-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# EXACT SAME CODE AS ORIGINAL - START
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("CEREBRAS_API_KEY")
from langchain_cerebras import ChatCerebras

llm = ChatCerebras(
    model="gpt-oss-120b",
)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

@st.cache_resource
def get_vector_store(_embeddings):
    vector_store = Chroma(
        collection_name="Resume_collection",
    embedding_function=_embeddings,
        persist_directory="./chroma_langchain_db", 
    )

    # Only populate if empty
    if vector_store._chroma_collection.count() == 0:
        # PDF
        pdf_loader = PyPDFLoader("Resume-Ramachandra_udupa.pdf")
        pdf_docs = pdf_loader.load()

        # Website
        web_loader = WebBaseLoader(
            web_paths=(
                'https://game-rama-portfolio.netlify.app/',
                'https://ramachandra.tech/',
                'https://ramachandra.tech/services',
                'https://ramachandra.tech/resume',
                'https://ramachandra.tech/work'
            ),
            requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}},
        )
        web_docs = web_loader.load()

        # Split all docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(pdf_docs + web_docs)

        # Add to DB
        vector_store.add_documents(all_splits)

    return vector_store

from langchain_core.tools import tool

# CRITICAL: These must be at module level like original
if 'retrieval_cache' not in st.session_state:
    st.session_state.retrieval_cache = {}
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = get_vector_store(embeddings)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    if query in st.session_state.retrieval_cache:
        return st.session_state.retrieval_cache[query]

    retrieved_docs = st.session_state.vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    st.session_state.retrieval_cache[query] = (serialized, retrieved_docs)
    return st.session_state.retrieval_cache[query]

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

set_llm_cache(InMemoryCache()) 

memory = MemorySaver()

# Initialize agent_executor in session state
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = create_react_agent(
        llm, 
        [retrieve], 
        checkpointer=memory
    )

system_prompt = """
You are Ramachandra Udupa. Speak only from your own perspective using information from your portfolio.
Never break character, never admit you are an AI, and never answer unrelated questions (except brief casual or humorous chats in your style).

Keep answers as short and natural as a real conversation. Do not add extra details unless the user directly asks for them. 
Answer only what is asked, clearly and concisely. Never fabricate information. Never answer unrelated questions.
Use tools only when needed for accuracy.
"""

initial_message = {'role':'system','content': system_prompt}
thread_id = "def234"

# CRITICAL: Initialize with system message like original - only run once
if 'system_initialized' not in st.session_state:
    st.session_state.agent_executor.invoke(
        {"messages": [initial_message]},
        config={"configurable": {"thread_id": thread_id}}
    )
    st.session_state.system_initialized = True
    st.session_state.thread_id = thread_id

# EXACT SAME CODE AS ORIGINAL - END

# Initialize session state for UI
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown('<h1 class="title-header">💬 Ramachandra Udupa</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Chat with me about my projects, experience, and portfolio</p>', unsafe_allow_html=True)

# Chat container
chat_container = st.container()

# Display messages
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "tool":
            st.markdown(f'''
            <div class="tool-message">
                <div class="tool-header">🔍 Searching my portfolio...</div>
                <div style="font-size: 12px; opacity: 0.9;">Query: {message.get("query", "")}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:  # assistant
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything about my work, projects, or experience..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
    
    # Show thinking indicator
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown('<div class="thinking-indicator">🤔 Thinking...</div>', unsafe_allow_html=True)
    
    # Process with agent - EXACTLY like original code
    try:
        tool_used = False
        final_response = ""
        
        for event in st.session_state.agent_executor.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="values",
            config={"configurable": {"thread_id": st.session_state.thread_id}},
        ):
            last_message = event["messages"][-1]
            
            # Check for tool usage
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                if not tool_used:
                    thinking_placeholder.empty()
                    st.session_state.messages.append({
                        "role": "tool", 
                        "content": "Searching portfolio...",
                        "query": prompt
                    })
                    st.markdown(f'''
                    <div class="tool-message">
                        <div class="tool-header">🔍 Searching my portfolio...</div>
                        <div style="font-size: 12px; opacity: 0.9;">Query: {prompt}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    tool_used = True
            
            # Get final response
            if hasattr(last_message, 'content') and last_message.content:
                final_response = last_message.content
        
        # Clear thinking indicator and show final response
        thinking_placeholder.empty()
        
        if final_response:
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            st.markdown(f'<div class="assistant-message">{final_response}</div>', unsafe_allow_html=True)
        else:
            error_msg = "Sorry, I couldn't generate a response. Please try again."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.markdown(f'<div class="assistant-message">{error_msg}</div>', unsafe_allow_html=True)
            
    except Exception as e:
        thinking_placeholder.empty()
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer controls
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.messages = []
        # Reset the system initialization to reinitialize agent
        if 'system_initialized' in st.session_state:
            del st.session_state.system_initialized
        st.rerun()

with col3:
    st.info(f"💬 {len([m for m in st.session_state.messages if m['role'] in ['user', 'assistant']])} messages")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Ramachandra Udupa's AI Assistant**
    
    I'm powered by:
    - 🧠 Cerebras GPT-OSS-120B
    - 📚 RAG with my portfolio data
    - 🔍 Real-time information retrieval
    
    **Data Sources:**
    - My resume (PDF)
    - Portfolio websites
    - Project documentation
    """)
    
    st.header("💡 Try asking:")
    st.markdown("""
    - "Who are you?"
    - "Tell me about your projects"
    - "What's your experience with AI?"
    - "What services do you offer?"
    """)
    
    # Debug info
    with st.expander("🐛 Debug Info"):
        st.write(f"Thread ID: {st.session_state.get('thread_id', 'Not set')}")
        st.write(f"System Initialized: {st.session_state.get('system_initialized', False)}")
        st.write(f"Cache entries: {len(st.session_state.get('retrieval_cache', {}))}")
    
    st.markdown("---")
    st.caption("Built with Streamlit & LangGraph")