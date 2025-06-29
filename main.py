import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- NEW: Import HuggingFace classes ---
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# --- App Configuration ---
st.set_page_config(page_title="Factful Health Chatbot", page_icon="🩺")
st.title("🩺 Factful Health Chatbot")
st.markdown("""
This chatbot is powered by a cloud-hosted, open-source LLM and provides answers based on information
from the World Health Organization (WHO) and the Centers for Disease Control and Prevention (CDC).

**Disclaimer:** This is an informational tool and not a substitute for professional medical advice.
""")

# --- NEW: Load API Key from Streamlit Secrets ---
# This will work both locally (from .streamlit/secrets.toml) and on Streamlit Cloud
try:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except FileNotFoundError:
    st.error("Hugging Face API token not found! Please add it to your Streamlit secrets.")
    st.stop()


# --- Core Functions (with Caching) ---

@st.cache_resource(show_spinner="Loading and Indexing Knowledge Base...")
def get_vectorstore_from_urls(urls):
    """
    Loads documents from URLs, splits them into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    """
    loader = WebBaseLoader(urls)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)

    # --- NEW: Use HuggingFaceInferenceAPIEmbeddings ---
    # This uses a free, high-quality sentence-transformer model
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    return vector_store

def get_context_retriever_chain(vector_store):
    """
    Creates a retriever chain that understands conversation history.
    """
    # --- NEW: Use HuggingFaceHub for the LLM ---
    # We are using Google's Gemma model, hosted on Hugging Face
    llm = HuggingFaceHub(
        repo_id="google/gemma-2b-it",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 1024},
        huggingfacehub_api_token=HF_TOKEN
    )
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    """
    Creates the main RAG chain for generating answers based on retrieved context.
    """
    # --- NEW: Use the same HuggingFaceHub LLM instance ---
    llm = HuggingFaceHub(
        repo_id="google/gemma-2b-it",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 1024},
        huggingfacehub_api_token=HF_TOKEN
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based ONLY on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_health_classifier_chain():
    """
    Creates a simple chain to classify if a question is health-related.
    """
    # --- NEW: Use the same HuggingFaceHub LLM instance ---
    llm = HuggingFaceHub(
        repo_id="google/gemma-2b-it",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 10}, # Smaller max tokens for classification
        huggingfacehub_api_token=HF_TOKEN
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly skilled classifier. Your only task is to determine if the following user question is related to human health, medicine, diseases, symptoms, or wellness. Respond with only 'yes' or 'no'."),
        ("user", "{input}")
    ])
    return prompt | llm

# --- Main Application Logic ---

FACTFUL_URLS = [
    "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    "https://www.cdc.gov/diabetes/basics/index.html",
    "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "https://www.cdc.gov/sleep/features/getting-enough-sleep.html"
]

# Create the vector store and chains
vector_store = get_vectorstore_from_urls(FACTFUL_URLS)
health_classifier_chain = get_health_classifier_chain()
context_retriever_chain = get_context_retriever_chain(vector_store)
conversational_rag_chain = get_conversational_rag_chain(context_retriever_chain)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a health chatbot. How can I help you today based on information from WHO and CDC?"),
    ]

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🩺"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="👤"):
            st.write(message.content)

# Get user input
user_query = st.chat_input("Ask a question about health...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human", avatar="👤"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        classification_result = health_classifier_chain.invoke({"input": user_query}).lower().strip()

        if "yes" in classification_result:
            response = conversational_rag_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": user_query
            })
            ai_response = response['answer']
        else:
            ai_response = "I can only answer questions related to human health. Please ask a different question."

    st.session_state.chat_history.append(AIMessage(content=ai_response))
    with st.chat_message("AI", avatar="🩺"):
        st.write(ai_response)
