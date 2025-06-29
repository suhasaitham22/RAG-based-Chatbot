# filename: main.py

import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from datasets import load_dataset
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- THIS IS THE CRITICAL CHANGE: IMPORT THE CORRECT CLASS ---
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings

# --- App Configuration ---
st.set_page_config(page_title="Factful Health Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Factful Health Chatbot")
st.markdown("""
This chatbot is powered by Google's Gemma model and provides answers based on data
from a cloud-native PubMed dataset.

**Disclaimer:** This is an informational tool and not a substitute for professional medical advice.
""")

# --- Load API Key AND Set Environment Variable ---
try:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
except (KeyError, FileNotFoundError):
    st.error("Hugging Face API token not found! Please add it to your Streamlit secrets.")
    st.stop()


# --- Core Functions (with Caching) ---
@st.cache_resource(show_spinner="Loading Data From Cloud-Native Medical Pool...")
def get_vectorstore_from_hf_dataset():
    """
    Loads a smaller, manageable slice of the dataset to fit within cloud memory limits.
    """
    dataset_name = "armanc/pubmed-rct20k"
    dataset = load_dataset(dataset_name, split="train[:1000]")

    documents = []
    for entry in dataset:
        page_content = entry.get("text", "")
        metadata = {"source": dataset_name}
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunked_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    return vector_store

# --- UPDATED LLM TO USE THE CORRECT ChatHuggingFace CLASS ---
def get_llm():
    """Returns an instance of the ChatHuggingFace model."""
    return ChatHuggingFace(
        repo_id="google/gemma-2b-it",
        temperature=0.1,
        max_new_tokens=1024
    )

def get_context_retriever_chain(_vector_store):
    llm = get_llm()
    retriever = _vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query...")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based ONLY on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_health_classifier_chain():
    """Creates the classifier chain with the correct ChatHuggingFace class."""
    llm = ChatHuggingFace(
        repo_id="google/gemma-2b-it",
        temperature=0.1,
        max_new_tokens=10
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly skilled classifier... Respond with only 'yes' or 'no'."),
        ("user", "{input}")
    ])
    return prompt | llm

# --- Main Application Logic ---
vector_store = get_vectorstore_from_hf_dataset()

health_classifier_chain = get_health_classifier_chain()
context_retriever_chain = get_context_retriever_chain(vector_store)
conversational_rag_chain = get_conversational_rag_chain(context_retriever_chain)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a health chatbot. How can I help you based on the PubMed dataset?"),
    ]

# The rest of the app logic remains unchanged...
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🩺"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="👤"):
            st.write(message.content)

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
