import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize LLM and embeddings
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
<context>

Question: {input}
""")

# --- Streamlit UI ---
st.set_page_config(page_title="Insurance Chatbot - PDF RAG", page_icon="🤖")
st.title("🤖 AI-Powered Insurance Q&A Chatbot")
st.markdown("Ask questions about insurance policies based on uploaded PDFs.")

# --- PDF Upload Section ---
pdf_dir = "pdf"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

uploaded_files = st.file_uploader("📄 Upload insurance policy PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(pdf_dir, file.name), "wb") as f:
            f.write(file.read())
    st.success(f"✅ Uploaded {len(uploaded_files)} file(s) successfully.")

# --- Vector Embedding Function ---
def create_vector_database():
    st.info("🔄 Creating vector embeddings. Please wait...")

    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    if not docs:
        st.error("❌ No documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("⚠️ No content extracted. Check if PDFs are scanned images.")
        return

    vector_store = FAISS.from_documents(chunks, embeddings)
    st.session_state.vectors = vector_store
    st.session_state.docs = docs
    st.success(f"✅ Indexed {len(docs)} document(s) and {len(chunks)} chunks.")

# Button to generate embeddings
if st.button("📥 Build Knowledge Base"):
    create_vector_database()

# --- Chat Interface ---
query = st.text_input("💬 Ask a question about the uploaded insurance policies:")

if query:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please build the knowledge base first.")
    else:
        # Retrieval Chain
        retriever = st.session_state.vectors.as_retriever()
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": query})
        end = time.process_time()

        # --- Output ---
        st.markdown("### 🧠 Response:")
        st.success(response["answer"])

        st.caption(f"⏱️ Answer generated in {round(end - start, 2)} seconds")

        # Optional: Show supporting chunks
        with st.expander("🔍 View Source Chunks"):
            for i, doc in enumerate(response.get("context", [])):
                st.markdown(f"**📄 Chunk {i+1}:**")
                st.write(doc.page_content)
                st.divider()
