# remove: bs4, hub, WebBaseLoader, Chroma, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# NEW: minimal imports + backend switch
import os
from dotenv import load_dotenv
from retrieval.embeddings.chromadb import chromadb_embeddings
from retrieval.embeddings.qdrant import qdrant_retriever, qdrant_ingest

load_dotenv()
BACKEND = os.getenv("VBOT_VECTOR_DB", "chroma").lower()     # "chroma" | "qdrant"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "qdrant_store")
QDRANT_BUILD = os.getenv("VBOT_BUILD", "0") == "1"          # set 1 to (re)ingest before querying

# defining the prompt for the rag pipeline
prompt = ChatPromptTemplate.from_template(
    """You are an admin operator at VIT Chennai. Answer the student's question *only* using the context.
If the answer is not in the context, say you don't know and suggest where they might check (e.g., Admissions Office or Registrar).
Always include brief citations like [doc_id pX].

Context:
{context}

Question:
{question}"""
)

def format_docs(docs):
    parts = []
    for d in docs:
        doc_id = d.metadata.get("doc_id", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"{d.page_content}\n\n[Source: {doc_id} p{page}]")
    return "\n\n---\n\n".join(parts)

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# RETRIEVER: choose backend (minimal change to your original structure)
if BACKEND == "qdrant":
    if QDRANT_BUILD:
        qdrant_ingest(collection_name=QDRANT_COLLECTION)   # one-time (re)build when docs change
    retriever = qdrant_retriever(collection_name=QDRANT_COLLECTION, k=5)
else:
    retriever = chromadb_embeddings()  # your existing build+retriever path

# RAG chain (unchanged)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
print(rag_chain.invoke("What is the last date for fee submission for UG students?"))
