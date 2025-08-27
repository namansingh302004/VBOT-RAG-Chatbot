from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from ..chunking.text_splitter import splitchunks
import os
from dotenv import load_dotenv

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in .env"

def chromadb_embeddings():
    text_split_chunks = splitchunks()
    persist_directory = "artifacts/chroma"

    vectorstore = Chroma.from_documents(
        documents=text_split_chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="college_docs_v1",
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever
