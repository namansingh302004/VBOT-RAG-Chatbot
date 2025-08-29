from langchain_openai import OpenAIEmbeddings
from ..chunking.text_splitter import splitchunks
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import hashlib, os

load_dotenv()
API = os.getenv("QDRANT_API")
URL = os.getenv("QDRANT_URL")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def _stable_ids(docs):
    ids = []
    for i, d in enumerate(docs):
        base = f"{d.metadata.get('doc_id','')}-{d.metadata.get('page','')}-{i}-{d.page_content}"
        ids.append(hashlib.md5(base.encode("utf-8")).hexdigest())
    return ids

def qdrant_ingest(collection_name="qdrant_store", batch_size=16):
    docs = splitchunks()
    client = QdrantClient(url=URL, api_key=API, timeout=120.0)
    store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    ids = _stable_ids(docs)
    store.add_documents(docs, ids=ids, batch_size=batch_size)

def qdrant_retriever(collection_name="qdrant_store", k=5, mmr=False):
    client = QdrantClient(url=URL, api_key=API, timeout=120.0)
    store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    if mmr:
        return store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5})
    return store.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    qdrant_ingest()
