import os, sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PERSIST_DIR = "artifacts/chroma"
COLLECTION  = "college_docs_v1"
EMB_MODEL   = "text-embedding-3-small"
LLM_MODEL   = "gpt-3.5-turbo"

def _load_retriever(k=5, mmr=False):
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in .env"
    if not Path(PERSIST_DIR).exists():
        raise FileNotFoundError(f"No Chroma store at {PERSIST_DIR}. Build it first.")

    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(model=EMB_MODEL),
    )
    if mmr:
        return store.as_retriever(search_type="mmr",
                                  search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5})
    return store.as_retriever(search_kwargs={"k": k})

def _build_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        "Answer ONLY from the context. If not found, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    def format_docs(docs):
        return "\n\n".join(
            f"{d.page_content}\n\n[Source: {d.metadata.get('doc_id','?')} p{d.metadata.get('page','?')}]"
            for d in docs
        )
    return ({"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser())

def answer(q: str) -> str:
    retriever = _load_retriever(k=5)
    chain = _build_chain(retriever)
    return chain.invoke(q)

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Ask: ")
    print(answer(q))
