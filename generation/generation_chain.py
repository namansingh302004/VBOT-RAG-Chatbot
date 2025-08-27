# remove: bs4, hub, WebBaseLoader, Chroma, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from retrieval.embeddings.chromadb import chromadb_embeddings


#defining the prompt for the rag pipeline
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

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
retriever = chromadb_embeddings()  # returns a retriever

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Question
print(rag_chain.invoke("What is the last date for fee submission for UG students?"))
