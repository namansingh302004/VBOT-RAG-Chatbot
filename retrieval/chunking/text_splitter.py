from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..document_handler.document_handler import load_all_pdfs

def splitchunks():
    pages = load_all_pdfs("docs")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    print(f"Loaded {len(pages)} pages, created {len(chunks)} chunks")
    print(chunks[0].page_content[:200])
    return chunks


